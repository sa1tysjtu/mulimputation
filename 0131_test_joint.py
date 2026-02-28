import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from relbench_loader import load_relbench_tables
from utils_core import produce_NA
from models.gen_imp import get_gen_imp
from models.imputaion_model import LinearHead


def normalize_table(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    for col_idx in range(out.shape[1]):
        col = out[:, col_idx]
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        min_val = np.min(col[valid])
        max_val = np.max(col[valid])
        if max_val == min_val:
            col[valid] = 0.0
        else:
            col[valid] = (col[valid] - min_val) / (max_val - min_val)
        out[:, col_idx] = col
    return out


def build_masks(
    values: np.ndarray,
    missing_ratio: float,
    missing_mechanism: str,
    mask_cols: np.ndarray | None = None,
) -> torch.Tensor:
    n_row, n_col = values.shape
    mask = produce_NA(
        torch.tensor(values, dtype=torch.float),
        p_miss=missing_ratio,
        mecha=missing_mechanism,
        n_row=n_row,
        n_col=n_col,
    ).view(-1)

    missing = np.isnan(values)
    if missing.any():
        mask = mask & ~torch.tensor(missing.reshape(-1), dtype=torch.bool)
    if mask_cols is not None:
        mask_cols = mask_cols.astype(bool)
        keep = np.repeat(mask_cols, n_row)
        mask = mask & torch.tensor(keep, dtype=torch.bool)
    return mask


def create_edge_node(nrow: int, ncol: int, n_target: int = 32) -> torch.Tensor:
    if ncol < 32:
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol, n_target))
        feature_node[np.arange(ncol), feature_ind + 1] = 1
    else:
        feature_node = np.zeros((ncol, n_target))
        for i in range(ncol):
            feature_node[i, i % n_target] = 1
            feature_node[i, (i + n_target // 2) % n_target] = 1
    sample_node = np.zeros((nrow, n_target))
    sample_node[:, 0] = 1
    node = sample_node.tolist() + feature_node.tolist()
    return torch.tensor(node, dtype=torch.float)


def create_value_node(values: np.ndarray) -> torch.Tensor:
    nrow, ncol = values.shape
    value_node = []
    for i in range(nrow):
        for j in range(ncol):
            value_node.append([values[i, j]])
    value_node = value_node + value_node
    return torch.tensor(value_node, dtype=torch.float)


def create_VE_affiliation(n_row: int, n_col: int) -> torch.Tensor:
    start = []
    end = []
    for x in range(n_row):
        start = start + [x] * n_col
        end = end + list(n_row + np.arange(n_col))
    start_dup = start + end
    end_dup = end + start
    return torch.tensor([start_dup, end_dup], dtype=int)


def get_data_from_table(
    values: np.ndarray,
    missing_ratio: float,
    missing_mechanism: str,
    seed: int,
    mask_cols: np.ndarray | None,
) -> Tuple:
    n_row, n_col = values.shape
    hyperedge = create_edge_node(n_row, n_col)
    values_filled = np.nan_to_num(values, nan=0.0)
    hyper_node = create_value_node(values_filled)
    ve_affiliation = create_VE_affiliation(n_row, n_col)

    torch.manual_seed(seed)
    train_mask = build_masks(values, missing_ratio, missing_mechanism, mask_cols=mask_cols)
    train_mask_dup = torch.cat((train_mask, train_mask), dim=0)
    test_mask = ~train_mask

    train_hyper_node = hyper_node.clone().detach()
    train_ve_affiliation = ve_affiliation.clone().detach()
    train_hyper_node = train_hyper_node[train_mask_dup]
    train_ve_affiliation = train_ve_affiliation[:, train_mask_dup]

    label_all = torch.tensor(values.reshape(-1), dtype=torch.float)
    train_labels = label_all[train_mask]

    test_hyper_node = hyper_node.clone().detach()
    test_ve_affiliation = ve_affiliation.clone().detach()
    test_hyper_node = test_hyper_node[~train_mask_dup]
    test_ve_affiliation = test_ve_affiliation[:, ~train_mask_dup]
    test_labels = label_all[test_mask]

    return (
        hyperedge,
        train_hyper_node,
        train_ve_affiliation,
        train_labels,
        test_hyper_node,
        test_ve_affiliation,
        test_labels,
    )


def train_joint(
    args,
    table_data: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    model = get_gen_imp(32, 1, args).to(device)
    input_dim = args.hyperedge_dim_hidden * 2
    impute_model = LinearHead(
        input_dim,
        1,
        hidden_layer_sizes=32,
        hidden_activation=args.impute_activation,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(impute_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    huber = nn.HuberLoss(delta=1.0)

    for _ in range(args.epochs):
        model.train()
        impute_model.train()
        for data in table_data:
            optimizer.zero_grad()

            train_hyper_node = data["train_hyper_node"]
            train_ve_affiliation = data["train_ve_affiliation"]
            train_labels = data["train_labels"]
            hyperedge = data["hyperedge"]

            known_mask = produce_NA(
                train_hyper_node[: int(train_hyper_node.shape[0] / 2)],
                p_miss=1 - args.known,
                mecha="Random",
            )
            known_mask_dup = torch.cat((known_mask, known_mask), dim=0)
            known_hyper_node = train_hyper_node[known_mask_dup]
            known_ve_affiliation = train_ve_affiliation[:, known_mask_dup]

            embedding, _ = model(hyperedge, known_hyper_node, known_ve_affiliation)
            half = int(train_hyper_node.shape[0] / 2)
            pred = impute_model(
                [embedding[train_ve_affiliation[0, :half]], embedding[train_ve_affiliation[1, :half]]],
                token_emb=[],
            )
            pred_train = pred[:half, 0]
            loss = huber(pred_train, train_labels)
            loss.backward()
            optimizer.step()

    return model, impute_model


def evaluate_tables(
    model: nn.Module,
    impute_model: nn.Module,
    table_data: List[Dict[str, torch.Tensor]],
) -> Tuple[List[float], List[float], List[int], List[str]]:
    rmse_list: List[float] = []
    mae_list: List[float] = []
    count_list: List[int] = []
    name_list: List[str] = []

    model.eval()
    impute_model.eval()
    with torch.no_grad():
        for data in table_data:
            hyperedge = data["hyperedge"]
            train_hyper_node = data["train_hyper_node"]
            train_ve_affiliation = data["train_ve_affiliation"]
            test_hyper_node = data["test_hyper_node"]
            test_ve_affiliation = data["test_ve_affiliation"]
            test_labels = data["test_labels"]

            embedding, _ = model(hyperedge, train_hyper_node, train_ve_affiliation)
            pred = impute_model(
                [embedding[test_ve_affiliation[0]], embedding[test_ve_affiliation[1]]],
                token_emb=[],
            )
            pred_test = pred[: int(test_hyper_node.shape[0] / 2), 0]
            label_test = test_labels
            valid = torch.isfinite(label_test)
            if valid.sum() == 0:
                rmse = float("nan")
                mae = float("nan")
                count = 0
            else:
                pred_test = pred_test[valid]
                label_test = label_test[valid]
                mse = F.mse_loss(pred_test, label_test)
                rmse = float(torch.sqrt(mse).item())
                mae = float(F.l1_loss(pred_test, label_test).item())
                count = int(valid.sum().item())

            rmse_list.append(rmse)
            mae_list.append(mae)
            count_list.append(count)
            name_list.append(data["name"])

    return rmse_list, mae_list, count_list, name_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--missing_ratio", type=float, default=0.3)
    parser.add_argument("--missing_mechanism", type=str, default="MCAR")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--known", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mask_fk", type=int, default=1)

    parser.add_argument("--hyperedge_dim_hidden", type=int, default=64)
    parser.add_argument("--hyper_node_dim_hidden", type=int, default=64)
    parser.add_argument("--gnn_layer_num", type=int, default=3)
    parser.add_argument("--gnn_activation", type=str, default="relu")
    parser.add_argument("--impute_activation", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    tables = load_relbench_tables(
        dataset_name=args.dataset,
        drop_text_cols=True,
        drop_time_col=True,
    )

    table_data: List[Dict[str, torch.Tensor]] = []
    for table_name, table in tables.items():
        feature_df = table.feature_df
        if feature_df.empty:
            print(f"skip {table_name}: zero features")
            continue
        values = feature_df.to_numpy(dtype=float, na_value=np.nan)
        values = normalize_table(values)
        if np.isnan(values).all():
            print(f"skip {table_name}: all missing")
            continue
        mask_cols = None
        if args.mask_fk == 0:
            mask_cols = np.ones(values.shape[1], dtype=bool)
            for idx, col in enumerate(feature_df.columns):
                if table.feature_types.get(col) == "fkey":
                    mask_cols[idx] = False

        (
            hyperedge,
            train_hyper_node,
            train_ve_affiliation,
            train_labels,
            test_hyper_node,
            test_ve_affiliation,
            test_labels,
        ) = get_data_from_table(values, args.missing_ratio, args.missing_mechanism, args.seed, mask_cols)

        table_data.append(
            {
                "name": table_name,
                "hyperedge": hyperedge.to(device),
                "train_hyper_node": train_hyper_node.to(device),
                "train_ve_affiliation": train_ve_affiliation.to(device),
                "train_labels": train_labels.to(device),
                "test_hyper_node": test_hyper_node.to(device),
                "test_ve_affiliation": test_ve_affiliation.to(device),
                "test_labels": test_labels.to(device),
            }
        )

    if not table_data:
        print("No tables evaluated.")
        return

    model, impute_model = train_joint(args, table_data, device)
    rmse_list, mae_list, count_list, name_list = evaluate_tables(model, impute_model, table_data)

    for name, rmse, mae in zip(name_list, rmse_list, mae_list):
        print(f"{name}: rmse={rmse:.4f} mae={mae:.4f}")

    total = sum(count_list)
    if total > 0:
        avg_rmse = float(np.sum(np.array(rmse_list) * np.array(count_list)) / total)
        avg_mae = float(np.sum(np.array(mae_list) * np.array(count_list)) / total)
        print(f"AVG: rmse={avg_rmse:.4f} mae={avg_mae:.4f}")
    else:
        avg_rmse = float(np.mean(rmse_list))
        avg_mae = float(np.mean(mae_list))
        print(f"AVG: rmse={avg_rmse:.4f} mae={avg_mae:.4f}")


if __name__ == "__main__":
    main()
