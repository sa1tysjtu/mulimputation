import argparse
from typing import List, Tuple

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
    if mask_cols is not None:
        keep = np.repeat(mask_cols.astype(bool), n_row)
        keep = torch.tensor(keep, dtype=torch.bool)
        train_mask = train_mask & keep
        train_mask_dup = torch.cat((train_mask, train_mask), dim=0)
        test_mask = (~train_mask) & keep

    train_hyper_node = hyper_node.clone().detach()
    train_ve_affiliation = ve_affiliation.clone().detach()
    train_hyper_node = train_hyper_node[train_mask_dup]
    train_ve_affiliation = train_ve_affiliation[:, train_mask_dup]

    label_all = torch.tensor(values.reshape(-1), dtype=torch.float)
    train_labels = label_all[train_mask]

    test_mask_dup = torch.cat((test_mask, test_mask), dim=0)
    test_hyper_node = hyper_node.clone().detach()
    test_ve_affiliation = ve_affiliation.clone().detach()
    test_hyper_node = test_hyper_node[test_mask_dup]
    test_ve_affiliation = test_ve_affiliation[:, test_mask_dup]
    test_labels = label_all[test_mask]
    test_flat_indices = torch.nonzero(test_mask, as_tuple=False).view(-1)

    return (
        hyperedge,
        train_hyper_node,
        train_ve_affiliation,
        train_labels,
        test_hyper_node,
        test_ve_affiliation,
        test_labels,
        test_flat_indices,
    )


def train_table(
    args,
    values: np.ndarray,
    device: torch.device,
    mask_cols: np.ndarray | None,
    fk_col_mask: np.ndarray,
) -> dict:
    (
        hyperedge,
        train_hyper_node,
        train_ve_affiliation,
        train_labels,
        test_hyper_node,
        test_ve_affiliation,
        test_labels,
        test_flat_indices,
    ) = get_data_from_table(values, args.missing_ratio, args.missing_mechanism, args.seed, mask_cols)

    model = get_gen_imp(hyperedge.shape[1], train_hyper_node.shape[1], args).to(device)
    input_dim = args.hyperedge_dim_hidden * 2
    impute_model = LinearHead(
        input_dim,
        1,
        hidden_layer_sizes=hyperedge.shape[1],
        hidden_activation=args.impute_activation,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(impute_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    huber = nn.HuberLoss(delta=1.0)

    train_hyper_node = train_hyper_node.to(device)
    hyperedge = hyperedge.to(device)
    train_ve_affiliation = train_ve_affiliation.to(device)
    train_labels = train_labels.to(device)
    test_ve_affiliation = test_ve_affiliation.to(device)

    for _ in range(args.epochs):
        model.train()
        impute_model.train()
        optimizer.zero_grad()

        known_mask = produce_NA(
            train_hyper_node[: int(train_hyper_node.shape[0] / 2)],
            p_miss=1 - args.known,
            mecha="Random",
        )
        known_mask_dup = torch.cat((known_mask, known_mask), dim=0)
        known_hyper_node = train_hyper_node.clone().detach()
        known_ve_affiliation = train_ve_affiliation.clone().detach()
        known_hyper_node = known_hyper_node[known_mask_dup]
        known_ve_affiliation = known_ve_affiliation[:, known_mask_dup]

        embedding, _ = model(hyperedge, known_hyper_node, known_ve_affiliation)
        pred = impute_model(
            [
                embedding[train_ve_affiliation[0, : int(train_hyper_node.shape[0] / 2)]],
                embedding[train_ve_affiliation[1, : int(train_hyper_node.shape[0] / 2)]],
            ],
            token_emb=[],
        )
        pred_train = pred[: int(train_hyper_node.shape[0] / 2), 0]
        loss = huber(pred_train, train_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    impute_model.eval()
    with torch.no_grad():
        embedding, _ = model(hyperedge, train_hyper_node, train_ve_affiliation)
        pred = impute_model(
            [embedding[test_ve_affiliation[0]], embedding[test_ve_affiliation[1]]],
            token_emb=[],
        )
        pred_test = pred[: int(test_hyper_node.shape[0] / 2), 0]
        label_test = test_labels.to(device)
        valid = torch.isfinite(label_test)
        if valid.sum() == 0:
            return {
                "all": (float("nan"), float("nan"), 0),
                "non_fk": (float("nan"), float("nan"), 0),
                "fk": (float("nan"), float("nan"), 0),
            }

        n_row, n_col = values.shape
        col_ids = (test_flat_indices % n_col).to(device)
        fk_col_mask_t = torch.tensor(fk_col_mask, dtype=torch.bool, device=device)
        is_fk_entry = fk_col_mask_t[col_ids]

        def metric_for(mask: torch.Tensor) -> Tuple[float, float, int]:
            m = valid & mask
            c = int(m.sum().item())
            if c == 0:
                return float("nan"), float("nan"), 0
            p = pred_test[m]
            y = label_test[m]
            mse = F.mse_loss(p, y)
            rmse = float(torch.sqrt(mse).item())
            mae = float(F.l1_loss(p, y).item())
            return rmse, mae, c

        all_mask = torch.ones_like(valid, dtype=torch.bool)
        non_fk_mask = ~is_fk_entry
        fk_mask = is_fk_entry

        metrics = {
            "all": metric_for(all_mask),
            "non_fk": metric_for(non_fk_mask),
            "fk": metric_for(fk_mask),
        }

    return metrics


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

    rmse_all_list: List[float] = []
    mae_all_list: List[float] = []
    count_all_list: List[int] = []
    rmse_non_fk_list: List[float] = []
    mae_non_fk_list: List[float] = []
    count_non_fk_list: List[int] = []
    rmse_fk_list: List[float] = []
    mae_fk_list: List[float] = []
    count_fk_list: List[int] = []

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
        fk_col_mask = np.array([table.feature_types.get(col) == "fkey" for col in feature_df.columns], dtype=bool)
        mask_cols = None
        if args.mask_fk == 0:
            mask_cols = np.ones(values.shape[1], dtype=bool)
            for idx, col in enumerate(feature_df.columns):
                if table.feature_types.get(col) == "fkey":
                    mask_cols[idx] = False

        metrics = train_table(args, values, device, mask_cols, fk_col_mask)
        rmse_all, mae_all, cnt_all = metrics["all"]
        rmse_non_fk, mae_non_fk, cnt_non_fk = metrics["non_fk"]
        rmse_fk, mae_fk, cnt_fk = metrics["fk"]

        rmse_all_list.append(rmse_all)
        mae_all_list.append(mae_all)
        count_all_list.append(cnt_all)
        if cnt_non_fk > 0:
            rmse_non_fk_list.append(rmse_non_fk)
            mae_non_fk_list.append(mae_non_fk)
            count_non_fk_list.append(cnt_non_fk)
        if cnt_fk > 0:
            rmse_fk_list.append(rmse_fk)
            mae_fk_list.append(mae_fk)
            count_fk_list.append(cnt_fk)

        print(
            f"{table_name}: all_rmse={rmse_all:.4f} all_mae={mae_all:.4f} "
            f"non_fk_rmse={rmse_non_fk:.4f} non_fk_mae={mae_non_fk:.4f} "
            f"fk_rmse={rmse_fk:.4f} fk_mae={mae_fk:.4f}"
        )

    def print_weighted_avg(tag: str, rmse_list: List[float], mae_list: List[float], count_list: List[int]) -> None:
        if not rmse_list:
            print(f"{tag}: no cells evaluated")
            return
        total = sum(count_list)
        if total > 0:
            avg_rmse = float(np.sum(np.array(rmse_list) * np.array(count_list)) / total)
            avg_mae = float(np.sum(np.array(mae_list) * np.array(count_list)) / total)
            print(f"{tag}: rmse={avg_rmse:.4f} mae={avg_mae:.4f}")
        else:
            avg_rmse = float(np.mean(rmse_list))
            avg_mae = float(np.mean(mae_list))
            print(f"{tag}: rmse={avg_rmse:.4f} mae={avg_mae:.4f}")

    if rmse_all_list:
        print_weighted_avg("AVG_ALL", rmse_all_list, mae_all_list, count_all_list)
        print_weighted_avg("AVG_NON_FK", rmse_non_fk_list, mae_non_fk_list, count_non_fk_list)
        print_weighted_avg("AVG_FK", rmse_fk_list, mae_fk_list, count_fk_list)
    else:
        print("No tables evaluated.")


if __name__ == "__main__":
    main()
