import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
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

    test_mask_dup = torch.cat((test_mask, test_mask), dim=0)
    test_hyper_node = hyper_node.clone().detach()
    test_ve_affiliation = ve_affiliation.clone().detach()
    test_hyper_node = test_hyper_node[test_mask_dup]
    test_ve_affiliation = test_ve_affiliation[:, test_mask_dup]
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


def build_fk_row_indices(
    table_df: pd.DataFrame,
    table_data: Dict[str, Dict],
    fkey_map: Dict[str, str],
) -> Dict[str, np.ndarray]:
    fk_row_indices: Dict[str, np.ndarray] = {}
    for fk_col, target_table in fkey_map.items():
        target_meta = table_data.get(target_table)
        if target_meta is None:
            continue
        target_df = target_meta["df"]
        target_pkey = target_meta["pkey_col"]
        if target_pkey is None or target_pkey not in target_df.columns:
            continue
        pkey_to_row = {val: idx for idx, val in enumerate(target_df[target_pkey].tolist())}
        fk_vals = table_df[fk_col].tolist()
        idxs = []
        for val in fk_vals:
            if pd.isna(val):
                idxs.append(-1)
            else:
                idxs.append(pkey_to_row.get(val, -1))
        fk_row_indices[fk_col] = np.array(idxs, dtype=int)
    return fk_row_indices


def compute_row_embeddings(
    model: nn.Module,
    hyperedge: torch.Tensor,
    hyper_node: torch.Tensor,
    ve_affiliation: torch.Tensor,
    n_row: int,
    use_known_mask: bool,
    known: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_known_mask:
        known_mask = produce_NA(
            hyper_node[: int(hyper_node.shape[0] / 2)],
            p_miss=1 - known,
            mecha="Random",
        )
        known_mask_dup = torch.cat((known_mask, known_mask), dim=0)
        hyper_node = hyper_node[known_mask_dup]
        ve_affiliation = ve_affiliation[:, known_mask_dup]

    embedding, _ = model(hyperedge, hyper_node, ve_affiliation)
    row_emb = embedding[:n_row]
    col_emb = embedding[n_row:]
    return row_emb, col_emb


def apply_fk_hard_propagation(
    row_self: torch.Tensor,
    col_self: torch.Tensor,
    table_meta: Dict,
    other_row_embs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    fk_indices = table_meta["fk_row_indices"]
    if not fk_indices:
        return torch.cat([row_self, col_self], dim=0)

    device = row_self.device
    row_fk = torch.zeros_like(row_self, device=device)
    counts = torch.zeros(row_self.shape[0], device=device)

    for fk_col, idxs in fk_indices.items():
        target_table = table_meta["fkey_map"].get(fk_col)
        if not target_table:
            continue
        target_rows = other_row_embs.get(target_table)
        if target_rows is None:
            continue
        idxs_tensor = torch.tensor(idxs, device=device)
        valid = idxs_tensor >= 0
        if not torch.any(valid):
            continue
        row_fk[valid] += target_rows[idxs_tensor[valid]]
        counts[valid] += 1.0

    valid_counts = counts > 0
    if torch.any(valid_counts):
        row_fk[valid_counts] = row_fk[valid_counts] / counts[valid_counts].unsqueeze(1)

    fk_mean = row_fk.mean(dim=0, keepdim=True)
    fk_std = row_fk.std(dim=0, keepdim=True).clamp_min(1e-6)
    row_fk = (row_fk - fk_mean) / fk_std

    gate_layer = table_meta.get("fk_gate")
    if gate_layer is not None:
        gate_in = torch.cat([row_self, row_fk], dim=1)
        gate = torch.sigmoid(gate_layer(gate_in))
        row_new = row_self + gate * row_fk
    else:
        row_new = row_self + row_fk
    return torch.cat([row_new, col_self], dim=0)


def init_models(args, table_meta: Dict, device: torch.device) -> None:
    hyperedge, train_hyper_node, *_ = table_meta["data"]
    model = get_gen_imp(hyperedge.shape[1], train_hyper_node.shape[1], args).to(device)
    input_dim = args.hyperedge_dim_hidden * 2
    impute_model = LinearHead(
        input_dim,
        1,
        hidden_layer_sizes=hyperedge.shape[1],
        hidden_activation=args.impute_activation,
        dropout=args.dropout,
    ).to(device)
    gate_layer = nn.Linear(2 * args.hyperedge_dim_hidden, args.hyperedge_dim_hidden).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(impute_model.parameters()) + list(gate_layer.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    table_meta["model"] = model
    table_meta["impute_model"] = impute_model
    table_meta["fk_gate"] = gate_layer
    table_meta["optimizer"] = optimizer


def train_all(
    args,
    all_table_meta: Dict[str, Dict],
    device: torch.device,
) -> None:
    huber = nn.HuberLoss(delta=1.0)
    for epoch in range(args.epochs):
        if epoch % args.log_every == 0:
            print(f"epoch {epoch}")
        for table_name, table_meta in all_table_meta.items():
            model = table_meta["model"]
            impute_model = table_meta["impute_model"]
            optimizer = table_meta["optimizer"]
            (
                hyperedge,
                train_hyper_node,
                train_ve_affiliation,
                train_labels,
                _,
                _,
                _,
            ) = table_meta["data"]

            model.train()
            impute_model.train()
            optimizer.zero_grad()

            hyperedge = hyperedge.to(device)
            train_hyper_node = train_hyper_node.to(device)
            train_ve_affiliation = train_ve_affiliation.to(device)
            train_labels = train_labels.to(device)

            row_self, col_self = compute_row_embeddings(
                model,
                hyperedge,
                train_hyper_node,
                train_ve_affiliation,
                table_meta["n_row"],
                use_known_mask=True,
                known=args.known,
            )

            other_row_embs: Dict[str, torch.Tensor] = {}
            for fk_col, target_table in table_meta["fkey_map"].items():
                target_meta = all_table_meta.get(target_table)
                if target_meta is None or target_table in other_row_embs:
                    continue
                tgt_model = target_meta["model"]
                tgt_hyperedge, tgt_train_hn, tgt_train_ve, *_ = target_meta["data"]
                tgt_row, _ = compute_row_embeddings(
                    tgt_model,
                    tgt_hyperedge.to(device),
                    tgt_train_hn.to(device),
                    tgt_train_ve.to(device),
                    target_meta["n_row"],
                    use_known_mask=False,
                    known=args.known,
                )
                other_row_embs[target_table] = tgt_row.detach()

            embedding = apply_fk_hard_propagation(
                row_self,
                col_self,
                table_meta,
                other_row_embs,
            )

            half = int(train_hyper_node.shape[0] / 2)
            pred = impute_model(
                [embedding[train_ve_affiliation[0, :half]], embedding[train_ve_affiliation[1, :half]]],
                token_emb=[],
            )
            pred_train = pred[:half, 0]
            loss = huber(pred_train, train_labels)
            loss.backward()
            optimizer.step()


def evaluate_table(
    args,
    table_meta: Dict,
    all_table_meta: Dict[str, Dict],
    device: torch.device,
) -> Tuple[float, float, int]:
    model = table_meta["model"]
    impute_model = table_meta["impute_model"]
    (
        hyperedge,
        train_hyper_node,
        train_ve_affiliation,
        _,
        test_hyper_node,
        test_ve_affiliation,
        test_labels,
    ) = table_meta["data"]

    model.eval()
    impute_model.eval()
    with torch.no_grad():
        row_self, col_self = compute_row_embeddings(
            model,
            hyperedge.to(device),
            train_hyper_node.to(device),
            train_ve_affiliation.to(device),
            table_meta["n_row"],
            use_known_mask=False,
            known=args.known,
        )
        other_row_embs: Dict[str, torch.Tensor] = {}
        for fk_col, target_table in table_meta["fkey_map"].items():
            target_meta = all_table_meta.get(target_table)
            if target_meta is None or target_table in other_row_embs:
                continue
            tgt_model = target_meta["model"]
            tgt_hyperedge, tgt_train_hn, tgt_train_ve, *_ = target_meta["data"]
            tgt_row, _ = compute_row_embeddings(
                tgt_model,
                tgt_hyperedge.to(device),
                tgt_train_hn.to(device),
                tgt_train_ve.to(device),
                target_meta["n_row"],
                use_known_mask=False,
                known=args.known,
            )
            other_row_embs[target_table] = tgt_row

        embedding = apply_fk_hard_propagation(
            row_self,
            col_self,
            table_meta,
            other_row_embs,
        )

        pred = impute_model(
            [embedding[test_ve_affiliation[0].to(device)], embedding[test_ve_affiliation[1].to(device)]],
            token_emb=[],
        )
        pred_test = pred[: int(test_hyper_node.shape[0] / 2), 0]
        valid = torch.isfinite(test_labels.to(device))
        if valid.sum() == 0:
            return float("nan"), float("nan"), 0
        pred_test = pred_test[valid]
        label_test = test_labels.to(device)[valid]
        mse = F.mse_loss(pred_test, label_test)
        rmse = float(torch.sqrt(mse).item())
        mae = float(F.l1_loss(pred_test, label_test).item())

    return rmse, mae, int(valid.sum().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--missing_ratio", type=float, default=0.3)
    parser.add_argument("--missing_mechanism", type=str, default="MCAR")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--known", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)

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

    table_meta: Dict[str, Dict] = {}
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
        mask_cols = np.ones(values.shape[1], dtype=bool)
        for idx, col in enumerate(feature_df.columns):
            if table.feature_types.get(col) == "fkey":
                mask_cols[idx] = False

        data = get_data_from_table(values, args.missing_ratio, args.missing_mechanism, args.seed, mask_cols=mask_cols)
        table_meta[table_name] = {
            "name": table_name,
            "df": table.df,
            "feature_df": feature_df,
            "pkey_col": table.pkey_col,
            "fkey_map": table.fkey_col_to_pkey_table,
            "feature_types": table.feature_types,
            "data": data,
            "n_row": values.shape[0],
            "n_col": values.shape[1],
        }

    for table_name, meta in table_meta.items():
        meta["fk_row_indices"] = build_fk_row_indices(
            meta["df"],
            table_meta,
            meta["fkey_map"],
        )

    for table_name, meta in table_meta.items():
        init_models(args, meta, device)

    train_all(args, table_meta, device)

    rmse_list: List[float] = []
    mae_list: List[float] = []
    count_list: List[int] = []

    for table_name, meta in table_meta.items():
        rmse, mae, count = evaluate_table(args, meta, table_meta, device)
        rmse_list.append(rmse)
        mae_list.append(mae)
        count_list.append(count)
        print(f"{table_name}: rmse={rmse:.4f} mae={mae:.4f}")

    if rmse_list:
        total = sum(count_list)
        if total > 0:
            avg_rmse = float(np.sum(np.array(rmse_list) * np.array(count_list)) / total)
            avg_mae = float(np.sum(np.array(mae_list) * np.array(count_list)) / total)
            print(f"AVG: rmse={avg_rmse:.4f} mae={avg_mae:.4f}")
        else:
            avg_rmse = float(np.mean(rmse_list))
            avg_mae = float(np.mean(mae_list))
            print(f"AVG: rmse={avg_rmse:.4f} mae={avg_mae:.4f}")
    else:
        print("No tables evaluated.")


if __name__ == "__main__":
    main()
