from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from utils_core import produce_NA


@dataclass
class TableGraph:
    name: str
    num_rows: int
    num_cols: int
    feature_cols: List[str]
    feature_types: Dict[str, str]
    pkey_col: Optional[str]
    fkey_col_to_pkey_table: Dict[str, str]
    numeric_stats: Dict[str, tuple]
    hyperedge: torch.Tensor
    hyper_node: torch.Tensor
    ve_affiliation: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor


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


def build_table_graph(
    name: str,
    feature_df: pd.DataFrame,
    feature_types: Dict[str, str],
    pkey_col: Optional[str],
    fkey_col_to_pkey_table: Dict[str, str],
    missing_ratio: float,
    missing_mechanism: str,
    seed: int = 0,
) -> TableGraph:
    if feature_df.empty:
        raise ValueError(f"Table {name} has no usable feature columns after filtering.")

    numeric_stats: Dict[str, tuple] = {}
    for col, col_type in feature_types.items():
        if col_type != "numerical":
            continue
        series = feature_df[col]
        if series.isna().all():
            continue
        min_val = series.min()
        max_val = series.max()
        numeric_stats[col] = (float(min_val), float(max_val))
        if max_val != min_val:
            feature_df[col] = (series - min_val) / (max_val - min_val)

    nrow, ncol = feature_df.shape
    numeric_values = feature_df.to_numpy(dtype=float, na_value=np.nan)
    hyperedge = create_edge_node(nrow, ncol)
    hyper_node = create_value_node(numeric_values)
    ve_affiliation = create_VE_affiliation(nrow, ncol)

    torch.manual_seed(seed)

    raw_values = numeric_values
    nan_mask = pd.isna(raw_values).reshape(-1)

    if missing_mechanism != "MCAR":
        train_mask = produce_NA(
            torch.tensor(raw_values, dtype=torch.float),
            p_miss=missing_ratio,
            mecha=missing_mechanism,
            n_row=nrow,
            n_col=ncol,
        ).view(-1)
        if nan_mask.any():
            train_mask = train_mask & ~torch.tensor(nan_mask, dtype=torch.bool)
        remaining = ~train_mask
        val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        test_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        rem_idx = remaining.nonzero(as_tuple=False).view(-1)
        if rem_idx.numel() > 0:
            perm = torch.randperm(rem_idx.numel())
            split = rem_idx.numel() // 2
            val_mask[rem_idx[perm[:split]]] = True
            test_mask[rem_idx[perm[split:]]] = True
    else:
        rng = np.random.default_rng(seed)
        rand_vals = rng.uniform(0.0, 1.0, size=(nrow * ncol))
        train_mask = rand_vals < (1.0 - missing_ratio)
        remaining = ~train_mask
        val_mask = np.zeros_like(train_mask, dtype=bool)
        test_mask = np.zeros_like(train_mask, dtype=bool)
        rem_idx = np.where(remaining)[0]
        if rem_idx.size > 0:
            rng.shuffle(rem_idx)
            split = rem_idx.size // 2
            val_mask[rem_idx[:split]] = True
            test_mask[rem_idx[split:]] = True
        if nan_mask.any():
            train_mask = train_mask & ~nan_mask
            val_mask = val_mask & ~nan_mask
            test_mask = test_mask & ~nan_mask
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        val_mask = torch.tensor(val_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)

    return TableGraph(
        name=name,
        num_rows=nrow,
        num_cols=ncol,
        feature_cols=list(feature_df.columns),
        feature_types=feature_types,
        pkey_col=pkey_col,
        fkey_col_to_pkey_table=fkey_col_to_pkey_table,
        numeric_stats=numeric_stats,
        hyperedge=hyperedge,
        hyper_node=hyper_node,
        ve_affiliation=ve_affiliation,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
