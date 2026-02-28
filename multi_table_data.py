from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from relbench_loader import RelBenchTable, load_relbench_tables
from table_graph import TableGraph, build_table_graph


@dataclass
class FKRelation:
    src_table: str
    src_row: int
    fk_col: str
    dst_table: str
    dst_row: int


@dataclass
class MissingFK:
    src_table: str
    src_row: int
    fk_col: str
    dst_table: str


@dataclass
class MultiTableData:
    tables: Dict[str, RelBenchTable]
    graphs: Dict[str, TableGraph]
    observed_fk: List[FKRelation]
    missing_fk: List[MissingFK]


def _normalize_key(value):
    """Canonicalize pkey/fkey scalar values for robust pkey<->fkey matching."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        fv = float(value)
        if fv.is_integer():
            return int(fv)
        return fv
    if isinstance(value, int):
        return value
    return value


def _build_pkey_to_row_index(tables: Dict[str, RelBenchTable]) -> Dict[str, Dict]:
    """Build mapping: table_name -> {pkey_value -> row_position}."""
    out: Dict[str, Dict] = {}
    for table_name, table in tables.items():
        df = table.df
        pkey_col = table.pkey_col
        if pkey_col and pkey_col in df.columns:
            mapping: Dict = {}
            for row_pos, pkey_value in enumerate(df[pkey_col].tolist()):
                key = _normalize_key(pkey_value)
                if key is None:
                    continue
                mapping.setdefault(key, row_pos)
            out[table_name] = mapping
        else:
            out[table_name] = {row_pos: row_pos for row_pos in range(len(df))}
    return out


def _collect_fk_relations(
    tables: Dict[str, RelBenchTable],
    graphs: Dict[str, TableGraph],
) -> Tuple[List[FKRelation], List[MissingFK]]:
    observed: List[FKRelation] = []
    missing: List[MissingFK] = []
    pkey_to_row = _build_pkey_to_row_index(tables)
    for table_name, table in tables.items():
        if table_name not in graphs:
            continue
        graph = graphs[table_name]
        mask_matrix = graph.train_mask.view(graph.num_rows, graph.num_cols)
        df = table.df
        col_to_idx = {col: idx for idx, col in enumerate(graph.feature_cols)}
        for fk_col, dst_table in table.fkey_col_to_pkey_table.items():
            if fk_col not in df.columns:
                continue
            if fk_col not in col_to_idx:
                continue
            if dst_table not in pkey_to_row:
                continue

            col_idx = col_to_idx[fk_col]
            src_series = df[fk_col]
            dst_map = pkey_to_row[dst_table]
            n_rows = min(len(src_series), mask_matrix.size(0))

            # IMPORTANT: row embeddings are positional (0..nrow-1), do not use df.index labels.
            for src_row in range(n_rows):
                fk_value = src_series.iloc[src_row]
                observed_mask = bool(mask_matrix[src_row, col_idx].item())

                if pd.isna(fk_value) or not observed_mask:
                    missing.append(
                        MissingFK(
                            src_table=table_name,
                            src_row=src_row,
                            fk_col=fk_col,
                            dst_table=dst_table,
                        )
                    )
                    continue

                key = _normalize_key(fk_value)
                dst_row = None if key is None else dst_map.get(key)
                if dst_row is None:
                    missing.append(
                        MissingFK(
                            src_table=table_name,
                            src_row=src_row,
                            fk_col=fk_col,
                            dst_table=dst_table,
                        )
                    )
                    continue

                observed.append(
                    FKRelation(
                        src_table=table_name,
                        src_row=src_row,
                        fk_col=fk_col,
                        dst_table=dst_table,
                        dst_row=int(dst_row),
                    )
                )
    return observed, missing


def load_multi_table_data(
    dataset_name: str,
    cache_dir: str | None = None,
    missing_ratio: float = 0.3,
    missing_mechanism: str = "MCAR",
    seed: int = 0,
    mask_fk: bool = True,
) -> MultiTableData:
    tables = load_relbench_tables(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        drop_text_cols=True,
        drop_time_col=True,
    )

    graphs: Dict[str, TableGraph] = {}
    for name, table in tables.items():
        if table.feature_df.empty:
            continue
        graphs[name] = build_table_graph(
            name=name,
            feature_df=table.feature_df,
            feature_types=table.feature_types,
            pkey_col=table.pkey_col,
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
            missing_ratio=missing_ratio,
            missing_mechanism=missing_mechanism,
            seed=seed,
            mask_fk=mask_fk,
        )

    observed_fk, missing_fk = _collect_fk_relations(tables, graphs)
    return MultiTableData(
        tables=tables,
        graphs=graphs,
        observed_fk=observed_fk,
        missing_fk=missing_fk,
    )
