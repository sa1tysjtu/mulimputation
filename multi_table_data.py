from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def _collect_fk_relations(
    tables: Dict[str, RelBenchTable],
    graphs: Dict[str, TableGraph],
) -> Tuple[List[FKRelation], List[MissingFK]]:
    observed: List[FKRelation] = []
    missing: List[MissingFK] = []
    for table_name, table in tables.items():
        if table_name not in graphs:
            continue
        graph = graphs[table_name]
        mask_matrix = graph.train_mask.view(graph.num_rows, graph.num_cols)
        df = table.df
        for fk_col, dst_table in table.fkey_col_to_pkey_table.items():
            if fk_col not in df.columns:
                continue
            if fk_col not in graph.feature_cols:
                continue
            col_idx = graph.feature_cols.index(fk_col)
            series = df[fk_col]
            for row_idx, fk_value in series.items():
                if row_idx >= mask_matrix.size(0):
                    continue
                observed_mask = bool(mask_matrix[row_idx, col_idx].item())
                if pd.isna(fk_value):
                    missing.append(
                        MissingFK(
                            src_table=table_name,
                            src_row=row_idx,
                            fk_col=fk_col,
                            dst_table=dst_table,
                        )
                    )
                elif observed_mask:
                    observed.append(
                        FKRelation(
                            src_table=table_name,
                            src_row=row_idx,
                            fk_col=fk_col,
                            dst_table=dst_table,
                            dst_row=int(fk_value),
                        )
                    )
                else:
                    missing.append(
                        MissingFK(
                            src_table=table_name,
                            src_row=row_idx,
                            fk_col=fk_col,
                            dst_table=dst_table,
                        )
                    )
    return observed, missing


def load_multi_table_data(
    dataset_name: str,
    cache_dir: str | None = None,
    missing_ratio: float = 0.3,
    missing_mechanism: str = "MCAR",
    seed: int = 0,
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
        )

    observed_fk, missing_fk = _collect_fk_relations(tables, graphs)
    return MultiTableData(
        tables=tables,
        graphs=graphs,
        observed_fk=observed_fk,
        missing_fk=missing_fk,
    )
