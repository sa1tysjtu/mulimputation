import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq


@dataclass
class RelBenchTable:
    name: str
    df: pd.DataFrame
    feature_df: pd.DataFrame
    pkey_col: Optional[str]
    time_col: Optional[str]
    fkey_col_to_pkey_table: Dict[str, str]
    feature_types: Dict[str, str]
    categorical_maps: Dict[str, Dict]


def _read_metadata(table: pq.ParquetFile) -> Tuple[Optional[str], Optional[str], Dict[str, str]]:
    meta = table.schema_arrow.metadata or {}

    def _get_json(key: str):
        raw = meta.get(key.encode("utf-8"))
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))

    pkey_col = _get_json("pkey_col")
    time_col = _get_json("time_col")
    fkey_map = _get_json("fkey_col_to_pkey_table") or {}
    return pkey_col, time_col, fkey_map


def _is_text_dtype(dtype) -> bool:
    return (
        pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_categorical_dtype(dtype)
        or pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_period_dtype(dtype)
    )


def _encode_categorical(series: pd.Series) -> Tuple[pd.Series, Dict]:
    categories = pd.Categorical(series)
    codes = categories.codes.astype("int64")
    mapping = {cat: idx for idx, cat in enumerate(categories.categories)}
    # pandas uses -1 for NaN in categorical codes
    return pd.Series(codes, index=series.index), mapping


def load_relbench_tables(
    dataset_name: str,
    cache_dir: Optional[str] = None,
    drop_text_cols: bool = True,
    drop_time_col: bool = True,
    cat_cardinality_threshold: int = 50,
) -> Dict[str, RelBenchTable]:
    root = Path(cache_dir or Path.home() / ".cache/relbench")
    db_dir = root / dataset_name / "db"
    if not db_dir.exists():
        raise FileNotFoundError(f"RelBench dataset not found: {db_dir}")

    tables: Dict[str, RelBenchTable] = {}
    for table_path in sorted(db_dir.glob("*.parquet")):
        pf = pq.ParquetFile(table_path)
        pkey_col, time_col, fkey_map = _read_metadata(pf)
        df = pf.read().to_pandas()

        keep_cols = set(df.columns)
        if drop_text_cols:
            for col in list(keep_cols):
                if col == pkey_col or col in fkey_map or col == time_col:
                    continue
                if _is_text_dtype(df[col].dtype):
                    keep_cols.remove(col)

        if drop_time_col and time_col in keep_cols:
            keep_cols.remove(time_col)

        df = df.loc[:, sorted(keep_cols)]

        feature_cols = [c for c in df.columns if c != pkey_col]
        feature_types: Dict[str, str] = {}
        categorical_maps: Dict[str, Dict] = {}
        feature_df = df[feature_cols].copy()

        for col in feature_cols:
            if col in fkey_map:
                feature_types[col] = "fkey"
                continue

            dtype = feature_df[col].dtype
            if pd.api.types.is_bool_dtype(dtype):
                feature_types[col] = "categorical"
                feature_df[col] = feature_df[col].astype("int64")
                continue

            if pd.api.types.is_integer_dtype(dtype):
                nunique = feature_df[col].nunique(dropna=True)
                if nunique <= cat_cardinality_threshold:
                    feature_types[col] = "categorical"
                else:
                    feature_types[col] = "numerical"
            elif pd.api.types.is_float_dtype(dtype):
                feature_types[col] = "numerical"
            else:
                feature_types[col] = "numerical"

            if feature_types[col] == "categorical":
                encoded, mapping = _encode_categorical(feature_df[col])
                feature_df[col] = encoded
                categorical_maps[col] = mapping
                feature_types[col] = "numerical"

        tables[table_path.stem] = RelBenchTable(
            name=table_path.stem,
            df=df,
            feature_df=feature_df,
            pkey_col=pkey_col,
            time_col=time_col,
            fkey_col_to_pkey_table=fkey_map,
            feature_types=feature_types,
            categorical_maps=categorical_maps,
        )

    return tables
