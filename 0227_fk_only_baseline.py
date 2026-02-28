import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from relbench_loader import load_relbench_tables


@dataclass
class Metric:
    top1: float
    recall_k: float
    mrr: float
    map_k: float
    n: int


def _relation_seed(base_seed: int, table_name: str, fk_col: str) -> int:
    return (base_seed + hash((table_name, fk_col)) % (2**31 - 1)) % (2**31 - 1)


def _build_missing_mask(
    valid_edge_mask: np.ndarray,
    missing_ratio: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    valid_idx = np.where(valid_edge_mask)[0]
    n_mask = int(round(len(valid_idx) * missing_ratio))
    n_mask = max(0, min(n_mask, len(valid_idx)))
    mask = np.zeros_like(valid_edge_mask, dtype=bool)
    if n_mask > 0:
        pick = rng.choice(valid_idx, size=n_mask, replace=False)
        mask[pick] = True
    return mask


def _evaluate_most_frequent(
    y_true: np.ndarray,
    train_values: pd.Series,
    topk: int,
) -> Metric:
    if len(y_true) == 0:
        return Metric(float("nan"), float("nan"), float("nan"), float("nan"), 0)

    # Frequency ranking over observed (non-missing) FK values.
    rank_list = train_values.value_counts(dropna=True).index.to_list()
    rank_map = {v: i + 1 for i, v in enumerate(rank_list)}

    ranks = np.array([rank_map.get(v, np.inf) for v in y_true], dtype=float)
    top1 = float(np.mean(ranks == 1))
    recall_k = float(np.mean(ranks <= topk))
    mrr = float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)))
    map_k = float(np.mean(np.where(ranks <= topk, 1.0 / ranks, 0.0)))
    return Metric(top1, recall_k, mrr, map_k, len(y_true))


def _evaluate_random_rank(
    y_true: np.ndarray,
    candidate_size: int,
    topk: int,
    seed: int,
) -> Metric:
    if len(y_true) == 0:
        return Metric(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    if candidate_size <= 0:
        return Metric(float("nan"), float("nan"), float("nan"), float("nan"), 0)

    # Random full-ranking baseline:
    # true rank is sampled uniformly from [1, candidate_size].
    rng = np.random.default_rng(seed)
    ranks = rng.integers(1, candidate_size + 1, size=len(y_true))
    top1 = float(np.mean(ranks == 1))
    recall_k = float(np.mean(ranks <= topk))
    mrr = float(np.mean(1.0 / ranks))
    map_k = float(np.mean(np.where(ranks <= topk, 1.0 / ranks, 0.0)))
    return Metric(top1, recall_k, mrr, map_k, len(y_true))


def _weighted_avg(metrics: List[Metric]) -> Metric:
    valid = [m for m in metrics if m.n > 0]
    if not valid:
        return Metric(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    n_all = sum(m.n for m in valid)
    top1 = float(sum(m.top1 * m.n for m in valid) / n_all)
    recall_k = float(sum(m.recall_k * m.n for m in valid) / n_all)
    mrr = float(sum(m.mrr * m.n for m in valid) / n_all)
    map_k = float(sum(m.map_k * m.n for m in valid) / n_all)
    return Metric(top1, recall_k, mrr, map_k, n_all)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--fk_missing_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    tables = load_relbench_tables(
        dataset_name=args.dataset,
        drop_text_cols=True,
        drop_time_col=True,
    )

    mf_metrics: List[Metric] = []
    rnd_metrics: List[Metric] = []

    print(
        f"FK-only baseline on dataset={args.dataset}, fk_missing_ratio={args.fk_missing_ratio}, topk={args.topk}"
    )

    for table_name, table in tables.items():
        if not table.fkey_col_to_pkey_table:
            continue

        for fk_col, target_table_name in table.fkey_col_to_pkey_table.items():
            if fk_col not in table.df.columns:
                continue
            if target_table_name not in tables:
                continue

            target_table = tables[target_table_name]
            if not target_table.pkey_col or target_table.pkey_col not in target_table.df.columns:
                continue

            fk_series = table.df[fk_col]
            target_keys = target_table.df[target_table.pkey_col].dropna().to_numpy()
            target_key_set = set(target_keys.tolist())

            is_valid_edge = fk_series.notna().to_numpy() & fk_series.isin(target_key_set).to_numpy()
            if int(is_valid_edge.sum()) == 0:
                continue

            rel_seed = _relation_seed(args.seed, table_name, fk_col)
            missing_mask = _build_missing_mask(is_valid_edge, args.fk_missing_ratio, rel_seed)
            eval_mask = is_valid_edge & missing_mask
            train_mask = is_valid_edge & (~missing_mask)

            y_true = fk_series[eval_mask].to_numpy()
            train_values = fk_series[train_mask]
            candidate_size = int(len(target_key_set))

            mf = _evaluate_most_frequent(y_true, train_values, args.topk)
            rnd = _evaluate_random_rank(y_true, candidate_size, args.topk, rel_seed + 13)
            if mf.n == 0:
                continue

            mf_metrics.append(mf)
            rnd_metrics.append(rnd)

            print(
                f"{table_name}.{fk_col}->{target_table_name}: "
                f"n_eval={mf.n} candidates={candidate_size} "
                f"mf(top1={mf.top1:.4f}, r@{args.topk}={mf.recall_k:.4f}, mrr={mf.mrr:.4f}, map@{args.topk}={mf.map_k:.4f}) "
                f"rand(top1={rnd.top1:.4f}, r@{args.topk}={rnd.recall_k:.4f}, mrr={rnd.mrr:.4f}, map@{args.topk}={rnd.map_k:.4f})"
            )

    mf_avg = _weighted_avg(mf_metrics)
    rnd_avg = _weighted_avg(rnd_metrics)
    print(
        f"MF AVG: top1={mf_avg.top1:.4f} recall@{args.topk}={mf_avg.recall_k:.4f} "
        f"mrr={mf_avg.mrr:.4f} map@{args.topk}={mf_avg.map_k:.4f} n={mf_avg.n}"
    )
    print(
        f"RAND AVG: top1={rnd_avg.top1:.4f} recall@{args.topk}={rnd_avg.recall_k:.4f} "
        f"mrr={rnd_avg.mrr:.4f} map@{args.topk}={rnd_avg.map_k:.4f} n={rnd_avg.n}"
    )


if __name__ == "__main__":
    main()
