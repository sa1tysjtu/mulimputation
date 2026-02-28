import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import k_hop_subgraph

from relbench_loader import RelBenchTable, load_relbench_tables


def normalize_np(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            out[:, j] = 0.0
            continue
        mn = np.min(col[valid])
        mx = np.max(col[valid])
        if mx == mn:
            col[valid] = 0.0
        else:
            col[valid] = (col[valid] - mn) / (mx - mn)
        col[~valid] = 0.0
        out[:, j] = col
    return out


def map_at_k_from_ranks(ranks: torch.Tensor, k: int) -> float:
    return float(((ranks <= k).float() / ranks.float()).mean().item())


@dataclass
class FKRelationData:
    rel_name: str
    rel_key: str
    src_table: str
    fk_col: str
    tgt_table: str
    src_row_idx_train: torch.Tensor
    tgt_row_idx_train: torch.Tensor
    src_row_idx_eval: torch.Tensor
    tgt_row_idx_eval: torch.Tensor
    src_x: torch.Tensor
    tgt_x: torch.Tensor


class RelationLinkModel(nn.Module):
    def __init__(self, src_in_dim: int, tgt_in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.src_proj = nn.Linear(src_in_dim, hidden_dim)
        self.tgt_proj = nn.Linear(tgt_in_dim, hidden_dim)
        self.dropout = dropout
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def encode(self, src_x: torch.Tensor, tgt_x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.src_proj(src_x), self.tgt_proj(tgt_x)], dim=0)
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i + 1 < len(self.convs):
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


def build_fk_true_row_indices(
    table_df: pd.DataFrame,
    table_data: Dict[str, RelBenchTable],
    fkey_map: Dict[str, str],
) -> Dict[str, np.ndarray]:
    fk_true_indices: Dict[str, np.ndarray] = {}
    for fk_col, target_table in fkey_map.items():
        tgt_meta = table_data.get(target_table)
        if tgt_meta is None:
            continue
        tgt_df = tgt_meta.df
        tgt_pkey = tgt_meta.pkey_col
        if tgt_pkey is None or tgt_pkey not in tgt_df.columns:
            continue
        pkey_to_row = {val: idx for idx, val in enumerate(tgt_df[tgt_pkey].tolist())}
        idxs = []
        for val in table_df[fk_col].tolist():
            if pd.isna(val):
                idxs.append(-1)
            else:
                idxs.append(pkey_to_row.get(val, -1))
        fk_true_indices[fk_col] = np.array(idxs, dtype=int)
    return fk_true_indices


def build_fk_observed_mask(n: int, valid_mask: np.ndarray, missing_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    valid_idx = np.where(valid_mask)[0]
    observed = np.zeros(n, dtype=bool)
    if len(valid_idx) == 0:
        return observed
    observed[valid_idx] = True
    n_miss = int(round(len(valid_idx) * missing_ratio))
    n_miss = max(0, min(n_miss, len(valid_idx)))
    if n_miss > 0:
        miss_idx = rng.choice(valid_idx, size=n_miss, replace=False)
        observed[miss_idx] = False
    return observed


def get_table_raw_features(table: RelBenchTable) -> pd.DataFrame:
    feat = table.feature_df.copy()
    # Keep only numeric-like columns in current loader output.
    return feat


def build_relations(args, tables: Dict[str, RelBenchTable], device: torch.device) -> List[FKRelationData]:
    relations: List[FKRelationData] = []
    fk_idx_cache: Dict[str, Dict[str, np.ndarray]] = {}
    raw_feat_cache: Dict[str, pd.DataFrame] = {}

    for tname, t in tables.items():
        raw_feat_cache[tname] = get_table_raw_features(t)
        fk_idx_cache[tname] = build_fk_true_row_indices(t.df, tables, t.fkey_col_to_pkey_table)

    for src_table, src_t in tables.items():
        src_feat_df = raw_feat_cache[src_table]
        if src_feat_df.empty:
            continue
        for fk_col, tgt_table in src_t.fkey_col_to_pkey_table.items():
            if tgt_table not in tables:
                continue
            if fk_col not in fk_idx_cache[src_table]:
                continue
            tgt_feat_df = raw_feat_cache[tgt_table]
            if tgt_feat_df.empty:
                continue

            true_idx = fk_idx_cache[src_table][fk_col]
            valid = true_idx >= 0
            if valid.sum() == 0:
                continue

            seed = args.seed + (hash((src_table, fk_col)) % 1_000_003)
            observed = build_fk_observed_mask(len(true_idx), valid, args.fk_missing_ratio, seed)
            train_mask = valid & observed
            eval_mask = valid & (~observed)
            if train_mask.sum() == 0 or eval_mask.sum() == 0:
                continue

            src_used = src_feat_df.copy()
            # Crucial: rows with masked FK should not leak FK value.
            if fk_col in src_used.columns:
                src_used.loc[~observed, fk_col] = np.nan

            src_np = src_used.to_numpy(dtype=float, na_value=np.nan)
            tgt_np = tgt_feat_df.to_numpy(dtype=float, na_value=np.nan)
            src_x = torch.tensor(normalize_np(src_np), dtype=torch.float, device=device)
            tgt_x = torch.tensor(normalize_np(tgt_np), dtype=torch.float, device=device)

            src_train = torch.tensor(np.where(train_mask)[0], dtype=torch.long, device=device)
            tgt_train = torch.tensor(true_idx[train_mask], dtype=torch.long, device=device)
            src_eval = torch.tensor(np.where(eval_mask)[0], dtype=torch.long, device=device)
            tgt_eval = torch.tensor(true_idx[eval_mask], dtype=torch.long, device=device)

            relations.append(
                FKRelationData(
                    rel_name=f"{src_table}.{fk_col}->{tgt_table}",
                    rel_key=f"{src_table}__{fk_col}__to__{tgt_table}",
                    src_table=src_table,
                    fk_col=fk_col,
                    tgt_table=tgt_table,
                    src_row_idx_train=src_train,
                    tgt_row_idx_train=tgt_train,
                    src_row_idx_eval=src_eval,
                    tgt_row_idx_eval=tgt_eval,
                    src_x=src_x,
                    tgt_x=tgt_x,
                )
            )
    return relations


def build_relation_train_graph(rel: FKRelationData) -> Tuple[torch.Tensor, int, int]:
    n_src = rel.src_x.size(0)
    n_tgt = rel.tgt_x.size(0)
    src = rel.src_row_idx_train
    tgt = rel.tgt_row_idx_train + n_src
    e1 = torch.stack([src, tgt], dim=0)
    e2 = torch.stack([tgt, src], dim=0)
    edge_index = torch.cat([e1, e2], dim=1)
    return edge_index, n_src, n_tgt


def evaluate_relations(
    args,
    relations: List[FKRelationData],
    models: nn.ModuleDict,
    edge_dict: Dict[str, torch.Tensor],
) -> Tuple[List[Tuple[str, float, float, float, float, int]], Dict[str, float]]:
    per_rel = []
    top1_sum = topk_sum = mrr_sum = map_sum = 0.0
    n_all = 0

    for rel in relations:
        model = models[rel.rel_key]
        model.eval()
        edge_index = edge_dict[rel.rel_key]
        with torch.no_grad():
            z = model.encode(rel.src_x, rel.tgt_x, edge_index)
        n_src = rel.src_x.size(0)
        src_eval = z[rel.src_row_idx_eval]
        tgt_all = z[n_src:]
        scores = src_eval @ tgt_all.t()
        true_idx = rel.tgt_row_idx_eval
        true_score = scores.gather(1, true_idx.unsqueeze(1)).squeeze(1)
        ranks = (scores > true_score.unsqueeze(1)).sum(dim=1) + 1

        top1 = float((ranks == 1).float().mean().item())
        rec_k = float((ranks <= args.topk).float().mean().item())
        mrr = float((1.0 / ranks.float()).mean().item())
        map_k = map_at_k_from_ranks(ranks, args.topk)
        n = int(rel.src_row_idx_eval.numel())
        per_rel.append((rel.rel_name, top1, rec_k, mrr, map_k, n))
        top1_sum += top1 * n
        topk_sum += rec_k * n
        mrr_sum += mrr * n
        map_sum += map_k * n
        n_all += n

    agg = {
        "top1": top1_sum / n_all if n_all else float("nan"),
        "recall_k": topk_sum / n_all if n_all else float("nan"),
        "mrr": mrr_sum / n_all if n_all else float("nan"),
        "map_k": map_sum / n_all if n_all else float("nan"),
        "n": n_all,
    }
    return per_rel, agg


def train_relations(args, relations: List[FKRelationData], models: nn.ModuleDict, edge_dict: Dict[str, torch.Tensor]) -> None:
    opt = torch.optim.AdamW(models.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        models.train()
        losses = []
        for rel in relations:
            model = models[rel.rel_key]
            edge_index = edge_dict[rel.rel_key]
            n_src = rel.src_x.size(0)
            n_tgt = rel.tgt_x.size(0)

            src_all = rel.src_row_idx_train
            pos_all = rel.tgt_row_idx_train
            perm = torch.randperm(src_all.numel(), device=src_all.device)
            src_all = src_all[perm]
            pos_all = pos_all[perm]

            bs = args.batch_size if args.batch_size > 0 else src_all.numel()
            for st in range(0, src_all.numel(), bs):
                ed = min(st + bs, src_all.numel())
                src = src_all[st:ed]
                pos = pos_all[st:ed]
                neg = torch.randint(low=0, high=n_tgt, size=pos.shape, device=pos.device)
                same = neg == pos
                if torch.any(same):
                    neg[same] = (neg[same] + 1) % n_tgt

                seed_nodes = torch.cat([src, pos + n_src, neg + n_src], dim=0)
                subset, sub_ei, mapping, _ = k_hop_subgraph(
                    seed_nodes,
                    num_hops=args.num_hops,
                    edge_index=edge_index,
                    num_nodes=n_src + n_tgt,
                    relabel_nodes=True,
                )

                src_sub = rel.src_x[subset[subset < n_src]]
                tgt_sub = rel.tgt_x[subset[subset >= n_src] - n_src]
                # Reconstruct local x with source-first then target layout:
                src_mask = subset < n_src
                tgt_mask = ~src_mask
                src_global = subset[src_mask]
                tgt_global = subset[tgt_mask] - n_src
                # map old local id -> new local id after source-first reorder
                remap = torch.empty_like(subset)
                remap[src_mask] = torch.arange(src_global.numel(), device=subset.device)
                remap[tgt_mask] = torch.arange(tgt_global.numel(), device=subset.device) + src_global.numel()
                sub_ei = remap[sub_ei]
                z = model.encode(src_sub, tgt_sub, sub_ei)

                b = src.size(0)
                src_z = z[remap[mapping[:b]]]
                pos_z = z[remap[mapping[b : 2 * b]]]
                neg_z = z[remap[mapping[2 * b :]]]

                loss = F.softplus(-(torch.sum(src_z * pos_z, dim=1) - torch.sum(src_z * neg_z, dim=1))).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

        if epoch % args.log_every == 0:
            _, agg = evaluate_relations(args, relations, models, edge_dict)
            tr_loss = float(np.mean(losses)) if losses else float("nan")
            print(
                f"[rdl_like] epoch {epoch} train_loss={tr_loss:.6f} "
                f"top1={agg['top1']:.4f} recall@{args.topk}={agg['recall_k']:.4f} "
                f"mrr={agg['mrr']:.4f} map@{args.topk}={agg['map_k']:.4f} n={agg['n']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--fk_missing_ratio", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_hops", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    tables = load_relbench_tables(dataset_name=args.dataset, drop_text_cols=True, drop_time_col=True)
    relations = build_relations(args, tables, device)
    if not relations:
        raise RuntimeError("No valid FK relations.")

    models = nn.ModuleDict()
    edge_dict: Dict[str, torch.Tensor] = {}
    print(f"num_relations={len(relations)}")
    for rel in relations:
        models[rel.rel_key] = RelationLinkModel(
            src_in_dim=rel.src_x.size(1),
            tgt_in_dim=rel.tgt_x.size(1),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        edge_index, n_src, n_tgt = build_relation_train_graph(rel)
        edge_dict[rel.rel_key] = edge_index
        print(
            f"{rel.rel_name}: n_train={rel.src_row_idx_train.numel()} n_eval={rel.src_row_idx_eval.numel()} "
            f"n_src={n_src} n_tgt={n_tgt} src_dim={rel.src_x.size(1)} tgt_dim={rel.tgt_x.size(1)}"
        )

    train_relations(args, relations, models, edge_dict)
    per_rel, agg = evaluate_relations(args, relations, models, edge_dict)
    for rel_name, top1, rec, mrr, map_k, n in per_rel:
        print(
            f"{rel_name}: top1={top1:.4f} recall@{args.topk}={rec:.4f} "
            f"mrr={mrr:.4f} map@{args.topk}={map_k:.4f} n={n}"
        )
    print(
        f"FK AVG: top1={agg['top1']:.4f} recall@{args.topk}={agg['recall_k']:.4f} "
        f"mrr={agg['mrr']:.4f} map@{args.topk}={agg['map_k']:.4f} n={agg['n']}"
    )


if __name__ == "__main__":
    main()

