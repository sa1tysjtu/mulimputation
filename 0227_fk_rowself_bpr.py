import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.gen_imp import get_gen_imp
from models.imputaion_model import LinearHead
from relbench_loader import load_relbench_tables
from utils_core import produce_NA


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


def get_data_from_table(values: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_row, n_col = values.shape
    hyperedge = create_edge_node(n_row, n_col)
    values_filled = np.nan_to_num(values, nan=0.0)
    hyper_node = create_value_node(values_filled)
    ve_affiliation = create_VE_affiliation(n_row, n_col)
    return hyperedge, hyper_node, ve_affiliation


def compute_row_embeddings(
    model: nn.Module,
    hyperedge: torch.Tensor,
    hyper_node: torch.Tensor,
    ve_affiliation: torch.Tensor,
    n_row: int,
) -> torch.Tensor:
    embedding, _ = model(hyperedge, hyper_node, ve_affiliation)
    return embedding[:n_row]


def init_models_for_ckpt(args, table_meta: Dict, device: torch.device) -> None:
    hyperedge, hyper_node, _ = table_meta["data"]
    model = get_gen_imp(hyperedge.shape[1], hyper_node.shape[1], args).to(device)
    # not used in this script, but required to load ckpt format from 0225 pretrain.
    input_dim = args.hyperedge_dim_hidden * 2
    impute_model = LinearHead(
        input_dim,
        1,
        hidden_layer_sizes=hyperedge.shape[1],
        hidden_activation=args.impute_activation,
        dropout=args.dropout,
    ).to(device)
    fk_gate = nn.Linear(2 * args.hyperedge_dim_hidden, args.hyperedge_dim_hidden).to(device)
    row_proj_shared = nn.Linear(args.hyperedge_dim_hidden, args.hyperedge_dim_hidden).to(device)
    row_proj_num = nn.Linear(args.hyperedge_dim_hidden, args.hyperedge_dim_hidden).to(device)
    row_proj_fk = nn.Linear(args.hyperedge_dim_hidden, args.hyperedge_dim_hidden).to(device)
    table_meta["model"] = model
    table_meta["impute_model"] = impute_model
    table_meta["fk_gate"] = fk_gate
    table_meta["row_proj_shared"] = row_proj_shared
    table_meta["row_proj_num"] = row_proj_num
    table_meta["row_proj_fk"] = row_proj_fk


def load_pretrain_checkpoint(all_table_meta: Dict[str, Dict], ckpt_dir: str, device: torch.device) -> None:
    for table_name, table_meta in all_table_meta.items():
        ckpt_path = os.path.join(ckpt_dir, f"{table_name}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint for table {table_name}: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        table_meta["model"].load_state_dict(ckpt["model"])
        table_meta["impute_model"].load_state_dict(ckpt["impute_model"])
        table_meta["fk_gate"].load_state_dict(ckpt["fk_gate"])
        if "row_proj_shared" in ckpt:
            table_meta["row_proj_shared"].load_state_dict(ckpt["row_proj_shared"])
        if "row_proj_num" in ckpt:
            table_meta["row_proj_num"].load_state_dict(ckpt["row_proj_num"])
        if "row_proj_fk" in ckpt:
            table_meta["row_proj_fk"].load_state_dict(ckpt["row_proj_fk"])


def build_fk_true_row_indices(
    table_df: pd.DataFrame,
    table_data: Dict[str, Dict],
    fkey_map: Dict[str, str],
) -> Dict[str, np.ndarray]:
    fk_true_indices: Dict[str, np.ndarray] = {}
    for fk_col, target_table in fkey_map.items():
        target_meta = table_data.get(target_table)
        if target_meta is None:
            continue
        target_df = target_meta["df"]
        target_pkey = target_meta["pkey_col"]
        if target_pkey is None or target_pkey not in target_df.columns:
            continue
        pkey_to_row = {val: idx for idx, val in enumerate(target_df[target_pkey].tolist())}
        idxs = []
        for val in table_df[fk_col].tolist():
            if pd.isna(val):
                idxs.append(-1)
            else:
                idxs.append(pkey_to_row.get(val, -1))
        fk_true_indices[fk_col] = np.array(idxs, dtype=int)
    return fk_true_indices


def build_fk_observed_mask_for_relation(
    n: int, valid_mask: np.ndarray, missing_ratio: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    valid_idx = np.where(valid_mask)[0]
    n_miss = int(round(len(valid_idx) * missing_ratio))
    n_miss = max(0, min(n_miss, len(valid_idx)))
    observed = np.zeros(n, dtype=bool)
    if len(valid_idx) == 0:
        return observed
    observed[valid_idx] = True
    if n_miss > 0:
        miss_idx = rng.choice(valid_idx, size=n_miss, replace=False)
        observed[miss_idx] = False
    return observed


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


def build_relations(args, table_meta: Dict[str, Dict], device: torch.device) -> List[FKRelationData]:
    relations: List[FKRelationData] = []
    for src_table, meta in table_meta.items():
        true_map = meta["fk_true_row_indices"]
        for fk_col, tgt_table in meta["fkey_map"].items():
            if tgt_table not in table_meta or fk_col not in true_map:
                continue
            true_idx = true_map[fk_col]
            valid = true_idx >= 0
            if valid.sum() == 0:
                continue
            seed = args.seed + (hash((src_table, fk_col)) % 1_000_003)
            observed = build_fk_observed_mask_for_relation(
                n=len(true_idx),
                valid_mask=valid,
                missing_ratio=args.fk_missing_ratio,
                seed=seed,
            )
            train_mask = valid & observed
            eval_mask = valid & (~observed)
            if train_mask.sum() == 0 or eval_mask.sum() == 0:
                continue
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
                )
            )
    return relations


def evaluate_relations(
    args,
    relations: List[FKRelationData],
    row_embs: Dict[str, torch.Tensor],
    rel_heads: nn.ModuleDict,
) -> Tuple[List[Tuple[str, float, float, float, float, int, float, float]], Dict[str, float]]:
    per_rel = []
    top1_sum = 0.0
    topk_sum = 0.0
    mrr_sum = 0.0
    map_sum = 0.0
    coarse_sum = 0.0
    n_all = 0
    pos_sum = 0.0
    neg_sum = 0.0
    hard_sum = 0.0
    diag_n = 0

    for rel in relations:
        src = row_embs[rel.src_table][rel.src_row_idx_eval]
        tgt_all = row_embs[rel.tgt_table]
        head = rel_heads[rel.rel_key]
        q = head(src)
        scores = q @ tgt_all.t()
        true_idx = rel.tgt_row_idx_eval
        true_score = scores.gather(1, true_idx.unsqueeze(1)).squeeze(1)
        greater = (scores > true_score.unsqueeze(1)).sum(dim=1)
        ranks = greater + 1

        top1 = float((ranks == 1).float().mean().item())
        recall_k = float((ranks <= args.topk).float().mean().item())
        mrr = float((1.0 / ranks.float()).mean().item())
        map_k = float(((ranks <= args.topk).float() / ranks.float()).mean().item())

        k_coarse = min(args.coarse_k, scores.size(1))
        if k_coarse > 0:
            coarse_idx = torch.topk(scores, k=k_coarse, dim=1).indices
            coarse_hit = (coarse_idx == true_idx.unsqueeze(1)).any(dim=1).float().mean().item()
        else:
            coarse_hit = float("nan")

        masked = scores.clone()
        masked.scatter_(1, true_idx.unsqueeze(1), float("-inf"))
        neg_mean = float(masked[torch.isfinite(masked)].mean().item())
        k_hard = min(5, masked.size(1) - 1)
        hard_mean = float(torch.topk(masked, k=k_hard, dim=1).values.mean().item()) if k_hard > 0 else float("nan")
        pos_mean = float(true_score.mean().item())
        hard_margin = pos_mean - hard_mean if np.isfinite(hard_mean) else float("nan")

        n = src.size(0)
        per_rel.append((rel.rel_name, top1, recall_k, mrr, map_k, n, coarse_hit, hard_margin))

        top1_sum += top1 * n
        topk_sum += recall_k * n
        mrr_sum += mrr * n
        map_sum += map_k * n
        coarse_sum += coarse_hit * n
        n_all += n
        pos_sum += pos_mean * n
        neg_sum += neg_mean * n
        hard_sum += hard_mean * n
        diag_n += n

    if n_all == 0:
        agg = {
            "top1": float("nan"),
            "recall_k": float("nan"),
            "mrr": float("nan"),
            "map_k": float("nan"),
            "coarse_recall": float("nan"),
            "n": 0,
            "pos_mean": float("nan"),
            "neg_mean": float("nan"),
            "margin": float("nan"),
            "hard_margin": float("nan"),
        }
    else:
        pos_mean = pos_sum / diag_n
        neg_mean = neg_sum / diag_n
        hard_mean = hard_sum / diag_n
        agg = {
            "top1": top1_sum / n_all,
            "recall_k": topk_sum / n_all,
            "mrr": mrr_sum / n_all,
            "map_k": map_sum / n_all,
            "coarse_recall": coarse_sum / n_all,
            "n": n_all,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "margin": pos_mean - neg_mean,
            "hard_margin": pos_mean - hard_mean,
        }
    return per_rel, agg


def train_fk_bpr(args, relations: List[FKRelationData], row_embs: Dict[str, torch.Tensor], rel_heads: nn.ModuleDict) -> None:
    optimizer = torch.optim.AdamW(rel_heads.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        rel_heads.train()
        rel_losses = []
        for rel in relations:
            src_all = rel.src_row_idx_train
            tgt_pos_all = rel.tgt_row_idx_train
            if src_all.numel() == 0:
                continue
            perm = torch.randperm(src_all.numel(), device=src_all.device)
            src_all = src_all[perm]
            tgt_pos_all = tgt_pos_all[perm]
            batch_size = args.batch_size if args.batch_size > 0 else src_all.numel()
            for st in range(0, src_all.numel(), batch_size):
                ed = min(st + batch_size, src_all.numel())
                src_idx = src_all[st:ed]
                pos_idx = tgt_pos_all[st:ed]
                src_emb = row_embs[rel.src_table][src_idx]
                tgt_all = row_embs[rel.tgt_table]
                head = rel_heads[rel.rel_key]

                q = head(src_emb)
                pos_emb = tgt_all[pos_idx]

                neg_idx = torch.randint(
                    low=0,
                    high=tgt_all.size(0),
                    size=pos_idx.shape,
                    device=pos_idx.device,
                )
                same = neg_idx == pos_idx
                if torch.any(same):
                    neg_idx[same] = (neg_idx[same] + 1) % tgt_all.size(0)
                neg_emb = tgt_all[neg_idx]

                pos_score = torch.sum(q * pos_emb, dim=1)
                neg_score = torch.sum(q * neg_emb, dim=1)
                loss = torch.nn.functional.softplus(-(pos_score - neg_score)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rel_losses.append(float(loss.item()))

        if epoch % args.log_every == 0:
            rel_heads.eval()
            _, agg = evaluate_relations(args, relations, row_embs, rel_heads)
            loss_str = np.mean(rel_losses) if rel_losses else float("nan")
            print(
                f"[fk_bpr] epoch {epoch} train_loss={loss_str:.6f} "
                f"top1={agg['top1']:.4f} recall@{args.topk}={agg['recall_k']:.4f} "
                f"mrr={agg['mrr']:.4f} map@{args.topk}={agg['map_k']:.4f} "
                f"coarse@{args.coarse_k}={agg['coarse_recall']:.4f} n={agg['n']}"
            )
            print(
                f"[fk_bpr] diag s_pos={agg['pos_mean']:.4f} s_neg={agg['neg_mean']:.4f} "
                f"margin={agg['margin']:.4f} hard_margin={agg['hard_margin']:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--load_pretrain_ckpt_dir", type=str, required=True)
    parser.add_argument("--fk_missing_ratio", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--coarse_k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--known", type=float, default=0.6)

    # Keep same defaults as 0225 for model checkpoint compatibility.
    parser.add_argument("--hyperedge_dim_hidden", type=int, default=64)
    parser.add_argument("--hyper_node_dim_hidden", type=int, default=64)
    parser.add_argument("--gnn_layer_num", type=int, default=3)
    parser.add_argument("--gnn_activation", type=str, default="relu")
    parser.add_argument("--impute_activation", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    tables = load_relbench_tables(dataset_name=args.dataset, drop_text_cols=True, drop_time_col=True)

    table_meta: Dict[str, Dict] = {}
    for table_name, table in tables.items():
        feature_df = table.feature_df
        if feature_df.empty:
            print(f"skip {table_name}: zero features")
            continue

        keep_cols = [c for c in feature_df.columns if table.feature_types.get(c) != "fkey"]
        if not keep_cols:
            print(f"skip {table_name}: no non-fk features")
            continue
        feature_df = feature_df[keep_cols]

        values = feature_df.to_numpy(dtype=float, na_value=np.nan)
        values = normalize_table(values)
        if np.isnan(values).all():
            print(f"skip {table_name}: all missing")
            continue

        data = get_data_from_table(values)
        table_meta[table_name] = {
            "name": table_name,
            "df": table.df,
            "feature_df": feature_df,
            "pkey_col": table.pkey_col,
            "fkey_map": table.fkey_col_to_pkey_table,
            "data": data,
            "n_row": values.shape[0],
        }

    for table_name, meta in table_meta.items():
        meta["fk_true_row_indices"] = build_fk_true_row_indices(
            meta["df"],
            table_meta,
            meta["fkey_map"],
        )
        init_models_for_ckpt(args, meta, device)

    load_pretrain_checkpoint(table_meta, args.load_pretrain_ckpt_dir, device)
    print(f"Loaded pretrain checkpoints from {args.load_pretrain_ckpt_dir}")

    # Freeze row embeddings extracted from pretrain model.
    row_embs: Dict[str, torch.Tensor] = {}
    for table_name, meta in table_meta.items():
        model = meta["model"]
        model.eval()
        with torch.no_grad():
            hyperedge, hyper_node, ve_aff = meta["data"]
            row_emb = compute_row_embeddings(
                model,
                hyperedge.to(device),
                hyper_node.to(device),
                ve_aff.to(device),
                meta["n_row"],
            )
            row_embs[table_name] = row_emb.detach()

    relations = build_relations(args, table_meta, device)
    if not relations:
        raise RuntimeError("No valid FK relations found for training/evaluation.")

    rel_heads = nn.ModuleDict()
    for rel in relations:
        rel_heads[rel.rel_key] = nn.Linear(args.hyperedge_dim_hidden, args.hyperedge_dim_hidden).to(device)

    print(f"num_relations={len(relations)}")
    for rel in relations:
        print(
            f"{rel.rel_name}: n_train={rel.src_row_idx_train.numel()} "
            f"n_eval={rel.src_row_idx_eval.numel()} "
            f"num_candidates={row_embs[rel.tgt_table].size(0)}"
        )

    train_fk_bpr(args, relations, row_embs, rel_heads)

    rel_heads.eval()
    per_rel, agg = evaluate_relations(args, relations, row_embs, rel_heads)
    for rel_name, top1, rec, mrr, map_k, n, coarse, hard_margin in per_rel:
        print(
            f"{rel_name}: top1={top1:.4f} recall@{args.topk}={rec:.4f} "
            f"mrr={mrr:.4f} map@{args.topk}={map_k:.4f} n={n} coarse@{args.coarse_k}={coarse:.4f} "
            f"hard_margin={hard_margin:.4f}"
        )
    print(
        f"FK AVG: top1={agg['top1']:.4f} recall@{args.topk}={agg['recall_k']:.4f} "
        f"mrr={agg['mrr']:.4f} map@{args.topk}={agg['map_k']:.4f} n={agg['n']}"
    )
    print(f"FK AVG: coarse_recall@{args.coarse_k}={agg['coarse_recall']:.4f} n={agg['n']}")
    print(
        f"FK AVG DIAG: s_pos={agg['pos_mean']:.4f} s_neg={agg['neg_mean']:.4f} "
        f"margin={agg['margin']:.4f} hard_margin={agg['hard_margin']:.4f}"
    )


if __name__ == "__main__":
    main()
