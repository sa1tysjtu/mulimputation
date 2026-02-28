import argparse
import os
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


def build_fk_observed_mask(
    series: pd.Series,
    missing_ratio: float,
    missing_mechanism: str,
    seed: int,
) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    values = values.reshape(-1, 1)
    torch.manual_seed(seed)
    mask = produce_NA(
        torch.tensor(values, dtype=torch.float),
        p_miss=missing_ratio,
        mecha=missing_mechanism,
        n_row=values.shape[0],
        n_col=values.shape[1],
    ).view(-1)
    missing = np.isnan(values[:, 0])
    if missing.any():
        mask = mask & ~torch.tensor(missing, dtype=torch.bool)
    return mask.cpu().numpy().astype(bool)


def build_fk_row_indices(
    table_df: pd.DataFrame,
    table_data: Dict[str, Dict],
    fkey_map: Dict[str, str],
    fk_observed: Dict[str, np.ndarray],
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
        obs_mask = fk_observed.get(fk_col)
        idxs = []
        for i, val in enumerate(fk_vals):
            if obs_mask is not None and not obs_mask[i]:
                idxs.append(-1)
            elif pd.isna(val):
                idxs.append(-1)
            else:
                idxs.append(pkey_to_row.get(val, -1))
        fk_row_indices[fk_col] = np.array(idxs, dtype=int)
    return fk_row_indices


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


class RelationBilinearScorer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.weight)

    def score_pair(self, query: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        # query: [B, D], cand: [B, D] -> [B, 1]
        q_w = query @ self.weight
        return (q_w * cand).sum(dim=1, keepdim=True) + self.bias

    def score_all(self, query: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
        # query: [B, D], cands: [N, D] -> [B, N]
        q_w = query @ self.weight
        return q_w @ cands.t() + self.bias


def compute_fk_matching_loss(
    table_meta: Dict,
    row_self: torch.Tensor,
    other_row_embs: Dict[str, torch.Tensor],
    neg_k: int,
    max_pairs: int,
) -> torch.Tensor:
    match_heads: nn.ModuleDict = table_meta["fk_match_heads"]
    total_loss = torch.tensor(0.0, device=row_self.device)
    rel_count = 0
    for fk_col, target_table in table_meta["fkey_map"].items():
        if fk_col not in match_heads:
            continue
        target_rows = other_row_embs.get(target_table)
        if target_rows is None or target_rows.size(0) <= 1:
            continue
        true_idx = torch.tensor(table_meta["fk_true_row_indices"][fk_col], device=row_self.device)
        obs_mask = torch.tensor(table_meta["fk_observed"][fk_col], dtype=torch.bool, device=row_self.device)
        valid = obs_mask & (true_idx >= 0)
        if valid.sum() == 0:
            continue
        src_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if max_pairs > 0 and src_idx.numel() > max_pairs:
            perm = torch.randperm(src_idx.numel(), device=row_self.device)[:max_pairs]
            src_idx = src_idx[perm]
        pos_idx = true_idx[src_idx]

        scorer: RelationBilinearScorer = match_heads[fk_col]
        src_emb = row_self[src_idx]
        pos_emb = target_rows[pos_idx]
        pos_scores = scorer.score_pair(src_emb, pos_emb)

        num_tgt = target_rows.size(0)
        k = min(neg_k, max(1, num_tgt - 1))
        neg_idx = torch.randperm(num_tgt, device=row_self.device)[:k]
        neg_emb = target_rows[neg_idx]
        neg_scores = scorer.score_all(src_emb, neg_emb)

        logits = torch.cat([pos_scores, neg_scores], dim=1)
        labels = torch.zeros(src_emb.size(0), dtype=torch.long, device=row_self.device)
        total_loss = total_loss + F.cross_entropy(logits, labels)
        rel_count += 1

    if rel_count > 0:
        total_loss = total_loss / rel_count
    return total_loss


def evaluate_fk_metrics(
    args,
    table_meta: Dict,
    row_self: torch.Tensor,
    other_row_embs: Dict[str, torch.Tensor],
) -> Tuple[float, float, float, int]:
    match_heads: nn.ModuleDict = table_meta["fk_match_heads"]
    top1_hit = 0.0
    topk_hit = 0.0
    mrr_sum = 0.0
    total = 0
    for fk_col, target_table in table_meta["fkey_map"].items():
        if fk_col not in match_heads:
            continue
        target_rows = other_row_embs.get(target_table)
        if target_rows is None or target_rows.size(0) == 0:
            continue
        true_idx_all = torch.tensor(table_meta["fk_true_row_indices"][fk_col], device=row_self.device)
        obs_mask = torch.tensor(table_meta["fk_observed"][fk_col], dtype=torch.bool, device=row_self.device)
        # Evaluate FK recovery on missing FK rows.
        eval_mask = (~obs_mask) & (true_idx_all >= 0)
        if eval_mask.sum() == 0:
            continue
        src_idx = torch.nonzero(eval_mask, as_tuple=False).squeeze(-1)
        scorer: RelationBilinearScorer = match_heads[fk_col]
        src_emb = row_self[src_idx]
        true_idx = true_idx_all[src_idx]

        batch_size = args.fk_eval_batch_size
        for st in range(0, src_emb.size(0), batch_size):
            ed = min(st + batch_size, src_emb.size(0))
            src_b = src_emb[st:ed]
            true_b = true_idx[st:ed]
            scores = scorer.score_all(src_b, target_rows)
            pred = torch.argmax(scores, dim=1)
            top1_hit += (pred == true_b).float().sum().item()

            k = min(args.fk_topk, scores.size(1))
            topk = torch.topk(scores, k=k, dim=1).indices
            topk_hit += (topk == true_b.unsqueeze(1)).any(dim=1).float().sum().item()

            true_scores = scores.gather(1, true_b.unsqueeze(1))
            rank = (scores > true_scores).sum(dim=1) + 1
            mrr_sum += (1.0 / rank.float()).sum().item()
            total += src_b.size(0)

    if total == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return top1_hit / total, topk_hit / total, mrr_sum / total, total


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
    fk_match_heads = nn.ModuleDict()
    for fk_col in table_meta["fkey_map"].keys():
        fk_match_heads[fk_col] = RelationBilinearScorer(args.hyperedge_dim_hidden).to(device)
    table_meta["model"] = model
    table_meta["impute_model"] = impute_model
    table_meta["fk_gate"] = gate_layer
    table_meta["fk_match_heads"] = fk_match_heads
    table_meta["optimizer"] = None


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad = enabled


def configure_stage_optimizers(args, all_table_meta: Dict[str, Dict], stage: str) -> None:
    for table_meta in all_table_meta.values():
        model = table_meta["model"]
        impute_model = table_meta["impute_model"]
        fk_gate = table_meta["fk_gate"]
        fk_match_heads = table_meta["fk_match_heads"]

        if stage == "joint":
            set_requires_grad(model, True)
            set_requires_grad(impute_model, True)
            set_requires_grad(fk_gate, True)
            set_requires_grad(fk_match_heads, True)
            params = (
                list(model.parameters())
                + list(impute_model.parameters())
                + list(fk_gate.parameters())
                + list(fk_match_heads.parameters())
            )
        elif stage == "pretrain_num":
            set_requires_grad(model, True)
            set_requires_grad(impute_model, True)
            set_requires_grad(fk_gate, True)
            set_requires_grad(fk_match_heads, False)
            params = list(model.parameters()) + list(impute_model.parameters()) + list(fk_gate.parameters())
        elif stage == "finetune_fk":
            set_requires_grad(model, False)
            set_requires_grad(impute_model, False)
            set_requires_grad(fk_gate, False)
            set_requires_grad(fk_match_heads, True)
            params = list(fk_match_heads.parameters())
        else:
            raise ValueError(f"Unsupported stage: {stage}")

        if len(params) == 0:
            table_meta["optimizer"] = None
        else:
            table_meta["optimizer"] = torch.optim.AdamW(
                params,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )


def save_pretrain_checkpoint(all_table_meta: Dict[str, Dict], ckpt_dir: str) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    for table_name, table_meta in all_table_meta.items():
        ckpt_path = os.path.join(ckpt_dir, f"{table_name}.pt")
        torch.save(
            {
                "model": table_meta["model"].state_dict(),
                "impute_model": table_meta["impute_model"].state_dict(),
                "fk_gate": table_meta["fk_gate"].state_dict(),
            },
            ckpt_path,
        )


def load_pretrain_checkpoint(all_table_meta: Dict[str, Dict], ckpt_dir: str, device: torch.device) -> None:
    for table_name, table_meta in all_table_meta.items():
        ckpt_path = os.path.join(ckpt_dir, f"{table_name}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint for table {table_name}: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        table_meta["model"].load_state_dict(ckpt["model"])
        table_meta["impute_model"].load_state_dict(ckpt["impute_model"])
        table_meta["fk_gate"].load_state_dict(ckpt["fk_gate"])


def train_all(
    args,
    all_table_meta: Dict[str, Dict],
    device: torch.device,
    stage: str,
    epochs: int,
) -> None:
    huber = nn.HuberLoss(delta=1.0)
    for epoch in range(epochs):
        if epoch % args.log_every == 0:
            print(f"[{stage}] epoch {epoch}")
            rmse_list: List[float] = []
            mae_list: List[float] = []
            count_list: List[int] = []
            for table_name, meta in all_table_meta.items():
                rmse, mae, count = evaluate_table(args, meta, all_table_meta, device)
                rmse_list.append(rmse)
                mae_list.append(mae)
                count_list.append(count)
            total = sum(count_list)
            if total > 0:
                avg_rmse = float(np.sum(np.array(rmse_list) * np.array(count_list)) / total)
                avg_mae = float(np.sum(np.array(mae_list) * np.array(count_list)) / total)
                print(f"  [{stage}] eval rmse={avg_rmse:.4f} mae={avg_mae:.4f}")
            if stage != "pretrain_num":
                fk_top1_list: List[float] = []
                fk_recall_list: List[float] = []
                fk_mrr_list: List[float] = []
                fk_count_list: List[int] = []
                for table_name, meta in all_table_meta.items():
                    f_top1, f_rec, f_mrr, f_n = evaluate_fk_table(args, meta, all_table_meta, device)
                    if f_n > 0:
                        fk_top1_list.append(f_top1)
                        fk_recall_list.append(f_rec)
                        fk_mrr_list.append(f_mrr)
                        fk_count_list.append(f_n)
                if fk_count_list:
                    fk_total = sum(fk_count_list)
                    avg_top1 = float(np.sum(np.array(fk_top1_list) * np.array(fk_count_list)) / fk_total)
                    avg_rec = float(np.sum(np.array(fk_recall_list) * np.array(fk_count_list)) / fk_total)
                    avg_mrr = float(np.sum(np.array(fk_mrr_list) * np.array(fk_count_list)) / fk_total)
                    print(
                        f"  [{stage}] eval fk_top1={avg_top1:.4f} fk_recall@{args.fk_topk}={avg_rec:.4f} "
                        f"fk_mrr={avg_mrr:.4f} n={fk_total}"
                    )

        for table_name, table_meta in all_table_meta.items():
            model = table_meta["model"]
            impute_model = table_meta["impute_model"]
            optimizer = table_meta["optimizer"]
            if optimizer is None:
                continue
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
            if stage == "pretrain_num":
                loss_match = torch.tensor(0.0, device=device)
            else:
                loss_match = compute_fk_matching_loss(
                    table_meta,
                    row_self,
                    other_row_embs,
                    neg_k=args.fk_neg_k,
                    max_pairs=args.fk_max_pairs,
                )

            half = int(train_hyper_node.shape[0] / 2)
            pred = impute_model(
                [embedding[train_ve_affiliation[0, :half]], embedding[train_ve_affiliation[1, :half]]],
                token_emb=[],
            )
            pred_train = pred[:half, 0]
            loss_num = huber(pred_train, train_labels)
            if stage == "joint":
                loss = loss_num + args.fk_match_weight * loss_match
            elif stage == "pretrain_num":
                loss = loss_num
            elif stage == "finetune_fk":
                loss = loss_match
            else:
                raise ValueError(f"Unsupported stage: {stage}")
            if not loss.requires_grad:
                continue
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


def evaluate_fk_table(
    args,
    table_meta: Dict,
    all_table_meta: Dict[str, Dict],
    device: torch.device,
) -> Tuple[float, float, float, int]:
    model = table_meta["model"]
    (
        hyperedge,
        train_hyper_node,
        train_ve_affiliation,
        _,
        _,
        _,
        _,
    ) = table_meta["data"]
    model.eval()
    with torch.no_grad():
        row_self, _ = compute_row_embeddings(
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
        return evaluate_fk_metrics(args, table_meta, row_self, other_row_embs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--missing_ratio", type=float, default=0.3)
    parser.add_argument("--missing_mechanism", type=str, default="MCAR")
    parser.add_argument("--fk_missing_ratio", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--pretrain_epochs", type=int, default=2000)
    parser.add_argument("--finetune_epochs", type=int, default=400)
    parser.add_argument("--stage", type=str, default="joint", choices=["joint", "pretrain_num", "finetune_fk", "two_stage"])
    parser.add_argument("--save_pretrain_ckpt_dir", type=str, default="")
    parser.add_argument("--load_pretrain_ckpt_dir", type=str, default="")
    parser.add_argument("--known", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--fk_match_weight", type=float, default=0.1)
    parser.add_argument("--fk_neg_k", type=int, default=32)
    parser.add_argument("--fk_max_pairs", type=int, default=4096)
    parser.add_argument("--fk_topk", type=int, default=5)
    parser.add_argument("--fk_eval_batch_size", type=int, default=1024)

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

    fk_missing_ratio = args.fk_missing_ratio
    if fk_missing_ratio is None:
        fk_missing_ratio = args.missing_ratio

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

        data = get_data_from_table(values, args.missing_ratio, args.missing_mechanism, args.seed, mask_cols=None)
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

        fk_observed = {}
        for fk_col in table.fkey_col_to_pkey_table.keys():
            fk_observed[fk_col] = build_fk_observed_mask(
                table.df[fk_col],
                missing_ratio=fk_missing_ratio,
                missing_mechanism=args.missing_mechanism,
                seed=args.seed,
            )
        table_meta[table_name]["fk_observed"] = fk_observed

    for table_name, meta in table_meta.items():
        meta["fk_row_indices"] = build_fk_row_indices(
            meta["df"],
            table_meta,
            meta["fkey_map"],
            meta["fk_observed"],
        )
        meta["fk_true_row_indices"] = build_fk_true_row_indices(
            meta["df"],
            table_meta,
            meta["fkey_map"],
        )

    for table_name, meta in table_meta.items():
        init_models(args, meta, device)

    if args.stage == "joint":
        configure_stage_optimizers(args, table_meta, "joint")
        train_all(args, table_meta, device, stage="joint", epochs=args.epochs)
    elif args.stage == "pretrain_num":
        configure_stage_optimizers(args, table_meta, "pretrain_num")
        train_all(args, table_meta, device, stage="pretrain_num", epochs=args.pretrain_epochs)
        if args.save_pretrain_ckpt_dir:
            save_pretrain_checkpoint(table_meta, args.save_pretrain_ckpt_dir)
            print(f"Saved pretrain checkpoints to {args.save_pretrain_ckpt_dir}")
    elif args.stage == "finetune_fk":
        if not args.load_pretrain_ckpt_dir:
            raise ValueError("--load_pretrain_ckpt_dir is required for stage=finetune_fk")
        load_pretrain_checkpoint(table_meta, args.load_pretrain_ckpt_dir, device)
        print(f"Loaded pretrain checkpoints from {args.load_pretrain_ckpt_dir}")
        configure_stage_optimizers(args, table_meta, "finetune_fk")
        train_all(args, table_meta, device, stage="finetune_fk", epochs=args.finetune_epochs)
    elif args.stage == "two_stage":
        configure_stage_optimizers(args, table_meta, "pretrain_num")
        train_all(args, table_meta, device, stage="pretrain_num", epochs=args.pretrain_epochs)
        if args.save_pretrain_ckpt_dir:
            save_pretrain_checkpoint(table_meta, args.save_pretrain_ckpt_dir)
            print(f"Saved pretrain checkpoints to {args.save_pretrain_ckpt_dir}")
        configure_stage_optimizers(args, table_meta, "finetune_fk")
        train_all(args, table_meta, device, stage="finetune_fk", epochs=args.finetune_epochs)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")

    rmse_list: List[float] = []
    mae_list: List[float] = []
    count_list: List[int] = []
    fk_top1_list: List[float] = []
    fk_recall_list: List[float] = []
    fk_mrr_list: List[float] = []
    fk_count_list: List[int] = []

    for table_name, meta in table_meta.items():
        rmse, mae, count = evaluate_table(args, meta, table_meta, device)
        rmse_list.append(rmse)
        mae_list.append(mae)
        count_list.append(count)
        print(f"{table_name}: rmse={rmse:.4f} mae={mae:.4f}")
        if args.stage != "pretrain_num":
            f_top1, f_rec, f_mrr, f_n = evaluate_fk_table(args, meta, table_meta, device)
            if f_n > 0:
                fk_top1_list.append(f_top1)
                fk_recall_list.append(f_rec)
                fk_mrr_list.append(f_mrr)
                fk_count_list.append(f_n)
                print(
                    f"{table_name}: fk_top1={f_top1:.4f} fk_recall@{args.fk_topk}={f_rec:.4f} "
                    f"fk_mrr={f_mrr:.4f} n={f_n}"
                )

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

    if args.stage != "pretrain_num" and fk_count_list:
        fk_total = sum(fk_count_list)
        avg_top1 = float(np.sum(np.array(fk_top1_list) * np.array(fk_count_list)) / fk_total)
        avg_rec = float(np.sum(np.array(fk_recall_list) * np.array(fk_count_list)) / fk_total)
        avg_mrr = float(np.sum(np.array(fk_mrr_list) * np.array(fk_count_list)) / fk_total)
        print(
            f"FK AVG: top1={avg_top1:.4f} recall@{args.fk_topk}={avg_rec:.4f} "
            f"mrr={avg_mrr:.4f} n={fk_total}"
        )


if __name__ == "__main__":
    main()
