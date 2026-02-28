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


def compute_fk_matching_loss(
    table_meta: Dict,
    row_self: torch.Tensor,
    other_row_embs: Dict[str, torch.Tensor],
    coarse_k: int,
    max_pairs: int,
    temperature: float,
    coarse_hard_k: int,
    coarse_loss_weight: float,
    rerank_loss_weight: float,
) -> torch.Tensor:
    match_heads: nn.ModuleDict = table_meta["fk_match_heads"]
    rerank_heads: nn.ModuleDict = table_meta["fk_rerank_heads"]
    total_loss = torch.tensor(0.0, device=row_self.device)
    rel_count = 0
    for fk_col, target_table in table_meta["fkey_map"].items():
        if fk_col not in match_heads or fk_col not in rerank_heads:
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

        src_emb = match_heads[fk_col](row_self[src_idx])
        all_scores = src_emb @ target_rows.t()
        if temperature > 0:
            all_scores = all_scores / temperature

        num_tgt = target_rows.size(0)
        k_pool = min(coarse_k, num_tgt)
        if k_pool <= 0:
            continue

        cand_idx = torch.topk(all_scores, k=k_pool, dim=1).indices
        # Ensure positive is inside candidate pool for supervised reranking.
        has_pos = (cand_idx == pos_idx.unsqueeze(1)).any(dim=1)
        missing_pos = ~has_pos
        if torch.any(missing_pos):
            cand_idx[missing_pos, -1] = pos_idx[missing_pos]

        cand_emb = target_rows[cand_idx]
        q_expand = src_emb.unsqueeze(1).expand(-1, k_pool, -1)
        feat = torch.cat(
            [q_expand, cand_emb, torch.abs(q_expand - cand_emb), q_expand * cand_emb],
            dim=2,
        )
        rerank_scores = rerank_heads[fk_col](feat.reshape(-1, feat.size(-1))).reshape(-1, k_pool)

        labels = torch.argmax((cand_idx == pos_idx.unsqueeze(1)).to(torch.int64), dim=1)

        loss_rerank = F.cross_entropy(rerank_scores, labels)

        # Auxiliary coarse-ranking loss to directly train recall quality of the match head.
        pos_scores = all_scores.gather(1, pos_idx.unsqueeze(1))
        masked_scores = all_scores.clone()
        masked_scores.scatter_(1, pos_idx.unsqueeze(1), float("-inf"))
        k_hard = min(coarse_hard_k, max(1, num_tgt - 1))
        hard_neg_scores = torch.topk(masked_scores, k=k_hard, dim=1).values
        coarse_logits = torch.cat([pos_scores, hard_neg_scores], dim=1)
        coarse_labels = torch.zeros(src_emb.size(0), dtype=torch.long, device=row_self.device)
        loss_coarse = F.cross_entropy(coarse_logits, coarse_labels)

        total_loss = total_loss + coarse_loss_weight * loss_coarse + rerank_loss_weight * loss_rerank
        rel_count += 1

    if rel_count > 0:
        total_loss = total_loss / rel_count
    return total_loss


def evaluate_fk_metrics(
    args,
    table_meta: Dict,
    row_self: torch.Tensor,
    other_row_embs: Dict[str, torch.Tensor],
) -> Tuple[float, float, float, int, float, float, float, float, float]:
    match_heads: nn.ModuleDict = table_meta["fk_match_heads"]
    rerank_heads: nn.ModuleDict = table_meta["fk_rerank_heads"]
    top1_hit = 0.0
    topk_hit = 0.0
    mrr_sum = 0.0
    coarse_hit = 0.0
    total = 0
    pos_score_sum = 0.0
    neg_score_sum = 0.0
    hard_neg_score_sum = 0.0
    diag_total = 0
    for fk_col, target_table in table_meta["fkey_map"].items():
        if fk_col not in match_heads or fk_col not in rerank_heads:
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
        src_emb = match_heads[fk_col](row_self[src_idx])
        true_idx = true_idx_all[src_idx]

        batch_size = args.fk_eval_batch_size
        for st in range(0, src_emb.size(0), batch_size):
            ed = min(st + batch_size, src_emb.size(0))
            src_b = src_emb[st:ed]
            true_b = true_idx[st:ed]
            coarse_scores = src_b @ target_rows.t()
            if args.fk_loss_temp > 0:
                coarse_scores = coarse_scores / args.fk_loss_temp
            k_pool = min(args.fk_coarse_k, coarse_scores.size(1))
            if k_pool <= 0:
                continue
            cand_idx = torch.topk(coarse_scores, k=k_pool, dim=1).indices
            cand_emb = target_rows[cand_idx]
            q_expand = src_b.unsqueeze(1).expand(-1, k_pool, -1)
            feat = torch.cat(
                [q_expand, cand_emb, torch.abs(q_expand - cand_emb), q_expand * cand_emb],
                dim=2,
            )
            rerank_scores = rerank_heads[fk_col](feat.reshape(-1, feat.size(-1))).reshape(-1, k_pool)

            true_in_pool = (cand_idx == true_b.unsqueeze(1))
            coarse_hit += true_in_pool.any(dim=1).float().sum().item()
            pred_local = torch.argmax(rerank_scores, dim=1)
            pred_global = cand_idx.gather(1, pred_local.unsqueeze(1)).squeeze(1)
            top1_hit += (pred_global == true_b).float().sum().item()

            k = min(args.fk_topk, k_pool)
            topk_local = torch.topk(rerank_scores, k=k, dim=1).indices
            topk_global = cand_idx.gather(1, topk_local)
            topk_hit += (topk_global == true_b.unsqueeze(1)).any(dim=1).float().sum().item()

            true_local = torch.argmax(true_in_pool.to(torch.int64), dim=1)
            in_pool = true_in_pool.any(dim=1)
            true_scores = rerank_scores.gather(1, true_local.unsqueeze(1)).squeeze(1)
            rank = (rerank_scores > true_scores.unsqueeze(1)).sum(dim=1) + 1
            mrr_sum += (in_pool.float() / rank.float()).sum().item()
            total += src_b.size(0)

            if torch.any(in_pool):
                scores_in = rerank_scores[in_pool]
                true_local_in = true_local[in_pool]
                true_scores_in = scores_in.gather(1, true_local_in.unsqueeze(1)).squeeze(1)
                pos_score_sum += true_scores_in.sum().item()

                if k_pool > 1:
                    neg_mean = (scores_in.sum(dim=1) - true_scores_in) / float(k_pool - 1)
                    neg_score_sum += neg_mean.sum().item()

                    neg_scores = scores_in.clone()
                    neg_scores.scatter_(1, true_local_in.unsqueeze(1), float("-inf"))
                    hard_k = min(5, k_pool - 1)
                    hard_topk_vals = torch.topk(neg_scores, k=hard_k, dim=1).values
                    hard_neg_mean = hard_topk_vals.mean(dim=1)
                else:
                    neg_score_sum += 0.0
                    hard_neg_mean = torch.zeros_like(true_scores_in)
                hard_neg_score_sum += hard_neg_mean.sum().item()
                diag_total += scores_in.size(0)

    if total == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    if diag_total > 0:
        pos_mean = pos_score_sum / diag_total
        neg_mean = neg_score_sum / diag_total
        margin = pos_mean - neg_mean
        hard_neg_mean = hard_neg_score_sum / diag_total
        hard_margin = pos_mean - hard_neg_mean
    else:
        pos_mean = float("nan")
        neg_mean = float("nan")
        margin = float("nan")
        hard_margin = float("nan")
    return (
        top1_hit / total,
        topk_hit / total,
        mrr_sum / total,
        total,
        pos_mean,
        neg_mean,
        margin,
        hard_margin,
        coarse_hit / total,
    )


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
    fk_rerank_heads = nn.ModuleDict()
    for fk_col in table_meta["fkey_map"].keys():
        fk_match_heads[fk_col] = nn.Linear(args.hyperedge_dim_hidden, args.hyperedge_dim_hidden).to(device)
        hidden = args.fk_rerank_hidden if args.fk_rerank_hidden > 0 else args.hyperedge_dim_hidden
        fk_rerank_heads[fk_col] = nn.Sequential(
            nn.Linear(args.hyperedge_dim_hidden * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(device)
    table_meta["model"] = model
    table_meta["impute_model"] = impute_model
    table_meta["fk_gate"] = gate_layer
    table_meta["fk_match_heads"] = fk_match_heads
    table_meta["fk_rerank_heads"] = fk_rerank_heads
    table_meta["optimizer"] = None


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad = enabled


def set_fk_finetune_backbone_trainable(model: nn.Module, mode: str) -> List[nn.Parameter]:
    """Return backbone params to train during FK finetune (and set requires_grad)."""
    set_requires_grad(model, False)
    if mode == "freeze":
        return []
    if mode == "all_gnn":
        set_requires_grad(model, True)
        return [p for p in model.parameters() if p.requires_grad]
    if mode == "last_gnn":
        params: List[nn.Parameter] = []
        if hasattr(model, "v2e_layers") and len(model.v2e_layers) > 0:
            set_requires_grad(model.v2e_layers[-1], True)
            params.extend(list(model.v2e_layers[-1].parameters()))
        if hasattr(model, "e2v_layers") and len(model.e2v_layers) > 0:
            set_requires_grad(model.e2v_layers[-1], True)
            params.extend(list(model.e2v_layers[-1].parameters()))
        return params
    raise ValueError(f"Unsupported fk finetune backbone mode: {mode}")


def configure_stage_optimizers(args, all_table_meta: Dict[str, Dict], stage: str) -> None:
    for table_meta in all_table_meta.values():
        model = table_meta["model"]
        impute_model = table_meta["impute_model"]
        fk_gate = table_meta["fk_gate"]
        fk_match_heads = table_meta["fk_match_heads"]
        fk_rerank_heads = table_meta["fk_rerank_heads"]

        if stage == "joint":
            set_requires_grad(model, True)
            set_requires_grad(impute_model, True)
            set_requires_grad(fk_gate, True)
            set_requires_grad(fk_match_heads, True)
            set_requires_grad(fk_rerank_heads, True)
            params = (
                list(model.parameters())
                + list(impute_model.parameters())
                + list(fk_gate.parameters())
                + list(fk_match_heads.parameters())
                + list(fk_rerank_heads.parameters())
            )
        elif stage == "pretrain_num":
            set_requires_grad(model, True)
            set_requires_grad(impute_model, True)
            set_requires_grad(fk_gate, True)
            set_requires_grad(fk_match_heads, False)
            set_requires_grad(fk_rerank_heads, False)
            params = list(model.parameters()) + list(impute_model.parameters()) + list(fk_gate.parameters())
        elif stage == "finetune_fk":
            backbone_params = set_fk_finetune_backbone_trainable(model, args.fk_finetune_backbone_mode)
            set_requires_grad(impute_model, False)
            set_requires_grad(fk_gate, False)
            set_requires_grad(fk_match_heads, True)
            set_requires_grad(fk_rerank_heads, True)
            head_params = list(fk_match_heads.parameters()) + list(fk_rerank_heads.parameters())
            if backbone_params:
                params = [
                    {"params": head_params, "lr": args.lr},
                    {"params": backbone_params, "lr": args.fk_backbone_lr},
                ]
            else:
                params = head_params
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
                fk_pos_list: List[float] = []
                fk_neg_list: List[float] = []
                fk_margin_list: List[float] = []
                fk_hard_margin_list: List[float] = []
                fk_coarse_recall_list: List[float] = []
                for table_name, meta in all_table_meta.items():
                    (
                        f_top1,
                        f_rec,
                        f_mrr,
                        f_n,
                        f_pos,
                        f_neg,
                        f_margin,
                        f_hard_margin,
                        f_coarse_rec,
                    ) = evaluate_fk_table(args, meta, all_table_meta, device)
                    if f_n > 0:
                        fk_top1_list.append(f_top1)
                        fk_recall_list.append(f_rec)
                        fk_mrr_list.append(f_mrr)
                        fk_count_list.append(f_n)
                        fk_pos_list.append(f_pos)
                        fk_neg_list.append(f_neg)
                        fk_margin_list.append(f_margin)
                        fk_hard_margin_list.append(f_hard_margin)
                        fk_coarse_recall_list.append(f_coarse_rec)
                if fk_count_list:
                    fk_total = sum(fk_count_list)
                    avg_top1 = float(np.sum(np.array(fk_top1_list) * np.array(fk_count_list)) / fk_total)
                    avg_rec = float(np.sum(np.array(fk_recall_list) * np.array(fk_count_list)) / fk_total)
                    avg_mrr = float(np.sum(np.array(fk_mrr_list) * np.array(fk_count_list)) / fk_total)
                    avg_pos = float(np.sum(np.array(fk_pos_list) * np.array(fk_count_list)) / fk_total)
                    avg_neg = float(np.sum(np.array(fk_neg_list) * np.array(fk_count_list)) / fk_total)
                    avg_margin = float(np.sum(np.array(fk_margin_list) * np.array(fk_count_list)) / fk_total)
                    avg_hard_margin = float(
                        np.sum(np.array(fk_hard_margin_list) * np.array(fk_count_list)) / fk_total
                    )
                    avg_coarse_rec = float(
                        np.sum(np.array(fk_coarse_recall_list) * np.array(fk_count_list)) / fk_total
                    )
                    print(
                        f"  [{stage}] eval fk_top1={avg_top1:.4f} fk_recall@{args.fk_topk}={avg_rec:.4f} "
                        f"fk_mrr={avg_mrr:.4f} n={fk_total}"
                    )
                    print(f"  [{stage}] eval fk_coarse_recall@{args.fk_coarse_k}={avg_coarse_rec:.4f}")
                    print(
                        f"  [{stage}] eval fk_diag s_pos={avg_pos:.4f} s_neg={avg_neg:.4f} "
                        f"margin={avg_margin:.4f} hard_margin={avg_hard_margin:.4f}"
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
                    coarse_k=args.fk_coarse_k,
                    max_pairs=args.fk_max_pairs,
                    temperature=args.fk_loss_temp,
                    coarse_hard_k=args.fk_coarse_hard_k,
                    coarse_loss_weight=args.fk_coarse_loss_weight,
                    rerank_loss_weight=args.fk_rerank_loss_weight,
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
) -> Tuple[float, float, float, int, float, float, float, float]:
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
    # Keep fk_neg_k for backward compatibility with existing scripts.
    parser.add_argument("--fk_neg_k", type=int, default=32)
    parser.add_argument("--fk_hard_k", type=int, default=-1)
    parser.add_argument("--fk_coarse_k", type=int, default=128)
    parser.add_argument("--fk_rerank_hidden", type=int, default=64)
    parser.add_argument("--fk_loss_temp", type=float, default=1.0)
    parser.add_argument("--fk_coarse_hard_k", type=int, default=32)
    parser.add_argument("--fk_coarse_loss_weight", type=float, default=1.0)
    parser.add_argument("--fk_rerank_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--fk_finetune_backbone_mode",
        type=str,
        default="freeze",
        choices=["freeze", "last_gnn", "all_gnn"],
    )
    parser.add_argument("--fk_backbone_lr", type=float, default=1e-4)
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
    if args.fk_hard_k <= 0:
        args.fk_hard_k = args.fk_neg_k
    if args.fk_coarse_k <= 0:
        args.fk_coarse_k = args.fk_hard_k
    if args.fk_coarse_hard_k <= 0:
        args.fk_coarse_hard_k = args.fk_hard_k

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
    fk_coarse_recall_list: List[float] = []

    for table_name, meta in table_meta.items():
        rmse, mae, count = evaluate_table(args, meta, table_meta, device)
        rmse_list.append(rmse)
        mae_list.append(mae)
        count_list.append(count)
        print(f"{table_name}: rmse={rmse:.4f} mae={mae:.4f}")
        if args.stage != "pretrain_num":
            (
                f_top1,
                f_rec,
                f_mrr,
                f_n,
                f_pos,
                f_neg,
                f_margin,
                f_hard_margin,
                f_coarse_rec,
            ) = evaluate_fk_table(args, meta, table_meta, device)
            if f_n > 0:
                fk_top1_list.append(f_top1)
                fk_recall_list.append(f_rec)
                fk_mrr_list.append(f_mrr)
                fk_count_list.append(f_n)
                fk_coarse_recall_list.append(f_coarse_rec)
                print(
                    f"{table_name}: fk_top1={f_top1:.4f} fk_recall@{args.fk_topk}={f_rec:.4f} "
                    f"fk_mrr={f_mrr:.4f} n={f_n}"
                )
                print(f"{table_name}: fk_coarse_recall@{args.fk_coarse_k}={f_coarse_rec:.4f}")
                print(
                    f"{table_name}: fk_diag s_pos={f_pos:.4f} s_neg={f_neg:.4f} "
                    f"margin={f_margin:.4f} hard_margin={f_hard_margin:.4f}"
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
        avg_coarse_rec = float(np.sum(np.array(fk_coarse_recall_list) * np.array(fk_count_list)) / fk_total)
        print(
            f"FK AVG: top1={avg_top1:.4f} recall@{args.fk_topk}={avg_rec:.4f} "
            f"mrr={avg_mrr:.4f} n={fk_total}"
        )
        print(f"FK AVG: coarse_recall@{args.fk_coarse_k}={avg_coarse_rec:.4f} n={fk_total}")


if __name__ == "__main__":
    main()
