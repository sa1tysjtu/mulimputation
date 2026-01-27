import argparse
import copy
from typing import Dict, List

import torch
import torch.nn as nn

from multi_model import MultiTableImputer, RelationSpec
from multi_table_data import load_multi_table_data, FKRelation, MissingFK


def build_relation_specs(data) -> List[RelationSpec]:
    specs = {}
    for table_name, table in data.tables.items():
        for fk_col, dst_table in table.fkey_col_to_pkey_table.items():
            if table_name not in data.graphs or dst_table not in data.graphs:
                continue
            spec = RelationSpec(src_table=table_name, fk_col=fk_col, dst_table=dst_table)
            specs[spec.key] = spec
            rev = RelationSpec(src_table=dst_table, fk_col=fk_col, dst_table=table_name)
            specs[rev.key] = rev
    return list(specs.values())


def compute_fk_loss(
    model: MultiTableImputer,
    row_embs: Dict[str, torch.Tensor],
    observed_fk: List[FKRelation],
    k_near: int,
    k_rand: int,
    pool_size: int,
    temperature: float,
) -> torch.Tensor:
    if not observed_fk:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    losses = []
    for rel in observed_fk:
        if rel.src_table not in row_embs or rel.dst_table not in row_embs:
            continue
        src_emb = row_embs[rel.src_table][rel.src_row]
        dst_emb = row_embs[rel.dst_table]
        cand_idx, _ = model.candidate_pool(src_emb, dst_emb, k_near, k_rand, pool_size=pool_size)
        if cand_idx.numel() == 0:
            continue
        if rel.dst_row not in cand_idx.tolist():
            cand_idx = torch.cat(
                [cand_idx, torch.tensor([rel.dst_row], device=cand_idx.device)]
            )
        logits = model.matching_logits(src_emb, dst_emb, cand_idx, temperature=temperature)
        target = (cand_idx == rel.dst_row).nonzero(as_tuple=False).view(-1)
        if target.numel() == 0:
            continue
        target = target[0]
        loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), target.unsqueeze(0))
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(losses).mean()


def subsample_known_mask(train_mask: torch.Tensor, known: float, seed: int) -> torch.Tensor:
    if known >= 1.0:
        return train_mask
    train_idx = torch.nonzero(train_mask.view(-1), as_tuple=False).view(-1).cpu()
    if train_idx.numel() == 0:
        return train_mask
    keep = max(1, int(train_idx.numel() * known))
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    perm = torch.randperm(train_idx.numel(), generator=g)[:keep]
    keep_idx = train_idx[perm]
    out = torch.zeros_like(train_mask.view(-1), dtype=torch.bool)
    out[keep_idx] = True
    return out.view_as(train_mask).to(train_mask.device)


def apply_soft_fk_propagation(
    model: MultiTableImputer,
    row_embs: Dict[str, torch.Tensor],
    missing_fk: List[MissingFK],
    k_near: int,
    k_rand: int,
    pool_size: int,
    temperature: float,
) -> Dict[str, torch.Tensor]:
    agg: Dict[str, torch.Tensor] = {}
    counts: Dict[str, torch.Tensor] = {}
    for rel in missing_fk:
        if rel.src_table not in row_embs or rel.dst_table not in row_embs:
            continue
        src_emb = row_embs[rel.src_table][rel.src_row]
        dst_emb = row_embs[rel.dst_table]
        cand_idx, cand_emb = model.candidate_pool(src_emb, dst_emb, k_near, k_rand, pool_size=pool_size)
        if cand_idx.numel() == 0:
            continue
        weights = torch.softmax(torch.matmul(cand_emb, src_emb) / temperature, dim=0)
        msg = (weights.unsqueeze(-1) * cand_emb).sum(dim=0)
        if rel.src_table not in agg:
            agg[rel.src_table] = torch.zeros_like(row_embs[rel.src_table])
            counts[rel.src_table] = torch.zeros(
                row_embs[rel.src_table].size(0), device=src_emb.device
            )
        agg[rel.src_table][rel.src_row] += msg
        counts[rel.src_table][rel.src_row] += 1

    for table_name, table_agg in agg.items():
        denom = counts[table_name].clamp(min=1).unsqueeze(-1)
        mean_msg = table_agg / denom
        gate = torch.sigmoid(
            model.shared_gate(torch.cat([row_embs[table_name], mean_msg], dim=-1))
        )
        row_embs[table_name] = gate * row_embs[table_name] + (1 - gate) * mean_msg
    return row_embs


def train(args):
    data = load_multi_table_data(
        dataset_name=args.dataset,
        missing_ratio=args.missing_ratio,
        missing_mechanism=args.missing_mechanism,
        seed=args.seed,
    )

    relation_specs = build_relation_specs(data)
    model = MultiTableImputer(
        relation_specs=relation_specs,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        activation=args.activation,
    )

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    huber = nn.HuberLoss(delta=1.0)

    def compute_value_losses(mask_attr: str):
        loss_num = torch.tensor(0.0, device=device)
        loss_cat = torch.tensor(0.0, device=device)
        num_count = 0
        cat_count = 0
        rmse_sum = 0.0
        mae_sum = 0.0
        rmse_count = 0
        mae_count = 0
        se_sum = 0.0
        ae_sum = 0.0
        num_total = 0
        for table_name, graph in data.graphs.items():
            mask = getattr(graph, mask_attr).view(graph.num_rows, graph.num_cols).to(device)
            row_emb = row_embs[table_name]
            col_emb = col_embs[table_name]
            feature_df = data.tables[table_name].feature_df

            for col_idx, col_name in enumerate(graph.feature_cols):
                col_type = graph.feature_types[col_name]
                if col_type == "fkey":
                    continue
                row_idx = torch.nonzero(mask[:, col_idx], as_tuple=False).squeeze(-1)
                if row_idx.numel() == 0:
                    continue
                inputs = [
                    row_emb[row_idx],
                    col_emb[col_idx].unsqueeze(0).expand(row_idx.size(0), -1),
                ]
                if col_type in ("numerical", "categorical"):
                    pred = model.value_head(inputs, token_emb=[])[:, 0]
                    target = torch.tensor(
                        feature_df.iloc[row_idx.cpu().numpy()][col_name].to_numpy(),
                        dtype=torch.float,
                        device=device,
                    )
                    valid = torch.isfinite(target)
                    if valid.sum() == 0:
                        continue
                    pred = pred[valid]
                    target = target[valid]
                    loss_num = loss_num + huber(pred, target)
                    num_count += 1
                    se = (pred - target) ** 2
                    ae = torch.abs(pred - target)
                    se_sum += se.sum().item()
                    ae_sum += ae.sum().item()
                    num_total += int(valid.sum().item())

        if num_count > 0:
            loss_num = loss_num / num_count
        if cat_count > 0:
            loss_cat = loss_cat / cat_count
        if num_total > 0:
            rmse = (se_sum / num_total) ** 0.5
            mae = ae_sum / num_total
        else:
            rmse = float("nan")
            mae = float("nan")
        return loss_num, loss_cat, rmse, mae

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(args.epochs):
        epoch_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        epoch_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if epoch_start is not None:
            epoch_start.record()
        else:
            import time
            wall_start = time.time()

        model.train()
        optimizer.zero_grad()

        row_embs: Dict[str, torch.Tensor] = {}
        col_embs: Dict[str, torch.Tensor] = {}
        known_masks: Dict[str, torch.Tensor] = {}
        for name, graph in data.graphs.items():
            hyperedge = graph.hyperedge.to(device)
            hyper_node = graph.hyper_node.to(device)
            ve_aff = graph.ve_affiliation.to(device)
            train_mask = graph.train_mask.to(device)
            if args.use_known_mask:
                known_mask = subsample_known_mask(
                    train_mask,
                    known=args.known,
                    seed=args.seed + epoch,
                )
            else:
                known_mask = train_mask
            known_masks[name] = known_mask
            hyperedge_emb = model.compute_table_embeddings(hyperedge, hyper_node, ve_aff, known_mask)
            row_embs[name] = hyperedge_emb[: graph.num_rows]
            col_embs[name] = hyperedge_emb[graph.num_rows :]

        obs_edges = [
            (rel.src_table, rel.src_row, rel.fk_col, rel.dst_table, rel.dst_row)
            for rel in data.observed_fk
        ]
        row_embs = model.propagate_observed_fk(row_embs, obs_edges)
        row_embs = apply_soft_fk_propagation(
            model,
            row_embs,
            data.missing_fk,
            k_near=args.k_near,
            k_rand=args.k_rand,
            pool_size=args.pool_size,
            temperature=args.temperature,
        )

        # Row->cell update after cross-table propagation.
        for name, graph in data.graphs.items():
            if name not in row_embs:
                continue
            hyperedge = graph.hyperedge.to(device)
            hyper_node = graph.hyper_node.to(device)
            ve_aff = graph.ve_affiliation.to(device)
            known_mask = known_masks.get(name, graph.train_mask.to(device))
            updated_hyperedge = torch.cat([row_embs[name], col_embs[name]], dim=0)
            updated_hyperedge = model.row_cell_update(updated_hyperedge, hyper_node, ve_aff, known_mask)
            row_embs[name] = updated_hyperedge[: graph.num_rows]
            col_embs[name] = updated_hyperedge[graph.num_rows :]

        loss_num, loss_cat, _, _ = compute_value_losses("train_mask")

        loss_fk = compute_fk_loss(
            model,
            row_embs,
            data.observed_fk,
            k_near=args.k_near,
            k_rand=args.k_rand,
            pool_size=args.pool_size,
            temperature=args.temperature,
        )

        total_loss = (
            args.weight_num * loss_num
            + args.weight_fk * loss_fk
        )
        total_loss.backward()
        optimizer.step()

        if epoch_end is not None:
            epoch_end.record()
            torch.cuda.synchronize()
            elapsed_ms = epoch_start.elapsed_time(epoch_end)
            elapsed_str = f"{elapsed_ms/1000:.2f}s"
        else:
            import time
            elapsed_str = f"{time.time()-wall_start:.2f}s"

        print(
            f"epoch {epoch} total={total_loss.item():.4f} "
            f"num={loss_num.item():.4f} cat={loss_cat.item():.4f} "
            f"fk={loss_fk.item():.4f} time={elapsed_str}"
        )

        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_num, val_cat, val_rmse, _ = compute_value_losses("val_mask")
                print(f"  val num={val_num.item():.4f} rmse={val_rmse:.4f}")

                if val_rmse + args.min_delta < best_val:
                    best_val = val_rmse
                    best_state = {
                        "model": copy.deepcopy(model.state_dict()),
                        "optimizer": copy.deepcopy(optimizer.state_dict()),
                    }
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if args.patience > 0 and bad_epochs >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    model.eval()
    with torch.no_grad():
        eval_num, eval_cat, eval_rmse, eval_mae = compute_value_losses("test_mask")
        print(
            f"final test num={eval_num.item():.4f} "
            f"rmse={eval_rmse:.4f} mae={eval_mae:.4f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--missing_ratio", type=float, default=0.3)
    parser.add_argument("--missing_mechanism", type=str, default="MCAR")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--k_near", type=int, default=50)
    parser.add_argument("--k_rand", type=int, default=30)
    parser.add_argument("--pool_size", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--weight_num", type=float, default=1.0)
    parser.add_argument("--weight_fk", type=float, default=0.2)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--known", type=float, default=0.6)
    parser.add_argument("--use_known_mask", type=int, default=1)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)
