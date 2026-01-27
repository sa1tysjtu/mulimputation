from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gen_imp import get_gen_imp
from models.e2v_layer import E2V_layer
from models.v2e_layer import V2E_layer
from models.imputaion_model import LinearHead


@dataclass
class RelationSpec:
    src_table: str
    fk_col: str
    dst_table: str

    @property
    def key(self) -> str:
        return f"{self.src_table}:{self.fk_col}->{self.dst_table}"


def _relation_key(src_table: str, fk_col: str, dst_table: str) -> str:
    return f"{src_table}:{fk_col}->{dst_table}"


class MultiTableImputer(nn.Module):
    def __init__(
        self,
        relation_specs: List[RelationSpec],
        hidden_dim: int = 64,
        gnn_layers: int = 3,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gnn_layers = gnn_layers
        self.dropout = dropout
        self.activation = activation

        # Base GNN shared across tables (edge dim=32, node dim=1).
        self.base_gnn = get_gen_imp(32, 1, _Args(
            hyperedge_dim_hidden=hidden_dim,
            hyper_node_dim_hidden=hidden_dim,
            gnn_layer_num=gnn_layers,
            dropout=dropout,
            gnn_activation=activation,
        ))

        self.value_head = LinearHead(
            input_dims=hidden_dim * 2,
            output_dim=1,
            hidden_layer_sizes=hidden_dim,
            hidden_activation=activation,
            dropout=dropout,
        )

        self.row_to_cell = E2V_layer(hidden_dim, 1, hidden_dim, activation)
        self.cell_to_edge = V2E_layer(hidden_dim, hidden_dim, hidden_dim, activation)

        self.relation_transforms = nn.ModuleDict()
        for spec in relation_specs:
            self.relation_transforms[spec.key] = nn.Linear(hidden_dim, hidden_dim)
        self.shared_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.cat_heads = nn.ModuleDict()

    def compute_table_embeddings(
        self,
        hyperedge: torch.Tensor,
        hyper_node: torch.Tensor,
        ve_affiliation: torch.Tensor,
        train_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_dup = torch.cat([train_mask, train_mask], dim=0)
        known_node = hyper_node[mask_dup]
        known_aff = ve_affiliation[:, mask_dup]
        hyperedge_emb, _ = self.base_gnn(hyperedge, known_node, known_aff)
        return hyperedge_emb

    def row_cell_update(
        self,
        hyperedge: torch.Tensor,
        hyper_node: torch.Tensor,
        ve_affiliation: torch.Tensor,
        train_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_dup = torch.cat([train_mask, train_mask], dim=0)
        known_node = hyper_node[mask_dup]
        known_aff = ve_affiliation[:, mask_dup]

        cell_emb = self.row_to_cell(hyperedge, known_node, known_aff)
        hyperedge_new = self.cell_to_edge(hyperedge, cell_emb, known_aff)
        return hyperedge_new

    def propagate_observed_fk(
        self,
        row_embs: Dict[str, torch.Tensor],
        edges: List[Tuple[str, int, str, str, int]],
    ) -> Dict[str, torch.Tensor]:
        agg: Dict[str, torch.Tensor] = {}
        counts: Dict[str, torch.Tensor] = {}
        for src_table, src_row, fk_col, dst_table, dst_row in edges:
            rel_key = _relation_key(src_table, fk_col, dst_table)
            if rel_key not in self.relation_transforms:
                continue
            if src_table not in row_embs or dst_table not in row_embs:
                continue

            src_vec = row_embs[src_table][src_row]
            msg = self.relation_transforms[rel_key](src_vec)
            if dst_table not in agg:
                agg[dst_table] = torch.zeros_like(row_embs[dst_table])
                counts[dst_table] = torch.zeros(row_embs[dst_table].size(0), device=src_vec.device)
            agg[dst_table][dst_row] += msg
            counts[dst_table][dst_row] += 1

            # reverse direction
            dst_vec = row_embs[dst_table][dst_row]
            rev_key = _relation_key(dst_table, fk_col, src_table)
            if rev_key in self.relation_transforms and src_table in row_embs:
                rev_msg = self.relation_transforms[rev_key](dst_vec)
                if src_table not in agg:
                    agg[src_table] = torch.zeros_like(row_embs[src_table])
                    counts[src_table] = torch.zeros(row_embs[src_table].size(0), device=src_vec.device)
                agg[src_table][src_row] += rev_msg
                counts[src_table][src_row] += 1

        for table_name, table_agg in agg.items():
            denom = counts[table_name].clamp(min=1).unsqueeze(-1)
            mean_msg = table_agg / denom
            if self.shared_gate is None:
                row_embs[table_name] = row_embs[table_name] + mean_msg
                continue
            gate = torch.sigmoid(
                self.shared_gate(torch.cat([row_embs[table_name], mean_msg], dim=-1))
            )
            row_embs[table_name] = gate * row_embs[table_name] + (1 - gate) * mean_msg

        return row_embs

    def candidate_pool(
        self,
        src_vec: torch.Tensor,
        dst_emb: torch.Tensor,
        k_near: int,
        k_rand: int,
        pool_size: int = 2000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_dst = dst_emb.size(0)
        if num_dst == 0:
            return torch.empty(0, dtype=torch.long, device=dst_emb.device), torch.empty(0, device=dst_emb.device)

        if num_dst > pool_size:
            rand_idx = torch.randperm(num_dst, device=dst_emb.device)[:pool_size]
            pool_emb = dst_emb[rand_idx]
            sim = torch.matmul(pool_emb, src_vec)
            topk = torch.topk(sim, min(k_near, sim.numel())).indices
            near_idx = rand_idx[topk]
        else:
            sim = torch.matmul(dst_emb, src_vec)
            near_idx = torch.topk(sim, min(k_near, sim.numel())).indices

        if k_rand > 0:
            rand_idx = torch.randperm(num_dst, device=dst_emb.device)[:k_rand]
            cand_idx = torch.unique(torch.cat([near_idx, rand_idx]))
        else:
            cand_idx = near_idx

        return cand_idx, dst_emb[cand_idx]

    def matching_logits(
        self,
        src_vec: torch.Tensor,
        dst_emb: torch.Tensor,
        cand_idx: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        cand_emb = dst_emb[cand_idx]
        logits = torch.matmul(cand_emb, src_vec) / temperature
        return logits


class _Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
