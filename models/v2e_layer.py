import torch

from torch.nn.init import xavier_uniform_, zeros_

import torch.nn as nn
import torch.nn.functional as F
from utils_core import get_activation
class V2E_layer(nn.Module):
    
    def __init__(self, edge_in_channels, edge_out_channels, node_in_channels, activation):
        super(V2E_layer, self).__init__()

        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels
        self.node_in_channels = node_in_channels

        self.v2e_lin = nn.Linear(node_in_channels, edge_out_channels)
        self.update_lin = nn.Linear(edge_in_channels + edge_out_channels, edge_out_channels)

        self.v2e_activation = get_activation(activation)
        self.update_activation = get_activation(activation)

    def forward(self, hyperedge, hyper_node, ve_affiliation):

        num_hyperedges = hyperedge.size(0)
        
        # Hypernode to hyperedge
        node_info = self.v2e_activation(self.v2e_lin(hyper_node))
        out = scatter_mean(node_info, ve_affiliation[0], dim_size=num_hyperedges)
        
        # Update hyperedge
        out = self.update_activation(self.update_lin(torch.cat([out, hyperedge], dim=-1)))
        
        out = F.normalize(out, p=2, dim=-1)

        return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    ones = torch.ones(index.size(0), device=src.device, dtype=src.dtype)
    count.index_add_(0, index, ones)
    count = count.clamp(min=1).unsqueeze(-1)
    return out / count
