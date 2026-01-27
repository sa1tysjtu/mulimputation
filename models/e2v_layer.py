import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_core import get_activation

class E2V_layer(nn.Module):
    
    def __init__(self, edge_in_channels, node_in_channels, node_out_channels, activation):
        super(E2V_layer, self).__init__()

        self.edge_in_channels = edge_in_channels
        self.node_in_channels = node_in_channels
        self.node_out_channels = node_out_channels

        self.e2v_lin = nn.Linear(edge_in_channels*2+node_in_channels, node_out_channels)
        self.e2v_activation = get_activation(activation)

    def forward(self, hyperedge, hyper_node, ve_affiliation):

        edge_i = hyperedge[ve_affiliation[0],:]
        edge_j = hyperedge[ve_affiliation[1],:]

        hyper_node = self.e2v_lin(torch.cat((edge_i, edge_j, hyper_node),dim=-1))
        out = self.e2v_activation(hyper_node)

        # out = F.normalize(out, p=2, dim=-1)

        return out
    
