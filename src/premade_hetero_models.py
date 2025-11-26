"""
Examples of pre-built heterogeneous GNN models from PyTorch Geometric.
You can swap these into backbone_trainer_clean.py to compare performance.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import (
    HeteroConv,
    RGCNConv,
    GATConv,
    SAGEConv,
    HGTConv,
    to_hetero
)
import torch.nn.functional as F

class RGCNEncoder(nn.Module):
    def __init__(self, metadata, x_dict, hidden_size=128, num_layers=2, num_bases=None):
        super().__init__()
        self.hidden_size = hidden_size

        # Input projections for each node type
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            in_channels = x_dict[node_type].size(1)
            self.lin_dict[node_type] = nn.Linear(in_channels, hidden_size)

        # Get all edge types for R-GCN
        num_relations = len(metadata[1])

        # num_bases controls parameter sharing across relations
        # Rule of thumb: num_bases = num_relations // 5 or None for no sharing
        if num_bases is None:
            num_bases = max(num_relations // 5, 1)

        # R-GCN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                RGCNConv(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    num_relations=num_relations,
                    num_bases=num_bases
                )
            )

        self.metadata = metadata

    def forward(self, x_dict, edge_index_dict):
        # Project input features
        x_dict = {key: self.lin_dict[key](x) for key, x in x_dict.items()}

        # Convert to homogeneous format for R-GCN
        # R-GCN expects a single node feature matrix and edge_type tensor
        node_types = list(x_dict.keys())
        edge_types = list(edge_index_dict.keys())

        # Create node type mapping
        node_offset = {}
        offset = 0
        x_list = []
        for node_type in node_types:
            node_offset[node_type] = offset
            x_list.append(x_dict[node_type])
            offset += x_dict[node_type].size(0)

        x = torch.cat(x_list, dim=0)

        # Create edge_index and edge_type
        edge_index_list = []
        edge_type_list = []
        for i, edge_type in enumerate(edge_types):
            src_type, _, dst_type = edge_type
            edge_index = edge_index_dict[edge_type]
            # Offset indices
            edge_index = edge_index.clone()
            edge_index[0] += node_offset[src_type]
            edge_index[1] += node_offset[dst_type]
            edge_index_list.append(edge_index)
            edge_type_list.append(torch.full((edge_index.size(1),), i, dtype=torch.long, device=edge_index.device))

        edge_index = torch.cat(edge_index_list, dim=1)
        edge_type = torch.cat(edge_type_list, dim=0)

        # Apply R-GCN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)

        # Split back to heterogeneous format
        x_dict_out = {}
        for node_type in node_types:
            start = node_offset[node_type]
            end = start + x_dict[node_type].size(0)
            x_dict_out[node_type] = x[start:end]

        return x_dict_out


class HANEncoder(nn.Module):
    def __init__(self, metadata, x_dict, hidden_size=128, num_layers=2, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size

        # Input projections
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            in_channels = x_dict[node_type].size(1)
            self.lin_dict[node_type] = nn.Linear(in_channels, hidden_size)

        # HeteroConv layers with GAT
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                src_type, _, dst_type = edge_type
                # Use multi-head attention
                conv_dict[edge_type] = GATConv(
                    in_channels=hidden_size,
                    out_channels=hidden_size // num_heads,
                    heads=num_heads,
                    concat=True,
                    add_self_loops=False
                )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        self.bns = nn.ModuleList([
            nn.ModuleDict({nt: nn.BatchNorm1d(hidden_size) for nt in metadata[0]})
            for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict):
        # Project input features
        x_dict = {key: self.lin_dict[key](x) for key, x in x_dict.items()}

        # Apply HeteroConv layers
        for conv, bn in zip(self.convs, self.bns):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: bn[key](x) for key, x in x_dict.items()}
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        return x_dict


# ============================================================================
# Option 3: HGT (Heterogeneous Graph Transformer)
# ============================================================================
class HGTEncoder(nn.Module):
    def __init__(self, metadata, x_dict, hidden_size=128, num_layers=2, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size

        # Input projections
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            in_channels = x_dict[node_type].size(1)
            self.lin_dict[node_type] = nn.Linear(in_channels, hidden_size)

        # HGT layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    metadata=metadata,
                    heads=num_heads
                )
            )

    def forward(self, x_dict, edge_index_dict):
        # Project input features
        x_dict = {key: self.lin_dict[key](x) for key, x in x_dict.items()}

        # Apply HGT layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict

class HeteroSAGEEncoder(nn.Module):
    def __init__(self, metadata, x_dict, hidden_size=128, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size

        # Input projections
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            in_channels = x_dict[node_type].size(1)
            self.lin_dict[node_type] = nn.Linear(in_channels, hidden_size)

        # HeteroConv layers with SAGE
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = SAGEConv(
                    in_channels=hidden_size,
                    out_channels=hidden_size
                )
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

        self.bns = nn.ModuleList([
            nn.ModuleDict({nt: nn.BatchNorm1d(hidden_size) for nt in metadata[0]})
            for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict):
        # Project input features
        x_dict = {key: self.lin_dict[key](x) for key, x in x_dict.items()}

        # Apply HeteroConv layers
        for conv, bn in zip(self.convs, self.bns):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: bn[key](x) for key, x in x_dict.items()}
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        return x_dict

