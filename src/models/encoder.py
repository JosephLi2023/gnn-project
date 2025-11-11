import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, HeteroConv, SAGEConv
from transformers import BertTokenizer, BertModel

# --- GAT Encoder with configurable layers ---
class GATEncoder(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=384, num_layers=3, num_heads=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        for i in range(num_layers):
            if i == 0:
                in_channels = (-1,-1) # lazy init helps with heterogenous graphs
                out_channels = hidden_dim
                heads = num_heads
                concat = True
            elif i == num_layers - 1:
                in_channels = hidden_dim * num_heads
                out_channels = out_dim
                heads = 1
                concat = False
            else:
                in_channels = hidden_dim * num_heads
                out_channels = hidden_dim
                heads = num_heads
                concat = True

            layer = HeteroConv({
                ('movie', 'has_genre', 'genre'): 
                    GATv2Conv(in_channels, out_channels, heads=heads, concat=concat, 
                           dropout=dropout, add_self_loops=False),
                ('genre', 'has_movie', 'movie'): 
                    GATv2Conv(in_channels, out_channels, heads=heads, concat=concat,
                           dropout=dropout, add_self_loops=False),
                ('user', 'rated_high', 'movie'):
                    GATv2Conv(in_channels, out_channels, heads=heads, concat=concat,
                           dropout=dropout, add_self_loops=False),
                ('movie', 'rated_by', 'user'):
                    GATv2Conv(in_channels, out_channels, heads=heads, concat=concat,
                           dropout=dropout, add_self_loops=False),
                ('conversation', 'mentions', 'movie'): 
                    GATv2Conv(in_channels, out_channels, heads=heads, concat=concat,
                           dropout=dropout, add_self_loops=False),
                ('movie', 'mentioned_in', 'conversation'):
                    GATv2Conv(in_channels, out_channels, heads=heads, concat=concat,
                           dropout=dropout, add_self_loops=False)
            }, aggr='sum')
            self.layers.append(layer)

    def forward(self, x_dict, edge_index_dict):
        """
        Args:
            x_dict: Dict[node_type, Tensor[num_nodes, feat_dim]]
            edge_index_dict: Dict[edge_type, Tensor[2, num_edges]]
        Returns:
            x_dict: Dict[node_type, Tensor[num_nodes, out_dim]]
        """
        for i, layer in enumerate(self.layers):
            x_dict = layer(x_dict, edge_index_dict)
            
            # Apply activation and dropout (except last layer)
            if i != self.num_layers - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                         for key, x in x_dict.items()}
        
        return x_dict
