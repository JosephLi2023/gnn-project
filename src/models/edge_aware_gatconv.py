import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
class EdgeAwareGATConv(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_type_count, heads=2, dropout=0.6):
        super().__init__(aggr='add')
        self.heads = heads
        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.edge_type_emb = nn.Embedding(edge_type_count, out_dim)
        self.attn = nn.Parameter(torch.Tensor(1, heads, out_dim * 2))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.attn)
    
    def forward(self, x, edge_index, edge_type, edge_weight):
        x_proj = self.lin(x).view(-1, self.heads, x.size(-1))
        edge_emb = self.edge_type_emb(edge_type)
        return self.propagate(edge_index, x=x_proj, edge_attr=edge_emb, edge_weight=edge_weight)
    
    def message(self, x_i, x_j, edge_attr, edge_weight):
        # x_i, x_j: [E, heads, out_dim]
        cat = torch.cat([x_i, x_j + edge_attr], dim=-1)
        alpha = (cat * self.attn).sum(dim=-1)
        alpha = torch.nn.functional.leaky_relu(alpha)
        alpha = torch.nn.functional.softmax(alpha, dim=1)
        alpha = self.dropout(alpha)
        return x_j * alpha.unsqueeze(-1) * edge_weight.unsqueeze(-1)
