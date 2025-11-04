import torch
import torch.nn as nn

class QueryGuidedPooling(nn.Module):
    def __init__(self, node_dim, query_dim, hidden_dim=128):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, node_embs, query_emb, batch=None):
        # node_embs: [N, node_dim], query_emb: [B, query_dim]
        node_proj = self.node_proj(node_embs)  # [N, H]
        query_proj = self.query_proj(query_emb)  # [B, H]
        attn_scores = self.attn(torch.tanh(node_proj.unsqueeze(0) + query_proj.unsqueeze(1))).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * node_embs.unsqueeze(0), dim=1)
        return self.dropout(pooled)
