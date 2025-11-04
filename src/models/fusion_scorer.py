import torch
import torch.nn as nn

class FusionScorer(nn.Module):
    def __init__(self, graph_dim, query_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(graph_dim + query_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.ReLU(),
        )
        self.skip = nn.Linear(graph_dim + query_dim, hidden_dims[-1])
        self.scoring_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, pooled_graph, query_emb):
        x = torch.cat([pooled_graph, query_emb], dim=-1)
        h = self.mlp(x) + self.skip(x)
        return self.scoring_head(h).squeeze(-1)
