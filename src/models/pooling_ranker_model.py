import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from pooling_with_projection import QueryGuidedPoolingWithProjection

class QueryFusionScorer(nn.Module):
    """
    Fusion scorer: concatenate [item_emb, query_emb] then score via MLP.
    Use for per-graph scoring with deep embeddings.
    """
    def __init__(self, item_dim: int, query_dim: int, fusion_dims: List[int] = [512, 256, 128]):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(item_dim + query_dim, fusion_dims[0]),
            nn.ReLU(),
            nn.Linear(fusion_dims[0], fusion_dims[1]),
            nn.ReLU(),
            nn.Linear(fusion_dims[1], fusion_dims[2]),
            nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(fusion_dims[2], fusion_dims[2] // 4),
            nn.ReLU(),
            nn.Linear(fusion_dims[2] // 4, 1),
        )

    def forward(self, item_embs: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """
        item_embs: [N, D_item]
        query_emb: [D_query] or [N, D_query]
        returns: [N] logits
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)  # [1, Dq]
        if query_emb.size(0) != item_embs.size(0):
            query_emb = query_emb.expand(item_embs.size(0), -1)
        fused = torch.cat([item_embs, query_emb], dim=-1)  # [N, D_item + D_query]
        h = self.fusion(fused)                             # [N, H]
        logits = self.score_head(h).squeeze(-1)           # [N]
        return logits
    
class PoolingMovieRanker(nn.Module):
    """
    Ranker for subgraphs:
      - Forward takes a query_emb and a list of subgraphs (each: {graph_id, [embeddings]})
      - Pools each subgraph's embeddings with query_emb
      - Scores each pooled embedding
      - Returns (logits, graph_ids), where logits align with graph_ids order
    """
    def __init__(self, deep_emb_dim: int, query_emb_dim: int, fusion_dims: List[int] = [512, 256, 128], device: torch.device | None = None):
        super().__init__()
        self.pooling = QueryGuidedPoolingWithProjection(deep_emb_dim, query_emb_dim, shared_dim=deep_emb_dim)
        self.scorer = QueryFusionScorer(item_dim=deep_emb_dim, query_dim=query_emb_dim, fusion_dims=fusion_dims)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    def forward(
        self,
        query_emb: torch.Tensor,
        subgraphs: List[Dict],  # Each dict: {"graph_id": int, "embeddings": List[torch.Tensor]}
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        query_emb: [D_q]
        subgraphs: List of {"graph_id": int, "embeddings": List[torch.Tensor[D_deep]]}
        returns:
           - logits: [N] tensor
           - graph_ids: List[int] of length N
        """
        graph_ids = []
        pooled_embs = []
        for subgraph in subgraphs:
            graph_id = subgraph["graph_id"]
            emb_list = subgraph["embeddings"]
            if len(emb_list) == 0:
                continue
            graph_embs = torch.stack(emb_list, dim=0).to(self.device).float()  # [num_embs, D_deep]
            pooled_emb = self.pooling(graph_embs, query_emb)  # [D_deep]
            pooled_embs.append(pooled_emb)
            graph_ids.append(graph_id)
        if len(pooled_embs) == 0:
            return torch.empty(0, device=self.device), []
        item_embs = torch.stack(pooled_embs, dim=0)  # [N, D_deep]
        query_emb = query_emb.to(self.device).float()
        logits = self.scorer(item_embs, query_emb)  # [N]
        return logits, graph_ids
