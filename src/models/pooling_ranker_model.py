import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from pooling_with_projection import QueryGuidedPoolingWithProjection
from subgraph import Subgraph

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
        query_emb: torch.Tensor,                # shape: [D_q]
        subgraphs: List[Subgraph]               # list of Subgraph objects
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            query_emb: [D_q] tensor
            subgraphs: List of Subgraph objects
        Returns:
            logits: [num_subgraphs] tensor (score for each subgraph)
            graph_ids: List[int] (subgraph root ids)
        """
        pooled_embs = [self.pooling(torch.stack(sg.embeddings, dim=0), query_emb) for sg in subgraphs]
        item_embs = torch.stack(pooled_embs, dim=0)
        logits = self.scorer(item_embs, query_emb)
        graph_ids = [sg.graph_id for sg in subgraphs]
        return logits, graph_ids
