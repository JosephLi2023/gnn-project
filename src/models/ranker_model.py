import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class QueryFusionScorer(nn.Module):
    """
    Fusion scorer: concatenate [item_emb, query_emb] then score via MLP.
    Use for per-movie scoring with deep embeddings.
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
    
class MovieRanker(nn.Module):
    """
    Simplified ranker:
      - Forward takes a dict of movie_id -> deep movie emb and a query_emb
      - Computes scores for each movie
      - Returns (logits, movie_ids), where logits align with movie_ids order
    """
    def __init__(self, deep_emb_dim: int, query_emb_dim: int, fusion_dims: List[int] = [512, 256, 128], device: torch.device | None = None):
        super().__init__()
        self.scorer = QueryFusionScorer(item_dim=deep_emb_dim, query_dim=query_emb_dim, fusion_dims=fusion_dims)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(
        self,
        query_emb: torch.Tensor,
        candidate_deep_embs: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        query_emb: [D_q]
        candidate_deep_embs: {movie_id: torch.Tensor[D_deep]}
        returns:
           - logits: [N] tensor
           - movie_ids: List[int] of length N
        """
        # Flatten dict → tensors
        movie_ids: List[int] = list(candidate_deep_embs.keys())
        if len(movie_ids) == 0:
            return torch.empty(0, device=self.device), []
        # Stack embeddings in original dict order
        item_embs = torch.stack([candidate_deep_embs[mid].to(self.device).float() for mid in movie_ids], dim=0)  # [N, D_deep]
        query_emb = query_emb.to(self.device).float()
        logits = self.scorer(item_embs, query_emb)  # [N]
        return logits, movie_ids
        
    @torch.no_grad()
    def topk(
        self,
        query_emb: torch.Tensor,
        candidate_deep_embs: Dict[int, torch.Tensor],
        k: int = 5,
    ) -> List[int]:
        logits, movie_ids = self.forward(query_emb, candidate_deep_embs)
        if logits.numel() == 0:
            return []
        topk_idx = torch.topk(logits, k=min(k, logits.size(0)), largest=True, sorted=True).indices.tolist()
        return [movie_ids[i] for i in topk_idx]
