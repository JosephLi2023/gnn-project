import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryGuidedPoolingWithProjection(nn.Module):
    """
    Query-guided pooling with learnable projection to shared embedding space.
    """
    def __init__(self, movie_emb_dim: int, query_emb_dim: int, shared_dim: int = 256):
        super(QueryGuidedPoolingWithProjection, self).__init__()

        # Project movie and query embeddings to shared space
        self.movie_proj = nn.Sequential(
            nn.Linear(movie_emb_dim, shared_dim),
            nn.ReLU(),
            nn.LayerNorm(shared_dim)
        )

        self.query_proj = nn.Sequential(
            nn.Linear(query_emb_dim, shared_dim),
            nn.ReLU(),
            nn.LayerNorm(shared_dim)
        )

    def forward(self, movie_embs: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            movie_embs: [num_movies, movie_emb_dim]
            query_emb: [query_emb_dim] or [1, query_emb_dim]

        Returns:
            pooled_emb: [shared_dim]
        """
        # Project to shared space
        movie_proj = self.movie_proj(movie_embs)  # [num_movies, shared_dim]

        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        query_proj = self.query_proj(query_emb)  # [1, shared_dim]

        # Compute attention scores
        attn_scores = torch.matmul(movie_proj, query_proj.T).squeeze(-1)  # [num_movies]
        attn_weights = F.softmax(attn_scores, dim=0)  # [num_movies]

        # Weighted sum
        pooled_emb = torch.sum(movie_proj * attn_weights.unsqueeze(-1), dim=0)  # [shared_dim]

        return pooled_emb

