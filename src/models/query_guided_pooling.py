import torch
import torch.nn.functional as F

def query_guided_pooling(movie_embs, query_emb):
    """
    Query-guided pooling using attention mechanism.
    Args:
        movie_embs (torch.Tensor): [num_movies_in_subgraph, emb_dim]
        query_emb (torch.Tensor): [emb_dim] or [1, emb_dim]
    Returns:
        pooled_emb (torch.Tensor): [emb_dim]
    """
    # Ensure query_emb is [1, emb_dim]
    if query_emb.dim() == 1:
        query_emb = query_emb.unsqueeze(0)
    # Compute attention scores (dot product)
    attn_scores = torch.matmul(movie_embs, query_emb.T).squeeze(-1)  # [num_movies_in_subgraph]
    attn_weights = F.softmax(attn_scores, dim=0)  # [num_movies_in_subgraph]
    # Weighted sum of movie embeddings
    pooled_emb = torch.sum(movie_embs * attn_weights.unsqueeze(-1), dim=0)  # [emb_dim]
    return pooled_emb

def pool_subgraph(subgraph, node_type='movie', method='qgp', query_emb=None):
    movie_embs = subgraph[node_type].x  # [num_movies_in_subgraph, emb_dim]
    if method == 'qgp':
        assert query_emb is not None, "Query embedding required for query-guided pooling"
        return query_guided_pooling(movie_embs, query_emb)
    elif method == 'mean':
        return movie_embs.mean(dim=0)
    elif method == 'max':
        return movie_embs.max(dim=0)[0]
