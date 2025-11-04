import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class QueryEncoder(nn.Module):
  """
  Encondes raw test query into a vector.
  """
  def __init__(self, model_name='all-MiniLM-L6-v2'):
    super().__init__()

    self.model = SentenceTransformer(model_name)
    self.embedding_dim = self.model.get_sentence_embedding_dimension()

  def forward(self, query_text):
    embeddings = self.model.encode(
        query_text,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    return embeddings
  
class CandidateRetriever(nn.Module):
  """
  Finds the Top-k most topically-relevant movies for a query.
  """
  def __init__(self, movie_tag_embeddings):
    super().__init__()

    self.register_buffer(
        'movie_tag_embeddings',
        F.normalize(movie_tag_embeddings, p=2, dim=-1)
    )

  def forward(self, query_embedding, k=100):
    query_emb_norm = F.normalize(query_embedding, p=2, dim=-1)
    scores = torch.matmul(query_emb_norm, self.movie_tag_embeddings.T)

    top_k_scores, top_k_indices = torch.topk(scores, k=k, dim=1)

    return top_k_scores, top_k_indices

class RetrievalStage(nn.Module):
  """
  The full Stage 1 model: QueryEncoder + CandidateRetriever
  """
  def __init__(self, movie_tag_embeddings, model_name='all-MiniLM-L6-v2'):
    super().__init__()
    self.query_encoder = QueryEncoder(model_name=model_name)

    tag_embedding_dim = movie_tag_embeddings.shape[1]
    query_embedding_dim = self.query_encoder.embedding_dim
    
    if tag_embedding_dim != query_embedding_dim:
      print(f"Warning: Query dim ({query_embedding_dim}) != Tag dim ({tag_embedding_dim})")
      self.projection = nn.Linear(query_embedding_dim, tag_embedding_dim)
    else:
      self.projection = nn.Identity()
            
    self.retriever = CandidateRetriever(movie_tag_embeddings)

  def forward(self, query_text, k=100):
      query_embedding = self.query_encoder(query_text)

      projected_query_embedding = self.projection(query_embedding)

      top_k_scores, top_k_indices = self.retriever(projected_query_embedding, k)
      return top_k_scores, top_k_indices
