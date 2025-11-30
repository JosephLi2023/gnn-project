import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss

# Graph Libraries
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, HeteroConv, Linear

# Text Libraries
from sentence_transformers import SentenceTransformer

print("Imports successful.")
print(f"Using Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

class HeteroSAGEEncoder(nn.Module):
    def __init__(self, metadata, hidden_size=128, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lin_dict = nn.ModuleDict()
        
        # Standardize Input: Assume 384 dimensions (from SentenceTransformers)
        for node_type in metadata[0]:
            self.lin_dict[node_type] = nn.Linear(384, hidden_size)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = SAGEConv(
                    in_channels=hidden_size,
                    out_channels=hidden_size
                )
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

    def forward(self, x_dict, edge_index_dict):
        # 1. Project all inputs to hidden_size
        x_dict = {key: self.lin_dict[key](x) for key, x in x_dict.items()}
        # 2. Run GNN
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return x_dict

class FusionScorer(nn.Module):
  def __init__(self, emb_dim, query_dim, fusion_dim=[512, 256, 128, 32, 1]):
    super().__init__()

    self.query_proj = nn.Linear(query_dim, emb_dim)

    self.net = nn.Sequential(
        nn.Linear(emb_dim + emb_dim, fusion_dim[0]),
        nn.ReLU(),

        nn.Linear(fusion_dim[0], fusion_dim[1]),
        nn.ReLU(),

        nn.Linear(fusion_dim[1], fusion_dim[2]),
        nn.ReLU(),

        nn.Linear(fusion_dim[2], fusion_dim[3]),
        nn.ReLU(),
        
        nn.Linear(fusion_dim[3], fusion_dim[4])
    )

  def forward(self, movie_emb, query_emb):
    query_proj = self.query_proj(query_emb)
    x = torch.cat((movie_emb, query_proj), dim=1)
    x = self.net(x)
    return x.squeeze(-1)

class ScoreHeadTrainer():
  def __init__(self, model, optimizer, margin):
    self.model = model
    self.optimizer = optimizer

    self.criterion = nn.MarginRankingLoss(margin=margin)

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  def get_hard_negative_weights(self, cur_epoch, total_epochs):
    # Mostly ignore hard negatives at low epochs
    return 0.1 + (0.9 * (cur_epoch/total_epochs))

  def train_step(self, data_loader, epoch_idx, total_epochs):
    self.model.train()
    
    alpha = self.get_hard_negative_weights(epoch_idx, total_epochs)

    running_loss = 0.0

    for query_emb, pos_emb, hard_neg_emb, soft_neg_emb in data_loader:
      query_emb = query_emb.to(self.device)
      pos_emb = pos_emb.to(self.device)
      hard_neg_emb = hard_neg_emb.to(self.device)
      soft_neg_emb = soft_neg_emb.to(self.device)

      self.optimizer.zero_grad()

      pos_score = self.model(pos_emb, query_emb)
      hard_neg_score = self.model(hard_neg_emb, query_emb)
      soft_neg_score = self.model(soft_neg_emb, query_emb)

      target = torch.ones_like(pos_score).to(self.device)
      
      loss_hard = self.criterion(pos_score, hard_neg_score, target)
      loss_soft = self.criterion(pos_score, soft_neg_score, target)

      final_loss = (alpha * loss_hard) + ((1 - alpha) * loss_soft)

      final_loss.backward()
      self.optimizer.step()

      running_loss += final_loss.item()

    return running_loss / len(data_loader)

class InferencePipeline():
  def __init__(self, graph_data, faiss_index_path, backbone_path, scorer_path, backbone_class=HeteroSAGEEncoder, backbone_params={}):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.graph = graph_data
    self.graph.to(self.device)
    self.faiss_index = faiss.read_index(faiss_index_path)
  
    self.backbone = backbone_class(metadata=graph_data.metadata(), **backbone_params)
    
    self.backbone.load_state_dict(torch.load(backbone_path, map_location=self.device))
    self.backbone.to(self.device)
    self.backbone.eval()

    self.scorer = FusionScorer(emb_dim=128, query_dim=384)
    self.scorer.load_state_dict(torch.load(scorer_path, map_location=self.device))
    self.scorer.to(self.device)
    self.scorer.eval()
    

    self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

  def extract_subgraph(self, movie_idx):
    """
    Extract a random 2-hop subgraph around a specific movie candidate
    """
    loader = NeighborLoader(
        self.graph,
        num_neighbors=[20, 10],
        input_nodes=('movie', torch.tensor([movie_idx], device=self.device)),
        batch_size=1,
        shuffle=False
    )
    subgraph = next(iter(loader))
    return subgraph.to(self.device)

  def extract_subgraph_nvidia(self, movie_idx, query_emb, depth=2, threshold=0.5):
    """
    Implements NVIDIA's Query-Guided Subgraph Expansion.
    Instead of random sampling, we only walk to neighbors that are semantically 
    similar to the query.
    """
    query_norm = F.normalize(query_emb.view(1, -1), p=2, dim=1)

    subset_dict = {t: set() for t in self.graph.node_types}
    subset_dict['movie'].add(movie_idx)

    frontier = [('movie', movie_idx)]
    visited = set([('movie', movie_idx)])

    for _ in range(depth):
      new_frontier = []
      for src_type, src_idx in frontier:
        for edge_type in self.graph.edge_types:
          src_t, _, dst_t = edge_type

          if src_type != src_t:
            continue
          
          edge_index = self.graph[edge_type].edge_index

          mask = edge_index[0] == src_idx
          neighbor_indices = edge_index[1][mask]

          if len(neighbor_indices) == 0:
            continue

          neighbor_embs = self.graph[dst_t].x[neighbor_indices]
          neighbor_embs_norm = F.normalize(neighbor_embs, p=2, dim=1)

          sims = (neighbor_embs_norm @ query_norm.T).squeeze(-1)
          
          if sims.dim() == 0:
            sims = sims.unsqueeze(0)
          
          is_relevant = sims > threshold
          relevant_indices = neighbor_indices[is_relevant]

          for idx in relevant_indices.tolist():
            node_key = (dst_t, idx)
            if node_key not in visited:
              visited.add(node_key)
              new_frontier.append(node_key)
              subset_dict[dst_t].add(idx)

      frontier = new_frontier
      if not frontier:
        break

    final_subset_dict = {
        k : torch.tensor(list(v), device=self.device)
        for k, v in subset_dict.items()
        if len(v) > 0
    }

    subgraph = self.graph.subgraph(final_subset_dict)
    return subgraph
  

  def recommend(self, query_text, k=10):
    query_vec_numpy = self.text_encoder.encode(query_text).reshape(1, -1)
    D, I = self.faiss_index.search(query_vec_numpy, k)
    candidate_indices = I[0]

    query_emb = torch.tensor(query_vec_numpy, device=self.device).float()
    projected_query = self.scorer.query_proj(query_emb)

    scored_candidates = []

    for movie_idx in candidate_indices:
      #subgraph = self.extract_subgraph(movie_idx)
      subgraph = self.extract_subgraph_nvidia(movie_idx, query_emb, depth=2, threshold=0.6)

      subgraph_embs = self.backbone(subgraph.x_dict, subgraph.edge_index_dict)

      attn_scores = torch.matmul(subgraph_embs['movie'], projected_query.T)
      weights = F.softmax(attn_scores, dim=0)
      
      pooled_emb = torch.sum(subgraph_embs['movie'] * weights, dim=0)
      pooled_emb = pooled_emb.unsqueeze(0)

      score = self.scorer(pooled_emb, query_emb)
      scored_candidates.append((movie_idx, score.item()))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    final_ids = [x[0] for x in scored_candidates]
    final_scores = [x[1] for x in scored_candidates]

    return final_ids, final_scores
