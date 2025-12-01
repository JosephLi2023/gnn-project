import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from subgraph_extraction import extract_subgraph
from query_guided_pooling import pool_subgraph

class TripletSubgraphDataset(Dataset):
    def __init__(self, triplets, encoder, data, pooling_fn, max_depth=2, threshold=0.6):
        self.triplets = triplets
        self.encoder = encoder
        self.data = data
        self.pooling_fn = pooling_fn
        self.max_depth = max_depth
        self.threshold = threshold
    def __len__(self):
        return len(self.triplets)
    def __getitem__(self, idx):
        query_id, pos_movie_id, neg_movie_id = self.triplets[idx]
        query_emb = self.encoder(self.data.x_dict, self.data.edge_index_dict)['conversation'][query_id]
        # Extract and pool positive subgraph
        pos_subgraph = extract_subgraph(self.data, [pos_movie_id], query_emb, similarity_threshold=self.threshold, max_depth=self.max_depth)
        pos_pool = self.pooling_fn(pos_subgraph, node_type='movie', query_emb=query_emb)
        # Extract and pool negative subgraph
        neg_subgraph = extract_subgraph(self.data, [neg_movie_id], query_emb, similarity_threshold=self.threshold, max_depth=self.max_depth)
        neg_pool = self.pooling_fn(neg_subgraph, node_type='movie', query_emb=query_emb)
        return query_emb, pos_pool, neg_pool

class QuerySubgraphFusionScorer(nn.Module):
    """
    Fusion head for scoring the relevance between a query embedding and a pooled subgraph embedding.
    Inputs:
        subgraph_emb: [batch, emb_dim] - pooled embedding from query-guided pooling over subgraph
        query_emb: [batch, query_dim] - query embedding (from text encoder or conversation node)
    Output:
        score: [batch] - relevance score for each (query, subgraph) pair
    """
    def __init__(self, subgraph_dim, query_dim, fusion_dims=[512, 256, 128]):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(subgraph_dim + query_dim, fusion_dims[0]),
            nn.ReLU(),
            nn.Linear(fusion_dims[0], fusion_dims[1]),
            nn.ReLU(),
            nn.Linear(fusion_dims[1], fusion_dims[2]),
            nn.ReLU()
        )
        self.score_head = nn.Sequential(
            nn.Linear(fusion_dims[2], fusion_dims[2]//4),
            nn.ReLU(),
            nn.Linear(fusion_dims[2]//4, 1)
        )
    def forward(self, subgraph_emb, query_emb):
        # subgraph_emb: [batch, emb_dim]
        # query_emb: [batch, query_dim]
        combined = torch.cat([subgraph_emb, query_emb], dim=-1)
        fused = self.fusion(combined)
        score = self.score_head(fused)
        return score.squeeze(-1)


def train_fusion_scorer(scorer, dataloader, optimizer, epochs=10, margin=1.0, device='cuda'):
    """
    Train the fusion scorer using triplet supervision:
    For each batch: (query_emb, pos_subgraph_emb, neg_subgraph_emb)
    """
    scorer.train()
    for epoch in range(epochs):
        total_loss = 0
        for query_emb, pos_subgraph_emb, neg_subgraph_emb in dataloader:
            query_emb = query_emb.to(device)
            pos_subgraph_emb = pos_subgraph_emb.to(device)
            neg_subgraph_emb = neg_subgraph_emb.to(device)
            pos_score = scorer(pos_subgraph_emb, query_emb)
            neg_score = scorer(neg_subgraph_emb, query_emb)
            loss = -torch.nn.functional.logsigmoid(pos_score - neg_score - margin).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}")

    
# --- FAISS Index for Retrieval ---
def build_faiss_index(movie_embs):
    index = faiss.IndexFlatL2(movie_embs.shape[1])
    index.add(movie_embs.cpu().numpy())
    return index

def retrieve_top_k(query_emb, movie_embs, faiss_index, k=5):
    D, I = faiss_index.search(query_emb.cpu().numpy().reshape(1, -1), k)
    return I[0]  # indices of top-k movies

# --- End-to-End Recommendation Function ---
def recommend_movies(query, encoder, scorer, data, faiss_index, movie_ids, top_k=5, device='cuda'):
    # Step 1: Encode query
    if isinstance(query, int):  # query is a conversation node id
        query_emb = encoder(data.x_dict, data.edge_index_dict)['conversation'][query].to(device)
    else:  # query is text, use your text encoder
        text_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your text encoder of choice
        query_emb = text_encoder.encode(query, convert_to_tensor=True).to(device)  # [query_dim]

    # Step 2: Retrieve candidate movies (e.g., top 100)
    movie_embs = encoder(data.x_dict, data.edge_index_dict)['movie'].to(device)
    topk_idx = retrieve_top_k(query_emb, movie_embs, faiss_index, k=num_candidates)
    candidate_movie_ids = [movie_ids[i] for i in topk_idx]

    # Step 3: For each candidate, extract subgraph and pool
    subgraph_scores = []
    subgraph_pools = []
    subgraphs = []
    for movie_id in candidate_movie_ids:
        subgraph = extract_subgraph(data, [movie_id], query_emb, similarity_threshold=similarity_threshold, max_depth=max_depth)
        movie_nodes = subgraph['movie'].x
        if movie_nodes.shape[0] == 0:
            continue  # skip empty subgraphs
        pooled_emb = query_guided_pooling(movie_nodes, query_emb)
        score = scorer(pooled_emb.unsqueeze(0), query_emb.unsqueeze(0))  # [1]
        subgraph_scores.append(score.item())
        subgraph_pools.append(pooled_emb)
        subgraphs.append(subgraph)
    
    # Step 4: Select subgraph with highest score
    if not subgraph_scores:
        return []  # No valid subgraphs found
    best_idx = torch.tensor(subgraph_scores).argmax().item()
    best_subgraph = subgraphs[best_idx]
    # Step 5: Rank movies within best subgraph
    best_movie_embs = best_subgraph['movie'].x
    best_movie_ids = best_subgraph['movie'].node_ids if hasattr(best_subgraph['movie'], 'node_ids') else list(range(best_movie_embs.shape[0]))
    movie_scores = torch.matmul(best_movie_embs, query_emb)
    ranked = movie_scores.argsort(descending=True)
    recommended = [best_movie_ids[i] for i in ranked[:top_k]]
    return recommended

# --- Example Main Script ---
if __name__ == "__main__":
    # Load pretrained GNN encoder, HeteroData, and triplet training data
    encoder = ...  # Your pretrained GNN encoder
    data = ...     # HeteroData object
    movie_ids = ...  # List of movie node ids
    triplets = ...   # List of (query_id, pos_movie_id, neg_movie_id)
    movie_embs = encoder(data.x_dict, data.edge_index_dict)['movie'].detach()
    query_embs = encoder(data.x_dict, data.edge_index_dict)['conversation'].detach()
    # Build FAISS index
    faiss_index = build_faiss_index(movie_embs)   
    # Prepare dataset and dataloader
    dataset = TripletSubgraphDataset(triplets, movie_embs, query_embs)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    # Initialize scorer
    scorer = QuerySubgraphFusionScorer(subgraph_dim=pooled_emb_dim,
    query_dim=query_emb_dim).to('cuda')

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=5e-4, weight_decay=1e-4)
    # Train scorer
    train_fusion_scorer(scorer, dataloader, optimizer, epochs=10, margin=1.0, device='cuda')
    # Recommend movies for a query
    query = "I want to watch a romance movie in civil war"
    recommended = recommend_movies(query, encoder, scorer, data, faiss_index, movie_ids, top_k=5, device='cuda')
    print("Top 5 recommended movies:", recommended)
