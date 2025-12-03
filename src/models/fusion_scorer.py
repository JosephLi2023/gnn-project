import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from subgraph_extraction import extract_subgraph
from query_guided_pooling import pool_subgraph

# --- Parameters ---
num_candidates = 100
similarity_threshold = 0.6
max_depth = 2
pooled_emb_dim = 128  # Set according to your pooling output
query_emb_dim = 384   # Set according to your encoder output

class TripletSubgraphDataset(Dataset):
    def __init__(self, samples, encoder, data, pooling_fn, max_depth=2, threshold=0.6):
        self.samples = samples  # List of (query_id, movie_id)
        self.encoder = encoder
        self.data = data
        self.pooling_fn = pooling_fn
        self.max_depth = max_depth
        self.threshold = threshold
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        query_id, movie_id = self.samples[idx]
        # Comment: Will run the NN on the entire graph many times, making it very slow
        query_emb = self.encoder(self.data.x_dict, self.data.edge_index_dict)['conversation'][query_id]
        subgraph = extract_subgraph(self.data, [movie_id], query_emb, similarity_threshold=self.threshold, max_depth=self.max_depth)
        movie_embs = subgraph['movie'].x
        pooled_emb = self.pooling_fn(subgraph, node_type='movie', query_emb=query_emb)
        # Comment: Comparing floats can fail dure to precision errors
        label = (movie_embs == self.data['movie'].x[movie_id]).all(dim=1).nonzero(as_tuple=True)[0].item()  # index of correct movie in subgraph
        return query_emb, pooled_emb, movie_embs, label

class QuerySubgraphFusionScorer(nn.Module):
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
        combined = torch.cat([subgraph_emb, query_emb], dim=-1)
        fused = self.fusion(combined)
        score = self.score_head(fused)
        return score.squeeze(-1)

def train_fusion_scorer(scorer, dataloader, optimizer, epochs=10, margin=1.0, device='cuda'):
    scorer.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for query_emb, subgraph_emb, label in dataloader:
            query_emb = query_emb.to(device)
            label = label.to(device)
            subgraph_emb = subgraph_emb.to(device)
            pos_score = scorer(subgraph_emb, query_emb)
            # Comment: "scores" is not defined
            loss = criterion(scores.unsqueeze(0), label.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}")

def build_faiss_index(movie_embs):
    index = faiss.IndexFlatL2(movie_embs.shape[1])
    index.add(movie_embs.cpu().numpy())
    return index

def retrieve_top_k(query_emb, movie_embs, faiss_index, k=5):
    D, I = faiss_index.search(query_emb.cpu().numpy().reshape(1, -1), k)
    return I[0]

def recommend_movies(query, encoder, scorer, data, faiss_index, movie_ids, top_k=5, device='cuda'):
    # Step 1: Encode query
    if isinstance(query, int):
        query_emb = encoder(data.x_dict, data.edge_index_dict)['conversation'][query].to(device)
    else:
        text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        query_emb = text_encoder.encode(query, convert_to_tensor=True).to(device)
    # Step 2: Retrieve candidate movies
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
            continue
        pooled_emb = pool_subgraph(subgraph, node_type='movie', query_emb=query_emb)
        score = scorer(pooled_emb.unsqueeze(0), query_emb.unsqueeze(0))
        subgraph_scores.append(score.item())
        subgraph_pools.append(pooled_emb)
        subgraphs.append(subgraph)
    if not subgraph_scores:
        return []
    # Comment: Picking only one single subgraph might limit diversity and recall, what if rank the movies intead, a top-k
    best_idx = torch.tensor(subgraph_scores).argmax().item()
    best_subgraph = subgraphs[best_idx]
    best_movie_embs = best_subgraph['movie'].x
    best_movie_ids = getattr(best_subgraph['movie'], 'node_ids', list(range(best_movie_embs.shape[0])))
    movie_scores = torch.matmul(best_movie_embs, query_emb)
    ranked = movie_scores.argsort(descending=True)
    recommended = [best_movie_ids[i] for i in ranked[:top_k]]
    return recommended

# --- Example Main Script ---
if __name__ == "__main__":
    encoder = ...  # Your pretrained GNN encoder
    data = ...     # HeteroData object
    movie_ids = ...  # List of movie node ids
    triplets = ...   # List of (query_id, pos_movie_id, neg_movie_id)
    pooling_fn = pool_subgraph  # Ensure this is defined
    movie_embs = encoder(data.x_dict, data.edge_index_dict)['movie'].detach()
    query_embs = encoder(data.x_dict, data.edge_index_dict)['conversation'].detach()
    faiss_index = build_faiss_index(movie_embs)
    dataset = TripletSubgraphDataset(triplets, encoder, data, pooling_fn, max_depth=max_depth, threshold=similarity_threshold)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    scorer = QuerySubgraphFusionScorer(subgraph_dim=pooled_emb_dim, query_dim=query_emb_dim).to('cuda')
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=5e-4, weight_decay=1e-4)
    train_fusion_scorer(scorer, dataloader, optimizer, epochs=10, margin=1.0, device='cuda')
    query = "I want to watch a romance movie in civil war"
    recommended = recommend_movies(query, encoder, scorer, data, faiss_index, movie_ids, top_k=5, device='cuda')
    print("Top 5 recommended movies:", recommended)
