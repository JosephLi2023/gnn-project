import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer

# --- Fusion Scorer ---
class FusionScorer(nn.Module):
    def __init__(self, emb_dim, query_dim, fusion_dims=[512, 256, 128]):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim + query_dim, fusion_dims[0]),
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
    def forward(self, movie_emb, query_emb):
        combined = torch.cat([movie_emb, query_emb], dim=-1)
        fused = self.fusion(combined)
        score = self.score_head(fused)
        return score.squeeze(-1)


def train_scorer(scorer, dataloader, optimizer, epochs=10, margin=1.0, device='cuda'):
    scorer.train()
    for epoch in range(epochs):
        total_loss = 0
        for query_emb, pos_emb, neg_emb in dataloader:
            query_emb = query_emb.to(device)
            pos_emb = pos_emb.to(device)
            neg_emb = neg_emb.to(device)
            pos_score = scorer(pos_emb, query_emb)
            neg_score = scorer(neg_emb, query_emb)
            loss = -F.logsigmoid(pos_score - neg_score - margin).mean()
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
        query_emb = text_encoder(query).to(device)  # [query_dim]

    # Step 2: Retrieve candidate movies
    movie_embs = encoder(data.x_dict, data.edge_index_dict)['movie'].to(device)
    
    topk_idx = retrieve_top_k(query_emb, movie_embs, faiss_index, k=top_k)
    # Step 3: Score and rank
    
    candidate_embs = movie_embs[topk_idx]
    
    scores = scorer(candidate_embs, query_emb.expand_as(candidate_embs))
    ranked = scores.argsort(descending=True)
    
    return [movie_ids[topk_idx[i]] for i in ranked]


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
    dataset = TripletMovieDataset(triplets, movie_embs, query_embs)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    # Initialize scorer
    scorer = FusionScorer(emb_dim=movie_embs.shape[1], query_dim=query_embs.shape[1]).to('cuda')
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=5e-4, weight_decay=1e-4)
    # Train scorer
    train_scorer(scorer, dataloader, optimizer, epochs=10, margin=1.0, device='cuda')
    # Recommend movies for a query
    query = "I want to watch a romance movie in civil war"
    recommended = recommend_movies(query, encoder, scorer, data, faiss_index, movie_ids, top_k=5, device='cuda')
    print("Top 5 recommended movies:", recommended)
