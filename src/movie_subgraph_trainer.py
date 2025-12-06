import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from ranker_model import MovieRanker
from movie_subgraph_dataloader import MovieSubgraphRecommendationDataset
import faiss
import pandas as pd
import numpy as np
import sentence_transformers
from utils.utils import load_graph

def build_topk_with_label(
    logits: torch.Tensor,
    movie_ids: List[int],
    label_ids: List[int],
    k: int,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Ensure all label movies are present among the K items:
      - take top-(k-len(label_ids))
      - add the label movies if they're not already there
    Returns:
      - selected_logits: [K] logits
      - selected_ids: List[int] length K
    """
    N = logits.size(0)
    if N == 0:
        return torch.empty(0, device=logits.device), []
    # Get top-(k-len(label_ids)) indices
    k_prime = max(1, min(k - len(label_ids), N))
    top_idx = torch.topk(logits, k=k_prime, largest=True, sorted=True).indices.tolist()
    # Add label movies to reach K items
    label_indices = [movie_ids.index(lid) for lid in label_ids if lid in movie_ids and movie_ids.index(lid) not in top_idx]
    selected_idx = top_idx + label_indices
    # If not enough, fill up to k
    if len(selected_idx) < k:
        all_sorted = torch.argsort(logits, descending=True).tolist()
        for i in all_sorted:
            if i not in selected_idx:
                selected_idx.append(i)
            if len(selected_idx) == k:
                break
    selected_logits = logits[selected_idx]
    selected_ids = [movie_ids[i] for i in selected_idx]
    return selected_logits, selected_ids

def train(
    model: MovieRanker,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    top_k: int = 5,
    log_every: int = 50,
):
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_used = 0
        total_skipped = 0
        for step, batch in enumerate(dataloader, start=1):
            batch_loss = 0.0
            batch_count = 0
            for query_emb, candidates_dict, label_ids in batch:
                # Skip if none of the labels are in the candidate set
                if not any(lid in candidates_dict for lid in label_ids):
                    total_skipped += 1
                    continue
                logits, movie_ids = model.forward(query_emb, candidates_dict)
                if logits.numel() == 0:
                    total_skipped += 1
                    continue
                selected_logits, selected_ids = build_topk_with_label(logits, movie_ids, label_ids, k=top_k)
                if selected_logits.numel() == 0:
                    total_skipped += 1
                    continue
                # Multi-hot targets
                targets = torch.zeros(selected_logits.size(0), device=device)
                for lid in label_ids:
                    try:
                        label_k_pos = selected_ids.index(lid)
                        targets[label_k_pos] = 1.0
                    except ValueError:
                        continue  # label not in selected_ids
                if targets.sum() == 0:
                    total_skipped += 1
                    continue
                loss = bce(selected_logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += float(loss.item())
                batch_count += 1
            if batch_count > 0:
                total_loss += batch_loss
                total_used += batch_count
            if log_every and step % log_every == 0:
                avg = batch_loss / max(1, batch_count)
                print(f"Epoch {epoch+1} Step {step}: batch_avg_loss={avg:.4f} used={batch_count} skipped={total_skipped}")
        epoch_avg = total_loss / max(1, total_used)
        print(f"Epoch {epoch+1}: avg_loss={epoch_avg:.4f} used={total_used} skipped={total_skipped}")
    
if __name__ == "__main__":
    # Example wiring — replace with your actual data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dimensions
    D_q = 384
    D_deep = 256
    # Build model
    model = MovieRanker(deep_emb_dim=D_deep, query_emb_dim=D_q, fusion_dims=[512, 256, 128], device=device)
    # Dummy dataset
    num_samples = 200
    num_candidates_per_sample = 50
    top_k = 5
    queries = [torch.randn(D_q) for _ in range(num_samples)]
    candidates = []
    labels = []
    for _ in range(num_samples):
        # build candidate dict
        movie_ids = torch.randint(0, 50000, (num_candidates_per_sample,)).tolist()
        cand_dict: Dict[int, torch.Tensor] = {mid: torch.randn(D_deep) for mid in movie_ids}
        # choose a label from candidates
        label_mid = movie_ids[0]
        candidates.append(cand_dict)
        labels.append(label_mid)

    train_path = 'data/processed/redial_ollama/train_summerized.tsv'
    embedding_path = 'TODO'

    movie_embs = pd.read_parquet('processed_data/movie_tags_embeddings.parquet').to_numpy()
    embeddings = np.vstack(movie_embs['embedding'].values).astype('float32')
    graph_data = load_graph(name='combined_graph_filtered').to(device)

    # Create node_id mapping (FAISS index position -> node_id)
    node_id_map = movie_embs['node_id'].tolist()
    # Get embedding dimension
    dim = embeddings.shape[1]
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    # Use inner product (higher = more similar)
    index = faiss.IndexFlatIP(dim)
    # Add embeddings to index
    index.add(embeddings)

    text_encoder = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

    ds = MovieSubgraphRecommendationDataset(training_data_path=train_path,
                                   emb_lookup_path=embedding_path,
                                   faiss_index=index,
                                   faiss_movie_ids=np.array(node_id_map),
                                   text_encoder=text_encoder,
                                   graph_data=graph_data)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    train(model, dl, epochs=5, lr=5e-4, weight_decay=1e-4, device=device, top_k=top_k)
