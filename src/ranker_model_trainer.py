import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from ranker_model import MovieRanker
from movie_dataloader import MovieRecommendationDataset
import faiss
import pandas as pd
import numpy as np
import sentence_transformers


def build_topk_with_label(
    logits: torch.Tensor,
    movie_ids: List[int],
    label_id: int,
    k: int,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Ensure the label is present among the K items:
      - take top-(k-1)
      - add the label movie if it's not already there
    Returns:
      - selected_logits: [K] logits
      - selected_ids: List[int] length K
    """
    N = logits.size(0)
    if N == 0:
        return torch.empty(0, device=logits.device), []
    # If label not in candidates, return empty to signal skip
    try:
        label_pos = movie_ids.index(label_id)
    except ValueError:
        return torch.empty(0, device=logits.device), []
    # Get top-(k-1) indices
    k_prime = max(1, min(k - 1, N))  # ensure at least 1 if k>=2
    top_idx = torch.topk(logits, k=k_prime, largest=True, sorted=True).indices.tolist()
    if label_pos not in top_idx:
        # Add label to reach K items
        selected_idx = top_idx + [label_pos]
    else:
        # Label already in top-(k-1); can add next best to keep K items if desired
        if len(top_idx) < k:
            # Find next best not already in top_idx
            all_sorted = torch.argsort(logits, descending=True).tolist()
            for i in all_sorted:
                if i not in top_idx:
                    selected_idx = top_idx + [i]
                    break
            else:
                selected_idx = top_idx  # fallback if N < k
        else:
            selected_idx = top_idx
    # Gather logits and ids
    selected_logits = logits[selected_idx]  # [K] (or <=K if not enough items)
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
            # process per-sample due to variable-sized candidate dicts
            batch_loss = 0.0
            batch_count = 0
            for query_emb, candidates_dict, label_id in batch:
                # Skip if label not in the provided candidate set
                if label_id not in candidates_dict:
                    total_skipped += 1
                    continue
                logits, movie_ids = model.forward(query_emb, candidates_dict)  # [N], List[int]
                if logits.numel() == 0:
                    total_skipped += 1
                    continue
                # Build K-set including label
                selected_logits, selected_ids = build_topk_with_label(logits, movie_ids, label_id, k=top_k)
                if selected_logits.numel() == 0:
                    total_skipped += 1
                    continue
                # Targets: binary vector over K items (1 at label position, else 0)
                targets = torch.zeros(selected_logits.size(0), device=device)
                try:
                    label_k_pos = selected_ids.index(label_id)
                    targets[label_k_pos] = 1.0
                except ValueError:
                    # Should not happen due to build_topk_with_label; skip defensively
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

    ds = MovieRecommendationDataset(training_data_path=train_path,
                                   emb_lookup_path=embedding_path,
                                   faiss_index=index,
                                   faiss_movie_ids=np.array(node_id_map),
                                   text_encoder=text_encoder,
                                   num_candidates=100)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    train(model, dl, epochs=5, lr=5e-4, weight_decay=1e-4, device=device, top_k=top_k)
