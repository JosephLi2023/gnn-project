import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pooling_ranker_model import PoolingMovieRanker
from movie_subgraph_pooling_dataloader import MovieSubgraphPoolingRecommendationDataset
import faiss
import pandas as pd
import numpy as np
import sentence_transformers
from utils.utils import load_graph

def train(
    model: SubgraphRanker,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    device: torch.device = None,
    log_every: int = 50,
):
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_used = 0
        for step, batch in enumerate(dataloader, start=1):
            batch_loss = 0.0
            batch_count = 0
            query_embs = batch['query_emb']  # [batch_size, D_q]
            candidates_list = batch['candidates']  # List[List[Subgraph]]
            label_ids_list = batch['positive_movie_ids']  # List[List[int]]
            for query_emb, subgraphs, label_ids in zip(query_embs, candidates_list, label_ids_list):
                subgraph_positive_counts = [sum(mid in label_ids for mid in sg.movie_ids) for sg in subgraphs]
                total_positives = sum(subgraph_positive_counts)
                logits, graph_ids = model(query_emb, subgraphs)
                if total_positives == 0:
                    # All targets should be zero (no recommended movies in any subgraph)
                    targets = torch.zeros(len(subgraphs), device=device, dtype=torch.float32)
                else:
                    targets = torch.tensor([c / total_positives for c in subgraph_positive_counts], device=device, dtype=torch.float32)
                logits = logits.to(device)
                loss = mse(logits, targets)
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
                print(f"Epoch {epoch+1} Step {step}: batch_avg_loss={avg:.4f} used={batch_count}")
        epoch_avg = total_loss / max(1, total_used)
        print(f"Epoch {epoch+1}: avg_loss={epoch_avg:.4f} used={total_used}")
    
if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(42)
    # Example wiring — replace with your actual data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dimensions
    D_q = 384
    D_deep = 128
    # Build model
    model = PoolingMovieRanker(deep_emb_dim=D_deep, query_emb_dim=D_q, fusion_dims=[512, 256, 128], device=device)

    train_path = f"{BASE_PATH}/data/processed/redial/train_ml_id.tsv"
    embedding_path = f'{BASE_PATH}/models/movie_embeddings_sage.csv'

    movie_embs = pd.read_parquet(f'{BASE_PATH}/data/processed/movielens/movie_tags_embeddings.parquet')
    embeddings = np.vstack(movie_embs['tag_embedding'].values).astype('float32')

    # Create node_id mapping (FAISS index position -> node_id)
    node_id_map = movie_embs['movieId'].tolist()
    # Get embedding dimension
    dim = embeddings.shape[1]
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    # Initialize CPU index first
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(embeddings)

    # Check for GPU and move FAISS index if available
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources() # use a single GPU
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        print("FAISS index moved to GPU.")
    else:
        index = cpu_index
        print("FAISS index running on CPU.")

    # Initialize text encoder and move to GPU if available
    text_encoder = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
    text_encoder.to(device)
    print(f"Sentence Transformer model moved to: {text_encoder.device}")

    ds = MovieSubgraphPoolingRecommendationDataset(training_data_path=train_path,
                                   emb_lookup_path=embedding_path,
                                   faiss_index=index,
                                   faiss_movie_ids=np.array(node_id_map),
                                   text_encoder=text_encoder,
                                   graph_data=graph_data)
    # Increase num_workers for faster CPU-side data loading
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    train(model, dl, epochs=5, lr=5e-4, weight_decay=1e-4, device=device, top_k=10)

    #save model
    torch.save(model.state_dict(), f'{BASE_PATH}/models/subgraph_movie_ranker.pt')
