"""
Custom DataLoader for movie recommendation training.

Handles:
1. Query text → embeddings (using sentence transformer)
2. FAISS candidate generation
3. Subgraph extraction (optional)
4. Ground truth labels

Supports both Path 1 (direct reranking) and Path 2 (subgraph ranking).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer



class MovieRecommendationDataset(Dataset):
    """
    Dataset for movie recommendation training.

    Each sample contains:
    - query_text: Original query string
    - query_embedding: Encoded query (384-dim)
    - candidate_movie_ids: Top-k candidates from FAISS
    - positive_movie_id: Ground truth movie
    - label_idx: Index of positive movie in candidates list
    """

    def __init__(
        self,
        training_data_path: str, # contains query, reccomeneded movie
        emb_lookup_path: str, #contains path to a df of movie embeddings from the pretrained model
        faiss_index,
        faiss_movie_ids: np.ndarray,
        text_encoder: SentenceTransformer,
        num_candidates: int = 100
    ):
        self.faiss_index = faiss_index
        self.faiss_movie_ids = faiss_movie_ids
        self.text_encoder = text_encoder
        self.num_candidates = num_candidates
        self.embs = pd.read_csv(emb_lookup_path, index_col=0)

        # Load data
        df = pd.read_csv(training_data_path, sep='\t')

        # Filter by label (only keep accepted movies)
        df = df[df['label'] == 'accepted']

        # Group by conversation to get all positive movies per query
        self.samples = []
        for conv_id, group in df.groupby('conversation_id'):
            query = group['user_query'].iloc[0]  # Same query for all in group
            positive_movie_ids = group['movie_id'].tolist()  # All positive movies
            self.samples.append({
                'conversation_id': conv_id, 
                'query': query,
                'positive_movie_ids': positive_movie_ids})

        print(f"Loaded {len(self.samples)} samples from {training_data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Encode query
        query_text = sample['query']
        query_emb = self.text_encoder.encode(
            query_text,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # 2. Get FAISS candidates
        query_np = query_emb.cpu().numpy().reshape(1, -1).astype(np.float32)
        import faiss as faiss_lib
        faiss_lib.normalize_L2(query_np)

        distances, indices = self.faiss_index.search(query_np, self.num_candidates)
        candidate_movie_ids = [int(self.faiss_movie_ids[idx]) for idx in indices[0]]

        # 3. Ensure positive is in candidates (important for training)
        positive_movie_ids = sample['positive_movie_ids']
        for id in positive_movie_ids:
            if id not in candidate_movie_ids:
                # Replace a random candidate with the positive (for training)
                replace_idx = np.random.randint(0, len(candidate_movie_ids))
                candidate_movie_ids[replace_idx] = id

        #4 Create dict with movie id and embedding for all candidate movies
        candidates = {}
        for movie_id in candidate_movie_ids:
            emb_row = self.embs[self.embs.index == movie_id]
            if not emb_row.empty:
                emb_values = emb_row.iloc[0].values.astype(np.float32)
                candidates[movie_id] = torch.tensor(emb_values)
    

        return {
            'query_text': query_text,
            'query_emb': query_emb,
            'candidates': candidates,
            'positive_movie_ids': positive_movie_ids
        }

