import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from subgraph_extract import extract_movie_subgraph

class MovieSubgraphRecommendationDataset(Dataset):
    def __init__(
        self,
        training_data_path: str,
        emb_lookup_path: str,
        faiss_index,
        faiss_movie_ids: np.ndarray,
        text_encoder: SentenceTransformer,
        graph_data,  # HeteroData
        num_faiss_candidates: int = 10,
        num_neighbors_per_candidate: int = 20,
        max_total_candidates: int = 200
    ):
        self.faiss_index = faiss_index
        self.faiss_movie_ids = faiss_movie_ids
        self.text_encoder = text_encoder
        self.graph_data = graph_data
        self.num_faiss_candidates = num_faiss_candidates
        self.num_neighbors_per_candidate = num_neighbors_per_candidate
        self.max_total_candidates = max_total_candidates
        self.embs = pd.read_csv(emb_lookup_path, index_col=0)
        # Load and preprocess training data
        df = pd.read_csv(training_data_path, sep='\t')
        df = df[df['label'] == 'accepted']
        self.samples = []
        for conv_id, group in df.groupby('conversation_id'):
            query = group['user_query'].iloc[0]
            positive_movie_ids = group['movie_id'].tolist()
            self.samples.append({
                'conversation_id': conv_id, 
                'query': query,
                'positive_movie_ids': positive_movie_ids
            })
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        import faiss as faiss_lib
        import random
        sample = self.samples[idx]
        query_text = sample['query']
        query_emb = self.text_encoder.encode(
            query_text,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        # 1. FAISS candidate retrieval
        query_np = query_emb.cpu().numpy().reshape(1, -1).astype(np.float32)
        faiss_lib.normalize_L2(query_np)
        distances, indices = self.faiss_index.search(query_np, self.num_faiss_candidates)
        candidate_movie_ids = [int(self.faiss_movie_ids[idx]) for idx in indices[0]]
        # 2. Ensure all positives are in candidates
        positive_movie_ids = sample['positive_movie_ids']
        for pid in positive_movie_ids:
            if pid not in candidate_movie_ids:
                replace_idx = random.randint(0, len(candidate_movie_ids) - 1)
                candidate_movie_ids[replace_idx] = pid
        # 3. For each candidate, get up to 20 neighbors from the graph
        expanded_movie_ids = set(candidate_movie_ids)
        for cid in candidate_movie_ids:
            neighbors = get_neighbors(self.graph_data, cid, limit=self.num_neighbors_per_candidate)
            expanded_movie_ids.update(neighbors)
        # 4. Truncate to max_total_candidates if needed
        expanded_movie_ids = list(expanded_movie_ids)
        if len(expanded_movie_ids) > self.max_total_candidates:
            expanded_movie_ids = random.sample(expanded_movie_ids, self.max_total_candidates)
        # 5. Get embeddings for all candidates
        candidates = {}
        for movie_id in expanded_movie_ids:
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
