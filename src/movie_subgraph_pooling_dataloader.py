import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import ast

class Subgraph:
    def __init__(self, graph_id, movie_ids, embeddings):
        self.graph_id = graph_id
        self.movie_ids = movie_ids
        self.embeddings = embeddings  # List[torch.Tensor] or torch.Tensor

class MovieSubgraphPoolingRecommendationDataset(Dataset):
    def __init__(
        self,
        training_data_path: str,
        emb_lookup_path: str,
        faiss_index,
        faiss_movie_ids: np.ndarray,
        text_encoder: SentenceTransformer,
        graph_data: HeteroData,
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
        self.precomputed_neighbors = torch.load(NEIGHBOR_PATH)
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

        # 3. For each candidate, build a subgraph: root + neighbors
        subgraphs = []
        for cid in candidate_movie_ids:
            movie_ids = [cid]
            neighbors = extract_movie_subgraph(graph_data, movie_ids, query_emb, self.precomputed_neighbors)
            movie_ids.extend(neighbors)
            # Truncate if too many
            if len(movie_ids) > self.max_total_candidates:
               # Always keep the root_id, sample the rest
               root_id = cid  # The candidate movie id (root of the subgraph)
               neighbors = [mid for mid in movie_ids if mid != root_id]
               num_to_sample = self.max_total_candidates - 1
               sampled_neighbors = random.sample(neighbors, min(num_to_sample, len(neighbors)))
               movie_ids = [root_id] + sampled_neighbors

            # Get embeddings for all movies in subgraph
            embeddings = []
            for mid in movie_ids:
                emb_row = self.embs[self.embs.index == mid]
                if not emb_row.empty:
                    emb_values = emb_row.iloc[0].values.astype(np.float32)
                    embeddings.append(torch.tensor(emb_values))
            subgraphs.append({
                'graph_id': cid,
                'movie_ids': movie_ids,
                'embeddings': embeddings
            })
        return {
            'query_text': query_text,
            'query_emb': query_emb,
            'candidates': subgraphs,  # List of subgraphs
            'positive_movie_ids': positive_movie_ids
        }
        
def custom_collate_fn(batch):
    """
    Custom collate to handle variable candidate subgraph lists.
    Each batch item:
      - 'query_emb': Tensor
      - 'candidates': List[Dict] (subgraphs)
      - 'positive_movie_ids': List[int]
    Returns:
      - 'query_emb': Tensor [batch_size, emb_dim]
      - 'candidates': List[List[Dict]] (batch_size lists of subgraphs)
      - 'positive_movie_ids': List[List[int]] (batch_size lists)
    """
    return {
        'query_emb': torch.stack([item['query_emb'] for item in batch]),
        'candidates': [item['candidates'] for item in batch],  # List[List[Dict]]
        'positive_movie_ids': [item['positive_movie_ids'] for item in batch]
    }
