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
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer

from models.subgraph_extract import extract_subgraph


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
        data_path: str,
        faiss_index,
        faiss_movie_ids: np.ndarray,
        text_encoder: SentenceTransformer,
        num_candidates: int = 100,
        mode: str = 'train',
        dataset_source: Optional[str] = None
    ):
        """
        Args:
            data_path: Path to TSV file with columns: conversation_id, user_query, movie_id, label
            faiss_index: Pre-built FAISS index for candidate generation
            faiss_movie_ids: Array mapping FAISS indices to movie IDs
            text_encoder: SentenceTransformer model for encoding queries
            num_candidates: Number of candidates to retrieve from FAISS
            mode: 'train' or 'val' (affects whether we force positive into candidates)
            dataset_source: Optional filter for 'dataset_source' column
        """
        self.faiss_index = faiss_index
        self.faiss_movie_ids = faiss_movie_ids
        self.text_encoder = text_encoder
        self.num_candidates = num_candidates
        self.mode = mode

        # Load data
        df = pd.read_csv(data_path, sep='\t')

        # Filter by dataset source if specified
        if dataset_source is not None:
            df = df[df['dataset_source'] == dataset_source]

        # Filter by label (only keep accepted movies)
        df = df[df['label'] == 'accepted']

        # Group by conversation to get all positive movies per query
        self.samples = []
        for conv_id, group in df.groupby('conversation_id'):
            query = group['user_query'].iloc[0]  # Same query for all in group
            positive_movie_ids = group['movie_id'].tolist()  # All positive movies

            # For training, create one sample per positive movie
            # For validation, could aggregate differently
            for pos_movie_id in positive_movie_ids:
                self.samples.append({
                    'conversation_id': conv_id,
                    'query': query,
                    'positive_movie_id': pos_movie_id,
                    'all_positives': positive_movie_ids  # Keep track of all positives
                })

        print(f"Loaded {len(self.samples)} samples from {data_path} (mode: {mode})")

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
        positive_movie_id = sample['positive_movie_id']

        if positive_movie_id in candidate_movie_ids:
            label_idx = candidate_movie_ids.index(positive_movie_id)
        else:
            # Replace a random candidate with the positive (for training)
            if self.mode == 'train':
                replace_idx = np.random.randint(0, len(candidate_movie_ids))
                candidate_movie_ids[replace_idx] = positive_movie_id
                label_idx = replace_idx
            else:
                # For validation, keep FAISS results but mark as not found
                label_idx = -1

        return {
            'conversation_id': sample['conversation_id'],
            'query_text': query_text,
            'query_emb': query_emb,
            'candidate_movie_ids': candidate_movie_ids,
            'positive_movie_id': positive_movie_id,
            'label_idx': label_idx,
            'all_positives': sample['all_positives']
        }



def collate_fn_candidates(batch):
    """
    Collate function for Path 1 (direct reranking).

    Returns batched queries and lists of candidates.
    """
    return {
        'conversation_ids': [item['conversation_id'] for item in batch],
        'query_texts': [item['query_text'] for item in batch],
        'query_embs': torch.stack([item['query_emb'] for item in batch]),
        'candidate_movie_ids': [item['candidate_movie_ids'] for item in batch],
        'positive_movie_ids': [item['positive_movie_id'] for item in batch],
        'label_indices': torch.tensor([item['label_idx'] for item in batch]),
        'all_positives': [item['all_positives'] for item in batch]
    }

def create_dataloaders(
    train_data_path: str,
    val_data_path: str,
    faiss_index,
    faiss_movie_ids: np.ndarray,
    text_encoder_name: str = 'all-MiniLM-L6-v2',
    num_candidates: int = 100,
    batch_size: int = 32,
    num_workers: int = 0,
    use_subgraphs: bool = False,
    graph_data = None,
    **subgraph_kwargs
):
    """
    Create train and validation dataloaders.

    Args:
        train_data_path: Path to training TSV
        val_data_path: Path to validation TSV
        faiss_index: FAISS index for candidate generation
        faiss_movie_ids: Movie IDs for FAISS index
        text_encoder_name: Name of SentenceTransformer model
        num_candidates: Number of candidates to retrieve
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        use_subgraphs: If True, use SubgraphDataset (Path 2). If False, use base dataset (Path 1)
        graph_data: HeteroData graph (required if use_subgraphs=True)
        **subgraph_kwargs: Additional args for SubgraphDataset

    Returns:
        train_loader, val_loader
    """
    # Load text encoder
    print(f"Loading text encoder: {text_encoder_name}")
    text_encoder = SentenceTransformer(text_encoder_name)

    # Create base datasets
    train_dataset = MovieRecommendationDataset(
        train_data_path,
        faiss_index,
        faiss_movie_ids,
        text_encoder,
        num_candidates=num_candidates,
        mode='train'
    )

    val_dataset = MovieRecommendationDataset(
        val_data_path,
        faiss_index,
        faiss_movie_ids,
        text_encoder,
        num_candidates=num_candidates,
        mode='val'
    )


    collate_fn = collate_fn_candidates

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    import faiss

    print("Example: Creating dataloaders for Path 1 (Direct Reranking)")

    # Assuming you have a FAISS index built
    # faiss_index = ...
    # faiss_movie_ids = ...

    # train_loader, val_loader = create_dataloaders(
    #     train_data_path='train_output_combined.tsv',
    #     val_data_path='test_output_combined.tsv',
    #     faiss_index=faiss_index,
    #     faiss_movie_ids=faiss_movie_ids,
    #     num_candidates=100,
    #     batch_size=32,
    #     use_subgraphs=False  # Path 1
    # )

    # for batch in train_loader:
    #     print(f"Batch size: {len(batch['query_embs'])}")
    #     print(f"Query embeddings shape: {batch['query_embs'].shape}")
    #     print(f"Candidates per query: {len(batch['candidate_movie_ids'][0])}")
    #     break

    print("\nExample: Creating dataloaders for Path 2 (Subgraph Ranking)")

    # train_loader, val_loader = create_dataloaders(
    #     train_data_path='train_output_combined.tsv',
    #     val_data_path='test_output_combined.tsv',
    #     faiss_index=faiss_index,
    #     faiss_movie_ids=faiss_movie_ids,
    #     num_candidates=100,
    #     batch_size=32,
    #     use_subgraphs=True,  # Path 2
    #     graph_data=data,
    #     similarity_threshold=0.6,
    #     max_depth=2,
    #     extract_on_the_fly=True
    # )

    # for batch in train_loader:
    #     print(f"Batch size: {len(batch['query_embs'])}")
    #     print(f"Subgraphs for first sample: {len(batch['subgraphs'][0])}")
    #     break
