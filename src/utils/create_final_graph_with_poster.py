import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree
import os
import pickle
import torch.nn.functional as F

BASE_PATH = '/content/drive/MyDrive/CS224W_Project/data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Input Files
GRAPH_PATH = f'{BASE_PATH}/processed/movielens/hetero_graph.pt'
MAPPINGS_PATH = f'{BASE_PATH}/processed/movielens/id_mappings.pt'
MOVIES_CSV_PATH = f'{BASE_PATH}/raw/movielens/movies.csv'
LINKS_PATH = f'{BASE_PATH}/raw/movielens/links.csv'

# Conversation Data 
CONV_DATA_PREFIX = f'{BASE_PATH}/processed/combined/graph_data'

# Visual Embeddings
VISUAL_EMB_PATH = f'{BASE_PATH}/processed/posters/imdb_id_to_visual_embedding.pkl'

# Output Files
OUTPUT_GRAPH_PATH = f'{BASE_PATH}/processed/final_graph.pt'
OUTPUT_DEV_EDGES = f'{BASE_PATH}/processed/dev_edges.pt'
OUTPUT_TEST_EDGES = f'{BASE_PATH}/processed/test_edges.pt'
OUTPUT_DEV_ATTRS = f'{BASE_PATH}/processed/dev_attrs.pt' # Outcomes for dev
OUTPUT_TEST_ATTRS = f'{BASE_PATH}/processed/test_attrs.pt' # Outcomes for test


def assemble_graph():
    print(f"Loading base graph from {GRAPH_PATH}...")
    data = torch.load(GRAPH_PATH, weights_only=False)
    mappings = torch.load(MAPPINGS_PATH, weights_only=False)

    # 1. UPGRADE USER FEATURES 
    print("\n[1/4] Upgrading User Features...")
    # Calculate degree from 'rated_high' edges
    if ('user', 'rated_high', 'movie') in data.edge_types:
        edge_index = data['user', 'rated_high', 'movie'].edge_index
        user_indices = edge_index[0]
        user_degrees = degree(user_indices, num_nodes=data['user'].num_nodes)
        # Log-transform: log(degree + 1)
        user_features = torch.log1p(user_degrees).view(-1, 1)
        data['user'].x = user_features
        print(f"  - Added log-degree features. User X shape: {data['user'].x.shape}")
    else:
        print("  - Warning: 'rated_high' edge not found. Skipping user features.")

    # 2. UPGRADE MOVIE FEATURES 
    print("\n[2/4] Upgrading Movie Features...")

    # A. Title Embeddings
    print("  - Generating Title Embeddings...")
    movies_df = pd.read_csv(MOVIES_CSV_PATH)
    ml_id_to_node_idx = mappings['movie_id_to_idx']

    # Sort titles to match node indices
    sorted_titles = [""] * data['movie'].num_nodes
    for ml_id, node_idx in ml_id_to_node_idx.items():
        title = movies_df[movies_df['movieId'] == ml_id]['title'].values
        sorted_titles[node_idx] = title[0] if len(title) > 0 else "Unknown"

    title_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    title_embeddings = title_model.encode(sorted_titles, show_progress_bar=True, convert_to_tensor=True).cpu()

    # B. Visual Embeddings 
    visual_embeddings = None
    if os.path.exists(VISUAL_EMB_PATH):
        print("  - Loading Poster Embeddings...")
        with open(VISUAL_EMB_PATH, 'rb') as f:
            visual_map = pickle.load(f)

        # Map Node_Idx -> MovieLens_ID -> IMDb_ID -> Visual_Vector
        links_df = pd.read_csv(LINKS_PATH, dtype={'imdbId': str})
        ml_to_imdb = dict(zip(links_df['movieId'], links_df['imdbId']))

        # Initialize with zeros (512 dim for CLIP ViT-B/32)
        visual_embeddings = torch.zeros((data['movie'].num_nodes, 512))

        found_count = 0
        for ml_id, node_idx in ml_id_to_node_idx.items():
            imdb_id_raw = ml_to_imdb.get(ml_id)
            if imdb_id_raw:
                imdb_key = 'tt' + str(imdb_id_raw).zfill(7)
                if imdb_key in visual_map:
                    visual_embeddings[node_idx] = torch.tensor(visual_map[imdb_key])
                    found_count += 1
        print(f"  - Attached posters for {found_count} / {data['movie'].num_nodes} movies.")
    else:
        print("  - Visual embeddings file not found. Skipping.")

    # C. Concatenate All Movie Features
    # Start with existing tags (if any)
    final_movie_feats = [data['movie'].x] if data['movie'].x is not None else []
    final_movie_feats.append(title_embeddings)
    if visual_embeddings is not None:
        final_movie_feats.append(visual_embeddings)

    data['movie'].x = torch.cat(final_movie_feats, dim=1)
    print(f"  - Final Movie X shape: {data['movie'].x.shape}")

    # 3. ADD CONVERSATION NODES
    print("\n[3/4] Adding Conversation Nodes...")

    # Load the pre-processed chunks
    try:
        train_x = torch.load(f"{CONV_DATA_PREFIX}_train_conversation_x.pt")
        dev_x   = torch.load(f"{CONV_DATA_PREFIX}_dev_conversation_x.pt")
        test_x  = torch.load(f"{CONV_DATA_PREFIX}_test_conversation_x.pt")

        # Concatenate all to make the full 'conversation' node set
        all_conv_x = torch.cat([train_x, dev_x, test_x], dim=0)
        data['conversation'].x = all_conv_x
        print(f"  - Added {all_conv_x.shape[0]} conversation nodes.")

    except FileNotFoundError as e:
        print(f"  - FATAL ERROR: Could not find conversation .pt files. Run preprocess_data.py first.\n  {e}")
        return

    # 4. ADD CONVERSATION EDGES
    print("\n[4/4] Wiring Edges...")

    # Load edges
    train_edge_index = torch.load(f"{CONV_DATA_PREFIX}_train_conv_movie_edge_index.pt")
    train_edge_attr  = torch.load(f"{CONV_DATA_PREFIX}_train_conv_movie_edge_attr.pt")

    dev_edge_index   = torch.load(f"{CONV_DATA_PREFIX}_dev_conv_movie_edge_index.pt")
    dev_edge_attr    = torch.load(f"{CONV_DATA_PREFIX}_dev_conv_movie_edge_attr.pt")

    test_edge_index  = torch.load(f"{CONV_DATA_PREFIX}_test_conv_movie_edge_index.pt")
    test_edge_attr   = torch.load(f"{CONV_DATA_PREFIX}_test_conv_movie_edge_attr.pt")

    # Offset the indices for dev/test
    # The nodes are stacked [Train | Dev | Test]
    # Train indices are 0 to N_train-1 (Correct)
    # Dev indices need to be shifted by N_train
    # Test indices need to be shifted by N_train + N_dev
    num_train = train_x.shape[0]
    num_dev = dev_x.shape[0]

    dev_edge_index[0] += num_train
    test_edge_index[0] += (num_train + num_dev)

    # Add TRAIN edges to the graph structure (for message passing)
    data['conversation', 'recommends', 'movie'].edge_index = train_edge_index
    data['conversation', 'recommends', 'movie'].edge_attr = train_edge_attr

    # Add REVERSE edges 
    # Flip source/target (0/1)
    rev_edge_index = torch.stack([train_edge_index[1], train_edge_index[0]], dim=0)
    data['movie', 'recommended_by', 'conversation'].edge_index = rev_edge_index
    data['movie', 'recommended_by', 'conversation'].edge_attr = train_edge_attr

    print(f"  - Wired {train_edge_index.shape[1]} training edges.")
    print(f"  - Held out {dev_edge_index.shape[1]} dev edges.")
    print(f"  - Held out {test_edge_index.shape[1]} test edges.")

    # 5. SAVE EVERYTHING
    print(f"\nSaving final graph to {OUTPUT_GRAPH_PATH}...")
    torch.save(data, OUTPUT_GRAPH_PATH)

    # Save held-out sets for the training script to load
    torch.save(dev_edge_index, OUTPUT_DEV_EDGES)
    torch.save(test_edge_index, OUTPUT_TEST_EDGES)
    torch.save(dev_edge_attr, OUTPUT_DEV_ATTRS)
    torch.save(test_edge_attr, OUTPUT_TEST_ATTRS)


if __name__ == "__main__":
    assemble_graph()
