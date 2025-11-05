import torch
import pandas as pd
from torch_geometric.data import HeteroData
from utils import load_graph

def filter_combined_graph(ratings_path='processed_data/high_ratings.parquet',
                         top_k_movies=5000,
                         top_k_users=2000,
                         output_path='processed_data/combined_graph_filtered.pt'):
    """
    Filter the combined graph to keep only:
    1. Top K movies by rating count
    2. All movies mentioned in conversations (from inspired)
    3. All users, genres, and conversations connected to these movies
    """
    print("Loading combined graph...")
    data = load_graph(name='combined_graph')
    
    print("Loading ratings data...")
    high_ratings = pd.read_parquet(ratings_path)
    
    # Get movie node indices that have conversations
    conv_movie_edges = data['conversation', 'mentions', 'movie'].edge_index
    inspired_movie_indices = set(conv_movie_edges[1].tolist())
    print(f"Found {len(inspired_movie_indices)} movies with conversations")
    
    # Count ratings per movie node index
    movie_rating_counts = high_ratings['movie_idx'].value_counts()
    
    # Get top K movie indices
    top_k_indices = set(movie_rating_counts.head(top_k_movies).index.tolist())
    
    # Combine: top K + inspired movies
    movies_to_keep = top_k_indices.union(inspired_movie_indices)
    
    print(f"\nMovie filtering:")
    print(f"  Top {top_k_movies} movies: {len(top_k_indices)}")
    print(f"  Movies with conversations: {len(inspired_movie_indices)}")
    print(f"  Overlap: {len(top_k_indices.intersection(inspired_movie_indices))}")
    print(f"  Total to keep: {len(movies_to_keep)}")
    
    # Create mapping from old to new indices
    movies_to_keep_list = sorted(list(movies_to_keep))
    old_to_new_movie_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(movies_to_keep_list)}
    
    # Create filtered graph
    filtered_data = HeteroData()
    
    # Filter movie nodes
    print("\nFiltering movie nodes...")
    old_movie_mask = torch.zeros(data['movie'].num_nodes, dtype=torch.bool)
    for idx in movies_to_keep:
        old_movie_mask[idx] = True
    
    filtered_data['movie'].num_nodes = len(movies_to_keep)
    filtered_data['movie'].x = data['movie'].x[old_movie_mask]
    filtered_data['movie'].movie_id = data['movie'].movie_id[old_movie_mask]
    filtered_data['movie'].movie_name = [data['movie'].movie_name[i] for i in movies_to_keep_list]
    
    # Filter movie-genre edges
    print("Filtering movie-genre edges...")
    movie_genre_edges = data['movie', 'has_genre', 'genre'].edge_index
    mask = torch.tensor([idx.item() in movies_to_keep for idx in movie_genre_edges[0]])
    filtered_edges = movie_genre_edges[:, mask]
    new_movie_indices = torch.tensor([old_to_new_movie_idx[idx.item()] for idx in filtered_edges[0]])
    filtered_data['movie', 'has_genre', 'genre'].edge_index = torch.stack([new_movie_indices, filtered_edges[1]])
    
    # Filter genre-movie edges
    print("Filtering genre-movie edges...")
    genre_movie_edges = data['genre', 'has_movie', 'movie'].edge_index
    mask = torch.tensor([idx.item() in movies_to_keep for idx in genre_movie_edges[1]])
    filtered_edges = genre_movie_edges[:, mask]
    new_movie_indices = torch.tensor([old_to_new_movie_idx[idx.item()] for idx in filtered_edges[1]])
    filtered_data['genre', 'has_movie', 'movie'].edge_index = torch.stack([filtered_edges[0], new_movie_indices])
    
    # Keep all genres (copy as-is)
    filtered_data['genre'].num_nodes = data['genre'].num_nodes
    filtered_data['genre'].x = data['genre'].x
    filtered_data['genre'].genre_name = data['genre'].genre_name
    
    # Filter users BEFORE processing edges - keep only top K most active users
    print(f"Filtering users to top {top_k_users} most active...")
    user_rating_counts = high_ratings['user_idx'].value_counts()
    top_users = set(user_rating_counts.head(top_k_users).index.tolist())
    
    # Further filter to only users who rated movies we're keeping
    high_ratings_filtered = high_ratings[
        (high_ratings['movie_idx'].isin(movies_to_keep)) & 
        (high_ratings['user_idx'].isin(top_users))
    ]
    
    print(f"User filtering:")
    print(f"  Original users: {len(user_rating_counts)}")
    print(f"  Top {top_k_users} active users: {len(top_users)}")
    print(f"  After movie filtering: {high_ratings_filtered['user_idx'].nunique()}")
    print(f"  Edges after filtering: {len(high_ratings_filtered)}")
    
    # Get final user and movie sets
    active_users = high_ratings_filtered['user_idx'].unique()
    old_to_new_user_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(active_users)}
    
    # Filter user-movie edges
    print("\nFiltering user-movie edges...")
    new_user_indices = torch.tensor([old_to_new_user_idx[idx] for idx in high_ratings_filtered['user_idx'].values])
    new_movie_indices = torch.tensor([old_to_new_movie_idx[idx] for idx in high_ratings_filtered['movie_idx'].values])
    
    filtered_data['user', 'rated_high', 'movie'].edge_index = torch.stack([new_user_indices, new_movie_indices])
    filtered_data['user', 'rated_high', 'movie'].edge_attr = torch.tensor(
        high_ratings_filtered['rating'].values, 
        dtype=torch.float
    ).unsqueeze(1)
    
    # Filter movie-user edges (reverse)
    print("Filtering movie-user edges...")
    filtered_data['movie', 'rated_by', 'user'].edge_index = torch.stack([new_movie_indices, new_user_indices])
    filtered_data['movie', 'rated_by', 'user'].edge_attr = torch.tensor(
        high_ratings_filtered['rating'].values, 
        dtype=torch.float
    ).unsqueeze(1)
    
    # Filter user nodes
    print("Filtering user nodes...")
    filtered_data['user'].num_nodes = len(active_users)
    filtered_data['user'].x = data['user'].x[active_users]
    filtered_data['user'].user_id = data['user'].user_id[active_users]
    
    # Filter conversation-movie edges
    print("Filtering conversation-movie edges...")
    conv_movie_edges = data['conversation', 'mentions', 'movie'].edge_index
    conv_movie_attr = data['conversation', 'mentions', 'movie'].edge_attr
    
    # Keep edges where movie is in our filtered set
    mask = torch.tensor([idx.item() in movies_to_keep for idx in conv_movie_edges[1]])
    filtered_edges = conv_movie_edges[:, mask]
    filtered_attr = conv_movie_attr[mask]
    
    # Get conversations that mention at least one filtered movie
    active_convs = filtered_edges[0].unique()
    old_to_new_conv_idx = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(active_convs)}
    
    # Remap both conversation and movie indices
    new_conv_indices = torch.tensor([old_to_new_conv_idx[idx.item()] for idx in filtered_edges[0]])
    new_movie_indices = torch.tensor([old_to_new_movie_idx[idx.item()] for idx in filtered_edges[1]])
    filtered_data['conversation', 'mentions', 'movie'].edge_index = torch.stack([new_conv_indices, new_movie_indices])
    filtered_data['conversation', 'mentions', 'movie'].edge_attr = filtered_attr
    
    # Filter movie-conversation edges
    movie_conv_edges = data['movie', 'mentioned_in', 'conversation'].edge_index
    movie_conv_attr = data['movie', 'mentioned_in', 'conversation'].edge_attr
    mask = torch.tensor([idx.item() in movies_to_keep for idx in movie_conv_edges[0]])
    filtered_edges = movie_conv_edges[:, mask]
    filtered_attr = movie_conv_attr[mask]
    
    new_movie_indices = torch.tensor([old_to_new_movie_idx[idx.item()] for idx in filtered_edges[0]])
    new_conv_indices = torch.tensor([old_to_new_conv_idx[idx.item()] for idx in filtered_edges[1]])
    filtered_data['movie', 'mentioned_in', 'conversation'].edge_index = torch.stack([new_movie_indices, new_conv_indices])
    filtered_data['movie', 'mentioned_in', 'conversation'].edge_attr = filtered_attr
    
    # Filter conversation nodes
    print("Filtering conversation nodes...")
    filtered_data['conversation'].num_nodes = len(active_convs)
    filtered_data['conversation'].x = data['conversation'].x[active_convs]
    
    # Print summary
    print("\n=== Filtered Graph Summary ===")
    print(filtered_data)
    print(f"\nOriginal movies: {data['movie'].num_nodes} -> Filtered: {filtered_data['movie'].num_nodes}")
    print(f"Original users: {data['user'].num_nodes} -> Filtered: {filtered_data['user'].num_nodes}")
    print(f"Original conversations: {data['conversation'].num_nodes} -> Filtered: {filtered_data['conversation'].num_nodes}")
    
    # Save
    print(f"\nSaving to {output_path}...")
    torch.save(filtered_data, output_path)
    print("Done!")
    
    return filtered_data


if __name__ == "__main__":
    filtered_graph = filter_combined_graph()