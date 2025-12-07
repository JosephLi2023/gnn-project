import torch
from collections import deque
from torch_geometric.data import HeteroData
import random

def get_neighbors(data: HeteroData, movie_node_id: int, limit: int = 100) -> list:
    quota = limit // 3
    neighbor_sets = {'genre': set(), 'user': set(), 'conversation': set()}

    # 1. Conversation-based neighbors
    if ('movie', 'mentioned_in', 'conversation') in data.edge_index_dict and \
       ('conversation', 'mentions', 'movie') in data.edge_index_dict:
        movie_to_conv = data['movie', 'mentioned_in', 'conversation'].edge_index
        conv_to_movie = data['conversation', 'mentions', 'movie'].edge_index
        conv_ids = movie_to_conv[1][movie_to_conv[0] == movie_node_id].tolist()
        for conv_id in conv_ids:
            movies = conv_to_movie[1][conv_to_movie[0] == conv_id].tolist()
            neighbor_sets['conversation'].update(movies)

    # 2. User-based neighbors
    if ('movie', 'rated_by', 'user') in data.edge_index_dict and \
       ('user', 'rated_high', 'movie') in data.edge_index_dict:
        movie_to_user = data['movie', 'rated_by', 'user'].edge_index
        user_to_movie = data['user', 'rated_high', 'movie'].edge_index
        user_ids = movie_to_user[1][movie_to_user[0] == movie_node_id].tolist()
        for user_id in user_ids:
            movies = user_to_movie[1][user_to_movie[0] == user_id].tolist()
            neighbor_sets['user'].update(movies)
        
    # 3. Genre-based neighbors
    if ('movie', 'has_genre', 'genre') in data.edge_index_dict and \
       ('genre', 'has_movie', 'movie') in data.edge_index_dict:
        movie_to_genre = data['movie', 'has_genre', 'genre'].edge_index
        genre_to_movie = data['genre', 'has_movie', 'movie'].edge_index
        # Comment: If movie_to_genre lives on the GPU, moving data from GPU to CPU can be slow
        genre_ids = movie_to_genre[1][movie_to_genre[0] == movie_node_id].tolist()
        for genre_id in genre_ids:
            movies = genre_to_movie[1][genre_to_movie[0] == genre_id].tolist()
            # Comment: Only saves the movie id, no edges, we are left with floating nodes
            neighbor_sets['genre'].update(movies)

    # Remove the original movie node from all sets
    for s in neighbor_sets.values():
        s.discard(movie_node_id)
    # Randomly sample up to quota from each set
    sampled_neighbors = []
    for edge_type in ['genre', 'user', 'conversation']:
        neighbors = list(neighbor_sets[edge_type])
        if len(neighbors) > quota:
            sampled = random.sample(neighbors, quota)
        else:
            sampled = neighbors
        sampled_neighbors.extend(sampled)

    # If we have fewer than limit, fill up with additional random samples from all sets
    if len(sampled_neighbors) < limit:
        all_neighbors = set().union(*neighbor_sets.values())
        remaining = list(all_neighbors - set(sampled_neighbors))
        extra_needed = limit - len(sampled_neighbors)
        if remaining:
            sampled_neighbors.extend(random.sample(remaining, min(extra_needed, len(remaining))))
    # Finally, include the original movie node itself
    sampled_neighbors.append(movie_node_id)
    # Truncate to limit+1 (since we add itself)
    return sampled_neighbors[:limit+1]

# add max number of nodes.
def extract_movie_subgraph(
    data, 
    candidate_movie_ids, 
    query_emb, 
    similarity_threshold=0.6, 
    max_depth=2
):
    """
    Expands candidate_movie_ids by traversing the graph up to max_depth,
    adding neighbors whose embeddings are sufficiently similar to the query.
    Returns a list of unique movie node IDs.
    """
    visited = set(candidate_movie_ids)
    queue = deque([(node_id, 0) for node_id in candidate_movie_ids])
    movie_embs = data['movie'].x
    while queue:
        node_id, depth = queue.popleft()
        if depth >= max_depth:
            continue
        neighbors = get_neighbors(data, node_id)
        for neighbor_id in neighbors:
            if neighbor_id in visited:
                continue
            # Only consider valid movie node indices
            if 0 <= neighbor_id < movie_embs.shape[0]:
                neighbor_emb = movie_embs[neighbor_id]
                sim = torch.dot(neighbor_emb, query_emb) / (neighbor_emb.norm() * query_emb.norm() + 1e-8)
                if sim > similarity_threshold:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))
    # Only return movie node IDs
    return list(visited)
