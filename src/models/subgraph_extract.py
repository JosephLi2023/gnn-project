import torch
from collections import deque

def get_neighbors(data, node_id):
    # Implement this for HeteroData structure
    # Should return a list of neighbor node IDs for the given movie node
    raise NotImplementedError


# add max number of nodes.
def extract_subgraph(data, candidate_movie_ids, query_emb, similarity_threshold=0.6, max_depth=2):
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
            if neighbor_id in range(movie_embs.shape[0]):
                neighbor_emb = movie_embs[neighbor_id]
                sim = torch.dot(neighbor_emb, query_emb) / (neighbor_emb.norm() * query_emb.norm() + 1e-8)
                if sim > similarity_threshold:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))
    subgraph = data.subgraph(list(visited))
    return subgraph
