import torch
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage
torch.serialization.add_safe_globals([HeteroData, BaseStorage, NodeStorage, EdgeStorage])
# Load existing outputs
print("Loading existing data...")
hetero_graph = torch.load('processed_data/hetero_graph.pt', weights_only=True)
conversation_x = torch.load('processed_data/conversation_x.pt', weights_only=True)
conv_movie_edge_index = torch.load('processed_data/conv_movie_edge_index.pt', weights_only=True)
conv_movie_edge_attr = torch.load('processed_data/conv_movie_edge_attr.pt', weights_only=True)

# Add conversation nodes to the graph
print("Adding conversation nodes...")
hetero_graph['conversation'].x = conversation_x
hetero_graph['conversation'].num_nodes = conversation_x.shape[0]

# Add conversation-movie edges (both directions)
print("Adding edges...")
hetero_graph['conversation', 'mentions', 'movie'].edge_index = conv_movie_edge_index
hetero_graph['conversation', 'mentions', 'movie'].edge_attr = conv_movie_edge_attr

# Reverse edges
movie_conv_edge_index = torch.stack([conv_movie_edge_index[1], conv_movie_edge_index[0]], dim=0)
hetero_graph['movie', 'mentioned_in', 'conversation'].edge_index = movie_conv_edge_index
hetero_graph['movie', 'mentioned_in', 'conversation'].edge_attr = conv_movie_edge_attr

print("\n=== Combined Graph ===")
print(hetero_graph)

# Save
torch.save(hetero_graph, 'processed_data/combined_graph.pt')
print("\nSaved to processed_data/combined_graph.pt")
