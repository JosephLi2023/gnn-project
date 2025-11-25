import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling
from hetero_encoder import HeteroGNN

# Load the data
data = torch.load("acm.pkl")
# Create HeteroData object
hetero_graph = HeteroData()
# Node features (example shapes)
num_conversations = 10000
conv_feat_dim = 64
num_users = 10000
user_feat_dim = 64
num_movies = 10000
movie_feat_dim = 64
num_genres = 100
genre_feat_dim = 64
hetero_graph['conversation'].x = torch.randn(num_conversations, conv_feat_dim)
hetero_graph['user'].x = torch.randn(num_users, user_feat_dim)
hetero_graph['movie'].x = torch.randn(num_movies, movie_feat_dim)
hetero_graph['genre'].x = torch.randn(num_genres, genre_feat_dim)
# Edge indices and attributes
hetero_graph['conversation', 'mentions', 'movie'].edge_index = conv_movie_edge_index
hetero_graph['conversation', 'mentions', 'movie'].edge_attr = conv_movie_edge_attr  # shape [num_edges, 6]
hetero_graph['movie', 'mentioned_in', 'conversation'].edge_index = movie_conv_edge_index
hetero_graph['movie', 'mentioned_in', 'conversation'].edge_attr = conv_movie_edge_attr  # shape [num_edges, 6]
hetero_graph['user', 'rated_high', 'movie'].edge_index = user_movie_edges
hetero_graph['user', 'rated_high', 'movie'].edge_attr = torch.tensor(
    high_ratings['rating'].values, dtype=torch.float
).unsqueeze(1)  # shape [num_edges, 1]
hetero_graph['movie', 'rated_by', 'user'].edge_index = movie_user_edges
hetero_graph['movie', 'rated_by', 'user'].edge_attr = torch.tensor(
    high_ratings['rating'].values, dtype=torch.float
).unsqueeze(1)  # shape [num_edges, 1]
hetero_graph['movie', 'has_genre', 'genre'].edge_index = movie_genre_edges
hetero_graph['genre', 'has_movie', 'movie'].edge_index = genre_movie_edges

# Move to device
device = args['device']
hetero_data = hetero_data.to(device)
# Train/val/test indices for link prediction (edges)
train_edge_index = data['train_edge_index'].to(device)  # [2, num_train_edges]
val_edge_index = data['val_edge_index'].to(device)
test_edge_index = data['test_edge_index'].to(device)

def train(model, optimizer, data, train_edge_index, neg_edge_sampler):
    model.train()
    optimizer.zero_grad()
    z_dict = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    # Positive samples
    pos_src = train_edge_index[0]
    pos_dst = train_edge_index[1]
    # Negative samples (corrupt tail)
    num_neg = pos_src.size(0)
    neg_edge_index = neg_edge_sampler(data, num_neg)
    neg_src = neg_edge_index[0]
    neg_dst = neg_edge_index[1]
    # Compute scores
    src_emb = z_dict['user']
    dst_emb = z_dict['movie']
    pos_score = (src_emb[pos_src] * dst_emb[pos_dst]).sum(dim=1)
    neg_score = (src_emb[neg_src] * dst_emb[neg_dst]).sum(dim=1)
    # Link prediction loss
    loss = -torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, edge_index, neg_edge_sampler):
    model.eval()
    z_dict = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    src_emb = z_dict['user']
    dst_emb = z_dict['movie']
    pos_src = edge_index[0]
    pos_dst = edge_index[1]
    num_neg = pos_src.size(0)
    neg_edge_index = neg_edge_sampler(data, num_neg)
    neg_src = neg_edge_index[0]
    neg_dst = neg_edge_index[1]
    pos_score = (src_emb[pos_src] * dst_emb[pos_dst]).sum(dim=1)
    neg_score = (src_emb[neg_src] * dst_emb[neg_dst]).sum(dim=1)
    # Compute metrics (AUC, etc.)
    y_true = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    y_pred = torch.cat([pos_score, neg_score])
    auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    return auc

def neg_edge_sampler(data, num_neg):
    edge_index = data['user', 'rated_high', 'movie'].edge_index
    num_nodes = data['user'].num_nodes
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
        method='sparse'
    )
    return neg_edge_index

metadata = (list(hetero_data.node_types), list(hetero_data.edge_types))
edge_attr_dims = {
    ('conversation', 'mentions', 'movie'): 1,
    ('movie', 'mentioned_in', 'conversation'): 1,
    ('user', 'rated_high', 'movie'): 1,
    ('movie', 'rated_by', 'user'): 1,
    # No entry for edges without attributes
}
model = HeteroGNN(
    metadata=metadata,
    x_dict={nt: hetero_graph[nt].x for nt in hetero_graph.node_types},
    args={'hidden_size': 64, 'attn_size': 32},
    edge_attr_dims=edge_attr_dims,
    aggr="mean"
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
best_auc = 0
best_model = None

for epoch in range(args['epochs']):
    loss = train(model, optimizer, hetero_data, train_edge_index, neg_edge_sampler)
    train_auc = test(model, hetero_data, train_edge_index, neg_edge_sampler)
    val_auc = test(model, hetero_data, val_edge_index, neg_edge_sampler)
    test_auc = test(model, hetero_data, test_edge_index, neg_edge_sampler)
    print(f"Epoch {epoch+1}: loss={loss:.4f}, train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, test_auc={test_auc:.4f}")
    if val_auc > best_auc:
        best_auc = val_auc
        best_model = copy.deepcopy(model)

# Final test
final_test_auc = test(best_model, hetero_data, test_edge_index, neg_edge_sampler)
print(f"Best model test AUC: {final_test_auc:.4f}")
