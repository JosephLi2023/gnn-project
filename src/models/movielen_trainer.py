import torch
import torch.nn as nn
from torch_geometric.datasets import MovieLens
from torch_geometric.data import DataLoader
from models import QueryAwareGATModel

# Load MovieLens dataset
dataset = MovieLens(root='/tmp/movielens', name='100k')
loader = DataLoader(dataset, batch_size=1, shuffle=True)
# Model hyperparameters
node_feat_dim = dataset.num_features
gat_hidden = 64
gat_out = 32
bert_dim = 768
fusion_hidden = 64
num_gat_layers = 3
model = QueryAwareGATModel(node_feat_dim, gat_hidden, gat_out, bert_dim, fusion_hidden, num_gat_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
# Training loop
for epoch in range(2):
    model.train()
    total_loss = 0
    for batch in loader:
        node_feats = batch.x
        edge_index = batch.edge_index
        query_text = ["recommend comedy movies"]
        label = batch.y.float()
        optimizer.zero_grad()
        score = model(node_feats, edge_index, query_text)
        loss = criterion(score, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
# Testing loop
model.eval()
with torch.no_grad():
    for batch in loader:
        node_feats = batch.x
        edge_index = batch.edge_index
        query_text = ["recommend comedy movies"]
        label = batch.y.float()
        score = model(node_feats, edge_index, query_text)
        prediction = torch.sigmoid(score)
        print(f"Predicted score: {prediction.item():.4f}, True label: {label.item()}")
        break
