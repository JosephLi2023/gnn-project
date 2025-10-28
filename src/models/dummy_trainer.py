from torch.utils.data import DataLoader
from models import QueryAwareGATModel

# Dummy dataset for illustration
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, node_feat_dim):
        self.num_samples = num_samples
        self.node_feat_dim = node_feat_dim
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        node_feats = torch.randn(100, self.node_feat_dim)  # 100 nodes
        edge_index = torch.randint(0, 100, (2, 500))       # 500 edges
        query_text = ["find action movies"]                # batch size 1
        label = torch.tensor([1.0])                        # dummy label
        return node_feats, edge_index, query_text, label

# Hyperparameters
node_feat_dim = 128
gat_hidden = 64
gat_out = 32
bert_dim = 768
fusion_hidden = 64
num_gat_layers = 3
model = QueryAwareGATModel(node_feat_dim, gat_hidden, gat_out, bert_dim, fusion_hidden, num_gat_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
dataset = DummyDataset(num_samples=100, node_feat_dim=node_feat_dim)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for node_feats, edge_index, query_text, label in dataloader:
        optimizer.zero_grad()
        score = model(node_feats.squeeze(0), edge_index.squeeze(0), query_text)
        loss = criterion(score, label.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

model.eval()
with torch.no_grad():
    for node_feats, edge_index, query_text, label in dataloader:
        score = model(node_feats.squeeze(0), edge_index.squeeze(0), query_text)
        prediction = torch.sigmoid(score)
        print(f"Predicted score: {prediction.item():.4f}, True label: {label.item()}")
