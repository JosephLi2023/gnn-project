import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from transformers import BertTokenizer, BertModel

# --- GAT Encoder with configurable layers ---
class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, num_heads=2, dropout=0.6):
        super().__init__()
        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout))
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        # Last layer
        self.layers.append(GATConv(hidden_dim * num_heads, out_dim, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                x = torch.relu(x)
        return x  # [num_nodes, out_dim]

# --- BERT Query Encoder ---
class BERTQueryEncoder(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)

    def forward(self, query_text):
        # query_text: list of strings (batch)
        inputs = self.tokenizer(query_text, return_tensors='pt', truncation=True, max_length=64, padding=True)
        outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]

# --- Query-Aware Pooling ---
class QueryAwarePooling(nn.Module):
    def __init__(self, graph_dim, query_dim):
        super().__init__()
        self.attn = nn.Linear(graph_dim + query_dim, 1)

    def forward(self, graph_embs, query_emb):
        # graph_embs: [num_nodes, graph_dim]
        # query_emb: [batch_size, query_dim]
        batch_size = query_emb.size(0)
        num_nodes = graph_embs.size(0)
        query_expanded = query_emb.unsqueeze(1).repeat(1, num_nodes, 1)
        graph_expanded = graph_embs.unsqueeze(0).repeat(batch_size, 1, 1)
        concat = torch.cat([graph_expanded, query_expanded], dim=-1)
        attn_scores = self.attn(concat).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * graph_expanded, dim=1)
        return pooled

# --- Fusion and Scoring ---
class FusionScorer(nn.Module):
    def __init__(self, graph_dim, query_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(graph_dim + query_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, pooled_graph_emb, query_emb):
        x = torch.cat([pooled_graph_emb, query_emb], dim=-1)
        x = torch.relu(self.fc1(x))
        score = self.fc2(x)
        return score.squeeze(-1)

# --- Full Model ---
class QueryAwareGATModel(nn.Module):
    def __init__(self, node_feat_dim, gat_hidden, gat_out, bert_dim, fusion_hidden, num_gat_layers=3):
        super().__init__()
        self.gat_encoder = GATEncoder(node_feat_dim, gat_hidden, gat_out, num_layers=num_gat_layers)
        self.query_encoder = BERTQueryEncoder()
        self.pooling = QueryAwarePooling(gat_out, bert_dim)
        self.fusion = FusionScorer(gat_out, bert_dim, fusion_hidden)

    def forward(self, node_feats, edge_index, query_text):
        graph_embs = self.gat_encoder(node_feats, edge_index)  # [num_nodes, gat_out]
        query_emb = self.query_encoder(query_text)             # [batch_size, bert_dim]
        pooled_graph_emb = self.pooling(graph_embs, query_emb) # [batch_size, gat_out]
        score = self.fusion(pooled_graph_emb, query_emb)       # [batch_size]
        return score
