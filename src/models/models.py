import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from transformers import BertTokenizer, BertModel

# --- GAT Encoder with configurable layers ---
class GATEncoder(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=384, num_layers=3, num_heads=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        for i in range(num_layers):
            if i == 0:
                in_channels = (-1,-1) # lazy init helps with heterogenous graphs
                out_channels = hidden_dim
                heads = num_heads
                concat = True
            elif i == num_layers - 1:
                in_channels = hidden_dim * num_heads
                out_channels = out_dim
                heads = 1
                concat = False
            else:
                in_channels = hidden_dim * num_heads
                out_channels = hidden_dim
                heads = num_heads
                concat = True

            layer = HeteroConv({
                ('movie', 'has_genre', 'genre'): 
                    GATConv(in_channels, out_channels, heads=heads, concat=concat, 
                           dropout=dropout, add_self_loops=False),
                            ('genre', 'has_movie', 'movie'): 
                    GATConv(in_channels, out_channels, heads=heads, concat=concat,
                           dropout=dropout, add_self_loops=False),
                            ('user', 'rated_high', 'movie'):
                    GATConv(in_channels, out_channels, heads=heads, concat=concat,
                           dropout=dropout, add_self_loops=False),
                            ('movie', 'rated_by', 'user'):
                    GATConv(in_channels, out_channels, heads=heads, concat=concat,
                           dropout=dropout, add_self_loops=False),
                            ('conversation', 'mentions', 'movie'): 
                    GATConv(in_channels, out_channels, heads=heads, concat=concat,
                           dropout=dropout, add_self_loops=False),
                            ('movie', 'mentioned_in', 'conversation'):
                    GATConv(in_channels, out_channels, heads=heads, concat=concat,
                           dropout=dropout, add_self_loops=False)
            }, aggr='sum')
            self.layers.append(layer)

    def forward(self, x_dict, edge_index_dict):
        """
        Args:
            x_dict: Dict[node_type, Tensor[num_nodes, feat_dim]]
            edge_index_dict: Dict[edge_type, Tensor[2, num_edges]]
        Returns:
            x_dict: Dict[node_type, Tensor[num_nodes, out_dim]]
        """
        for i, layer in enumerate(self.layers):
            x_dict = layer(x_dict, edge_index_dict)
            
            # Apply activation and dropout (except last layer)
            if i != self.num_layers - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                         for key, x in x_dict.items()}
        
        return x_dict


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
