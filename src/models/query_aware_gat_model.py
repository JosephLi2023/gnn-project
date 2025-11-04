import torch
import torch.nn as nn
from edge_aware_gatconv import EdgeAwareGATConv
from query_guided_pooling import QueryGuidedPooling
from fusion_scorer import FusionScorer

class QueryAwareGATModel(nn.Module):
    def __init__(self, node_feat_dim, edge_type_count, gat_hidden, gat_out, bert_dim, fusion_hidden, num_gat_layers=3):
        super().__init__()
        self.gat_layers = nn.ModuleList([
            EdgeAwareGATConv(node_feat_dim if i == 0 else gat_hidden, gat_hidden, edge_type_count)
            for i in range(num_gat_layers)
        ])
        self.query_encoder = BERTQueryEncoder(bert_model_name='bert-base-uncased')
        self.pooling = QueryGuidedPooling(gat_hidden, bert_dim)
        self.fusion = FusionScorer(gat_hidden, bert_dim)
    
    def forward(self, node_feats, edge_index, edge_type, edge_weight, query_text):
        x = node_feats
        for layer in self.gat_layers:
            x = layer(x, edge_index, edge_type, edge_weight)
            x = torch.relu(x)
        query_emb = self.query_encoder(query_text)
        pooled = self.pooling(x, query_emb)
        score = self.fusion(pooled, query_emb)
        return score
