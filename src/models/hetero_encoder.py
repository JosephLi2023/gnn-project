import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class HeteroGNNConv(MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels, edge_attr_dim=None):
        super().__init__(aggr="mean")
        self.lin_dst = nn.Linear(in_channels_dst, out_channels)
        self.lin_src = nn.Linear(in_channels_src, out_channels)
        self.lin_update = nn.Linear(2 * out_channels, out_channels)
        if edge_attr_dim is not None:
            self.lin_edge = nn.Linear(edge_attr_dim, out_channels)
        else:
            self.lin_edge = None

    def forward(self, node_feature_src, node_feature_dst, edge_index, edge_attr=None, size=None):
        return self.propagate(
            edge_index,
            node_feature_src=node_feature_src,
            node_feature_dst=node_feature_dst,
            edge_attr=edge_attr,
            size=size)

    def message(self, node_feature_src_j, edge_attr):
        msg=node_feature_src_j
        if self.lin_edge is not None and edge_attr is not None:
            msg = msg + self.lin_edge(msg)
        return msg

    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter_mean(inputs, index, dim=0, dim_size=dim_size)

    def update(self, aggr_out, node_feature_dst):
        h_dst = self.lin_dst(node_feature_dst)
        h_src = self.lin_src(aggr_out)
        h_cat = torch.cat([h_dst, h_src], dim=1)
        aggr_out = self.lin_update(h_cat)
        
        return aggr_out

class HeteroGNNWrapperConv(nn.Module):
    def __init__(self, convs, aggr="mean", hidden_size=None, attn_size=None):
        super().__init__()
        self.convs = nn.ModuleDict({str(k): v for k, v in convs.items()})
        self.aggr = aggr
        if aggr == "attn":
            self.attn_proj = nn.Sequential(
                nn.Linear(hidden_size, attn_size),
                nn.Tanh(),
                nn.Linear(attn_size, 1, bias=False)
            )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # Compute per-message-type embeddings
        message_type_emb = {}
        for message_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = message_type
            conv = self.convs[str(message_type)]
            edge_attr = None
            if edge_attr_dict is not None and message_type in edge_attr_dict:
                edge_attr = edge_attr_dict[message_type]
            message_type_emb[message_type] = conv(
                x_dict[src_type], x_dict[dst_type], edge_index, edge_attr=edge_attr
            )
        # Aggregate per destination node type
        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        for (src, edge, dst), emb in message_type_emb.items():
            node_emb[dst].append(emb)
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
        return node_emb

    def aggregate(self, xs):
        if self.aggr == "mean":
            x = torch.stack(xs, dim=0)
            return x.mean(dim=0)
        elif self.aggr == "attn":
            M, N, D = len(xs), xs[0].shape[0], xs[0].shape[1]
            x = torch.stack(xs, dim=0)  # [M, N, D]
            z = self.attn_proj(x)       # [M, N, 1]
            z = z.mean(1)               # [M, 1]
            alpha = torch.softmax(z, dim=0)  # [M, 1]
            x = x * alpha.view(M, 1, 1)
            return x.sum(dim=0)

def generate_convs(metadata, conv_class, hidden_size, x_dict, edge_attr_dims, first_layer=False):
    convs = {}
    for message_type in metadata[1]:
        src_type, _, dst_type = message_type
        if first_layer:
            in_channels_src = x_dict[src_type].size(1)
            in_channels_dst = x_dict[dst_type].size(1)
        else:
            in_channels_src = hidden_size
            in_channels_dst = hidden_size
        edge_attr_dim = edge_attr_dims.get(message_type, None)
        convs[message_type] = conv_class(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            out_channels=hidden_size,
            edge_attr_dim=edge_attr_dim
        )
    return convs

class HeteroGNN(nn.Module):
    def __init__(self, metadata, x_dict, args, edge_attr_dims, aggr="mean"):
        super().__init__()
        self.aggr = aggr
        self.hidden_size = args['hidden_size']
        self.attn_size = args.get('attn_size', 32)

        # Layer 1
        convs1_dict = generate_convs(metadata, HeteroGNNConv, self.hidden_size, x_dict, edge_attr_dims, first_layer=True)
        self.convs1 = HeteroGNNWrapperConv(convs1_dict, aggr=self.aggr, hidden_size=self.hidden_size, attn_size=self.attn_size)
        self.bns1 = nn.ModuleDict({nt: nn.BatchNorm1d(self.hidden_size) for nt in x_dict})
        self.relus1 = nn.ModuleDict({nt: nn.LeakyReLU() for nt in x_dict})
        # Layer 2
        convs2_dict = generate_convs(metadata, HeteroGNNConv, self.hidden_size, x_dict, edge_attr_dims, first_layer=False)
        self.convs2 = HeteroGNNWrapperConv(convs2_dict, aggr=self.aggr, hidden_size=self.hidden_size, attn_size=self.attn_size)
        self.bns2 = nn.ModuleDict({nt: nn.BatchNorm1d(self.hidden_size) for nt in x_dict})
        self.relus2 = nn.ModuleDict({nt: nn.LeakyReLU() for nt in x_dict})
        # Post-MPS (for link prediction, output embeddings)
        self.post_mps = nn.ModuleDict({nt: nn.Identity() for nt in x_dict})

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x = self.convs1(x_dict, edge_index_dict, edge_attr_dict)
        x = {k: self.bns1[k](v) for k, v in x.items()}
        x = {k: self.relus1[k](v) for k, v in x.items()}
        x = self.convs2(x, edge_index_dict, edge_attr_dict)
        x = {k: self.bns2[k](v) for k, v in x.items()}
        x = {k: self.relus2[k](v) for k, v in x.items()}
        x = {k: self.post_mps[k](v) for k, v in x.items()}

        return x
