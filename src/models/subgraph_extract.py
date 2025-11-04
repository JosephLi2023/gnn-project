from torch_geometric.utils import from_networkx

def subgraph_to_pyg(nx_subgraph, node_feat_dim):
    if nx_subgraph.number_of_edges() == 0:
        return None
    for n in nx_subgraph.nodes:
        emb = nx_subgraph.nodes[n].get('embedding', np.zeros(node_feat_dim))
        nx_subgraph.nodes[n]['x'] = torch.tensor(emb, dtype=torch.float)
    data = from_networkx(nx_subgraph, node_attrs=['x'], edge_attrs=['weight', 'type'])
    data.edge_weight = data.weight
    data.edge_type = data.type
    return data
