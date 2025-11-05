def load_graph(dir = 'processed_data', name='combined_graph'):
    import torch
    from torch_geometric.data.hetero_data import HeteroData
    from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage
    torch.serialization.add_safe_globals([HeteroData, BaseStorage, NodeStorage, EdgeStorage])
    return torch.load(f'{dir}/{name}.pt', weights_only=True)