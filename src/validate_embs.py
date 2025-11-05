import torch
import torch_geometric
from utils.utils import load_graph
from models.models import GATEncoder
import inspect
import _operator
import typing
import torch.nn.functional as F
import numpy


def check_embedding_quality(encoder, data, device='cuda'):
    """
    Diagnose potential model collapse
    """
    print("\n" + "="*60)
    print("Embedding Quality Diagnostics")
    print("="*60)
    encoder.to(device)
    data.to(device)
    encoder.eval()
    with torch.no_grad():
        z_dict = encoder(data.x_dict, data.edge_index_dict)
        
        for node_type, emb in z_dict.items():
            # 1. Variance (collapse → near 0)
            variance = emb.var(dim=0).mean().item()
            
            # 2. Norm (check if embeddings are normalized)
            norms = emb.norm(dim=1)
            mean_norm = norms.mean().item()
            std_norm = norms.std().item()
            
            # 3. Pairwise similarity (collapse → all similar)
            sample_size = min(1000, emb.shape[0])
            sample_idx = torch.randperm(emb.shape[0])[:sample_size]
            sample_emb = F.normalize(emb[sample_idx], p=2, dim=1)
            
            sim_matrix = torch.matmul(sample_emb, sample_emb.t())
            # Remove diagonal
            mask = ~torch.eye(sample_size, device=device).bool()
            pairwise_sim = sim_matrix[mask].cpu()
            
            mean_sim = pairwise_sim.mean().item()
            std_sim = pairwise_sim.std().item()
            
            print(f"\n{node_type}:")
            print(f"  Variance: {variance:.4f} {'✅' if variance > 0.01 else '❌ TOO LOW!'}")
            print(f"  Norm: {mean_norm:.3f} ± {std_norm:.3f} {'✅' if mean_norm > 0.1 else '❌ TOO LOW!'}")
            print(f"  Pairwise similarity: {mean_sim:.3f} ± {std_sim:.3f} {'✅' if abs(mean_sim) < 0.3 else '⚠️ HIGH!'}")
            
            # Check for collapse
            if variance < 0.001:
                print('variance too small')
            if abs(mean_sim) > 0.8:
                print('very high similarity')
            if mean_norm < 0.01:
                print(f"norm too small")
    
    encoder.train()
    print("="*60 + "\n")

# Run after training:

    
    # You would then call this in your main validation function
    # after you calculate `movie_embs = F.normalize(...)`
if __name__ == '__main__':
    torch.serialization.add_safe_globals([GATEncoder,#FIXME: there has to be a way around this
                                          torch.nn.modules.container.ModuleList,
                                          torch_geometric.nn.conv.hetero_conv.HeteroConv,
                                          torch_geometric.nn.module_dict.ModuleDict,
                                          torch_geometric.nn.conv.gat_conv.GATConv,
                                          torch_geometric.nn.aggr.basic.SumAggregation,
                                          torch_geometric.nn.dense.linear.Linear,
                                          torch_geometric.inspector.Inspector,
                                          torch_geometric.inspector.Signature,
                                          torch_geometric.inspector.Parameter,
                                          inspect._empty,
                                          _operator.getitem,
                                          typing.Union,
                                          type,
                                          int,
                                          typing.OrderedDict,
                                          numpy._core.multiarray.scalar,
                                          numpy.dtype,
                                          numpy.dtypes.Float64DType])
    
    encoder = GATEncoder()
    encoder.parameters = torch.load('checkpoints/best_anti_collapse.pt', weights_only=True)
    data = load_graph(name='combined_graph_filtered')
    # Check current movie embeddings
    movie_embs = data['movie'].x
    sample = movie_embs[torch.randperm(len(movie_embs))[:1000]]
    sample_norm = F.normalize(sample, p=2, dim=1)
    sim_matrix = torch.matmul(sample_norm, sample_norm.t())
    mask = ~torch.eye(1000).bool()
    mean_sim = sim_matrix[mask].mean().item()

    sim = F.cosine_similarity(movie_embs[0:1], movie_embs[1:1000]).mean()
    print(f"Initial movie similarity: {sim:.3f}")

    print(f"Movie embedding similarity: {mean_sim:.3f}")

    if mean_sim > 0.7:
        print('too similar!')
    else:
        print('good diversity')
    
    check_embedding_quality(encoder, data)