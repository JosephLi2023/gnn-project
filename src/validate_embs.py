import torch
import torch_geometric
from utils.utils import load_graph
from models.encoder import GATEncoder
import inspect
import _operator
import typing
import torch.nn.functional as F
import numpy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

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

def visualize_movie_embeddings_multi_genre(encoder, data, top_n_genres=6, sample_size=2000, device='cuda'):
    """
    Create multiple TSNE plots, one highlighting each major genre
    """
    encoder.to(device)
    data = data.to(device)
    encoder.eval()
    
    with torch.no_grad():
        z_dict = encoder(data.x_dict, data.edge_index_dict)
        movie_emb = z_dict['movie']
        
        n_movies = movie_emb.shape[0]
        movie_genre_edges = data['movie', 'has_genre', 'genre'].edge_index
        
        # Get genre names from node attributes
        genre_names = data['genre'].genre_name  # This should be a list or tensor
        print(f"Total genres in graph: {len(genre_names)}")
        
        # Sample if needed
        if n_movies > sample_size:
            sample_idx = torch.randperm(n_movies)[:sample_size].sort()[0]
            tsne_emb = movie_emb[sample_idx].cpu().numpy()
        else:
            sample_idx = torch.arange(n_movies)
            tsne_emb = movie_emb.cpu().numpy()
            sample_size = n_movies
        
        # Build movie -> genres mapping (multi-label)
        from collections import defaultdict
        movie_to_genres = defaultdict(list)
        sample_idx_set = set(sample_idx.cpu().numpy())
        
        for movie_idx, genre_idx in zip(movie_genre_edges[0].cpu().numpy(), 
                                        movie_genre_edges[1].cpu().numpy()):
            if movie_idx in sample_idx_set:
                movie_to_genres[movie_idx].append(genre_idx)
        
        # Count genre frequencies to find top genres
        genre_counts = defaultdict(int)
        for genres in movie_to_genres.values():
            for g in genres:
                genre_counts[g] += 1
        
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:top_n_genres]
        print(f"\nTop {top_n_genres} genres by frequency:")
        for genre_idx, count in top_genres:
            # Get the actual genre name
            genre_name = genre_names[genre_idx]
            print(f"  {genre_name}: {count} movies")
        
        # Run TSNE once
        perplexity = min(30, max(5, sample_size // 3 - 1))
        print(f"\nRunning TSNE with perplexity={perplexity}...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, verbose=1)
        X_tsne = tsne.fit_transform(tsne_emb)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (genre_idx, count) in enumerate(top_genres):
            ax = axes[idx]
            
            # Get genre name
            genre_name = genre_names[genre_idx]
            
            # Create mask for this genre
            has_genre = np.array([genre_idx in movie_to_genres.get(m.item(), []) 
                                  for m in sample_idx])
            
            # Plot background (movies without this genre)
            ax.scatter(X_tsne[~has_genre, 0], X_tsne[~has_genre, 1], 
                      s=5, alpha=0.2, c='lightgray', edgecolors='none')
            
            # Highlight movies with this genre
            ax.scatter(X_tsne[has_genre, 0], X_tsne[has_genre, 1], 
                      s=15, alpha=0.7, c='red', edgecolors='none')
            
            ax.set_title(f'{genre_name}\n({count} movies, {count/sample_size*100:.1f}%)', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('TSNE-1')
            ax.set_ylabel('TSNE-2')
        
        plt.suptitle(f'Movie Embeddings by Genre\n({sample_size} movies)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('tsne_movie_embeddings_multi_genre.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    encoder.train()

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
    encoder.parameters = torch.load('checkpoints/best_pretrained_encoder.pt', weights_only=True)
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
        print('too similar')
    else:
        print('good diversity')
    
    check_embedding_quality(encoder, data)
    visualize_movie_embeddings_multi_genre(encoder, data, top_n_genres=6, sample_size=2000)