
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm


class LinkPredictionPretrainer:
    """
    Simple link prediction pretraining across all edge types
    """
    def __init__(self, encoder, device='cuda', batch_size = 4096):
        self.encoder = encoder
        self.device = device
        self.batch_size = batch_size
    
    def negative_sampling_safe(self, pos_edge_index, num_src_nodes, num_dst_nodes, num_neg_samples=None):
        """
        Safe negative sampling that handles bipartite graphs
        """
        if num_neg_samples is None:
            num_neg_samples = pos_edge_index.shape[1]
        
        # Sample random negative edges
        neg_src = torch.randint(0, num_src_nodes, (num_neg_samples,), device=self.device)
        neg_dst = torch.randint(0, num_dst_nodes, (num_neg_samples,), device=self.device)
        neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
        
        return neg_edge_index
    
    def link_prediction_loss_batched(self, src_emb, dst_emb, pos_edge_index, neg_edge_index, batch_size):
        """
        Batched link prediction loss to save memory
        """
        num_edges = pos_edge_index.shape[1]
        total_loss = 0
        total_pos_score = 0
        total_neg_score = 0
        num_batches = 0
        
        for start_idx in range(0, num_edges, batch_size):
            end_idx = min(start_idx + batch_size, num_edges)
            
            # Batch positive edges
            batch_pos_idx = pos_edge_index[:, start_idx:end_idx]
            pos_src = src_emb[batch_pos_idx[0]]
            pos_dst = dst_emb[batch_pos_idx[1]]
            pos_scores = (pos_src * pos_dst).sum(dim=1)
            
            # Batch negative edges
            batch_neg_idx = neg_edge_index[:, start_idx:end_idx]
            neg_src = src_emb[batch_neg_idx[0]]
            neg_dst = dst_emb[batch_neg_idx[1]]
            neg_scores = (neg_src * neg_dst).sum(dim=1)
            
            # BPR loss
            batch_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            
            total_loss += batch_loss
            total_pos_score += pos_scores.mean().item()
            total_neg_score += neg_scores.mean().item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_pos = total_pos_score / num_batches
        avg_neg = total_neg_score / num_batches
        
        return avg_loss, avg_pos, avg_neg
    
    def pretrain_step(self, data):
        """
        Memory-efficient pretraining step with batched processing
        """
        z_dict = self.encoder(data.x_dict, data.edge_index_dict)
        
        losses = {}
        metrics = {}
        
        # 1. User-Movie edges (60% weight)
        if ('user', 'rated_high', 'movie') in data.edge_types:
            pos_edges = data['user', 'rated_high', 'movie'].edge_index
            
            num_pos = pos_edges.shape[1]
            num_neg = min(num_pos, 100000)  # Cap at 100k negatives
            
            neg_edges = self.negative_sampling_safe(
                pos_edges,
                data['user'].num_nodes,
                data['movie'].num_nodes,
                num_neg_samples=num_neg
            )
            
            # Use subset of positive edges if too many
            if num_pos > 100000:
                perm = torch.randperm(num_pos, device=self.device)[:100000]
                pos_edges = pos_edges[:, perm]
            
            loss, pos_score, neg_score = self.link_prediction_loss_batched(
                z_dict['user'], z_dict['movie'],
                pos_edges, neg_edges,
                batch_size=self.batch_size
            )
            losses['user_movie'] = loss
            metrics['user_movie_pos'] = pos_score
            metrics['user_movie_neg'] = neg_score
        
        # 2. Movie-Genre edges (30% weight)
        if ('movie', 'has_genre', 'genre') in data.edge_types:
            pos_edges = data['movie', 'has_genre', 'genre'].edge_index
            neg_edges = self.negative_sampling_safe(
                pos_edges,
                data['movie'].num_nodes,
                data['genre'].num_nodes,
                num_neg_samples=pos_edges.shape[1]
            )
            
            loss, pos_score, neg_score = self.link_prediction_loss_batched(
                z_dict['movie'], z_dict['genre'],
                pos_edges, neg_edges,
                batch_size=self.batch_size
            )
            losses['movie_genre'] = loss
            metrics['movie_genre_pos'] = pos_score
            metrics['movie_genre_neg'] = neg_score
        
        # 3. Conversation-Movie edges (10% weight)
        if ('conversation', 'mentions', 'movie') in data.edge_types:
            pos_edges = data['conversation', 'mentions', 'movie'].edge_index
            
            if pos_edges.shape[1] > 0:
                neg_edges = self.negative_sampling_safe(
                    pos_edges,
                    data['conversation'].num_nodes,
                    data['movie'].num_nodes,
                    num_neg_samples=pos_edges.shape[1]
                )
                
                loss, pos_score, neg_score = self.link_prediction_loss_batched(
                    z_dict['conversation'], z_dict['movie'],
                    pos_edges, neg_edges,
                    batch_size=self.batch_size
                )
                losses['conv_movie'] = loss
                metrics['conv_movie_pos'] = pos_score
                metrics['conv_movie_neg'] = neg_score
        
        # Weighted combination
        weights = {
            'user_movie': 0.6,
            'movie_genre': 0.3,
            'conv_movie': 0.1
        }
        
        total_loss = sum(weights.get(k, 0) * v for k, v in losses.items())
        
        # Add individual losses to metrics
        for k, v in losses.items():
            metrics[f'{k}_loss'] = v.item()
        
        return total_loss, metrics


def pretrain_link_prediction(encoder, data, epochs=15, lr=5e-4, 
                             warmup_epochs=3, device='cuda',
                             checkpoint_dir='checkpoints',
                             batch_size=4096,
                             accumulation_steps=1):
    """
    Memory-efficient pretraining
    """
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize with batch size
    pretrainer = LinkPredictionPretrainer(encoder, device=device, batch_size=batch_size)
    
    # Use more aggressive memory settings
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    history = {
        'total_loss': [],
        'user_movie_loss': [],
        'movie_genre_loss': [],
        'conv_movie_loss': [],
        'lr': []
    }
    
    # Training loop
    print(f"\n=== Training ===")
    encoder.train()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Clear cache at start of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Forward pass
        total_loss, metrics = pretrainer.pretrain_step(data)
        
        # Scale loss for gradient accumulation
        scaled_loss = total_loss / accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        # Update weights every accumulation_steps
        if (epoch + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Record history
        history['total_loss'].append(total_loss.item())
        history['lr'].append(optimizer.param_groups[0]['lr'])
        for k, v in metrics.items():
            if k.endswith('_loss'):
                if k not in history:
                    history[k] = []
                history[k].append(v)
        
        # Logging
        if epoch % 5 == 0 or epoch == epochs - 1:
            log_str = f"Epoch {epoch:2d}/{epochs}: Loss={total_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}"
            
            if torch.cuda.is_available():
                log_str += f", GPU={torch.cuda.memory_allocated() / 1e9:.2f}GB"
            
            if 'user_movie_loss' in metrics:
                log_str += f"\n  User-Movie: loss={metrics['user_movie_loss']:.4f}, pos={metrics['user_movie_pos']:.3f}, neg={metrics['user_movie_neg']:.3f}"
            if 'movie_genre_loss' in metrics:
                log_str += f"\n  Movie-Genre: loss={metrics['movie_genre_loss']:.4f}, pos={metrics['movie_genre_pos']:.3f}, neg={metrics['movie_genre_neg']:.3f}"
            if 'conv_movie_loss' in metrics:
                log_str += f"\n  Conv-Movie: loss={metrics['conv_movie_loss']:.4f}, pos={metrics['conv_movie_pos']:.3f}, neg={metrics['conv_movie_neg']:.3f}"
            
            print(log_str)
        
        # Save best checkpoint
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            checkpoint_path = os.path.join(checkpoint_dir, 'best_pretrained_encoder.pt')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'history': history
            }, checkpoint_path)
    
    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, 'final_pretrained_encoder.pt')
    torch.save({
        'epoch': epochs,
        'encoder_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss.item(),
        'history': history
    }, final_path)
    
    print(f"\n{'='*70}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Saved checkpoints:")
    print(f"  - Best: {checkpoint_path}")
    print(f"  - Final: {final_path}")
    print(f"{'='*70}\n")
    
    return encoder, history


def plot_training_history(history, save_path='pretrain_history.png'):
    """
    Plot training curves
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Total loss
    axes[0].plot(history['total_loss'], label='Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Pretraining Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Individual losses
    for key in history.keys():
        if key.endswith('_loss') and key != 'total_loss':
            axes[1].plot(history[key], label=key.replace('_loss', ''))
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Individual Task Losses')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved training plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    from utils.utils import load_graph
    from models.models import GATEncoder
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading graph...")
    data = load_graph(name='combined_graph_filtered').to(device)
    print("✓ Graph loaded\n")
    
    # Initialize encoder
    print("Initializing encoder...")
    encoder = GATEncoder(num_layers=2).to(device)
    
    # Pretrain
    encoder, history = pretrain_link_prediction(
        encoder=encoder,
        data=data,
        epochs=15,
        lr=5e-4,
        warmup_epochs=3,
        device=device,
        checkpoint_dir='checkpoints'
    )
    
    # Plot results
    try:
        plot_training_history(history)
    except ImportError:
        print("Matplotlib not available, skipping plot")
    