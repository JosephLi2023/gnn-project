import torch
import torch.nn.functional as F
from utils.utils import load_graph


def check_embedding_quality(encoder, data, device='cuda'):
    """
    Diagnose potential model collapse by checking:
    - Variance: Collapsed embeddings have near-zero variance
    - Norm: Check if embeddings have reasonable magnitude
    - Pairwise similarity: Collapsed embeddings are all similar to each other
    """
    print("\n" + "="*60)
    print("Embedding Quality Diagnostics")
    print("="*60)
    encoder.to(device)
    data = data.to(device)
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
            print(f"  Shape: {emb.shape}")
            print(f"  Variance: {variance:.4f} {'good' if variance > 0.01 else 'too low'}")
            print(f"  Norm: {mean_norm:.3f} ± {std_norm:.3f} {'good' if mean_norm > 0.1 else 'too low'}")
            print(f"  Pairwise similarity: {mean_sim:.3f} ± {std_sim:.3f} {'good' if abs(mean_sim) < 0.3 else 'high'}")

            # Check for collapse
            issues = []
            if variance < 0.001:
                issues.append('collapse: low variance')
            if abs(mean_sim) > 0.8:
                issues.append('collapse: high similarity')
            if mean_norm < 0.01:
                issues.append('collapse: norm')

            if issues:
                for issue in issues:
                    print(issue)
            else:
                print(f"No collapse detected!")

    encoder.train()
    print("="*60 + "\n")

def validate_pretrained_encoder(encoder_class, checkpoint_path, data, device='cuda'):
    """
    Load a pretrained encoder and validate its embeddings.

    Args:
        encoder_class: The encoder class (e.g., HGTEncoder, RGCNEncoder)
        checkpoint_path: Path to the saved checkpoint
        data: The graph data
        device: Device to run on
    """
    print(f"\n{'='*60}")
    print(f"Validating: {encoder_class.__name__}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    # Initialize encoder with same args as training
    encoder = encoder_class(
        metadata=data.metadata(),
        x_dict=data.x_dict,
        hidden_size=128,
        num_layers=2,
        **({'num_heads': 4} if 'HGT' in encoder_class.__name__ or 'HAN' in encoder_class.__name__ else {}),
        **({'num_bases': 10} if 'RGCN' in encoder_class.__name__ else {})
    ).to(device)

    # Load trained weights
    encoder.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    # Run diagnostics
    check_embedding_quality(encoder, data, device=device)

    return encoder


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    data = load_graph(name='combined_graph_filtered')

    # Check initial feature quality before training
    print("\n" + "="*60)
    print("Initial Feature Quality (Before Training)")
    print("="*60)
    for node_type, features in data.x_dict.items():
        sample_size = min(1000, features.shape[0])
        sample = features[torch.randperm(features.shape[0])[:sample_size]]
        sample_norm = F.normalize(sample, p=2, dim=1)
        sim_matrix = torch.matmul(sample_norm, sample_norm.t())
        mask = ~torch.eye(sample_size).bool()
        mean_sim = sim_matrix[mask].mean().item()

        print(f"\n{node_type}:")
        print(f"  Shape: {features.shape}")
        print(f"  Mean similarity: {mean_sim:.3f}")
        print(f"  Status: {'Good diversity' if mean_sim < 0.7 else 'High similarity'}")
    print("="*60 + "\n")

    # Dictionary of available encoders and their checkpoint paths
    from premade_hetero_models import HGTEncoder, RGCNEncoder, HANEncoder, HeteroSAGEEncoder

    encoders_to_validate = [
        (HGTEncoder, 'best_hgt_encoder.pt'),
        (RGCNEncoder, 'best_rgcn_encoder.pt'),
        (HANEncoder, 'best_han_encoder.pt'),
        (HeteroSAGEEncoder, 'best_sage_encoder.pt'),
    ]

    # Validate each encoder that has a checkpoint
    import os
    results = {}

    for encoder_class, checkpoint_path in encoders_to_validate:
        if os.path.exists(checkpoint_path):
            try:
                encoder = validate_pretrained_encoder(
                    encoder_class,
                    checkpoint_path,
                    data,
                    device=device
                )
                results[encoder_class.__name__] = encoder
                print(f"{encoder_class.__name__} validated successfully!\n")
            except Exception as e:
                print(f"Error validating {encoder_class.__name__}: {e}\n")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            print(f"Skipping {encoder_class.__name__}\n")

    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    print(f"Total encoders validated: {len(results)}")
    for name in results.keys():
        print(name)
    print("="*60)