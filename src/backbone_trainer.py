import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit
import gc


def clear_cuda_memory():
    """Clear CUDA memory cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def split_edges(data, val_split=0.1, test_split=0.1, neg_sampling_ratio=1.0):

    # Get all edge types from the graph
    all_edge_types = list(data.edge_types)
    edge_type_set = set(all_edge_types)
    processed = set()

    # Split edge types into forward and reverse pairs
    edge_types = []  # Will contain one edge from each pair (e.g., 'user rates movie')
    rev_edge_types = []  # Will contain corresponding reverse edges

    for edge_type in all_edge_types:
        if edge_type in processed:
            continue

        src, rel, dst = edge_type

        # Check if there's a reverse edge with flipped source and destination types
        reverse_edge = None
        for other_edge_type in edge_type_set:
            other_src, other_rel, other_dst = other_edge_type

            # If source and destination are flipped, it's a reverse edge
            if src == other_dst and dst == other_src and edge_type != other_edge_type:
                reverse_edge = other_edge_type
                break

        if reverse_edge:
            # Found a bidirectional pair
            edge_types.append(edge_type)
            rev_edge_types.append((edge_type, reverse_edge))
            processed.add(edge_type)
            processed.add(reverse_edge)
        else:
            # No reverse edge found, just add as a regular edge
            edge_types.append(edge_type)

    transform = RandomLinkSplit(
        num_val=val_split,
        num_test=test_split,
        is_undirected=False,
        edge_types=edge_types,
        rev_edge_types=rev_edge_types if rev_edge_types else None,
        neg_sampling_ratio=neg_sampling_ratio,  # Auto-generate negative samples
        add_negative_train_samples=True,
    )

    # Apply transform - returns 3 data objects
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data


def compute_loss(z_src, z_dst, edge_label_index, edge_label, margin=1.0, num_neg_samples=5):
    # Compute dot product scores for all edges (positive + negative)
    src_emb = z_src[edge_label_index[0]]
    dst_emb = z_dst[edge_label_index[1]]
    scores = (src_emb * dst_emb).sum(dim=1)

    # Split into positive and negative samples
    pos_mask = edge_label == 1
    neg_mask = edge_label == 0

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]

    # Handle edge cases
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        # Fallback to BCE if we don't have both positive and negative samples
        return F.binary_cross_entropy_with_logits(scores, edge_label.float())

    # Memory-efficient: sample K negatives per positive instead of all pairs
    num_pos = pos_scores.size(0)
    num_neg = neg_scores.size(0)

    # Sample negative indices for each positive (with replacement if needed)
    K = min(num_neg_samples, num_neg)
    neg_indices = torch.randint(0, num_neg, (num_pos, K), device=pos_scores.device)

    # Get sampled negative scores [num_pos, K]
    sampled_neg_scores = neg_scores[neg_indices]

    # Compute BPR loss: -log(sigmoid(pos - neg - margin))
    pos_expanded = pos_scores.unsqueeze(1)  # [num_pos, 1]
    diff = pos_expanded - sampled_neg_scores - margin  # [num_pos, K]

    loss = -F.logsigmoid(diff).mean()

    return loss


def train_epoch(encoder, data, margin=1.0):
    encoder.train()
    z = encoder(data.x_dict, data.edge_index_dict)

    total_loss = 0
    num_edge_types = 0

    # Iterate over edge types that have supervision labels
    for edge_type in data.edge_label_index_dict.keys():
        edge_label_index = data.edge_label_index_dict[edge_type]
        edge_label = data.edge_label_dict[edge_type]

        if edge_label_index.shape[1] == 0:
            continue

        src_type, _, dst_type = edge_type

        loss = compute_loss(
            z[src_type],
            z[dst_type],
            edge_label_index,
            edge_label,
            margin=margin
        )
        total_loss += loss
        num_edge_types += 1

    # Average loss across edge types
    if num_edge_types > 0:
        total_loss = total_loss / num_edge_types

    return total_loss


def eval_epoch(encoder, data, margin=1.0):
    """Single evaluation epoch"""
    encoder.eval()
    with torch.no_grad():
        return train_epoch(encoder, data, margin=margin)


def pretrain(encoder, data, epochs=50, lr=1e-3, patience=10, neg_sampling_ratio=1.0, margin=1.0, file_name='best_encoder.pt'):

    # Split edges using PyG's RandomLinkSplit
    train_data, val_data, test_data = split_edges(data, neg_sampling_ratio=neg_sampling_ratio)

    # Setup training
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(encoder, train_data, margin=margin)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Validate
        val_loss = eval_epoch(encoder, val_data, margin=margin)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(encoder.state_dict(), file_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}")

    # Test
    print("\n=== Final Test Evaluation ===")
    encoder.load_state_dict(torch.load(file_name, weights_only=True))
    test_loss = eval_epoch(encoder, test_data, margin=margin)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}\n")

    return encoder


if __name__ == "__main__":
    from utils.utils import load_graph
    from models.hetero_encoder import HeteroGNN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    data = load_graph(name='combined_graph_filtered').to(device)

    # # Define args for HeteroGNN
    # args = {
    #     'hidden_size': 128,  # Adjust as needed
    #     'attn_size': 32      # Only used if aggr='attn'
    # }
    print(' starting suite of heterogeneous GNN encoders...\n')

    # HGT
    try:
        print("=== Training HGTEncoder ===")
        clear_cuda_memory()

        from premade_hetero_models import HGTEncoder
        encoder = HGTEncoder(
            metadata=data.metadata(),
            x_dict=data.x_dict,
            hidden_size=128,
            num_layers=2,
            num_heads=4
        ).to(device)
        pretrain(encoder, data, epochs=100, lr=5e-4, patience=15, neg_sampling_ratio=1.0, margin=1.0, file_name='best_hgt_encoder.pt')

        # Clean up after training
        del encoder
        clear_cuda_memory()
        print("HGTEncoder training complete. Memory cleared.\n")
    except Exception as e:
        print(f"HGTEncoder failed to run: {e}. Skipping...")
        clear_cuda_memory()
    
    #R-GCN
    try:
        print("=== Training RGCNEncoder ===")
        clear_cuda_memory()

        from premade_hetero_models import RGCNEncoder
        encoder = RGCNEncoder(
            metadata=data.metadata(),
            x_dict=data.x_dict,
            hidden_size=128,
            num_layers=2,
            num_bases=10  # Adjust based on number of edge types
        ).to(device)
        pretrain(encoder, data, epochs=100, lr=5e-4, patience=15, neg_sampling_ratio=1.0, margin=1.0, file_name='best_rgcn_encoder.pt')

        # Clean up after training
        del encoder
        clear_cuda_memory()
        print("RGCNEncoder training complete. Memory cleared.\n")
    except Exception as e:
        print(f"RGCNEncoder failed to run: {e}. Skipping...")
        clear_cuda_memory()

    

    # HAN
    try:
        print("=== Training HANEncoder ===")
        clear_cuda_memory()

        from premade_hetero_models import HANEncoder
        encoder = HANEncoder(
            metadata=data.metadata(),
            x_dict=data.x_dict,
            hidden_size=128,
            num_layers=2,
            num_heads=4
        ).to(device)

        pretrain(encoder, data, epochs=100, lr=5e-4, patience=15, neg_sampling_ratio=1.0, margin=1.0, file_name='best_han_encoder.pt')

        # Clean up after training
        del encoder
        clear_cuda_memory()
        print("HANEncoder training complete. Memory cleared.\n")
    except Exception as e:
        print(f"HANEncoder failed to run: {e}. Skipping...")
        clear_cuda_memory()
    

    # HeteroSAGE
    try:
        print("=== Training HeteroSAGEEncoder ===")
        clear_cuda_memory()

        from premade_hetero_models import HeteroSAGEEncoder
        encoder = HeteroSAGEEncoder(
            metadata=data.metadata(),
            x_dict=data.x_dict,
            hidden_size=128,
            num_layers=2
        ).to(device)

        pretrain(encoder, data, epochs=100, lr=5e-4, patience=15, neg_sampling_ratio=1.0, margin=1.0, file_name='best_sage_encoder.pt')

        # Clean up after training
        del encoder
        clear_cuda_memory()
        print("HeteroSAGEEncoder training complete. Memory cleared.\n")
    except Exception as e:
        print(f"HeteroSAGEEncoder failed to run: {e}. Skipping...")
        clear_cuda_memory()