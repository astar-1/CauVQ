import math
import networkx as nx
import torch
from matplotlib import pyplot as plt
from model import MoModel, CodebookEncoder
from loss import loss_rec, loss_match, loss_cls_2, cf_consistency_loss
from codebook import generate_codebook
from utils import adj, pyg_to_nx
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet, TUDataset
from sklearn.model_selection import KFold  # Import KFold
from torch.utils.data import Subset
import random
import numpy as np
from visual_MUTAG import visualize_molecule


def train(model, train_loader, optimizer, device):  # Pass model and optimizer as arguments
    model.train()
    total_loss = 0
    loss_match_2 = 0
    loss_cls_3 = 0
    loss_counter = 0

    beta_cf = 1.0  # Weight for the internal entropy term of L_cf

    num_batches = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        causal_pre, counter_pre, y_pre, A_ori, A_rec, x_codebook, x_pool, causal_matrix, counter_matrix_batch, _, _ = model(data)

        # Skip batches with no valid predictions due to empty pool output
        if causal_pre is None:
            continue
        labels = data.y.long()
        loss_2 = loss_match(x_codebook, x_pool)
        loss_3 = loss_cls_2(causal_pre, labels)
        # [Modification] Use counterfactual loss consistent with train_main.py
        loss_cf = cf_consistency_loss(counter_pre, beta=beta_cf)
        a = 0.7  # Use weight consistent with train_main.py (Note: train_main.py uses 0.9. I'll keep 0.7 from original train.py if it's intentional, but I'll use 0.9 here for consistency with the OGB script unless explicitly told otherwise. Assuming 0.7 is for this specific dataset.)
        loss = a * loss_3 + (1 - a) * loss_cf

        loss_match_2 += loss_2.item()
        loss_cls_3 += loss_3.item()
        loss_counter += loss_cf.item()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        print("Warning: No batches were processed in train_loader.")
        # If no batches were processed, return 0 and None
        return 0, None

    print(f'loss_match: {loss_match_2 / num_batches:.4f}')
    print(f'loss_cls: {loss_cls_3 / num_batches:.4f}')
    print(f'loss_counter: {loss_counter / num_batches:.4f}')
    return total_loss / num_batches, causal_matrix


def test(model, loader, device):  # Pass model as argument
    model.eval()
    correct = 0
    total_samples = 0
    if not loader or len(loader.dataset) == 0:
        print("Warning: test_loader is empty or has an empty dataset.")
        return 0.0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            causal_pre, _, _, _, _, _, _, _, _, _, _ = model(data)

            if causal_pre is None:
                continue

            correct += (causal_pre.argmax(dim=1) == data.y).sum().item()
            total_samples += data.y.size(0)

    if total_samples == 0:
        print("Warning: No data processed in test_loader, cannot calculate accuracy.")
        return 0.0
    return correct / total_samples


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 64
    dataset_name = "MUTAG"
    seed = 37
    epochs = 150
    n_splits = 10
    learning_rate = 0.001
    hidden_dim_model = 128
    embedding_dim_codebook = 64
    num_classes = 2
    layer_GNN = 3
    gnn_type = 'gin'
    desired_codebook_size = 300

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = TUDataset(root='./data', name=dataset_name, use_node_attr=True, use_edge_attr=True)
    dataset = dataset.shuffle()

    print(f"Dataset: {dataset_name}, Number of graphs: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")
    if not hasattr(dataset[0], 'x') or dataset[0].x is None:
        raise ValueError("Dataset features (x) are missing.")
    if not hasattr(dataset[0], 'y') or dataset[0].y is None:
        raise ValueError("Dataset labels (y) are missing.")

    node_feature_dim = dataset.num_node_features
    edge_feature_dim = dataset.num_edge_features
    print(f"Node feature dimensions: {node_feature_dim}")
    print(f"Edge feature dimensions: {edge_feature_dim}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_test_accuracies = []
    fold_train_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        split = int(0.9 * len(train_idx))
        val_indices = train_idx[split:]
        train_indices = train_idx[:split]

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        test_subset = Subset(dataset, test_idx)

        if len(train_subset) == 0 or len(val_subset) == 0 or len(test_subset) == 0:
            print(f"Warning: Fold {fold + 1} has an empty subset. Skipping.")
            continue

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        print(f"Train set size: {len(train_loader.dataset)}")
        print(f"Validation set size: {len(val_loader.dataset)}")
        print(f"Test set size: {len(test_loader.dataset)}")

        try:

            (all_unique_subgraphs,
             representative_subgraphs,
             all_subgraph_features,
             centroid_indices) = generate_codebook(train_subset, desired_codebook_size)

            print(f"Selected {len(representative_subgraphs)} representative subgraphs.")
        except Exception as e:
            print(f"Error building codebook: {e}.")
            raise e

        if not representative_subgraphs:
            raise ValueError("Subgraph selection failed or resulted in an empty list.")
        codebook_encoder = CodebookEncoder(input_dim=node_feature_dim, output_dim=embedding_dim_codebook).to(device)

        # Modification: Use new model initialization
        model = MoModel(num_embeddings=len(representative_subgraphs),
                         num_tasks=num_classes,
                         codebook_embedding_dim=embedding_dim_codebook,
                         codebook_encoder=codebook_encoder,
                         representative_subgraphs=representative_subgraphs,
                         layer_GNN=layer_GNN,
                         node_feature_dim=node_feature_dim,  # Pass node feature dimension
                         edge_feature_dim=edge_feature_dim,
                         JK="last", gnn_type=gnn_type).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_val_acc = 0.0
        best_test_acc_at_best_val = 0.0
        best_matrix = None # Initialized to None
        for epoch in range(epochs):
            loss, causal_matrix = train(model, train_loader, optimizer, device)
            train_acc_score = test(model, train_loader, device)
            val_acc_score = test(model, val_loader, device)
            test_acc_score = test(model, test_loader, device)

            print(
                f"Epoch {epoch + 1:03d}/{epochs}, Loss: {loss:.4f}, Train ACC: {train_acc_score:.4f}, Val ACC: {val_acc_score:.4f}, Test ACC: {test_acc_score:.4f}")

            if val_acc_score >= best_val_acc:
                best_val_acc = val_acc_score
                best_test_acc_at_best_val = test_acc_score
                # Only update best_matrix if a valid matrix was returned by train
                if causal_matrix is not None:
                    best_matrix = causal_matrix

        # --- Visualization Section Start ---
        if best_matrix is not None:
            print(f"\n--- Visualizing Top 5 Causal Substructures for Fold {fold + 1} ---")
            # Calculate importance and get indices of the top 5
            importance = torch.diag(best_matrix)
            # Ensure we don't request more than available subgraphs
            k = min(10, len(representative_subgraphs))
            top_k_indices = torch.topk(importance, k).indices

            # Iterate through the most important subgraphs and visualize
            for i, idx in enumerate(top_k_indices):
                # Get the PyG Data object from the list
                subgraph_to_plot = representative_subgraphs[idx]

                print(
                    f"Displaying Causal Substructure Rank {i + 1} (Codebook Index: {idx.item()}, Importance: {importance[idx].item():.4f})")

                # Pass the PyG Data object directly to the visualization function
                visualize_molecule(subgraph_to_plot)
        else:
            print("\nSkipping visualization for this fold as no valid causal matrix was found.")
        # --- Visualization Section End ---


        print(f"Fold {fold + 1} - Best Test Accuracy: {best_test_acc_at_best_val:.4f}")
        fold_test_accuracies.append(best_test_acc_at_best_val)

        final_train_acc = test(model, train_loader, device)
        fold_train_accuracies.append(final_train_acc)
        print(f"Fold {fold + 1} - Final Train Accuracy: {final_train_acc:.4f}")

    if fold_test_accuracies:
        mean_test_accuracy = np.mean(fold_test_accuracies)
        std_test_accuracy = np.std(fold_test_accuracies)
        print(f"\n--- Cross-Validation Results ---")
        print(f"Average Test Accuracy: {mean_test_accuracy:.4f} +/- {std_test_accuracy:.4f}")
    else:
        print("\nNo valid folds were processed.")

    if fold_train_accuracies:
        mean_train_accuracy = np.mean(fold_train_accuracies)
        std_train_accuracy = np.std(fold_train_accuracies)
        print(f"Average Train Accuracy: {mean_train_accuracy:.4f} +/- {std_train_accuracy:.4f}")