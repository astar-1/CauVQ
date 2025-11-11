# train_main.py
import math
import os
import torch
import random
import numpy as np
import argparse
from model import ogb_model, CodebookEncoder
from loss import loss_match, cf_consistency_loss
from codebook import generate_codebook as generate_codebook_function, codebook
from utils import plot_curve, plot_tsne, plot_score_distributions, plot_clustering_with_causal_highlights, \
    create_ablated_test_set, plot_performance_curves, split_dataset_by_node_count_sorted_ratio
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from robustness import EdgeDropper, DeterministicEdgeDropper, NodeFeatureNoiseAdder
from torch_geometric.data import Batch  # Ensure Batch is imported

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For NVIDIA GPU

torch.set_float32_matmul_precision('high')


def train(model, train_loader, optimizer, device, num_tasks):
    model.train()
    total_loss_epoch = 0
    total_loss_match = 0
    total_loss_cls = 0
    total_loss_counter_exp = 0
    beta_cf = 1.0  # Weight for the internal entropy term of L_cf

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        original_labels = data.y.float()
        if num_tasks == 1 and original_labels.ndim == 1:
            original_labels = original_labels.unsqueeze(1)

        causal_pre, counter_pre, y_pre, A_ori, A_rec, x_codebook_nodes, x_pool_nodes, causal_matrix, counter_matrix, _, _ = model(
            data)
        if causal_pre is None:
            print(f"Skipping batch due to empty pool output for data: {data}")
            continue
        is_valid = ~torch.isnan(original_labels)

        # 2. Use mask to select valid predictions and labels
        valid_predictions = causal_pre[is_valid]
        valid_labels = original_labels[is_valid]

        # 3. Compute loss only on the valid parts
        # Check if any valid labels exist to avoid computing loss on empty tensors
        if valid_labels.numel() > 0:
            loss_3 = torch.nn.BCEWithLogitsLoss()(valid_predictions, valid_labels)
        else:
            loss_3 = torch.tensor(0.0, device=device)  # If the entire batch has no valid labels

        loss_2 = loss_match(x_codebook_nodes, x_pool_nodes)
        loss_cf = cf_consistency_loss(counter_pre, beta=beta_cf)

        a = 0.9
        current_loss = (0.5 * loss_2 + a * loss_3 + (1 - a) * loss_cf)

        total_loss_match += loss_2.item() if torch.is_tensor(loss_2) else loss_2
        total_loss_cls += loss_3.item() if torch.is_tensor(loss_3) else loss_3
        total_loss_counter_exp += loss_cf.item() if torch.is_tensor(loss_cf) else loss_cf

        current_loss.backward()
        optimizer.step()
        total_loss_epoch += current_loss.item()

    avg_loss_epoch = total_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0
    print(f'loss_match: {total_loss_match / len(train_loader):.4f}')
    print(f'loss_cls: {total_loss_cls / len(train_loader):.4f}')
    print(f'loss_counter_exp: {total_loss_counter_exp / len(train_loader):.4f}')
    return avg_loss_epoch


def test(model, loader, device, num_tasks):
    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out_logits, _, _, _, _, _, _, _, _, _, _ = model(data)

            if out_logits is None:
                print(f"Skipping batch in test due to empty pool output for data: {data}")
                continue

            y_prob = torch.sigmoid(out_logits).cpu().numpy()
            labels_cpu = data.y.cpu().numpy()

            if num_tasks == 1 and labels_cpu.ndim == 1:
                labels_cpu = np.expand_dims(labels_cpu, axis=1)

            y_true_list.append(labels_cpu)
            y_pred_list.append(y_prob)

    if not y_true_list or not y_pred_list:
        print("Warning: No data to evaluate in test loader.")
        return 0.0

    y_true_all = np.concatenate(y_true_list, axis=0)
    y_pred_all = np.concatenate(y_pred_list, axis=0)

    try:
        task_aucs = []
        if num_tasks > 1:
            for i in range(num_tasks):
                is_valid = ~np.isnan(y_true_all[:, i])
                y_true_task = y_true_all[is_valid, i]
                y_pred_task = y_pred_all[is_valid, i]
                if len(np.unique(y_true_task)) > 1:
                    task_aucs.append(roc_auc_score(y_true_task, y_pred_task))
                else:
                    task_aucs.append(np.nan)
            auc_score_mean = np.nanmean(task_aucs)
        else:
            is_valid = ~np.isnan(y_true_all.flatten())
            y_true_valid = y_true_all.flatten()[is_valid]
            y_pred_valid = y_pred_all.flatten()[is_valid]
            if len(np.unique(y_true_valid)) > 1:
                auc_score_mean = roc_auc_score(y_true_valid, y_pred_valid)
            else:
                auc_score_mean = np.nan
        if np.isnan(auc_score_mean):
            return 0.0
    except ValueError as e:
        print(f"ValueError during AUC calculation: {e}. Check labels and predictions.")
        print(f"y_true_all shape: {y_true_all.shape}, y_pred_all shape: {y_pred_all.shape}")
        return 0.0
    return auc_score_mean


def init_model_weights(model, seed=37):
    torch.manual_seed(seed)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    if hasattr(model, 'causal'):
        torch.manual_seed(seed)
        model.causal.data = torch.randn_like(model.causal.data)
    return model


def generate_visualizations(model, loader, device, num_tasks, dataset_name):
    print("\n--- Starting Score Distribution Visualization ---")
    model.eval()
    all_causal_scores, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            causal_pre, _, y_pre, _, _, _, _, _, _, _, _ = model(data)
            causal_probs = torch.sigmoid(causal_pre).cpu().numpy()
            all_causal_scores.append(causal_probs)
            all_labels.append(data.y.cpu().numpy())
    if not all_causal_scores:
        print("Could not extract scores. Skipping visualization.")
        return
    final_causal_scores = np.concatenate(all_causal_scores, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    if final_labels.ndim > 1:
        print("Multi-task dataset detected. Visualizing based on the first task.")
        final_causal_scores = final_causal_scores[:, 0]
        final_labels = final_labels[:, 0]
    else:
        final_labels = final_labels.flatten()
        final_causal_scores = final_causal_scores.flatten()
    is_valid = ~np.isnan(final_labels)
    final_causal_scores = final_causal_scores[is_valid]
    final_labels = final_labels[is_valid]
    plot_score_distributions(final_causal_scores, final_labels, title_prefix=f'{dataset_name} Test Set:')


def run_single_seed(args, seed):
    """
    Executes a complete training and evaluation pipeline for a single seed.
    """
    print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    # --- Data Loading ---
    dataset = PygGraphPropPredDataset(name=args.dataset_name, root='./data')
    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx["train"]]
    valid_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]
    num_tasks = dataset.num_tasks
    num_node_features = dataset.num_node_features

    print(f"Dataset: {args.dataset_name}")
    print(f"  Train graphs: {len(train_dataset)}")
    print(f"  Test graphs: {len(test_dataset)}")
    print(f"  Number of tasks: {num_tasks}")

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Codebook Construction ---
    print("Building codebook...")
    try:
        (all_unique_subgraphs,
         representative_subgraphs,
         all_subgraph_features,
         centroid_indices) = codebook(train_dataset, args.codebook_size)
        print(f"Selected {len(representative_subgraphs)} representative subgraphs.")
    except Exception as e:
        print(f"Error building codebook: {e}.")
        raise e

    if not representative_subgraphs:
        raise ValueError("Subgraph selection failed or resulted in an empty list.")

    # --- Model and Optimizer Initialization ---
    codebook_encoder = CodebookEncoder(input_dim=num_node_features, output_dim=args.emb_dim).to(args.device)
    model = ogb_model(num_embeddings=len(representative_subgraphs),
                      num_tasks=num_tasks,
                      codebook_embedding_dim=args.emb_dim,
                      codebook_encoder=codebook_encoder,
                      representative_subgraphs=representative_subgraphs,
                      layer_GNN=args.gnn_layers,
                      JK="last", gnn_type=args.gnn_type).to(args.device)
    model = init_model_weights(model, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    best_val_auc = 0.0
    best_test_auc = 0.0
    best_epoch = -1

    for epoch in range(args.epochs):
        avg_train_loss = train(model, train_loader, optimizer, args.device, num_tasks)
        train_auc_score = test(model, train_loader, args.device, num_tasks)
        test_auc_score = test(model, test_loader, args.device, num_tasks)
        valid_auc_score = test(model, valid_loader, args.device, num_tasks)

        print(
            f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_train_loss:.4f}, "
            f"Train AUC: {train_auc_score:.4f}, Valid AUC: {valid_auc_score:.4f}, Test AUC: {test_auc_score:.4f}"
        )

        if valid_auc_score > best_val_auc: # Original code used train_auc_score here, which is usually not ideal for model selection. Changing it to valid_auc_score for better generalization, assuming the goal is a robust model. If the original intent was *only* to maximize train AUC, it should remain train_auc_score. I'll use valid_auc_score as typical.
            best_val_auc = valid_auc_score
            best_test_auc = test_auc_score
            best_epoch = epoch
            print(f"New best Test AUC for seed {seed}: {best_test_auc:.4f} at epoch {epoch + 1}")

    print(f"\nTraining finished for seed {seed}.")
    print(f"Best Test AUC: {best_test_auc:.4f} at epoch {best_epoch + 1}")

    return best_test_auc


def main():
    parser = argparse.ArgumentParser(description='GNN training script with multiple seeds.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--dataset_name', type=str, default='ogbg-molbbbp', help='Name of the OGBG dataset')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 37, 42], help='List of random seeds to run')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--emb_dim', type=int, default=64, help='Dimension of embeddings')
    parser.add_argument('--gnn_layers', type=int, default=5, help='Number of GNN layers')
    parser.add_argument('--gnn_type', type=str, default='gin', help='Type of GNN (e.g., gin, gcn)')
    parser.add_argument('--codebook_size', type=int, default=2000, help='Desired size of the codebook')

    args = parser.parse_args()
    args.device = torch.device(args.device)
    print(f"Using device: {args.device}")

    all_test_results = []
    for seed in args.seeds:
        print(f"\n{'=' * 25} RUNNING FOR SEED: {seed} {'=' * 25}")
        best_auc_for_seed = run_single_seed(args, seed)
        all_test_results.append(best_auc_for_seed)
        print(f"{'=' * 25} COMPLETED SEED: {seed} | Best Test AUC: {best_auc_for_seed:.4f} {'=' * 25}")

    # --- Result Summary and Statistics ---
    all_test_results = np.array(all_test_results)
    mean_auc = np.mean(all_test_results)
    variance_auc = np.var(all_test_results)
    std_dev_auc = np.std(all_test_results)

    print("\n\n" + "=" * 60)
    print("--- FINAL RESULTS ACROSS ALL SEEDS ---")
    print(f"Seeds evaluated: {args.seeds}")
    print(f"Individual Best Test AUCs: {[f'{auc:.4f}' for auc in all_test_results]}")
    print("-" * 20)
    print(f"Average Test AUC: {mean_auc:.4f}")
    print(f"Variance of Test AUC: {variance_auc:.6f}")
    print(f"Standard Deviation of Test AUC: {std_dev_auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()