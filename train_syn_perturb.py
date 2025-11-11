# train_syn_perturb.py
import math
import os
import argparse
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from matplotlib import pyplot as plt

from model import SynModel, CodebookEncoder
from loss import loss_rec, loss_match, cf_consistency_loss
from codebook import generate_codebook
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_adj

from robustness import DeterministicEdgeDropper
from spmotif_dataset import SpMotif
import os.path as osp

from utils import plot_multiclass_score_distributions


def perturb_training_set(dataset, perturbation_ratio=0.5, edge_drop_p=0.2):
    """
    Perturbs a portion of the training set by randomly dropping edges.
    """
    num_graphs = len(dataset)
    num_to_perturb = int(num_graphs * perturbation_ratio)
    all_indices = list(range(num_graphs))
    random.shuffle(all_indices)
    perturb_indices = set(all_indices[:num_to_perturb])
    new_dataset = []
    print(f"Starting to perturb training set: {num_to_perturb} out of {num_graphs} graphs will have edges dropped...")
    for i, data in enumerate(dataset):
        if i in perturb_indices:
            new_data = data.clone()
            edge_attr = new_data.edge_attr if hasattr(new_data,
                                                      'edge_attr') and new_data.edge_attr is not None else None
            # Dropout edges
            edge_index_perturbed, edge_attr_perturbed = dropout_adj(
                edge_index=new_data.edge_index, edge_attr=edge_attr, p=edge_drop_p, force_undirected=True, training=True
            )
            new_data.edge_index = edge_index_perturbed
            new_data.edge_attr = edge_attr_perturbed
            new_dataset.append(new_data)
        else:
            new_dataset.append(data)
    print("Training set perturbation completed.")
    return new_dataset


def train(train_loader, model, optimizer, loss_fn_cls):
    model.train()
    total_loss, total_loss_rec, total_loss_match, total_loss_cls, total_loss_counter = 0, 0, 0, 0, 0
    beta_cf = 1.0
    for data in train_loader:
        data = data.to('cuda')
        optimizer.zero_grad()
        # Renamed syn_model to SynModel
        (causal_pre, counter_pre, y_pre, A_ori, A_rec, x_codebook_nodes, x_pool_nodes, causal_matrix, counter_matrix, _,
         _) = model(data)
        labels = data.y.long()
        loss_1 = loss_rec(A_ori, A_rec) if A_ori and A_rec else torch.tensor(0.0, device=data.device)
        loss_2 = loss_match(x_codebook_nodes, x_pool_nodes)
        loss_3 = loss_fn_cls(causal_pre, labels)
        loss_cf = cf_consistency_loss(counter_pre, beta=beta_cf)
        a = 0.9
        # Note: Original code uses 0 * loss_1, so reconstruction loss is disabled.
        loss = 0 * loss_1 + 0.1 * loss_2 + a * loss_3 + (1 - a) * loss_cf
        total_loss_rec += loss_1.item() if torch.is_tensor(loss_1) else loss_1
        total_loss_match += loss_2.item() if torch.is_tensor(loss_2) else loss_2
        total_loss_cls += loss_3.item() if torch.is_tensor(loss_3) else loss_3
        total_loss_counter += loss_cf.item() if torch.is_tensor(loss_cf) else loss_cf
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'loss_rec: {total_loss_rec / len(train_loader):.4f}')
    print(f'loss_match: {total_loss_match / len(train_loader):.4f}')
    print(f'loss_cls: {total_loss_cls / len(train_loader):.4f}')
    print(f'loss_counter_exp: {total_loss_counter / len(train_loader):.4f}')
    return total_loss / len(train_loader), causal_matrix


def test(loader, model, device):
    model.eval()
    correct, total_graphs = 0, 0
    for data in loader:
        data = data.to(device)
        causal_pre, _, _, _, _, _, _, _, _, _, _ = model(data)
        correct += (causal_pre.argmax(dim=1) == data.y).sum().item()
        total_graphs += data.num_graphs
    return correct / total_graphs if total_graphs > 0 else 0.0


def generate_syn_visualizations(model, loader, device, num_classes, dataset_name):
    print("\n--- Starting Score Distribution Visualization for SPMotif ---")
    model.eval()
    all_causal_scores, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            causal_pre, _, y_pre, _, _, _, _, _, _, _, _ = model(data)
            causal_probs = F.softmax(causal_pre, dim=1).cpu().numpy()
            all_causal_scores.append(causal_probs)
            all_labels.append(data.y.cpu().numpy())
    final_causal_scores = np.concatenate(all_causal_scores, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    plot_multiclass_score_distributions(
        final_causal_scores, final_labels, num_classes, title_prefix=f'{dataset_name} Test Set:'
    )


def main(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    datadir = './data'
    try:
        # Renamed SPMotif to SpMotif
        original_train_dataset = SpMotif(osp.join(datadir, f'SPMotif-{args.bias}/'), mode='train')
        val_dataset = SpMotif(osp.join(datadir, f'SPMotif-{args.bias}/'), mode='val')
        test_dataset = SpMotif(osp.join(datadir, f'SPMotif-{args.bias}/'), mode='test')
    except FileNotFoundError:
        print(f"Dataset not found at {datadir}/SPMotif-{args.bias}/. Ensure the path is correct.")
        exit()

    perturbed_train_dataset = perturb_training_set(
        original_train_dataset, perturbation_ratio=args.perturbation_ratio, edge_drop_p=args.edge_drop_p
    )

    train_loader = DataLoader(perturbed_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train dataset size (after perturbation): {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    try:
        # Renamed codebook to generate_codebook
        # Note: Codebook is built from the *original* training set for fair comparison
        (_, representative_subgraphs, _, _) = generate_codebook(original_train_dataset, args.codebook_size)
        print(f"Selected {len(representative_subgraphs)} representative subgraphs.")
    except Exception as e:
        print(f"Error building codebook: {e}.")
        raise e

    if not representative_subgraphs:
        raise ValueError("Subgraph selection failed or resulted in an empty list.")

    codebook_encoder = CodebookEncoder(input_dim=4, output_dim=args.embedding_dim).to(device)
    # Renamed syn_model to SynModel
    model = SynModel(
        num_embeddings=len(representative_subgraphs), num_tasks=args.num_classes,
        codebook_embedding_dim=args.embedding_dim, codebook_encoder=codebook_encoder,
        representative_subgraphs=representative_subgraphs, layer_GNN=args.layer_GNN,
        JK="last", gnn_type=args.gnn_type
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    classification_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_test_acc_at_best_val = 0
    best_epoch = -1
    last_causal_matrix = None

    for epoch in range(args.epochs):
        loss, causal_matrix_from_epoch = train(train_loader, model, optimizer, classification_loss_fn)
        train_acc_score = test(train_loader, model, device)
        val_acc_score = test(val_loader, model, device)
        test_acc_score = test(test_loader, model, device)
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs}, Loss: {loss:.4f}, "
            f"Train ACC: {train_acc_score:.4f}, Val ACC: {val_acc_score:.4f}, Test ACC: {test_acc_score:.4f}"
        )
        if val_acc_score >= best_val_acc:
            best_val_acc = val_acc_score
            best_test_acc_at_best_val = test_acc_score
            best_epoch = epoch
            last_causal_matrix = causal_matrix_from_epoch
            torch.save(model.state_dict(), f'best_model_syn_{args.bias}_perturbed.pth')
            print(f"New best Test ACC: {best_test_acc_at_best_val:.4f} at epoch {epoch + 1}")

    print("\n--- Training Finished on Perturbed Data ---")
    print(f'Best Epoch: {best_epoch + 1:03d}')
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    print(f'Test Accuracy at Best Validation: {best_test_acc_at_best_val:.4f}')

    best_model_path = f'best_model_syn_{args.bias}_perturbed.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"\nSuccessfully loaded best perturbed model from {best_model_path}")
        generate_syn_visualizations(model, test_loader, device, args.num_classes, f'SPMotif-{args.bias}')
    else:
        print("Could not find saved perturbed model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GNN on perturbed SPMotif dataset.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for training')
    parser.add_argument('--seed', type=int, default=37, help='Random seed')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--bias', type=float, default=0.7, help='Bias for SPMotif dataset')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--layer_GNN', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--gnn_type', type=str, default='gcn', help='GNN type (e.g., gcn, gin)')
    parser.add_argument('--codebook_size', type=int, default=1000, help='Desired codebook size')
    parser.add_argument('--perturbation_ratio', type=float, default=0.5, help='Ratio of training graphs to perturb')
    parser.add_argument('--edge_drop_p', type=float, default=0.1,
                        help='Probability of dropping an edge in perturbed graphs')

    args = parser.parse_args()
    main(args)