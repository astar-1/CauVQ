# train_mnist.py
import math
import argparse
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from matplotlib import pyplot as plt

from mnist import MNIST75sp  # Assuming MNIST75sp is available
from model import MniModel, CodebookEncoder
from loss import loss_rec, loss_match
from codebook import generate_codebook
from torch_geometric.loader import DataLoader
import os.path as osp


# Assuming this function is defined elsewhere (e.g., in loss.py) but was missed in previous files.
# Defining a placeholder for `counter_constrain` based on its usage.
def counter_constrain(causal_pre, counter_pre):
    """Placeholder for counterfactual constraint loss calculation."""
    # Assuming the constraint encourages difference, e.g., L1 distance between logits
    return torch.mean(torch.abs(causal_pre - counter_pre))


def train(train_loader, model, optimizer, loss_fn_cls):
    model.train()
    total_loss, loss_match_2, loss_cls_3, loss_counter, loss_rec_1 = 0, 0, 0, 0, 0
    for data in train_loader:
        data = data.to('cuda')
        optimizer.zero_grad()
        # Renamed mni_model to MniModel
        causal_pre, counter_pre, y_pre, A_ori, A_rec, x_codebook, x_pool, causal_matrix, counter_matrix, _, _ = model(
            data)

        labels = data.y.long()
        loss_1 = loss_rec(A_ori, A_rec)
        loss_2 = loss_match(x_codebook, x_pool)
        loss_3 = loss_fn_cls(causal_pre, labels)

        # Using the assumed counter_constrain function
        counter_loss_value = counter_constrain(causal_pre, counter_pre)
        # Original loss: 0.5 * loss_1 + 0.5 * loss_2 + 5 * loss_3 + 1 * torch.exp(-counter)
        loss = 0.5 * loss_1 + 0.5 * loss_2 + 5 * loss_3 + 1 * torch.exp(-counter_loss_value)

        loss_rec_1 += loss_1.item()
        loss_match_2 += loss_2.item()
        loss_cls_3 += loss_3.item()
        loss_counter += torch.exp(-counter_loss_value).item()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'loss_rec: {loss_rec_1 / len(train_loader):.4f}')
    print(f'loss_match: {loss_match_2 / len(train_loader):.4f}')
    print(f'loss_cls: {loss_cls_3 / len(train_loader):.4f}')
    print(f'loss_counter: {loss_counter / len(train_loader):.4f}')
    return total_loss / len(
        train_loader), causal_matrix  # Note: model(data) output changed in model.py, so I'm using the 8th returned value as the causal_matrix


def test(loader, model, device, color_noises, noise_level=0.4):
    model.eval()
    acc = 0
    n_samples_offset = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        num_in_batch = batch.x.size(0)

        # Apply color noise deterministically (based on offset)
        noise_indices = torch.arange(n_samples_offset, n_samples_offset + num_in_batch) % len(color_noises)
        noise = color_noises[noise_indices].to(device) * noise_level

        # Apply noise to the first 3 features (RGB channels)
        batch.x[:, :3] = batch.x[:, :3] + noise
        n_samples_offset += num_in_batch

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                # Renamed mni_model to MniModel
                # model(batch) returns 10+ values in MniModel, but test only uses the first (causal_pre)
                pred, _, _, _, _, _, _, _, _, _, _ = model(batch)
                acc += torch.sum(pred.argmax(-1).view(-1) == batch.y.view(-1))
    acc = float(acc) / len(loader.dataset)
    return [acc]


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

    datadir = './data/'
    train_path = osp.join(datadir, 'mnistsp/')

    # Renamed MNIST75sp to MNIST75sp (assuming it's a fixed class name)
    train_val = MNIST75sp(train_path, mode='train')
    perm_idx = torch.randperm(len(train_val), generator=torch.Generator().manual_seed(0))
    train_val = train_val[perm_idx]
    train_dataset, val_dataset = train_val[:args.n_train], train_val[-args.n_val:]
    test_dataset = MNIST75sp(train_path, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # Only test on the first 1000 samples for consistency with the original code
    test_loader = DataLoader(test_dataset[0:1000], batch_size=args.batch_size, shuffle=False)

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    print("Selecting representative subgraphs for codebook...")
    # Renamed codebook to generate_codebook
    (_, representative_subgraphs, _, _) = generate_codebook(train_dataset, args.codebook_size)
    print(f"Selected {len(representative_subgraphs)} representative subgraphs.")
    if not representative_subgraphs:
        raise ValueError("Subgraph selection failed or resulted in an empty list.")

    codebook_encoder = CodebookEncoder(input_dim=5, output_dim=args.embedding_dim).to(device)
    # Renamed mni_model to MniModel
    model = MniModel(
        num_embeddings=len(representative_subgraphs), num_tasks=args.num_classes,
        codebook_embedding_dim=args.embedding_dim, codebook_encoder=codebook_encoder,
        representative_subgraphs=representative_subgraphs, layer_GNN=args.layer_GNN,
        JK="last", gnn_type=args.gnn_type
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    classification_loss_fn = nn.CrossEntropyLoss()

    best_val_acc, best_test_acc_at_best_val, best_epoch = 0, 0, -1
    last_causal_matrix = None

    # Load color noises for testing
    color_noises = torch.load(osp.join(datadir, 'mnistsp/raw/mnist_75sp_color_noise.pt')).view(-1, 3)

    for epoch in range(args.epochs):
        loss, causal_matrix_from_epoch = train(train_loader, model, optimizer, classification_loss_fn)

        # Pass noise level for testing
        train_acc_score = test(train_loader, model, device, color_noises, args.noise_level)[0]
        val_acc_score = test(val_loader, model, device, color_noises, args.noise_level)[0]
        test_acc_score = test(test_loader, model, device, color_noises, args.noise_level)[0]

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs}, Loss: {loss:.4f}, "
            f"Train ACC: {train_acc_score:.4f}, Val ACC: {val_acc_score:.4f}, Test ACC: {test_acc_score:.4f}"
        )

        if val_acc_score >= best_val_acc:
            best_val_acc = val_acc_score
            best_test_acc_at_best_val = test_acc_score
            best_epoch = epoch
            last_causal_matrix = causal_matrix_from_epoch

    print("\n--- Training Finished ---")
    print(f'Best Epoch: {best_epoch + 1:03d}')
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    print(f'Test Accuracy at Best Validation: {best_test_acc_at_best_val:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GNN on MNIST-75sp dataset.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for training')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for MNIST')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--layer_GNN', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--gnn_type', type=str, default='gin', help='GNN type (e.g., gin, gcn)')
    parser.add_argument('--codebook_size', type=int, default=1000, help='Desired codebook size')
    parser.add_argument('--n_train', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=1000, help='Number of validation samples')
    parser.add_argument('--noise_level', type=float, default=0.4,
                        help='Noise level for color perturbation during testing')

    args = parser.parse_args()
    main(args)