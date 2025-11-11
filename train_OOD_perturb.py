# train_OOD_perturb.py
import math
import os
import torch
import random
import numpy as np
import argparse
import logging
from model import OgbModel, CodebookEncoder
from loss import loss_rec, loss_match, loss_cls, loss_dag, cf_consistency_loss
from codebook import generate_codebook
from utils import plot_curve, plot_tsne, plot_score_distributions
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from torch_geometric.utils import dropout_adj

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.set_float32_matmul_precision('high')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def reduce_training_set(dataset, keep_ratio=0.5):
    """
    Randomly reduces the size of the training dataset.
    """
    num_graphs = len(dataset)
    num_to_keep = int(num_graphs * keep_ratio)
    all_indices = list(range(num_graphs))
    random.shuffle(all_indices)
    indices_to_keep = all_indices[:num_to_keep]
    logger.info(f"Reducing training set: randomly keeping {num_to_keep} out of {num_graphs} graphs (ratio: {keep_ratio}).")
    reduced_dataset = dataset[indices_to_keep]
    return reduced_dataset


def train(model, train_loader, optimizer, device, num_tasks):
    model.train()
    total_loss_epoch, total_loss_rec, total_loss_match, total_loss_cls, total_loss_counter_exp = 0, 0, 0, 0, 0
    beta_cf = 1.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        original_labels = data.y.float()
        if num_tasks == 1 and original_labels.ndim == 1:
            original_labels = original_labels.unsqueeze(1)
        # Renamed ogb_model to OgbModel
        causal_pre, counter_pre, y_pre, A_ori, A_rec, x_codebook_nodes, x_pool_nodes, causal_matrix, counter_matrix, _, _ = model(data)
        if causal_pre is None:
            continue
        is_valid = ~torch.isnan(original_labels)
        valid_predictions = causal_pre[is_valid]
        valid_labels = original_labels[is_valid]
        if valid_labels.numel() > 0:
            loss_3 = torch.nn.BCEWithLogitsLoss()(valid_predictions, valid_labels)
        else:
            loss_3 = torch.tensor(0.0, device=device)
        loss_1 = loss_rec(A_ori, A_rec)
        loss_2 = loss_match(x_codebook_nodes, x_pool_nodes)
        loss_cf = cf_consistency_loss(counter_pre, beta=beta_cf)
        a = 1
        # Note: Original code uses 0 * loss_1 and 0 * loss_2, disabling reconstruction and matching loss.
        current_loss = (0 * loss_1 + 0 * loss_2 + a * loss_3 + (1 - a) * loss_cf)
        total_loss_rec += loss_1.item() if torch.is_tensor(loss_1) else loss_1
        total_loss_match += loss_2.item() if torch.is_tensor(loss_2) else loss_2
        total_loss_cls += loss_3.item() if torch.is_tensor(loss_3) else loss_3
        total_loss_counter_exp += loss_cf.item() if torch.is_tensor(loss_cf) else loss_cf
        current_loss.backward()
        optimizer.step()
        total_loss_epoch += current_loss.item()
    avg_loss_epoch = total_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0
    print(f'loss_rec: {total_loss_rec / len(train_loader):.4f}')
    print(f'loss_match: {total_loss_match / len(train_loader):.4f}')
    print(f'loss_cls: {total_loss_cls / len(train_loader):.4f}')
    print(f'loss_counter_exp: {total_loss_counter_exp / len(train_loader):.4f}')
    return avg_loss_epoch


def test(model, loader, device, num_tasks):
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out_logits, _, _, _, _, _, _, _, _, _, _ = model(data)
            if out_logits is None: continue
            y_prob = torch.sigmoid(out_logits).cpu().numpy()
            labels_cpu = data.y.cpu().numpy()
            if num_tasks == 1 and labels_cpu.ndim == 1:
                labels_cpu = np.expand_dims(labels_cpu, axis=1)
            y_true_list.append(labels_cpu)
            y_pred_list.append(y_prob)
    if not y_true_list: return 0.0
    y_true_all = np.concatenate(y_true_list, axis=0)
    y_pred_all = np.concatenate(y_pred_list, axis=0)
    try:
        task_aucs = []
        if num_tasks > 1:
            for i in range(num_tasks):
                is_valid = ~np.isnan(y_true_all[:, i])
                y_true_task, y_pred_task = y_true_all[is_valid, i], y_pred_all[is_valid, i]
                if len(np.unique(y_true_task)) > 1:
                    task_aucs.append(roc_auc_score(y_true_task, y_pred_task))
                else:
                    task_aucs.append(np.nan)
            auc_score_mean = np.nanmean(task_aucs)
        else:
            is_valid = ~np.isnan(y_true_all.flatten())
            y_true_valid, y_pred_valid = y_true_all.flatten()[is_valid], y_pred_all.flatten()[is_valid]
            if len(np.unique(y_true_valid)) > 1:
                auc_score_mean = roc_auc_score(y_true_valid, y_pred_valid)
            else:
                auc_score_mean = np.nan
        return 0.0 if np.isnan(auc_score_mean) else auc_score_mean
    except ValueError as e:
        print(f"ValueError during AUC calculation: {e}")
        return 0.0


def init_model_weights(model, seed=37):
    torch.manual_seed(seed)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    return model


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    dataset = PygGraphPropPredDataset(name=args.dataset_name, root='./data')
    split_idx = dataset.get_idx_split()
    original_train_dataset = dataset[split_idx["train"]]

    logger.info("Reducing OGBG training set for robustness experiment...")
    perturbed_train_dataset = reduce_training_set(original_train_dataset, keep_ratio=args.keep_ratio)

    valid_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]
    num_tasks = dataset.num_tasks
    num_node_features = dataset.num_node_features

    print(f"Dataset: {args.dataset_name}")
    print(f"  Train graphs (after reduction): {len(perturbed_train_dataset)}")
    print(f"  Test graphs: {len(test_dataset)}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(perturbed_train_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    try:
        # Renamed codebook to generate_codebook
        (_, representative_subgraphs, _, _) = generate_codebook(perturbed_train_dataset, args.codebook_size)
        print(f"Selected {len(representative_subgraphs)} representative subgraphs.")
    except Exception as e:
        print(f"Error building codebook: {e}.")
        raise e

    if not representative_subgraphs:
        raise ValueError("Subgraph selection failed or resulted in an empty list.")

    codebook_encoder = CodebookEncoder(input_dim=num_node_features, output_dim=args.codebook_output_dim).to(device)
    # Renamed ogb_model to OgbModel
    model = OgbModel(
        num_embeddings=len(representative_subgraphs), num_tasks=num_tasks,
        codebook_embedding_dim=args.codebook_output_dim, codebook_encoder=codebook_encoder,
        representative_subgraphs=representative_subgraphs, layer_GNN=args.gnn_layers,
        JK="last", gnn_type=args.gnn_type
    ).to(device)
    model = init_model_weights(model, args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_train_auc, best_test_auc, best_epoch = 0, 0, -1
    for epoch in range(args.epochs):
        avg_train_loss = train(model, train_loader, optimizer, device, num_tasks)
        train_auc_score = test(model, train_loader, device, num_tasks)
        test_auc_score = test(model, test_loader, device, num_tasks)
        valid_auc_score = test(model, valid_loader, device, num_tasks)
        print(
            f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_train_loss:.4f}, "
            f"Train AUC: {train_auc_score:.4f}, Valid AUC: {valid_auc_score:.4f}, Test AUC: {test_auc_score:.4f}"
        )
        if train_auc_score > best_train_auc:
            best_train_auc = train_auc_score
            best_test_auc = test_auc_score
            best_epoch = epoch
            torch.save(model.state_dict(), f'best_model_{args.dataset_name}_perturbed.pth')
            print(f"New best Test AUC: {best_test_auc:.4f} at epoch {epoch + 1}")

    print(f"\nTraining finished for {args.dataset_name} with perturbed training data.")
    print(f"Best Test AUC: {best_test_auc:.4f} at epoch {best_epoch + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GNN on a reduced OGBG dataset.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for training')
    parser.add_argument('--seed', type=int, default=37, help='Random seed')
    parser.add_argument('--dataset_name', type=str, default='ogbg-molbbbp', help='Name of the OGBG dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--codebook_output_dim', type=int, default=64, help='Output dimension for codebook encoder')
    parser.add_argument('--gnn_layers', type=int, default=5, help='Number of GNN layers')
    parser.add_argument('--gnn_type', type=str, default='gcn', help='GNN type (e.g., gcn, gin)')
    parser.add_argument('--codebook_size', type=int, default=10000, help='Desired codebook size')
    parser.add_argument('--keep_ratio', type=float, default=0.7, help='Ratio of training data to keep')

    args = parser.parse_args()
    main(args)