import math
import random
import community
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.data import Batch
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
import warnings
from itertools import product
from torch_geometric.utils import degree
import torch.nn.functional as F
from collections import defaultdict
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from collections import defaultdict, Counter
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
from torch_geometric.data import Data
import torch
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem.Draw import MolToImage

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit is not installed. Chemical structure visualization will not be available.")


def get_substructure(G):
    """
    Extracts substructures (communities) from a NetworkX graph G using the SLPA algorithm,
    and converts them into a list of PyG Data objects.

    Args:
        G (networkx.Graph): Input undirected graph.

    Returns:
        list: List of PyG Data objects, each representing a substructure (community).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        # G is a networkx graph
        communities = slpa_community_dict(G)

    # Map community ID to node list
    result_dict = {}
    for key, value in communities.items():
        result_dict[key] = value

    substructure_nodes = list(result_dict.values())

    # Subgraph index alignment
    original_graph = nx_to_pyg(G)
    substructures = []

    for nodes in substructure_nodes:
        if len(nodes) <= 1:
            continue

        # Create mask for selected nodes
        node_mask = torch.zeros(original_graph.num_nodes, dtype=torch.bool)
        node_mask[nodes] = True

        # Generate subgraph (automatically filters edges and re-indexes nodes)
        sub_graph = original_graph.subgraph(node_mask)
        substructures.append(sub_graph)

    return substructures


# Renaming 'substructure' to 'get_substructure' for clarity (and keeping 'substructure' alias if needed)
substructure = get_substructure


def slpa_community_dict(G, max_iter=50, threshold=0.15, r=0.5):
    """
    Implementation of the SLPA algorithm, outputting in the format {community_ID: [node_list]}.

    Args:
        G (networkx.Graph): Input undirected graph.
        max_iter (int): Maximum number of iterations (default is 50).
        threshold (float): Label retention threshold (default is 0.15).
        r (float): Probability of randomly selecting a neighbor (default is 0.5).

    Returns:
        dict: {Community ID: sorted node list}
            Example: {0: [0,1,2], 1: [1,2,3], 2: [4,5,6]}
    """
    # 1. Initialization
    nodes = sorted(G.nodes())
    memory = {node: [node] for node in nodes}  # Each node records label history

    # 2. Label Propagation Process
    for _ in range(max_iter):
        order = list(nodes)
        random.shuffle(order)  # Random sequence

        for node in order:
            neighbors = list(G.neighbors(node))  # Get the list of neighbors first
            if random.random() < r and len(neighbors) > 0:

                # --- Start Modification (for determinism) ---
                # Sort the neighbor list before random choice to ensure determinism
                sorted_neighbors = sorted(neighbors)
                neighbor = random.choice(sorted_neighbors)
                # --- End Modification ---

                if memory[neighbor]:
                    received_label = random.choice(memory[neighbor])
                    memory[node].append(received_label)

    # 3. Post-processing: Filter labels based on threshold
    communities = defaultdict(list)

    for node in nodes:
        # Count the frequency of all labels appearing for the node
        label_counts = Counter(memory[node])
        total = sum(label_counts.values())

        # Retain labels whose frequency exceeds the threshold
        for label, count in label_counts.items():
            if count / total >= threshold:
                communities[label].append(node)

    # 4. Merge overlapping communities (communities with the same set of nodes)
    unique_communities = {}
    for comm in communities.values():
        comm_tuple = tuple(sorted(comm))
        unique_communities[comm_tuple] = list(comm_tuple)

    # 5. Renumber community IDs to continuous integers (0, 1, 2, ...)
    renumbered_communities = {
        new_id: nodes
        for new_id, (_, nodes) in enumerate(sorted(unique_communities.items()))
    }

    return renumbered_communities


def nx_to_pyg(nx_graph):
    """Converts a NetworkX graph to a PyG Data object."""
    # Extract features according to node order
    node_list = sorted(nx_graph.nodes())  # Sort to ensure consistent order
    x = torch.tensor([nx_graph.nodes[n]['feature'] for n in node_list], dtype=torch.float)

    # Generate edge index
    edge_index = torch.tensor(
        [[u, v] for u, v in nx_graph.edges()],
        dtype=torch.long
    ).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def pyg_to_nx(data, node_id_mapping=None):
    """Converts a PyG Data object to a NetworkX graph, preserving node features.

    Args:
        data (Data): PyG data object, must contain x and edge_index.
        node_id_mapping (list/dict): Optional, original node ID mapping table.

    Returns:
        nx.Graph: NetworkX graph with features.
    """
    # Create an empty graph
    G = nx.Graph()

    # Handle node ID mapping
    if node_id_mapping is None:
        node_ids = list(range(data.num_nodes))
    elif isinstance(node_id_mapping, list):
        node_ids = node_id_mapping
    else:
        node_ids = list(node_id_mapping.values())

    # Add nodes with features
    for i, feat in enumerate(data.x.numpy()):
        original_id = node_ids[i]  # Get original node ID
        G.add_node(original_id, feature=feat.tolist())  # Convert to Python list for storage

    # Add edges
    edges = data.edge_index.t().numpy()
    for u_idx, v_idx in edges:
        u = node_ids[u_idx]
        v = node_ids[v_idx]
        G.add_edge(u, v)

    return G


def sample(graph_list, k):
    """Randomly samples k graphs from a list and returns them as a PyG Batch object."""
    batch_size = k
    sampled_indices = random.sample(range(len(graph_list)), batch_size)  # Randomly select indices
    sampled_data = [graph_list[i] for i in sampled_indices]  # Get corresponding graphs
    batch = Batch.from_data_list(sampled_data)
    return batch


def get_adjacency_matrix(data):
    """Computes the dense adjacency matrix for a single PyG Data object."""
    num_nodes = data.num_nodes  # Get number of nodes
    edge_index = data.edge_index  # Get edge index

    # Create an all-zero matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

    # Fill in edges
    # For undirected graphs, each edge needs to be filled twice
    edge_list = edge_index.tolist()
    for i in range(len(edge_list[0])):
        adj_matrix[edge_list[0][i], edge_list[1][i]] = 1

    # Ensure it is a dense tensor (redundant if already zeros, but harmless)
    adj_tensor = adj_matrix.to_dense() if adj_matrix.is_sparse else adj_matrix

    return adj_tensor


# Renaming 'adj' to 'get_adjacency_matrix'
adj = get_adjacency_matrix


def size_split_idx(dataset, mode='ld'):
    """Split dataset based on graph size.

    Args:
        dataset: PyG dataset object
        mode: 'ls' (train on large graphs, test on small) or 'lb' (train on small graphs, test on large - this should probably be 'sl' for consistency with common OOD naming, but keeping 'ld'/'ls' for now).

    Returns:
        Dictionary with train/valid/test indices
    """
    num_graphs = len(dataset)
    num_val = int(0.1 * num_graphs)
    num_test = int(0.1 * num_graphs)

    # Get sorted indices by graph size
    num_nodes = [data.num_nodes for data in dataset]
    sorted_indices = np.argsort(num_nodes)

    # Split based on mode
    if mode == 'ls':  # Train on large graphs, test on small graphs
        train_indices = sorted_indices[2 * num_val:]
        val_test_indices = sorted_indices[:2 * num_val]
    else:  # 'ld' (default - assumed to mean 'small' train, 'large' test in context of original code split logic, which often happens in OOD setups)
        train_indices = sorted_indices[:-2 * num_val]
        val_test_indices = sorted_indices[-2 * num_val:]

    # Randomly split val_test into validation and test
    np.random.shuffle(val_test_indices)
    val_indices = val_test_indices[:num_val]
    test_indices = val_test_indices[num_val:]

    return {
        'train': torch.tensor(train_indices, dtype=torch.long),
        'valid': torch.tensor(val_indices, dtype=torch.long),
        'test': torch.tensor(test_indices, dtype=torch.long)
    }


def min_max_normalize(tensor):
    """Min-Max normalization of the diagonal elements of a square matrix."""
    # Assuming input is a square matrix
    diagonal_elements = torch.diag(tensor)
    tensor_min = diagonal_elements.min()  # Get minimum value
    tensor_max = diagonal_elements.max()  # Get maximum value
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor


def split_dataset_by_node_count_sorted_ratio(dataset, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """
    Sorts the dataset based on the number of nodes in each graph and splits it
    into train, validation, and test sets according to fixed ratios.

    Splitting logic:
    1. Get the node count for each graph in the dataset.
    2. Sort the graph indices in ascending order based on node count.
    3. Split the sorted index list according to the given ratios.
       - Train set: The first (smallest node count) portion.
       - Valid set: The middle portion.
       - Test set: The last (largest node count) portion.

    Args:
    dataset (torch_geometric.data.Dataset): The complete PyG dataset.
    train_ratio (float): Ratio for the training set (default is 0.8).
    valid_ratio (float): Ratio for the validation set (default is 0.1).
    test_ratio (float): Ratio for the test set (default is 0.1).

    Returns:
    dict: A dictionary containing 'train', 'valid', 'test' keys,
          each mapping to a PyTorch LongTensor of graph indices.
    """
    # Ensure ratios sum to 1
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    print("Splitting dataset by sorted node count with ratio...")
    print(f"  - Ratios: Train={train_ratio * 100}%, Valid={valid_ratio * 100}%, Test={test_ratio * 100}%")

    # Step 1: Get node counts and original indices
    print("Scanning dataset to get node counts...")
    node_counts_with_indices = []
    for i in tqdm(range(len(dataset)), desc="Scanning graphs"):
        node_counts_with_indices.append((dataset[i].num_nodes, i))

    # Step 2: Sort the list based on node count (first element of the tuple)
    print("Sorting graphs by node count...")
    node_counts_with_indices.sort(key=lambda x: x[0])

    # Extract the sorted original indices
    sorted_indices = [index for count, index in node_counts_with_indices]

    # Step 3: Calculate split points and divide indices
    total_size = len(dataset)
    train_end_idx = int(total_size * train_ratio)
    valid_end_idx = train_end_idx + int(total_size * valid_ratio)

    train_indices = sorted_indices[:train_end_idx]
    valid_indices = sorted_indices[train_end_idx:valid_end_idx]
    test_indices = sorted_indices[valid_end_idx:]

    split_idx = {
        'train': torch.LongTensor(train_indices),
        'valid': torch.LongTensor(valid_indices),
        'test': torch.LongTensor(test_indices)
    }

    print("\nSplit summary (after sorting by node count):")
    print(f"  - Number of training graphs: {len(split_idx['train'])}")
    print(f"  - Number of validation graphs: {len(split_idx['valid'])}")
    print(f"  - Number of test graphs: {len(split_idx['test'])}")

    return split_idx


def plot_curve(loss, train_acc):
    """Plots training loss and a performance metric (e.g., ROC-AUC difference)."""
    # Create a figure and a subplot
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot loss curve
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(loss, label='Train Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Automatically adjust y-axis range (expand top by 10% space)
    y1_min, y1_max = min(loss), max(loss)
    ax1.set_ylim(y1_min, y1_max + 0.1 * (y1_max - y1_min))

    # Instantiate a second y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('ROC-AUC', color='tab:blue')
    ax2.plot(train_acc, label='ROC-AUC diff (train vs test)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Automatically adjust second y-axis range (expand top by 15% space)
    y2_min, y2_max = min(train_acc), max(train_acc)
    ax2.set_ylim(y2_min, y2_max + 0.15 * (y2_max - y2_min))

    # Merge legends and position precisely
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels,
               loc='upper left',
               bbox_to_anchor=(0.02, 0.98),  # Top left inside
               framealpha=0.8)

    plt.tight_layout()
    plt.show()


def plot_performance_curves(train_perfs, test_perfs, metric_name='ROC-AUC', dataset_name='Dataset'):
    """
    Plots the training performance, test performance, and the difference between the two over epochs.

    Args:
    train_perfs (list): List of training set performance (e.g., accuracy) per epoch.
    test_perfs (list): List of test set performance per epoch.
    metric_name (str): Name of the performance metric (e.g., 'Accuracy', 'ROC-AUC', 'RMSE').
    dataset_name (str): Name of the dataset, used for generating the filename.
    """
    # Ensure inputs are numpy arrays for easy calculation
    train_perfs_np = np.array(train_perfs)
    test_perfs_np = np.array(test_perfs)

    # Calculate performance difference
    perf_diff = train_perfs_np - test_perfs_np

    # Create a figure and subplots
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # --- Left Y-axis: Plot Train and Test Performance ---
    color_train = 'tab:blue'
    color_test = 'tab:green'
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel(metric_name, color=color_train, fontsize=20)

    ax1.plot(train_perfs_np, label=f'Train {metric_name}', color=color_train, linestyle='-')
    ax1.plot(test_perfs_np, label=f'Test {metric_name}', color=color_test, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color_train, labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # --- Right Y-axis: Plot Performance Difference ---
    ax2 = ax1.twinx()
    color_diff = 'tab:red'
    ax2.set_ylabel('Difference (Train - Test)', color=color_diff, fontsize=20)
    ax2.plot(perf_diff, label=f'{metric_name} Difference', color=color_diff, linestyle=':')
    ax2.tick_params(axis='y', labelcolor=color_diff, labelsize=14)

    # Draw a horizontal dashed line at y=0 as a reference baseline for the difference
    ax2.axhline(0, color=color_diff, linestyle='--', linewidth=1)

    # --- Legend and Title ---
    # Merge the legends of both axes (The correct way)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

    plt.title(f'Training & Test Performance ({metric_name})', fontsize=20)
    fig.tight_layout()

    # Save the image instead of displaying it directly, to avoid blocking the program in a server environment
    filename = f'performance_curve_{dataset_name}.png'
    plt.savefig(filename)
    print(f"Performance curve plot saved to {filename}")
    plt.close()


def plot_tsne(embeddings, labels, title='t-SNE Visualization of Graph Embeddings'):
    """
    Performs dimensionality reduction on graph embeddings using t-SNE and visualizes the result.

    Args:
        embeddings (np.array): Numpy array of shape (N, D), where N is the number of graphs and D is the embedding dimension.
        labels (np.array): Numpy array of shape (N,), containing the label for each graph.
        title (str): Title of the image.
    """
    print(f"Running t-SNE for '{title}'...")
    # Initialize t-SNE model
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca', learning_rate='auto')

    # Reduce dimensionality of embeddings
    tsne_results = tsne.fit_transform(embeddings)

    # Use Seaborn to create a visually appealing scatter plot
    plt.figure(figsize=(12, 10))

    # Ensure labels are integer type for categorization
    unique_labels = np.unique(labels)

    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=labels.astype(int),  # Use labels for color differentiation
        palette=sns.color_palette("deep", len(unique_labels)),  # Use the deep palette
        legend="full",
        alpha=0.7
    )

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title='Class')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Display the image
    print("Displaying t-SNE plot.")
    plt.show()


def plot_score_distributions(causal_scores, labels, title_prefix=''):
    """
    Plots the score distribution histogram for causal predictions (binary classification).

    Args:
        causal_scores (np.array): Predicted probabilities based on causal representation.
        labels (np.array): True labels.
        title_prefix (str): Prefix for the image title.
    """
    print("Generating score distribution histogram for causal embeddings...")

    # Create a single figure
    plt.figure(figsize=(10, 7))
    plt.title(f'{title_prefix} Predicted Probability Distribution \n(Non-Causal Embeddings)', fontsize=20)

    # Separate scores for the two classes
    scores_class_0 = causal_scores[labels == 0]
    scores_class_1 = causal_scores[labels == 1]

    colors = sns.color_palette("deep", 2)
    # Use Seaborn to plot the distribution histogram for both classes
    # Class 0 (Negative)
    sns.histplot(scores_class_0, bins=40, color=colors[0], label='Class 0 (Negative)',
                 stat='density', common_norm=False, alpha=0.7)

    # Class 1 (Positive)
    sns.histplot(scores_class_1, bins=40, color=colors[1], label='Class 1 (Positive)',
                 stat='density', common_norm=False, alpha=0.7)

    # Increase label font size
    plt.xlabel('Predicted Probability of being Class 1', fontsize=20)
    plt.ylabel('Density', fontsize=30)

    # Increase legend font size
    plt.legend(title='Classes', fontsize=30, title_fontsize=20)

    # Optional: Increase tick label font size
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    print("Displaying plot.")
    plt.show()


def plot_multiclass_score_distributions(scores, labels, num_classes, title_prefix=''):
    """
    Plots the score distribution histogram for the true class predictions in a multi-class task (single plot).

    Args:
        scores (np.array): Predicted probabilities (softmax outputs), shape [N, num_classes].
        labels (np.array): True labels, shape [N,].
        num_classes (int): Total number of classes in the dataset.
        title_prefix (str): Prefix for the image title.
    """
    print("Generating multi-class score distribution histogram...")

    # Create a single figure
    plt.figure(figsize=(12, 7))
    plt.title(f'{title_prefix} Predicted Probability of True Class\n(Based on Causal Embeddings)',
              fontsize=20, pad=20)

    # Define colors for different classes
    colors = sns.color_palette("deep", num_classes)

    # Iterate through each class
    for i in range(num_classes):
        # Find all samples whose true label is the current class i
        class_mask = (labels == i)

        if np.any(class_mask):
            # Extract the predicted probability for the *correct* class i
            correct_class_probs = scores[class_mask, i]

            # Plot the probability distribution histogram
            sns.histplot(correct_class_probs, bins=30, color=colors[i],
                         label=f'True Class {i}', stat='density', alpha=0.7)

    plt.xlabel('Predicted Probability of True Class', fontsize=20)
    plt.ylabel('Density', fontsize=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(title='Classes', fontsize=15, title_fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, 1)  # Probability range is 0 to 1

    plt.tight_layout()
    print("Displaying plot.")
    plt.show()


def get_atom_symbol_from_feature(atom_feature):
    """Retrieves the atom symbol from the OGB dataset's atom feature vector (used for BBBP)."""
    atomic_num = int(atom_feature[0].item())
    return Chem.GetPeriodicTable().GetElementSymbol(atomic_num)


def get_bond_type_from_feature(bond_feature):
    """Retrieves the RDKit bond type from the OGB dataset's bond feature vector."""
    bond_idx = int(bond_feature[0].item())
    bond_map = {0: Chem.rdchem.BondType.AROMATIC, 1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE,
                3: Chem.rdchem.BondType.TRIPLE}
    return bond_map.get(bond_idx, Chem.rdchem.BondType.UNSPECIFIED)


def pyg_to_rdkit_mol(pyg_data):
    """Converts a PyG Data object to an RDKit Mol object."""
    if not RDKIT_AVAILABLE: return None
    editable_mol = Chem.EditableMol(Chem.Mol())

    # Add atoms
    for i in range(pyg_data.num_nodes):
        # Assumes node features follow OGB's molecular structure (atomic number first)
        atom_symbol = get_atom_symbol_from_feature(pyg_data.x[i])
        atom = Chem.Atom(atom_symbol)
        editable_mol.AddAtom(atom)

    added_edges = set()

    # Add bonds
    if pyg_data.edge_index is not None:
        for i in range(pyg_data.edge_index.size(1)):  # Iterate over edges
            u, v = pyg_data.edge_index[:, i].tolist()
            if u > v: u, v = v, u
            if (u, v) in added_edges: continue

            bond_type = Chem.rdchem.BondType.SINGLE
            if pyg_data.edge_attr is not None:
                # Assumes edge attributes follow OGB's molecular structure (bond type first)
                bond_type = get_bond_type_from_feature(pyg_data.edge_attr[i])

            editable_mol.AddBond(u, v, bond_type)
            added_edges.add((u, v))

    mol = editable_mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
        Chem.rdDepictor.Compute2DCoords(mol)
    except Exception:
        pass
    return mol


# --- [UPGRADED] Main Visualization Function ---

def plot_clustering_with_causal_highlights(
        all_subgraph_features,
        all_subgraph_structures,
        centroid_indices,
        causal_weights,
        representative_subgraphs,
        num_to_highlight=5,
        title='t-SNE of Subgraph Clustering for Codebook Construction'
):
    """
    [UPGRADED] Visualizes the subgraph clustering process and highlights the chemical
    structure of the subgraphs with the highest causal weights using RDKit.
    """
    print("Visualizing codebook clustering process with causal highlights...")

    # 1. Perform t-SNE dimensionality reduction on all subgraph features
    perplexity = min(30, len(all_subgraph_features) - 1)
    if perplexity <= 0:
        print("Not enough data points for t-SNE. Skipping visualization.")
        return

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(all_subgraph_features)

    # 2. Find the subgraphs with the highest causal weights
    # Note: causal_weights are assumed to be a 1D tensor of size len(representative_subgraphs)
    top_causal_weights, top_indices_in_repr = torch.topk(causal_weights, k=num_to_highlight)
    # Map back to the index in the original all_subgraph_structures list
    top_causal_original_indices = [centroid_indices[i] for i in top_indices_in_repr]

    # --- Plot t-SNE Scatter Plot ---
    plt.figure(figsize=(16, 12))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c='lightblue', alpha=0.5, label='All Unique Subgraphs')

    # Scatter the centroids (Codebook Members)
    plt.scatter(tsne_results[centroid_indices, 0], tsne_results[centroid_indices, 1], c='orange', s=50,
                edgecolors='gray', label='Cluster Centroids (Codebook Members)')

    # Highlight the top causal subgraphs
    plt.scatter(tsne_results[top_causal_original_indices, 0], tsne_results[top_causal_original_indices, 1], c='red',
                s=200, marker='*', edgecolors='black', linewidth=1, label=f'Top {num_to_highlight} Causal Subgraphs')

    plt.title(title, fontsize=18)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # --- [CORE UPGRADE] Plotting Chemical Structures with RDKit ---
    if not RDKIT_AVAILABLE:
        print("Skipping chemical structure visualization because RDKit is not installed.")
        return

    print("\nDisplaying the chemical structures of the most important causal subgraphs...")
    cols = min(num_to_highlight, 5)
    rows = math.ceil(num_to_highlight / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
    axes = np.array(axes).flatten()

    for i in range(num_to_highlight):
        ax = axes[i]
        original_idx = top_causal_original_indices[i]
        subgraph_data = all_subgraph_structures[original_idx]

        # Use the conversion and plotting helper function
        mol = pyg_to_rdkit_mol(subgraph_data)

        if mol:
            img = MolToImage(mol, size=(300, 300), legend=f'Rank {i + 1}')
            ax.imshow(img)
        else:
            # Fallback to networkx if RDKit rendering fails
            G = pyg_to_nx(subgraph_data)
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='coral', node_size=400, edge_color='gray')
            ax.text(0.5, 0.5, 'RDKit Render Failed', ha='center', va='center', color='red', fontsize=10)

        ax.set_title(f"Causal Rank {i + 1}\nWeight: {top_causal_weights[i]:.4f}", fontsize=12)
        ax.axis('off')

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def create_ablated_test_set(original_dataset, important_subgraphs):
    """
    Creates a new test set where important substructures contained in each graph are removed.

    Args:
        original_dataset (Dataset): The original PyG test dataset.
        important_subgraphs (list): A list of important substructures (PyG Data objects).

    Returns:
        list: A list of new ablated graphs (PyG Data objects).
        int: The number of graphs that were successfully modified.
    """
    print(
        f"Starting ablation experiment: searching for and removing {len(important_subgraphs)} important substructures from the test set...")

    ablated_graphs = []
    modified_graph_count = 0

    # Pre-convert important PyG subgraphs to NetworkX graphs for efficiency
    important_nx_subgraphs = [pyg_to_nx(sub) for sub in important_subgraphs]

    # Use tqdm to show processing progress
    for graph_data in tqdm(original_dataset, desc="Ablating test graphs"):
        # Copy the original graph data to avoid modifying the original dataset
        new_graph_data = graph_data.clone()

        # Convert the test graph to NetworkX format
        nx_G = pyg_to_nx(new_graph_data)

        was_modified = False
        for nx_S in important_nx_subgraphs:
            # Define node matcher function: requires node features to be exactly the same
            node_matcher = lambda n1, n2: torch.equal(
                torch.tensor(n1['feature'], dtype=torch.float32),
                torch.tensor(n2['feature'], dtype=torch.float32)
            )

            # Use GraphMatcher for subgraph isomorphism search
            matcher = nx.isomorphism.GraphMatcher(nx_G, nx_S, node_match=node_matcher)

            if matcher.subgraph_is_isomorphic():
                # Remove only the first match found, then proceed to the next test graph
                # mapping: {subgraph_node: graph_node}
                mapping = next(matcher.subgraph_isomorphisms_iter())

                edges_to_remove = []
                subgraph_edges = nx_S.edges()

                for u, v in subgraph_edges:
                    # Map edges in the subgraph back to edges in the main graph
                    mapped_u, mapped_v = mapping[u], mapping[v]
                    if nx_G.has_edge(mapped_u, mapped_v):
                        edges_to_remove.append((mapped_u, mapped_v))

                # Remove these edges from the graph
                if edges_to_remove:
                    nx_G.remove_edges_from(edges_to_remove)
                    was_modified = True
                    break  # Found and removed one, break the inner loop

        if was_modified:
            modified_graph_count += 1
            # Convert the modified NetworkX graph back to a PyG Data object
            # Note: Node count remains unchanged, only edge_index is modified
            new_edge_list = list(nx_G.edges())
            if new_edge_list:
                new_edge_index = torch.tensor(new_edge_list, dtype=torch.long).t().contiguous()
            else:  # If no edges remain after removal
                new_edge_index = torch.empty((2, 0), dtype=torch.long)

            new_graph_data.edge_index = new_edge_index

        ablated_graphs.append(new_graph_data)

    print(f"Ablation complete. {modified_graph_count} out of {len(original_dataset)} graphs were modified.")
    return ablated_graphs, modified_graph_count