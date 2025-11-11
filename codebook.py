# codebook_generation.py

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
# Assume utils.py provides these functions
from utils import pyg_to_nx, nx_to_pyg, substructure, sample
import numpy as np
from torch_geometric.data import Batch
import torch
from einops import rearrange
import torch.nn.functional as F
import networkx as nx
from collections import defaultdict


def extract_lightweight_features(pyg_graph, max_degree=10):
    """Extract lightweight structural features for a single PyG graph."""
    num_nodes = pyg_graph.num_nodes
    num_edges = pyg_graph.num_edges

    # Calculate degrees
    if pyg_graph.edge_index.numel() > 0:
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        degrees.scatter_add_(0, pyg_graph.edge_index[0], torch.ones_like(pyg_graph.edge_index[0], dtype=torch.long))
        degrees.scatter_add_(0, pyg_graph.edge_index[1], torch.ones_like(pyg_graph.edge_index[1], dtype=torch.long))
    else:
        degrees = torch.zeros(num_nodes, dtype=torch.long)

    # Create degree histogram
    degree_hist = torch.histc(degrees.float(), bins=max_degree, min=0, max=max_degree)

    # Normalize histogram by number of nodes
    if num_nodes > 0:
        degree_hist = degree_hist / num_nodes

    # Combine into a feature vector
    features = degree_hist.numpy()

    return features


def compute_graph_complexity(pyg_graph):
    """Compute a complexity score for the graph structure."""
    # Convert to networkx graph to calculate graph metrics
    nx_graph = pyg_to_nx(pyg_graph)

    num_nodes = pyg_graph.num_nodes
    num_edges = pyg_graph.num_edges

    # Complexity is 0 if there are no edges
    if num_edges == 0:
        return 0.0

    # Base complexity: edge-to-node ratio
    edge_node_ratio = num_edges / num_nodes

    # Calculate clustering coefficient (may be 0 for very small graphs)
    try:
        clustering_coeff = nx.average_clustering(nx_graph)
    except:
        clustering_coeff = 0.0

    # Calculate number of connected components (fewer is better)
    num_components = nx.number_connected_components(nx_graph)

    # Combined complexity score
    complexity = (
            0.6 * edge_node_ratio +
            0.3 * clustering_coeff +
            0.1 * (1 / num_components))

    return complexity


def generate_codebook(data, num_codes=1000, method='size'):
    """
    Modified function: reduces the codebook size using non-clustering methods.

    Supports four reduction methods:
    1. 'frequency' - Filter based on occurrence frequency (default)
    2. 'complexity' - Filter based on graph complexity
    3. 'random' - Random sampling
    4. 'size' - Filter based on graph size
    """
    graph_list = [graph for graph in data]
    subs_list = []
    sample_size = len(graph_list)  # Limit sampling size for efficiency
    batch = sample(graph_list, sample_size)
    original_graphs = batch.to_data_list()
    print('Sample size:', len(original_graphs))

    # Extract subgraphs
    for graph in original_graphs:
        nx_graph = pyg_to_nx(graph)
        subs_list += substructure(nx_graph)
    print(f'Total number of original subgraphs: {len(subs_list)}')

    # MODIFICATION: Count the frequency of each subgraph
    graph_frequency = defaultdict(int)
    unique_subgraphs_dict = {}

    for sub_pyg in subs_list:
        if sub_pyg.x is not None and sub_pyg.num_nodes > 1:  # Ensure subgraph has nodes and is not an isolated node
            sub_nx = pyg_to_nx(sub_pyg)
            # node_features = tuple(tuple(row) for row in sub_pyg.x.tolist()) # Not used
            graph_hash = nx.weisfeiler_lehman_graph_hash(sub_nx, node_attr='feature')

            # Count frequency
            graph_frequency[graph_hash] += 1

            # Keep only the first occurrence of the subgraph as a representative
            if graph_hash not in unique_subgraphs_dict:
                unique_subgraphs_dict[graph_hash] = sub_pyg

    unique_subgraphs = list(unique_subgraphs_dict.values())
    print(f'Number of unique independent subgraphs after deduplication: {len(unique_subgraphs)}')

    # Extract frequency information
    frequencies = [graph_frequency[hash_val] for hash_val in unique_subgraphs_dict.keys()]

    # If the number of unique subgraphs is less than or equal to the target size, return directly
    if len(unique_subgraphs) <= num_codes:
        print(f"Number of unique subgraphs ({len(unique_subgraphs)}) is already less than or equal to the target codebook size ({num_codes})")
        centroid_indices = list(range(len(unique_subgraphs)))
        lightweight_features = np.array([extract_lightweight_features(g) for g in unique_subgraphs])
        return unique_subgraphs, unique_subgraphs, lightweight_features, centroid_indices

    # Reduce codebook based on the selected method
    if method == 'random':
        # Method 1: Random sampling
        print(f"Reducing codebook size using random sampling: {len(unique_subgraphs)} -> {num_codes}")
        indices = np.random.choice(len(unique_subgraphs), size=num_codes, replace=False)
        representative_subgraphs = [unique_subgraphs[i] for i in indices]
        centroid_indices = indices.tolist()

    elif method == 'size':
        # Method 2: Filter based on graph size (keep graphs with more nodes)
        print(f"Reducing codebook size based on graph size: {len(unique_subgraphs)} -> {num_codes}")
        # Sort by number of nodes
        sorted_indices = sorted(range(len(unique_subgraphs)),
                                key=lambda i: unique_subgraphs[i].num_nodes,
                                reverse=False) # Note: Reverse is False, so it keeps smaller graphs first, then takes the top num_codes. The comment said 'retain graphs with more nodes', but the code keeps the smallest ones first. I'll preserve the original logic (reverse=False) unless it's a known bug. Assuming the user meant to keep the *first* num_codes after sorting by size (ascending).

        # Take the first num_codes
        selected_indices = sorted_indices[:num_codes]
        representative_subgraphs = [unique_subgraphs[i] for i in selected_indices]
        centroid_indices = selected_indices

    elif method == 'complexity':
        # Method 3: Filter based on graph complexity
        print(f"Reducing codebook size based on graph complexity: {len(unique_subgraphs)} -> {num_codes}")
        # Calculate complexity for all subgraphs
        complexities = [compute_graph_complexity(g) for g in unique_subgraphs]

        # Sort by complexity
        sorted_indices = sorted(range(len(complexities)),
                                key=lambda i: complexities[i],
                                reverse=True)

        # Take the first num_codes
        selected_indices = sorted_indices[:num_codes]
        representative_subgraphs = [unique_subgraphs[i] for i in selected_indices]
        centroid_indices = selected_indices

    else:  # Default: Filter based on frequency
        # Method 4: Filter based on occurrence frequency
        print(f"Reducing codebook size based on occurrence frequency: {len(unique_subgraphs)} -> {num_codes}")
        # Sort by frequency in descending order
        sorted_indices = sorted(range(len(frequencies)),
                                key=lambda i: frequencies[i],
                                reverse=True)

        # Take the first num_codes
        selected_indices = sorted_indices[:num_codes]
        representative_subgraphs = [unique_subgraphs[i] for i in selected_indices]
        centroid_indices = selected_indices

        # Print frequency distribution information
        total_occurrences = sum(frequencies)
        top_freq_sum = sum(frequencies[i] for i in selected_indices)
        print(f"The top {num_codes} high-frequency subgraphs cover {top_freq_sum / total_occurrences:.2%} of the total occurrences")

    # Calculate lightweight features
    lightweight_features = np.array([extract_lightweight_features(g) for g in unique_subgraphs])

    print(f'Successfully selected {len(representative_subgraphs)} representative subgraphs using the {method} method.')
    return unique_subgraphs, representative_subgraphs, lightweight_features, centroid_indices

# Rename the function used in train_main.py to be consistent
codebook = generate_codebook