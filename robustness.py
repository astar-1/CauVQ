# robustness.py

import torch
from torch_geometric.data import Data
from typing import Optional


class EdgeDropper:
    """
    A PyG transform that randomly removes a portion of edges from a graph's edge set.
    This implementation correctly handles undirected graphs, ensuring that both (u, v) and (v, u) are removed simultaneously.

    Args:
        p (float): The proportion of edges to remove, ranging from [0, 1].
                   E.g., p=0.2 means removing 20% of the edges.
    """

    def __init__(self, p: float):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Removal ratio p must be in the range [0, 1], but got {p}")
        self.p = p

    def __call__(self, data: Data) -> Data:
        """
        Applies the transform to the input PyG Data object.

        Args:
            data (Data): The input graph data.

        Returns:
            Data: The new graph data with edges removed.
        """
        if self.p == 0.0 or data.edge_index is None or data.num_edges == 0:
            # If removal ratio is 0 or the graph has no edges, return the original data directly
            return data

        # Clone data to avoid modifying the object in the original dataset
        new_data = data.clone()

        edge_index = new_data.edge_index
        edge_attr = new_data.edge_attr

        # --- Correct handling for undirected graphs ---
        # 1. Identify unique undirected edges. We only consider (u, v) where u < v.
        row, col = edge_index
        mask = row < col
        canonical_edge_index = edge_index[:, mask]

        # If edge attributes exist, filter them correspondingly
        canonical_edge_attr: Optional[torch.Tensor] = None
        if edge_attr is not None:
            canonical_edge_attr = edge_attr[mask]

        num_canonical_edges = canonical_edge_index.size(1)

        # 2. Calculate the number of edges to keep
        num_to_keep = int(num_canonical_edges * (1 - self.p))

        # 3. Randomly select edges to keep
        perm = torch.randperm(num_canonical_edges, device=edge_index.device)
        keep_indices = perm[:num_to_keep]

        kept_canonical_edge_index = canonical_edge_index[:, keep_indices]

        # 4. Reconstruct the complete, symmetric edge index from the kept canonical edges
        new_edge_index = torch.cat(
            [kept_canonical_edge_index, kept_canonical_edge_index.flip(0)],
            dim=1
        )

        # 5. Handle edge attributes similarly
        if canonical_edge_attr is not None:
            kept_canonical_edge_attr = canonical_edge_attr[keep_indices]
            # Edge attributes for undirected graphs are typically symmetric, so we just duplicate
            new_edge_attr = torch.cat([kept_canonical_edge_attr, kept_canonical_edge_attr], dim=0)
            new_data.edge_attr = new_edge_attr

        new_data.edge_index = new_edge_index

        return new_data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class DeterministicEdgeDropper:
    """
    A PyG transform that deterministically removes a portion of edges from a graph's edge set.

    For the same graph and the same seed, this transform will always remove the exact same edges,
    which is crucial for fair comparison across models or experiments.

    Args:
        p (float): The proportion of edges to remove, ranging from [0, 1].
        seed (int): The seed used to initialize the random number generator, ensuring reproducibility.
    """

    def __init__(self, p: float, seed: int):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Removal ratio p must be in the range [0, 1], but got {p}")
        self.p = p
        self.seed = seed

    def __call__(self, data: Data) -> Data:
        """
        Applies the transform to the input PyG Data object.

        Args:
            data (Data): The input graph data.

        Returns:
            Data: The new graph data with edges removed.
        """
        if self.p == 0.0 or data.edge_index is None or data.num_edges == 0:
            return data

        new_data = data.clone()
        edge_index = new_data.edge_index
        edge_attr = new_data.edge_attr

        row, col = edge_index
        mask = row < col
        canonical_edge_index = edge_index[:, mask]

        canonical_edge_attr: Optional[torch.Tensor] = None
        if edge_attr is not None:
            canonical_edge_attr = edge_attr[mask]

        num_canonical_edges = canonical_edge_index.size(1)
        if num_canonical_edges == 0:
            return new_data  # Return directly if there are no canonical edges

        # --- Deterministic Core ---
        # 1. Create a deterministic signature for the graph, which is always the same for the same graph.
        graph_signature = data.num_nodes + data.num_edges + int(edge_index.sum())

        # 2. Combine the base seed and the graph signature to create a local seed specific to this graph.
        local_seed = self.seed + graph_signature

        # 3. Use this specific seed to initialize a local, independent random number generator.
        #    This will not affect or be affected by the global torch.manual_seed.
        generator = torch.Generator(device=edge_index.device).manual_seed(local_seed)

        # 4. Use this local generator to create the permutation, guaranteeing reproducible results.
        perm = torch.randperm(num_canonical_edges, device=edge_index.device, generator=generator)
        # --- Deterministic Core End ---

        num_to_keep = int(num_canonical_edges * (1 - self.p))
        keep_indices = perm[:num_to_keep]

        kept_canonical_edge_index = canonical_edge_index[:, keep_indices]

        new_edge_index = torch.cat(
            [kept_canonical_edge_index, kept_canonical_edge_index.flip(0)],
            dim=1
        )

        if canonical_edge_attr is not None:
            kept_canonical_edge_attr = canonical_edge_attr[keep_indices]
            new_edge_attr = torch.cat([kept_canonical_edge_attr, kept_canonical_edge_attr], dim=0)
            new_data.edge_attr = new_edge_attr

        new_data.edge_index = new_edge_index

        return new_data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p}, seed={self.seed})'


def node_feature_noise_adder(std, seed, data):
    """
    Deterministically adds Gaussian noise to node features.

    Args:
        std (float): The standard deviation of the Gaussian noise.
        seed (int): The base seed for the random number generator.
        data (Data): The PyG Data object.

    Returns:
        Data: The PyG Data object with added noise.
    """
    if std == 0.0 or data.x is None:
        # If noise standard deviation is 0 or there are no node features, return the original data directly
        return data

    # --- Deterministic Core ---
    # 1. Create a deterministic signature for the graph
    graph_signature = data.num_nodes + int(data.x.sum())

    # 2. Combine the base seed and the graph signature to create a local seed
    local_seed = seed + graph_signature

    # 3. Use this specific seed to initialize a local, independent random number generator
    generator = torch.Generator(device=data.x.device).manual_seed(local_seed)

    # 4. Generate noise with the same shape as node features, mean 0, and standard deviation 1
    noise = torch.randn(data.x.shape, generator=generator, device=data.x.device)

    # 5. Scale the noise to the specified standard deviation and add it to the node features
    scaled_noise = noise * std
    # --- Deterministic Core End ---
    data.x = data.x + scaled_noise
    # Note: If the original features were not float (e.g., long), this conversion might be an issue.
    # The original code cast to long here. Assuming this is intentional for the dataset.
    data.x = data.x.long()
    return data