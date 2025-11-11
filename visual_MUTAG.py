import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx


def visualize_molecule(data):
    """
    Visualizes a molecular graph or subgraph from PyG Data object.
    Used specifically for MUTAG visualization in train.py.
    """
    # Atom types available in MUTAG
    atom_types = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
    G = to_networkx(data, to_undirected=True)

    # Assign atom labels and colors
    # data.x is one-hot encoded for MUTAG, argmax() gets the index.
    node_labels = {i: atom_types[data.x[i].argmax().item()] for i in G.nodes()}
    colors = []
    for node in G.nodes():
        atom_type = node_labels[node]
        if atom_type == 'C':
            colors.append('gray')
        elif atom_type == 'N':
            colors.append('lightcyan')
        elif atom_type == 'O':
            colors.append('red')
        elif atom_type == 'F':
            colors.append('green')
        elif atom_type == 'I':
            colors.append('purple')
        elif atom_type == 'Cl':
            colors.append('lime')
        elif atom_type == 'Br':
            colors.append('orange')
        else:
            colors.append('yellow') # Default color for safety

    # Key optimization: use Kamada-Kawai or spring layout
    pos = nx.kamada_kawai_layout(G)  # More suitable for small molecules
    # pos = nx.spring_layout(G, k=0.3, iterations=100)  # Adjust k and iterations

    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        labels=node_labels,
        node_size=1000,
        node_color=colors,
        font_size=16,
        font_weight='bold',
        edge_color='red',
        width=2.0
    )
    plt.title("Optimized Molecular Graph")
    plt.show()