# CauVQ


CauVQ/
├── data/                       # Data storage directory
├── causal.py                   # Causal Core: DegreeLayer for masking
├── codebook.py                 # Subgraph extraction (SLPA) and codebook filtering
├── conv.py                     # Standard GNN layers (GIN/GCN)
├── mo_conv.py                  # GNN layers for models with continuous features (MoModel)
├── GCN_Conv.py                 # Simplified GCNConv
├── loss.py                     # Custom loss functions (L_rec, L_cf, L_match)
├── model.py                    # Main network architectures (SynModel, OgbModel, VectorQuantizer)
├── robustness.py               # Deterministic EdgeDropper and NodeFeatureNoiseAdder transforms
├── train.py                    # MUTAG K-Fold training and visualization
├── train_main.py               # OGB standard training
├── train_mnist.py              # MNIST-75sp training
├── train_syn.py                # SPMotif baseline training
├── train_syn_perturb.py        # SPMotif training on perturbed data
└── train_OOD_perturb.py        # OGB training on reduced training set


