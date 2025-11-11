# CauVQ
Overview
This project implements a Causal Substructure Learning framework designed to enhance the Out-of-Distribution (OOD) generalization and interpretability of Graph Neural Networks (GNNs) in graph classification tasks. The core idea is to identify and leverage invariant causal subgraphs by decoupling predictive features from spurious correlations using a Codebook and Causal Intervention inspired by Structural Causal Models (SCMs).

The framework is tested across synthetic (SPMotif), real-world molecular (OGBG, MUTAG) and non-molecular (MNIST-75sp) datasets.

File,Description
model.py,"Defines the main network architectures: SynModel, OgbModel, MoModel, CodebookEncoder, VectorQuantizer (the causal core), and ClassLayer."
conv.py,"Standard GNN layers (GinConv, GcnConv) and GnnNode implementations, primarily for OGB datasets."
mo_conv.py,"Modified GNN layers (MoGnnNode) supporting continuous node/edge features, used for datasets like MUTAG."
GCN_Conv.py,"Simple, independent GcnConv implementation."
causal.py,Implements the differentiable DegreeLayer for generating the counterfactual mask.
loss.py,"Contains custom loss functions: loss_rec, loss_match, loss_cls_2, and the key cf_consistency_loss."
codebook.py,"Defines the generate_codebook function, responsible for extracting and filtering representative subgraphs based on frequency, size, or complexity."
utils.py,"Contains utility functions for graph conversion (nx_to_pyg, pyg_to_nx), subgraph extraction (get_substructure), dataset splitting, and visualization (plot_multiclass_score_distributions)."
robustness.py,"Contains graph perturbation transforms (EdgeDropper, DeterministicEdgeDropper, NodeFeatureNoiseAdder) for robustness evaluation."

Script,Dataset,Domain,Training Feature,Robustness Test
train.py,MUTAG (TUDataset),Molecule,Cross-Validation,Visualization of top causal subgraphs
train_main.py,OGBG-Mol***,Molecule,Standard OGB split,AUC score
train_syn.py,SPMotif,Synthetic,Standard ID split (Bias: 0.7),Deterministic Edge Dropping
train_syn_perturb.py,SPMotif,Synthetic,Perturbed Training Set (Edge dropout applied to  50% of training graphs),Accuracy on clean test set
train_OOD_perturb.py,OGBG-Mol***,Molecule,Reduced Training Set (Subset of training data used to simulate low-resource/OOD training),AUC score on OGB test set
train_mnist.py,MNIST-75sp,Vision/Graph,Standard split,Testing with color noise perturbation


