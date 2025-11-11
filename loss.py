import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_cls(causal_y, y):
    # Use BCELoss
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(causal_y.sigmoid(), y.sigmoid())
    return loss


def loss_cls_2(causal_y, y):
    """
    Computes the classification loss.

    Args:
        causal_y (torch.Tensor): The raw logits output of the model, shape [batch_size, num_classes].
        y (torch.Tensor): The true class indices, shape [batch_size], type Long.
    """
    # For multi-class classification, CrossEntropyLoss should be used.
    # It internally integrates log_softmax and NLLLoss, thus requires raw logits.
    criterion = nn.CrossEntropyLoss()

    # Pass the raw output (logits) and Long-type labels directly
    loss = criterion(causal_y, y)
    return loss


def loss_rec(A_ori_list, A_rec_list):
    """
    Computes the reconstruction loss between original and reconstructed adjacency matrices.

    Args:
        A_ori_list (list of torch.Tensor): List of original (pooled) adjacency matrices.
        A_rec_list (list of torch.Tensor): List of reconstructed adjacency matrices.

    Returns:
        torch.Tensor: The summed reconstruction loss.
    """
    loss_r = 0
    for tensor1, tensor2 in zip(A_ori_list, A_rec_list):
        # Ensure the shapes of the two tensors are consistent
        if tensor1.shape == tensor2.shape:
            # Choose a loss function, e.g., L2 loss or L1 loss
            loss = F.mse_loss(tensor1, tensor2)  # Use Mean Squared Error (MSE) as loss
            loss_r += loss
        else:
            raise ValueError(f"Tensor shapes are inconsistent: {tensor1.shape} vs {tensor2.shape}")

    return loss_r


def loss_dag(causal_matrix):
    """
    Computes the DAG constraint loss (non-cyclicity loss) using the trace of matrix exponential.

    Args:
        causal_matrix (torch.Tensor): The causal influence matrix, shape (d, d).

    Returns:
        torch.Tensor: The scalar DAG loss.
    """
    # causal_matrix shape is (d, d)
    d = causal_matrix.shape[0]
    # Compute matrix exponential e^(W o W)
    # o denotes element-wise product
    m_exp = torch.matrix_exp(causal_matrix * causal_matrix)
    # Compute the trace and subtract dimension d
    loss = torch.trace(m_exp) - d
    return loss


def loss_match(codebook, x_pool):
    """
    Computes the matching loss between the quantized codebook vectors and the pooled graph features.

    Args:
        codebook (torch.Tensor): The selected codebook vectors (z_nodes in model.py), shape [num_pooled_nodes, dim].
        x_pool (torch.Tensor): The pooled graph features (data_pool_x), shape [num_pooled_nodes, dim].

    Returns:
        torch.Tensor: The scalar matching loss (MSE).
    """
    loss = F.mse_loss(codebook, x_pool)
    return loss


def cf_consistency_loss(counter_logits: torch.Tensor,
                        beta: float = 1.0,
                        is_logits: bool = True) -> torch.Tensor:
    """
    Counterfactual Consistency Regularization Loss L_cf

    Args:
        counter_logits (torch.Tensor): The model's output (logits) on counterfactual representations.
        beta (float): Weight for the entropy term.
        is_logits (bool): Whether the input is raw logits.

    Returns:
        torch.Tensor: The calculated scalar loss.
    """
    if counter_logits is None or counter_logits.numel() == 0:
        # Check if the tensor is on CUDA before accessing device
        device = counter_logits.device if counter_logits is not None else 'cpu'
        return torch.tensor(0.0, device=device)

    # 1. Get probabilities p
    if is_logits:
        if counter_logits.size(1) == 1:  # Binary classification
            p = torch.sigmoid(counter_logits)
        else:  # Multi-class classification
            p = F.softmax(counter_logits, dim=1)
    else:
        p = counter_logits

    B, C = p.shape[0], p.shape[1]

    # 2. Compute the average probability across samples p_bar
    p_bar = p.mean(dim=0, keepdim=True)

    # 3. Variance term L_var
    L_var = ((p - p_bar) ** 2).mean()

    # 4. Entropy term L_ent
    eps = 1e-9
    if C == 1:  # Explicit entropy for binary classification
        p_pos = p_bar.squeeze()
        p_neg = 1.0 - p_pos
        L_ent = -(p_pos * torch.log(p_pos + eps) + p_neg * torch.log(p_neg + eps))
    else:  # Multi-class entropy
        L_ent = -(p_bar * torch.log(p_bar + eps)).sum()

    # 5. Final loss
    L_cf = L_var + beta * L_ent
    return L_cf