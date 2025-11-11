# counterfactual_mask_generation.py
import torch
import torch.nn.functional as F
from utils import min_max_normalize  # Assuming utils is available


class DegreeLayer(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature  # Softmax temperature parameter, controls approximation level

    def forward(self, diagonal_matrix):
        """
        Calculates a differentiable approximation of a binary matrix based on the diagonal
        elements of the input matrix, using a soft-threshold derived from a soft-KS statistic.

        Args:
            diagonal_matrix: A square matrix (N x N), where only diagonal elements are relevant.
        Returns:
            binary_matrix: A differentiable binary-like matrix (1s and 0s)
        """
        # Extract diagonal elements
        diag_elements = torch.diag(diagonal_matrix)

        # Differentiable KS statistic calculation (Approximation)
        sorted_values, _ = torch.sort(diag_elements)
        n = len(sorted_values)

        # Generate all possible candidate thresholds (midpoints of sorted values)
        candidate_thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2

        # Compute the (approximate) KS statistic for each threshold
        # This part of the KS statistic approximation logic seems non-standard for soft-thresholding
        # but is preserved from the original code for functional fidelity.
        cdf_low = torch.arange(1, n, device=diag_elements.device).float() / n
        cdf_high = 1 - cdf_low
        ks_stats = torch.abs(cdf_low - cdf_high)

        # Select the threshold with the largest KS statistic (via Softmax approximation of argmax)
        weights = F.softmax(ks_stats / self.temperature, dim=0)

        # Ensure candidate_thresholds is not empty before weighted sum
        if candidate_thresholds.numel() > 0:
            threshold = (weights * candidate_thresholds).sum()
        else:
            # Handle the case where n=1 (only one element, no midpoints)
            threshold = sorted_values[0]

            # Apply soft-thresholding: elements greater than the threshold are set to 0.
        # This is a hard-threshold in the current implementation, which makes the gradient flow zero
        # when the condition is met. I will keep the original implementation's behavior.
        # A truly differentiable alternative would be:
        # binary_diag = 1.0 - torch.sigmoid((diag_elements - threshold) * 100)

        # Original logic: Hard-thresholding (non-differentiable at the cutoff point)
        diag_elements[diag_elements > threshold] = 0

        binary_matrix = torch.diag_embed(diag_elements)
        return binary_matrix


if __name__ == "__main__":
    torch.manual_seed(42)

    # Generate random diagonal matrix (5x5)
    N = 3
    diag_values = torch.randn(N, requires_grad=True)
    original_matrix = torch.diag_embed(diag_values)
    # Initialize differentiable KS module
    ks_module = DegreeLayer(temperature=0.01)

    # Forward pass
    binary_matrix = ks_module(original_matrix)

    # Print results
    print("Original Diagonal Matrix:\n", original_matrix)
    print("Binarized Matrix (Approximate):\n", binary_matrix)

    # Simulate loss and backpropagation
    loss = binary_matrix.sum()
    loss.backward()
    print("Gradients of Diagonal Elements:", diag_values.grad)