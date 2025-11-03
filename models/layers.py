"""
OADA Core Components

This module implements the core layers for the Domain-Adaptive EHR Prediction Framework:
- MMDAlignmentLayer: Maximum Mean Discrepancy for domain alignment
- SparseAutoencoder: Sparse representation learning for invariant features
- OrthogonalDomainExtractor: Orthogonal projection to isolate domain features
- DomainClassifier: Domain prediction from domain features

Reference: OADA.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MMDAlignmentLayer(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) alignment layer.

    Computes the MMD loss between source and target domain features using
    multi-kernel Gaussian kernels for robust domain alignment.

    Args:
        kernel_mul: Multiplicative factor for kernel bandwidth (default: 2.0)
        kernel_num: Number of kernels with different bandwidths (default: 5)

    Reference: Gretton et al. "A Kernel Two-Sample Test". JMLR 2012.
    """

    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5):
        super(MMDAlignmentLayer, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-kernel Gaussian kernel matrix.

        Args:
            source: Source domain features [n_source, d]
            target: Target domain features [n_target, d]

        Returns:
            Kernel matrix [n_source + n_target, n_source + n_target]
        """
        n_source = source.size(0)
        n_target = target.size(0)
        n_samples = n_source + n_target

        # Concatenate source and target
        total = torch.cat([source, target], dim=0)  # [n_samples, d]

        # Compute pairwise L2 distances
        total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1)
        total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1)
        L2_distance = ((total0 - total1) ** 2).sum(2)  # [n_samples, n_samples]

        # Compute bandwidth for kernels
        bandwidth = torch.sum(L2_distance.detach()) / (n_samples ** 2 - n_samples)
        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))

        # Multi-kernel
        kernel_val = [torch.exp(-L2_distance / (bandwidth * (self.kernel_mul ** i)))
                      for i in range(self.kernel_num)]

        return sum(kernel_val)  # [n_samples, n_samples]

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD loss between source and target distributions.

        Args:
            source: Source domain features [n_source, d]
            target: Target domain features [n_target, d]

        Returns:
            MMD loss (scalar)
        """
        n_source = source.size(0)
        n_target = target.size(0)

        if n_source == 0 or n_target == 0:
            return torch.tensor(0.0, device=source.device)

        # Compute kernel matrix
        kernels = self.gaussian_kernel(source, target)

        # Split kernel matrix into blocks
        XX = kernels[:n_source, :n_source]
        YY = kernels[n_source:, n_source:]
        XY = kernels[:n_source, n_source:]
        YX = kernels[n_source:, :n_source]

        # Compute MMD^2
        mmd_loss = (XX.sum() / (n_source ** 2) +
                    YY.sum() / (n_target ** 2) -
                    2 * XY.sum() / (n_source * n_target))

        return mmd_loss.clamp(min=0.0)  # Ensure non-negative


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for learning invariant feature representations.

    Learns a sparse latent code s that can reconstruct the invariant features p.
    Uses L1 regularization to encourage sparsity in the latent representation.

    Args:
        input_dim: Dimension of input features (d)
        latent_dim: Dimension of sparse latent code (k), typically k > d for overcomplete dictionary
        sparsity_weight: Weight for L1 sparsity regularization (λ_sp)

    Architecture:
        Encoder: Linear(d, k) + ReLU
        Decoder: W^T (tied with encoder weights transposed)
    """

    def __init__(self, input_dim: int, latent_dim: int, sparsity_weight: float = 0.01):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight

        # Encoder
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)

        # Decoder (tied weights with encoder for efficiency)
        # Weight matrix W: [input_dim, latent_dim]
        # Reconstruction: x_hat = s @ W^T where s is [batch, latent_dim]
        self.decoder_weight = nn.Parameter(torch.empty(input_dim, latent_dim))
        nn.init.xavier_uniform_(self.decoder_weight)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse latent code.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            s: Sparse latent code [batch, latent_dim]
        """
        return F.relu(self.encoder(x))

    def decode(self, s: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse latent code to reconstruction.

        Args:
            s: Sparse latent code [batch, latent_dim]

        Returns:
            x_hat: Reconstructed features [batch, input_dim]
        """
        # Compute: x_hat = s @ W^T + bias
        # s: [batch, latent_dim], W: [input_dim, latent_dim]
        # W^T: [latent_dim, input_dim]
        # Result: [batch, input_dim]
        return s @ self.decoder_weight.t() + self.decoder_bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode and decode with loss computation.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            x_hat: Reconstructed features [batch, input_dim]
            s: Sparse latent code [batch, latent_dim]
            loss: Total SAE loss (reconstruction + sparsity)
        """
        # Encode
        s = self.encode(x)

        # Decode
        x_hat = self.decode(s)

        # Compute losses
        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = torch.mean(torch.abs(s))

        total_loss = recon_loss + self.sparsity_weight * sparsity_loss

        return x_hat, s, total_loss


class OrthogonalDomainExtractor(nn.Module):
    """
    Orthogonal Domain Feature Extractor.

    Extracts domain-specific features z that are orthogonal to the invariant features p̂.
    Uses a dense encoder to project the raw embedding v to the SAE basis, then computes
    the orthogonal residual: z = v̂ - α * p̂

    Args:
        feature_dim: Dimension of input features (d)
        latent_dim: Dimension of SAE latent space (k)

    Key Equations:
        u = encoder_dense(v)         # Dense latent code
        v̂ = W^T @ u                  # Project to SAE basis
        α = <v̂, p̂> / ||p̂||²         # Projection coefficient
        z = v̂ - α * p̂                # Orthogonal residual
    """

    def __init__(self, feature_dim: int, latent_dim: int):
        super(OrthogonalDomainExtractor, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Dense encoder for raw embedding v
        self.dense_encoder = nn.Linear(feature_dim, latent_dim, bias=True)

    def forward(
        self,
        v: torch.Tensor,
        p_hat: torch.Tensor,
        decoder_weight: torch.Tensor,
        gradient_block: bool = False,
        eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract orthogonal domain features.

        Args:
            v: Raw embedding from feature extractor [batch, d]
            p_hat: Reconstructed invariant features [batch, d]
            decoder_weight: SAE decoder weight matrix W [d, k]
            gradient_block: Whether to block gradients to p_hat (Scheme A) [default: False]
            eps: Small constant for numerical stability [default: 1e-6]

        Returns:
            z: Domain features (orthogonal to p_hat) [batch, d]
            v_hat: Projection of v in SAE basis [batch, d]
            alpha: Projection coefficients [batch, 1]
        """
        # Encode v to dense latent code
        u = F.relu(self.dense_encoder(v))  # [batch, k]

        # Project u to SAE basis using decoder weights
        # decoder_weight: W is [d, k], we want u @ W^T
        # u: [batch, k], W^T: [k, d]
        # Result: [batch, d]
        v_hat = u @ decoder_weight.t()  # [batch, d]

        # Apply gradient blocking if in stage 3
        if gradient_block:
            p_hat = p_hat.detach()

        # Compute projection coefficient: α = <v̂, p̂> / ||p̂||²
        p_hat_norm2 = torch.sum(p_hat ** 2, dim=-1, keepdim=True) + eps  # [batch, 1]
        alpha = torch.sum(v_hat * p_hat, dim=-1, keepdim=True) / p_hat_norm2  # [batch, 1]

        # Compute orthogonal residual: z = v̂ - α * p̂
        z = v_hat - alpha * p_hat  # [batch, d]

        return z, v_hat, alpha


class DomainClassifier(nn.Module):
    """
    Domain Classifier for predicting domain labels from domain features.

    A multi-layer perceptron that classifies whether features come from
    source domain (0) or target domain (1). Used in training stage 3 to
    encourage domain-specific information to concentrate in z.

    Args:
        input_dim: Dimension of input domain features (d)
        hidden_dim: Dimension of hidden layers (default: 256)
        dropout: Dropout rate for regularization (default: 0.3)

    Architecture:
        Linear(d, hidden_dim) → ReLU → Dropout
        → Linear(hidden_dim, hidden_dim // 2) → ReLU → Dropout
        → Linear(hidden_dim // 2, 2)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super(DomainClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification: source vs target
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict domain label from domain features.

        Args:
            z: Domain features [batch, input_dim]

        Returns:
            logits: Domain classification logits [batch, 2]
        """
        return self.net(z)


# Helper function for testing
def test_layers():
    """Test all OADA layers with dummy data."""
    batch_size = 32
    feature_dim = 128
    latent_dim = 256

    print("Testing OADA Layers...")
    print("=" * 50)

    # Test MMD Alignment Layer
    print("\n1. Testing MMDAlignmentLayer...")
    mmd_layer = MMDAlignmentLayer()
    source = torch.randn(batch_size, feature_dim)
    target = torch.randn(batch_size, feature_dim)
    mmd_loss = mmd_layer(source, target)
    print(f"   Input: source {source.shape}, target {target.shape}")
    print(f"   Output: MMD loss = {mmd_loss.item():.6f}")
    assert mmd_loss.dim() == 0, "MMD loss should be scalar"
    print("   ✓ Passed")

    # Test Sparse Autoencoder
    print("\n2. Testing SparseAutoencoder...")
    sae = SparseAutoencoder(feature_dim, latent_dim, sparsity_weight=0.01)
    x = torch.randn(batch_size, feature_dim)
    x_hat, s, loss = sae(x)
    print(f"   Input: x {x.shape}")
    print(f"   Output: x_hat {x_hat.shape}, s {s.shape}, loss = {loss.item():.6f}")
    print(f"   Sparsity: {(s == 0).float().mean().item():.2%} zeros")
    assert x_hat.shape == x.shape, "Reconstruction shape mismatch"
    assert s.shape == (batch_size, latent_dim), "Latent code shape mismatch"
    print("   ✓ Passed")

    # Test Orthogonal Domain Extractor
    print("\n3. Testing OrthogonalDomainExtractor...")
    extractor = OrthogonalDomainExtractor(feature_dim, latent_dim)
    v = torch.randn(batch_size, feature_dim)
    p_hat = torch.randn(batch_size, feature_dim)
    decoder_weight = sae.decoder_weight

    # Without gradient blocking
    z, v_hat, alpha = extractor(v, p_hat, decoder_weight, gradient_block=False)
    print(f"   Input: v {v.shape}, p_hat {p_hat.shape}")
    print(f"   Output: z {z.shape}, v_hat {v_hat.shape}, alpha {alpha.shape}")

    # Check orthogonality: <z, p_hat> should be close to 0
    orthogonality = torch.abs(torch.sum(z * p_hat, dim=-1)).mean().item()
    print(f"   Orthogonality check: |<z, p_hat>| = {orthogonality:.6f} (should be ~0)")

    # With gradient blocking
    z_blocked, _, _ = extractor(v, p_hat, decoder_weight, gradient_block=True)
    print(f"   With gradient blocking: z {z_blocked.shape}")
    print("   ✓ Passed")

    # Test Domain Classifier
    print("\n4. Testing DomainClassifier...")
    classifier = DomainClassifier(feature_dim, hidden_dim=256, dropout=0.3)
    domain_logits = classifier(z)
    print(f"   Input: z {z.shape}")
    print(f"   Output: domain_logits {domain_logits.shape}")
    assert domain_logits.shape == (batch_size, 2), "Domain logits shape mismatch"
    print("   ✓ Passed")

    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == '__main__':
    test_layers()
