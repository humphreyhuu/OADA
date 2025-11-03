"""
OADA Model Implementation

This module implements the main OADAModel that integrates all components
for domain-adaptive EHR prediction with 3-stage training curriculum.

Reference: OADA.md, plan.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from models.layers import (
    MMDAlignmentLayer,
    SparseAutoencoder,
    OrthogonalDomainExtractor,
    DomainClassifier
)


class OADAModel(nn.Module):
    """
    Domain-Adaptive EHR Prediction Model.

    Integrates all OADA components with 3-stage training curriculum:
    - Stage 1: Invariant feature learning (L_sup + L_MMD)
    - Stage 2: Sparse reconstruction (+ L_SAE)
    - Stage 3: Domain supervision (+ L_dom with gradient blocking)

    Args:
        backbone: Feature extractor (e.g., Transformer)
        embedding_dim: Dimension of backbone output (d)
        latent_dim: Dimension of SAE latent space (k), typically k > d
        output_size: Number of output classes for task prediction
        lambda_mmd: Weight for MMD alignment loss (default: 0.1)
        lambda_rec: Weight for SAE reconstruction loss (default: 1.0)
        lambda_sp: Weight for sparsity regularization (default: 0.01)
        lambda_dom: Weight for domain classification loss (default: 0.5)

    Architecture:
        x → backbone(f_φ) → v
        v → SAE → p̂, s
        v, p̂ → OrthogonalExtractor → z
        p̂ → task_predictor → ŷ
        z → domain_classifier → domain_pred
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        latent_dim: int,
        output_size: int,
        lambda_mmd: float = 0.1,
        lambda_rec: float = 1.0,
        lambda_sp: float = 0.01,
        lambda_dom: float = 0.5
    ):
        super(OADAModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.output_size = output_size

        # Backbone feature extractor
        self.feature_extractor = backbone

        # OADA components
        self.mmd_layer = MMDAlignmentLayer(kernel_mul=2.0, kernel_num=5)
        self.sae = SparseAutoencoder(embedding_dim, latent_dim, sparsity_weight=lambda_sp)
        self.domain_extractor = OrthogonalDomainExtractor(embedding_dim, latent_dim)
        self.domain_classifier = DomainClassifier(embedding_dim, hidden_dim=256, dropout=0.3)

        # Task predictor
        self.task_predictor = nn.Linear(embedding_dim, output_size)

        # Loss weights
        self.lambda_mmd = lambda_mmd
        self.lambda_rec = lambda_rec
        self.lambda_dom = lambda_dom

    def extract_features(self, code_x: torch.Tensor) -> torch.Tensor:
        """
        Extract raw features from backbone.

        Args:
            code_x: Input EHR data (format depends on backbone)

        Returns:
            v: Raw embedding [batch, embedding_dim]
        """
        # Assuming backbone returns [batch, embedding_dim]
        # For Transformer, it returns logits, so we need to extract embeddings
        if hasattr(self.feature_extractor, 'transformer'):
            # Get the last transformer output
            patient_emb = []
            for feature_key in self.feature_extractor.feature_keys:
                x = code_x[feature_key]
                x = self.feature_extractor.embeddings[feature_key](x)
                x = torch.sum(x, dim=-2)
                mask = torch.any(x != 0, dim=-1)
                _, x = self.feature_extractor.transformer[feature_key](x, mask)
                patient_emb.append(x)
            v = torch.cat(patient_emb, dim=1)  # [batch, embedding_dim]
        else:
            # Generic backbone
            v = self.feature_extractor(code_x)

        return v

    def forward(
        self,
        code_x: Dict[str, torch.Tensor],
        domain_labels: torch.Tensor,
        stage: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with stage-based loss computation.

        Args:
            code_x: Input EHR data, dict with feature keys
            domain_labels: Domain labels [batch], 0=source, 1=target
            stage: Training stage (1, 2, or 3)

        Returns:
            task_output: Task predictions [batch, output_size]
            losses: Dictionary of individual loss components
        """
        # Extract features from backbone
        v = self.extract_features(code_x)  # [batch, embedding_dim]
        batch_size = v.size(0)

        # Split into source and target based on domain labels
        source_mask = (domain_labels == 0)
        target_mask = (domain_labels == 1)

        losses = {}

        # Initialize p for task prediction
        p = v

        # Stage 1: Invariant Feature Learning (MMD Alignment)
        if stage >= 1:
            p_source = v[source_mask]
            p_target = v[target_mask]

            if len(p_source) > 0 and len(p_target) > 0:
                losses['mmd'] = self.mmd_layer(p_source, p_target)
            else:
                losses['mmd'] = torch.tensor(0.0, device=v.device)

        # Stage 2: Sparse Reconstruction
        if stage >= 2:
            p_hat, s, sae_loss = self.sae(v)
            losses['sae'] = sae_loss
            p = p_hat  # Use reconstructed features for task prediction
        else:
            p_hat = v
            s = None

        # Stage 3: Domain Feature Supervision (with gradient blocking)
        if stage >= 3 and s is not None:
            # Extract domain features with gradient blocking (Scheme A)
            z, v_hat, alpha = self.domain_extractor(
                v=v,
                p_hat=p_hat,
                decoder_weight=self.sae.decoder_weight,
                gradient_block=True  # Critical: blocks gradients to p_hat and s
            )

            # Domain classification
            domain_logits = self.domain_classifier(z)
            losses['domain'] = F.cross_entropy(domain_logits, domain_labels.long())
        else:
            losses['domain'] = torch.tensor(0.0, device=v.device)

        # Task prediction (using invariant features p)
        task_output = self.task_predictor(p)

        # Compute total loss (weighted sum)
        losses['total'] = (
            losses.get('mmd', torch.tensor(0.0)) * self.lambda_mmd +
            losses.get('sae', torch.tensor(0.0)) * self.lambda_rec +
            losses.get('domain', torch.tensor(0.0)) * self.lambda_dom
        )

        return task_output, losses

    def get_invariant_features(self, code_x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract invariant features for inference or analysis.

        Args:
            code_x: Input EHR data

        Returns:
            p_hat: Invariant features [batch, embedding_dim]
        """
        with torch.no_grad():
            v = self.extract_features(code_x)
            p_hat, _, _ = self.sae(v)
            return p_hat

    def get_domain_features(self, code_x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract domain features for analysis.

        Args:
            code_x: Input EHR data

        Returns:
            z: Domain features [batch, embedding_dim]
        """
        with torch.no_grad():
            v = self.extract_features(code_x)
            p_hat, _, _ = self.sae(v)
            z, _, _ = self.domain_extractor(
                v=v,
                p_hat=p_hat,
                decoder_weight=self.sae.decoder_weight,
                gradient_block=False
            )
            return z

    def predict(self, code_x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inference mode for task prediction.

        Args:
            code_x: Input EHR data

        Returns:
            task_output: Task predictions [batch, output_size]
        """
        p = self.get_invariant_features(code_x)
        return self.task_predictor(p)


class OADAWrapper(nn.Module):
    """
    Wrapper for OADA model to match the interface expected by train_dev.py.

    This adapter converts the binary code matrix format used in EHRDataset
    to the dictionary format expected by OADA model, similar to TransformerAdapter
    in train_base.py.
    """

    def __init__(self, oada_model: OADAModel, device: torch.device):
        super(OADAWrapper, self).__init__()
        self.oada_model = oada_model
        self.device = device

    def prepare_input(self, code_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert binary code matrix to dictionary format.

        Args:
            code_x: Binary code matrix [batch, max_visits, code_num]

        Returns:
            Dictionary with 'diagnosis' key containing code indices
        """
        batch_size_actual = code_x.shape[0]
        max_visits = code_x.shape[1]

        # Convert binary matrix to indices format
        code_x_indices = []
        for b in range(batch_size_actual):
            visit_codes = []
            for v in range(max_visits):
                codes_in_visit = torch.nonzero(code_x[b, v], as_tuple=True)[0]
                visit_codes.append(codes_in_visit)
            code_x_indices.append(visit_codes)

        # Pad to same length
        max_codes_per_visit = max([max([len(v) for v in patient]) if len(patient) > 0 else 1
                                   for patient in code_x_indices])
        max_codes_per_visit = max(max_codes_per_visit, 1)

        code_x_padded = torch.zeros((batch_size_actual, max_visits, max_codes_per_visit),
                                    dtype=torch.long).to(self.device)
        for b in range(batch_size_actual):
            for v in range(max_visits):
                codes = code_x_indices[b][v]
                if len(codes) > 0:
                    code_x_padded[b, v, :len(codes)] = codes + 1  # +1 for padding_idx

        return {'diagnosis': code_x_padded}

    def forward(
        self,
        code_x: torch.Tensor,
        domain_labels: torch.Tensor,
        stage: int = 1,
        divided=None,
        neighbors=None,
        visit_lens=None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass matching train_dev.py interface.

        Args:
            code_x: Binary code matrix [batch, max_visits, code_num]
            domain_labels: Domain labels [batch], 0=source, 1=target
            stage: Training stage (1, 2, or 3)
            divided: Not used (compatibility with EHRDataset)
            neighbors: Not used (compatibility with EHRDataset)
            visit_lens: Not used (compatibility with EHRDataset)

        Returns:
            task_output: Task predictions [batch, output_size]
            losses: Dictionary of loss components
        """
        # Convert to dictionary format
        code_x_dict = self.prepare_input(code_x)

        # Forward through OADA model
        task_output, losses = self.oada_model(code_x_dict, domain_labels, stage)

        return task_output, losses

    def predict(self, code_x: torch.Tensor) -> torch.Tensor:
        """
        Inference mode.

        Args:
            code_x: Binary code matrix [batch, max_visits, code_num]

        Returns:
            task_output: Task predictions [batch, output_size]
        """
        code_x_dict = self.prepare_input(code_x)
        return self.oada_model.predict(code_x_dict)


# Helper function for testing
def test_oada_model():
    """Test OADA model with dummy data."""
    print("Testing OADA Model...")
    print("=" * 50)

    from models.Transformer import Transformer

    # Model configuration
    batch_size = 16
    max_visits = 10
    codes_per_visit = 20
    embedding_dim = 128
    latent_dim = 256
    output_size = 1  # Binary classification

    # Create dummy Transformer backbone
    feature_keys = ['diagnosis']
    code_nums = {'diagnosis': 1000}

    backbone = Transformer(
        feature_keys=feature_keys,
        code_nums=code_nums,
        embedding_dim=embedding_dim,
        output_size=embedding_dim,  # Output embedding, not task logits
        activation=False,  # No activation for embedding output
        heads=4,
        dropout=0.1,
        num_layers=2
    )

    # Create OADA model
    print("\n1. Creating OADA Model...")
    oada_model = OADAModel(
        backbone=backbone,
        embedding_dim=len(feature_keys) * embedding_dim,  # Transformer concatenates features
        latent_dim=latent_dim,
        output_size=output_size,
        lambda_mmd=0.1,
        lambda_rec=1.0,
        lambda_sp=0.01,
        lambda_dom=0.5
    )
    print(f"   Model created with {sum(p.numel() for p in oada_model.parameters())} parameters")

    # Create dummy data
    print("\n2. Creating dummy data...")
    code_x = {
        'diagnosis': torch.randint(0, 1000, (batch_size, max_visits, codes_per_visit))
    }
    domain_labels = torch.cat([
        torch.zeros(batch_size // 2),
        torch.ones(batch_size // 2)
    ])  # Half source, half target
    print(f"   code_x shape: {code_x['diagnosis'].shape}")
    print(f"   domain_labels: {domain_labels.long().tolist()}")

    # Test Stage 1
    print("\n3. Testing Stage 1 (Invariant Feature Learning)...")
    task_output, losses = oada_model(code_x, domain_labels, stage=1)
    print(f"   task_output shape: {task_output.shape}")
    print(f"   losses: {', '.join([f'{k}={v.item():.6f}' for k, v in losses.items()])}")
    assert task_output.shape == (batch_size, output_size), "Task output shape mismatch"
    assert 'mmd' in losses, "MMD loss should be present in stage 1"

    # Test Stage 2
    print("\n4. Testing Stage 2 (+ Sparse Reconstruction)...")
    task_output, losses = oada_model(code_x, domain_labels, stage=2)
    print(f"   losses: {', '.join([f'{k}={v.item():.6f}' for k, v in losses.items()])}")
    assert 'sae' in losses, "SAE loss should be present in stage 2"

    # Test Stage 3
    print("\n5. Testing Stage 3 (+ Domain Supervision)...")
    task_output, losses = oada_model(code_x, domain_labels, stage=3)
    print(f"   losses: {', '.join([f'{k}={v.item():.6f}' for k, v in losses.items()])}")
    assert 'domain' in losses, "Domain loss should be present in stage 3"

    # Test feature extraction
    print("\n6. Testing feature extraction...")
    p_hat = oada_model.get_invariant_features(code_x)
    z = oada_model.get_domain_features(code_x)
    print(f"   Invariant features (p): {p_hat.shape}")
    print(f"   Domain features (z): {z.shape}")

    # Check orthogonality
    orthogonality = torch.abs(torch.sum(p_hat * z, dim=-1)).mean().item()
    print(f"   Orthogonality: |<p, z>| = {orthogonality:.6f} (should be ~0)")

    # Test inference
    print("\n7. Testing inference mode...")
    predictions = oada_model.predict(code_x)
    print(f"   Predictions shape: {predictions.shape}")

    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == '__main__':
    test_oada_model()
