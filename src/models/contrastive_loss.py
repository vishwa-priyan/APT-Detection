import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologicalContrastiveLoss(nn.Module):
    """
    Topological Contrastive Loss (TCL) with InfoNCE for robust GNN representation
    learning under edge/graph perturbations and evasion attempts.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        """
        z_anchor: [N, D] embeddings from original graph
        z_positive: [N, D] embeddings from perturbed/augmented graph
        """
        # Normalize
        z_a = F.normalize(z_anchor, p=2, dim=-1)
        z_p = F.normalize(z_positive, p=2, dim=-1)

        # Cosine similarity matrix: [N, N]
        sim_matrix = torch.matmul(z_a, z_p.T) / self.temperature
        
        # Targets: diagonal elements are positive pairs
        targets = torch.arange(z_anchor.size(0), device=z_anchor.device)
        
        loss = F.cross_entropy(sim_matrix, targets)
        return loss

class MaskedFeatureRegularizer(nn.Module):
    """
    Masked Representation Learning (MAGIC) regularizer that forces the GNN
    to reconstruct masked flow features from topological neighborhood context.
    """
    def __init__(self, feat_dim: int, hidden_dim: int = 64, mask_ratio: float = 0.15):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )

    def forward(self, original_feats: torch.Tensor, penultimate_embeds: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs original flow features from penultimate embeddings.
        """
        reconstructed = self.reconstructor(penultimate_embeds)
        recon_loss = F.mse_loss(reconstructed, original_feats)
        return recon_loss
