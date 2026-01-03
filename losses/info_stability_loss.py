import torch
import torch.nn as nn
from typing import Dict, List


class InformationStabilityLoss(nn.Module):
    """
    Computes per-sample information stability loss by
    penalizing representation drift under small perturbations.

    This loss is:
    - Attack-agnostic
    - Per-sample
    - Layer-wise
    - GPU-efficient
    """

    def __init__(
        self,
        layers: List[str],
        epsilon: float = 8 / 255,
        p_norm: str = "inf",
        reduction: str = "mean",
    ):
        """
        Args:
            layers: list of layer names to monitor
            epsilon: perturbation radius
            p_norm: 'inf' or 'l2'
            reduction: 'mean' or 'sum'
        """
        super().__init__()

        assert p_norm in ["inf", "l2"]
        assert reduction in ["mean", "sum"]

        self.layers = layers
        self.epsilon = epsilon
        self.p_norm = p_norm
        self.reduction = reduction

    # ============================================================
    # Perturbation Generator (Attack-Agnostic)
    # ============================================================
    def _generate_perturbation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates random norm-bounded perturbation.
        No gradients involved.
        """
        if self.p_norm == "inf":
            delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.randn_like(x)
            delta = delta / (delta.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-8)
            delta = delta * self.epsilon

        return delta.detach()

    # ============================================================
    # Forward
    # ============================================================
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes information stability loss.

        Args:
            model: MobileNetWrapper
            x: input batch (B, C, H, W)

        Returns:
            scalar loss
        """

        # --------------------------------------------------------
        # Clean Forward
        # --------------------------------------------------------
        with torch.no_grad():
            _ = model(x)
            clean_acts = {
                k: v.clone()
                for k, v in model.get_activations().items()
            }

        # --------------------------------------------------------
        # Perturbed Forward
        # --------------------------------------------------------
        delta = self._generate_perturbation(x)
        x_perturbed = torch.clamp(x + delta, 0.0, 1.0)

        _ = model(x_perturbed)
        pert_acts: Dict[str, torch.Tensor] = model.get_activations()

        # --------------------------------------------------------
        # Compute Layer-wise Drift
        # --------------------------------------------------------
        total_loss = 0.0

        for layer in self.layers:
            z_clean = clean_acts[layer]
            z_pert = pert_acts[layer]

            # Flatten per sample
            z_clean = z_clean.view(z_clean.size(0), -1)
            z_pert = z_pert.view(z_pert.size(0), -1)

            # L2 drift per sample
            drift = torch.norm(z_clean - z_pert, p=2, dim=1)

            if self.reduction == "mean":
                total_loss += drift.mean()
            else:
                total_loss += drift.sum()

        return total_loss
