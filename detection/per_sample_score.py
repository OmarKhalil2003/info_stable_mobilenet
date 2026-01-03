import torch
import torch.nn as nn
from typing import List, Dict


class PerSampleStabilityScorer:
    """
    Computes per-sample information stability scores
    using internal representations of a trained model.
    """

    def __init__(
        self,
        model: nn.Module,
        layers: List[str],
        epsilon: float = 8 / 255,
        num_samples: int = 5,
        p_norm: str = "inf",
        device: str = "cuda",
    ):
        """
        Args:
            model: trained MobileNetWrapper
            layers: list of layer names to monitor
            epsilon: perturbation radius
            num_samples: number of random perturbations
            p_norm: 'inf' or 'l2'
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.layers = layers
        self.epsilon = epsilon
        self.num_samples = num_samples
        self.p_norm = p_norm
        self.device = device

        self.model.set_eval()

    # --------------------------------------------------------
    # Perturbation Generator
    # --------------------------------------------------------
    def _sample_delta(self, x: torch.Tensor) -> torch.Tensor:
        if self.p_norm == "inf":
            delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.randn_like(x)
            delta = delta / (
                delta.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-8
            )
            delta = delta * self.epsilon

        return delta

    # --------------------------------------------------------
    # Core Scoring Function
    # --------------------------------------------------------
    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes stability score for each sample in batch.

        Args:
            x: input tensor (B, C, H, W)

        Returns:
            scores: tensor of shape (B,)
        """

        x = x.to(self.device)
        batch_size = x.size(0)

        # Clean forward
        _ = self.model(x)
        clean_acts: Dict[str, torch.Tensor] = {
            k: v.clone()
            for k, v in self.model.get_activations().items()
            if k in self.layers
        }

        scores = torch.zeros(batch_size, device=self.device)

        # Monte Carlo perturbations
        for _ in range(self.num_samples):
            delta = self._sample_delta(x)
            x_pert = torch.clamp(x + delta, 0.0, 1.0)

            _ = self.model(x_pert)
            pert_acts = self.model.get_activations()

            for layer in self.layers:
                z_clean = clean_acts[layer]
                z_pert = pert_acts[layer]

                z_clean = z_clean.view(batch_size, -1)
                z_pert = z_pert.view(batch_size, -1)

                drift = torch.norm(z_clean - z_pert, p=2, dim=1)
                scores += drift

        scores = scores / self.num_samples
        return scores
