import torch
import torch.nn as nn
from typing import List, Dict


class DirectionalStabilityScorer:
    """
    Directional information stability using logit-gradient directions.
    """

    def __init__(
        self,
        model: nn.Module,
        layers: List[str],
        epsilon: float = 8 / 255,
        device: str = "cuda",
    ):
        self.model = model
        self.layers = layers
        self.epsilon = epsilon
        self.device = device

        self.model.set_eval()

    def score(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).detach()
        x.requires_grad_(True)
        batch_size = x.size(0)

        # Forward (with grad)
        logits = self.model(x)

        proxy = logits.norm(p=2, dim=1).sum()

        grad = torch.autograd.grad(
            proxy,
            x,
            retain_graph=False,
            create_graph=False,
        )[0]

        delta = self.epsilon * grad.sign()
        x_pert = torch.clamp(x + delta, 0.0, 1.0)

        with torch.no_grad():
            _ = self.model(x)
            clean_acts = {
                k: v
                for k, v in self.model.get_activations().items()
                if k in self.layers
            }

            _ = self.model(x_pert)
            pert_acts = self.model.get_activations()

        scores = torch.zeros(batch_size, device=self.device)

        for layer in self.layers:
            z_clean = clean_acts[layer].view(batch_size, -1)
            z_pert = pert_acts[layer].view(batch_size, -1)
            scores += torch.norm(z_clean - z_pert, p=2, dim=1)

        # IMPORTANT: cleanup
        x = x.detach()

        return scores

