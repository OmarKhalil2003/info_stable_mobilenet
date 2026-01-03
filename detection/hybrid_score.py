import torch
from typing import List

from detection.per_sample_score import PerSampleStabilityScorer
from detection.directional_score import DirectionalStabilityScorer


class HybridStabilityScorer:
    """
    Hybrid information stability:
    combines random and directional stability scores.
    """

    def __init__(
        self,
        model,
        layers: List[str],
        epsilon: float = 8 / 255,
        num_random: int = 5,
        alpha: float = 0.5,
        device: str = "cuda",
    ):
        self.alpha = alpha
        self.device = device

        self.random_scorer = PerSampleStabilityScorer(
            model=model,
            layers=layers,
            epsilon=epsilon,
            num_samples=num_random,
            device=device,
        )

        self.directional_scorer = DirectionalStabilityScorer(
            model=model,
            layers=layers,
            epsilon=epsilon,
            device=device,
        )

    @staticmethod
    def _normalize(scores: torch.Tensor) -> torch.Tensor:
        """
        Z-score normalization (per batch).
        """
        mean = scores.mean()
        std = scores.std() + 1e-8
        return (scores - mean) / std

    #@torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:

        """
        Computes hybrid stability score.

        Args:
            x: input tensor (B, C, H, W)

        Returns:
            hybrid scores (B,)
        """

        # Random stability
        s_rand = self.random_scorer.score(x)

        # Directional stability (needs grad internally)
        s_dir = self.directional_scorer.score(x)

        # Normalize
        s_rand_n = self._normalize(s_rand)
        s_dir_n = self._normalize(s_dir)

        # Hybrid score
        s_hybrid = self.alpha * s_rand_n + (1.0 - self.alpha) * s_dir_n

        return s_hybrid
