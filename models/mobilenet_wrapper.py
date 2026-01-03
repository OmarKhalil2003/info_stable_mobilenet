import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, List


class MobileNetWrapper(nn.Module):
    """
    Wrapper around pretrained MobileNet (V2 / V3) that:
    - Runs fully on GPU
    - Exposes intermediate representations
    - Supports forward hooks (non-invasive)
    - Is compatible with adversarial training and certification
    """

    def __init__(
        self,
        version: str = "v2",
        pretrained: bool = True,
        device: str = "cuda",
        hook_layers: List[str] = None,
        num_classes: int = 1000,
    ):
        super().__init__()

        assert version in ["v2", "v3"], "version must be 'v2' or 'v3'"
        self.version = version
        self.device = device

        # -----------------------------
        # Load Pretrained MobileNet
        # -----------------------------
        if version == "v2":
            self.model = models.mobilenet_v2(pretrained=pretrained)
            feature_dim = 1280
        else:
            self.model = models.mobilenet_v3_large(pretrained=pretrained)
            feature_dim = 960

        # -----------------------------
        # Replace Classifier (Optional)
        # -----------------------------
        if num_classes != 1000:
            self.model.classifier[-1] = nn.Linear(feature_dim, num_classes)

        self.model.to(self.device)
        self.model.eval()

        # -----------------------------
        # Hook Configuration
        # -----------------------------
        if hook_layers is None:
            self.hook_layers = [
                "features.0",        # early conv (low-level)
                "features.6",        # mid bottleneck
                "features.12",       # deep bottleneck
            ]
        else:
            self.hook_layers = hook_layers

        # Storage for activations
        self.activations: Dict[str, torch.Tensor] = {}

        # Register hooks
        self._register_hooks()

    # ============================================================
    # Hook Registration
    # ============================================================
    def _register_hooks(self):
        """
        Registers forward hooks on selected layers.
        Hooks are NON-DESTRUCTIVE and GPU-safe.
        """

        for name, module in self.model.named_modules():
            if name in self.hook_layers:
                module.register_forward_hook(self._hook_fn(name))

    def _hook_fn(self, layer_name: str):
        def hook(module, input, output):
            # Detach but keep on GPU
            self.activations[layer_name] = output.detach()
        return hook

    # ============================================================
    # Forward Pass
    # ============================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.
        Activations are collected automatically by hooks.
        """
        self.activations.clear()
        return self.model(x)

    # ============================================================
    # Access Internal Representations
    # ============================================================
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """
        Returns dictionary:
        { layer_name : activation_tensor }
        """
        return self.activations

    # ============================================================
    # Utility: Enable / Disable Training Mode
    # ============================================================
    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()
