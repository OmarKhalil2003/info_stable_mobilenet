import torch
from autoattack.autoattack import AutoAttack


def autoattack_generate(
    wrapper_model,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8 / 255,
    device: str = "cuda",
):
    """
    Generates AutoAttack adversarial examples.
    """

    wrapper_model.set_eval()

    base_model = wrapper_model.model
    base_model.eval()

    adversary = AutoAttack(
        base_model,
        norm="Linf",
        eps=epsilon,
        version="standard",
        device=device,
    )

    adv_images = adversary.run_standard_evaluation(
        images, labels, bs=images.size(0)
    )

    return adv_images
