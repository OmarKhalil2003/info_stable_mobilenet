import torch
import foolbox as fb


def fgsm_attack(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8 / 255,
    device: str = "cuda",
):
    """
    Generates FGSM adversarial examples.
    """
    model.eval()

    fmodel = fb.PyTorchModel(
        model.model,
        bounds=(0, 1),
        device=device
    )

    attack = fb.attacks.FGSM()
    adv_images, _, _ = attack(
        fmodel,
        images,
        labels,
        epsilons=epsilon
    )

    return adv_images
