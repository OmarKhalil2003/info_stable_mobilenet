import torch
import foolbox as fb


def pgd_attack(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8 / 255,
    steps: int = 40,
    step_size: float = 2 / 255,
    device: str = "cuda",
):
    """
    Generates PGD adversarial examples.
    """
    model.eval()

    fmodel = fb.PyTorchModel(
        model.model,
        bounds=(0, 1),
        device=device
    )

    attack = fb.attacks.LinfPGD(
        steps=steps,
        abs_stepsize=step_size
    )

    adv_images, _, _ = attack(
        fmodel,
        images,
        labels,
        epsilons=epsilon
    )

    return adv_images
