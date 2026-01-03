import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.mobilenet_wrapper import MobileNetWrapper
from detection.hybrid_score import HybridStabilityScorer
from detection.thresholds import (
    compute_threshold,
    compute_roc_metrics,
    tpr_at_fpr,
)
from adv_attacks.fgsm import fgsm_attack
from adv_attacks.pgd import pgd_attack


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Load Model
    # -----------------------------
    model = MobileNetWrapper(
        version="v2",
        pretrained=False,
        device=device,
        hook_layers=["features.0", "features.6"],
        num_classes=10
    )

    model.load_state_dict(torch.load("mobilenet_info_stable.pth"))
    model.set_eval()

    # -----------------------------
    # Data
    # -----------------------------
    transform = transforms.ToTensor()

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )

    loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    # -----------------------------
    # Stability Scorer
    # -----------------------------
    scorer = HybridStabilityScorer(
        model=model,
        layers=["features.0", "features.6"],
        epsilon=8 / 255,
        num_random=5,
        alpha=0.5,
        device=device
    )

    # -----------------------------
    # Clean Scores
    # -----------------------------
    clean_scores = scorer.score(images)
    images_fb = images.detach().clone()
    tau = compute_threshold(clean_scores, alpha=0.05)

    # -----------------------------
    # FGSM
    # -----------------------------
    adv_fgsm = fgsm_attack(model, images_fb, labels, device=device)
    fgsm_scores = scorer.score(adv_fgsm)

    fpr, tpr, auc = compute_roc_metrics(clean_scores, fgsm_scores)
    print("FGSM AUROC:", auc)
    print("FGSM TPR@5%FPR:", tpr_at_fpr(fpr, tpr, 0.05))

    # -----------------------------
    # PGD
    # -----------------------------
    adv_pgd = pgd_attack(model, images_fb, labels, device=device)
    pgd_scores = scorer.score(adv_pgd)

    fpr, tpr, auc = compute_roc_metrics(clean_scores, pgd_scores)
    print("PGD AUROC:", auc)
    print("PGD TPR@5%FPR:", tpr_at_fpr(fpr, tpr, 0.05))


if __name__ == "__main__":
    main()
