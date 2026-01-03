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
from adv_attacks.autoattack_eval import autoattack_generate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Load trained model
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
    # Stability scorer
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
    # Clean scores & threshold
    # -----------------------------
    clean_scores = scorer.score(images)
    tau = compute_threshold(clean_scores, alpha=0.05)

    # -----------------------------
    # AutoAttack
    # -----------------------------
    adv_auto = autoattack_generate(
        model,
        images,
        labels,
        device=device
    )

    adv_scores = scorer.score(adv_auto)

    # -----------------------------
    # Metrics
    # -----------------------------
    fpr, tpr, auc = compute_roc_metrics(clean_scores, adv_scores)

    print("AutoAttack AUROC:", auc)
    print("AutoAttack TPR@5%FPR:", tpr_at_fpr(fpr, tpr, 0.05))

    detect_rate = (adv_scores > tau).float().mean()
    print("AutoAttack detection rate @5% clean FPR:", detect_rate.item())


if __name__ == "__main__":
    main()
