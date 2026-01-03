import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

from models.mobilenet_wrapper import MobileNetWrapper
from detection.per_sample_score import PerSampleStabilityScorer


def get_loader(dataset, batch_size=64):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

def load_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    cifar10_test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )

    svhn_test = datasets.SVHN(
        root="./data",
        split="test",
        download=False,
        transform=transform
    )

    cifar100_test = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return cifar10_test, svhn_test, cifar100_test


def generate_gaussian_ood(num_samples, device):
    x = torch.randn(num_samples, 3, 32, 32, device=device)
    return torch.clamp(x, 0.0, 1.0)


@torch.no_grad()
def collect_scores(model, scorer, loader, device):
    scores = []

    for x, _ in loader:
        x = x.to(device)
        s = scorer.score(x)
        scores.append(s.cpu())

    return torch.cat(scores).numpy()


def evaluate_ood(id_scores, ood_scores, name):
    y_true = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores))
    ])

    y_scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_scores)
    print(f"OOD AUROC ({name}): {auroc:.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Model
    # -----------------------------
    model = MobileNetWrapper(
        version="v2",
        pretrained=False,
        device=device,
        hook_layers=["features.0", "features.6"],
        num_classes=10
    )

    model.load_state_dict(torch.load("mobilenet_info_stable.pth", map_location=device))
    model.set_eval()

    # -----------------------------
    # Stability Scorer (RANDOM)
    # -----------------------------
    scorer = PerSampleStabilityScorer(
        model=model,
        layers=["features.0", "features.6"],
        epsilon=8 / 255,
        num_samples=5,
        device=device,
    )

    # -----------------------------
    # Data
    # -----------------------------
    cifar10, svhn, cifar100 = load_datasets()

    id_loader = get_loader(cifar10)
    svhn_loader = get_loader(svhn)
    cifar100_loader = get_loader(cifar100)

    print("Computing ID scores (CIFAR-10)...")
    id_scores = collect_scores(model, scorer, id_loader, device)

    print("Computing SVHN OOD scores...")
    svhn_scores = collect_scores(model, scorer, svhn_loader, device)
    evaluate_ood(id_scores, svhn_scores, "SVHN")

    print("Computing CIFAR-100 OOD scores...")
    cifar100_scores = collect_scores(model, scorer, cifar100_loader, device)
    evaluate_ood(id_scores, cifar100_scores, "CIFAR-100")

    print("Computing Gaussian noise OOD scores...")
    gauss = generate_gaussian_ood(len(id_scores), device)
    gauss_scores = scorer.score(gauss).cpu().numpy()
    evaluate_ood(id_scores, gauss_scores, "Gaussian")


if __name__ == "__main__":
    main()
