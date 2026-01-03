import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

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


def accuracy_coverage(scores, correct, reject_fracs):
    """
    scores: instability scores (higher = more risky)
    correct: boolean array (True if prediction correct)
    reject_fracs: list of rejection rates
    """

    order = np.argsort(scores)[::-1]  # most unstable first
    results = []

    N = len(scores)

    for r in reject_fracs:
        k = int(r * N)
        keep_idx = order[k:]

        acc = correct[keep_idx].mean()
        coverage = len(keep_idx) / N

        results.append((coverage, acc))

    return results

@torch.no_grad()
def collect_scores_and_preds(model, scorer, loader, device):
    scores = []
    correct = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1)

        s = scorer.score(x)

        scores.append(s.cpu())
        correct.append((preds == y).cpu())

    return (
        torch.cat(scores).numpy(),
        torch.cat(correct).numpy()
    )

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
    # Stability Scorer (Random)
    # -----------------------------
    scorer = PerSampleStabilityScorer(
        model=model,
        layers=["features.0", "features.6"],
        epsilon=8 / 255,
        num_samples=5,
        device=device,
    )

    # -----------------------------
    # Data (CIFAR-10 test)
    # -----------------------------
    transform = transforms.Compose([transforms.ToTensor()])

    testset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )

    loader = get_loader(testset)

    print("Collecting stability scores and predictions...")
    scores, correct = collect_scores_and_preds(model, scorer, loader, device)

    # -----------------------------
    # Rejection Analysis
    # -----------------------------
    reject_fracs = [0.0, 0.05, 0.1, 0.2, 0.3]

    results = accuracy_coverage(scores, correct, reject_fracs)

    print("\nAccuracy–Coverage Tradeoff:")
    for (cov, acc), r in zip(results, reject_fracs):
        print(f"Reject {int(r*100):2d}% | Coverage: {cov:.2f} | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
