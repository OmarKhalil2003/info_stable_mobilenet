import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

def load_id_ood():
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10 = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )

    svhn = datasets.SVHN(
        root="./data",
        split="test",
        download=False,
        transform=transform
    )

    return cifar10, svhn

@torch.no_grad()
def collect_all(model, scorer, loader, device, is_id: bool):
    scores = []
    correct = []
    id_flags = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1)

        s = scorer.score(x)

        scores.append(s.cpu())
        correct.append((preds == y).cpu())
        id_flags.append(torch.full((x.size(0),), is_id))

    return (
        torch.cat(scores).numpy(),
        torch.cat(correct).numpy(),
        torch.cat(id_flags).numpy()
    )

def rejection_analysis(scores, correct, id_flags, reject_fracs):
    order = np.argsort(scores)[::-1]  # most unstable first
    N = len(scores)

    print("\nOOD Selective Rejection Results:")
    for r in reject_fracs:
        k = int(r * N)
        keep = order[k:]

        kept_correct = correct[keep]
        kept_id = id_flags[keep]

        # ID accuracy on remaining samples
        id_mask = kept_id == 1
        if id_mask.sum() > 0:
            id_acc = kept_correct[id_mask].mean()
        else:
            id_acc = float("nan")

        # OOD rejection rate
        rejected = order[:k]
        ood_rej = (id_flags[rejected] == 0).mean() if k > 0 else 0.0

        print(
            f"Reject {int(r*100):2d}% | "
            f"ID Acc: {id_acc:.4f} | "
            f"OOD Rejected: {ood_rej:.2f}"
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
    # Stability Scorer
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
    cifar10, svhn = load_id_ood()

    id_loader = get_loader(cifar10)
    ood_loader = get_loader(svhn)

    print("Collecting ID samples...")
    s_id, c_id, f_id = collect_all(model, scorer, id_loader, device, is_id=True)

    print("Collecting OOD samples...")
    s_ood, c_ood, f_ood = collect_all(model, scorer, ood_loader, device, is_id=False)

    # -----------------------------
    # Merge
    # -----------------------------
    scores = np.concatenate([s_id, s_ood])
    correct = np.concatenate([c_id, c_ood])
    id_flags = np.concatenate([f_id, f_ood])

    # -----------------------------
    # Rejection
    # -----------------------------
    reject_fracs = [0.0, 0.05, 0.1, 0.2, 0.3]
    rejection_analysis(scores, correct, id_flags, reject_fracs)

if __name__ == "__main__":
    main()
