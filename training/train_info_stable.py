import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.mobilenet_wrapper import MobileNetWrapper
from losses.info_stability_loss import InformationStabilityLoss


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Training Loop
# ============================================================
def train(
    model,
    loader,
    optimizer,
    ce_loss,
    info_loss,
    device,
    stability_interval = 4,
    lambda_info=0.1,
):
    scaler = torch.cuda.amp.GradScaler()
    model.set_train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step, (x, y) in enumerate(tqdm(loader, desc="Training")):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward (clean)
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss_ce = ce_loss(logits, y)

        # Info Stability Loss
        if info_loss is not None and step % stability_interval == 0:
            loss_info = info_loss(model, x)
        else:
            loss_info = torch.tensor(0.0, device=device)

        loss = loss_ce + lambda_info * loss_info
        #loss = loss_ce
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples

    return avg_loss, acc


# ============================================================
# Main
# ============================================================
def main():
    # -----------------------------
    # Configuration
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16   # safe for 6GB GPU
    epochs = 10
    lr = 1e-4
    lambda_info = 0.1
    seed = 42

    set_seed(seed)

    # -----------------------------
    # Data (CIFAR-10 for now)
    # -----------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = MobileNetWrapper(
        version="v2",
        pretrained=True,
        device=device,
        hook_layers=["features.0", "features.6"],
        num_classes=10
    )

    # -----------------------------
    # Losses
    # -----------------------------
    ce_loss = nn.CrossEntropyLoss()
    info_loss = InformationStabilityLoss(
        layers=["features.0", "features.6"],
        epsilon=8/255
    )

    # -----------------------------
    # Optimizer
    # -----------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    # -----------------------------
    # Training
    # -----------------------------
    stability_start_epoch = int(0.8 * epochs)  # last 20%

    for epoch in range(epochs):
        use_info = epoch >= stability_start_epoch

        loss, acc = train(
            model,
            train_loader,
            optimizer,
            ce_loss,
            info_loss if use_info else None,
            device,
            lambda_info
        )

        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"Loss: {loss:.4f} | Acc: {acc:.4f} | "
            #"Info: OFF (training)"
            f"Info: {'ON' if use_info else 'OFF'}"
        )
    # -----------------------------
    # Save Model
    # -----------------------------
    torch.save(model.state_dict(), "mobilenet_info_stable.pth")
    print("Model saved.")


if __name__ == "__main__":
    main()
