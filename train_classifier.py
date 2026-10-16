import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from app.cnn_model import AssignmentCNN, get_transform


ROOT = Path(__file__).resolve().parent


def main():
    torch.manual_seed(42)

    # Use True for an initial setup check.
    # Change to False before final training and submission.
    QUICK_CHECK = False

    epochs = 1 if QUICK_CHECK else 10
    batch_size = 64

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    train_dataset = CIFAR10(
        root=str(ROOT / "data"),
        train=True,
        download=True,
        transform=get_transform(),
    )

    test_dataset = CIFAR10(
        root=str(ROOT / "data"),
        train=False,
        download=True,
        transform=get_transform(),
    )

    if QUICK_CHECK:
        train_dataset = Subset(
            train_dataset, range(512)
        )
        test_dataset = Subset(
            test_dataset, range(128)
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = AssignmentCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            count = labels.size(0)
            total_loss += loss.detach().item() * count
            total_correct += (
                logits.detach().argmax(dim=1) == labels
            ).sum().item()
            total_examples += count

        metrics = {
            "epoch": epoch,
            "train_loss": total_loss / total_examples,
            "train_accuracy_percent": (
                100 * total_correct / total_examples
            ),
        }
        history.append(metrics)

        print(
            f"Epoch {epoch}/{epochs}: "
            f"loss={metrics['train_loss']:.4f}, "
            f"accuracy="
            f"{metrics['train_accuracy_percent']:.2f}%"
        )

    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_examples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            count = labels.size(0)
            test_loss += loss.item() * count
            test_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            test_examples += count

    test_loss /= test_examples
    test_accuracy = 100 * test_correct / test_examples

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2f}%")

    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    # Store CPU weights so the API can load them on CPU.
    cpu_weights = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }

    torch.save(
        cpu_weights,
        model_dir / "cifar10_cnn.pth",
    )

    report = {
        "quick_check": QUICK_CHECK,
        "epochs": epochs,
        "train_examples": len(train_dataset),
        "test_examples": len(test_dataset),
        "history": history,
        "test_loss": test_loss,
        "test_accuracy_percent": test_accuracy,
    }

    (ROOT / "training_metrics.json").write_text(
        json.dumps(report, indent=2)
    )

    print("Saved models/cifar10_cnn.pth")

    if QUICK_CHECK:
        print(
            "Setup check only. Set QUICK_CHECK = False "
            "and rerun for final training."
        )


if __name__ == "__main__":
    main()