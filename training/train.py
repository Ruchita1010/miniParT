from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import MiniParT

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.npz"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Training configuration
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 3
CLASS_NAMES = ["Hbb", "Hcc", "QCD"]


def load_data():
    """Load the prepared dataset and create PyTorch DataLoaders."""

    data = np.load(DATASET_PATH)

    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.long)

    X_val = torch.tensor(data["X_val"], dtype=torch.float32)
    y_val = torch.tensor(data["y_val"], dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader, X_train.shape[-1], y_train


def evaluate(model, data_loader, criterion, device):
    """Evaluate the model on the validation dataset."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_y.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train():
    """Train MiniParT and save the best validation checkpoint."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, input_dim, y_train = load_data()

    print("Input features:", input_dim)
    print("Training events:", len(y_train))
    print("Training batches:", len(train_loader))

    # Calculate class weights from the training labels.
    class_counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
    class_weights = len(y_train) / (NUM_CLASSES * class_counts)

    class_weights = class_weights.to(device)

    print("Class counts:", class_counts.tolist())
    print("Class weights:", class_weights.tolist())

    # Initialize the model.
    model = MiniParT(input_dim=input_dim).to(device)

    print("\nModel architecture:")
    print(model)

    # Loss function and optimizer.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass.
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backpropagation and parameter update.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_y.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        # Evaluate on the validation dataset.
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        history_path = ARTIFACTS_DIR / "training_history.npz"

        np.savez_compressed(
            history_path,
            train_loss=np.asarray(history["train_loss"], dtype=np.float64),
            train_accuracy=np.asarray(history["train_accuracy"], dtype=np.float64),
            val_loss=np.asarray(history["val_loss"], dtype=np.float64),
            val_accuracy=np.asarray(history["val_accuracy"], dtype=np.float64),
        )

        print("Training history saved to:", history_path)

        # Save the checkpoint with the lowest validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "input_dim": input_dim,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "best_val_loss": best_val_loss,
            }

            torch.save(
                checkpoint,
                ARTIFACTS_DIR / "minipart_best.pt",
            )

            print("  Saved new best model checkpoint.")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Best model saved to:", ARTIFACTS_DIR / "minipart_best.pt")


if __name__ == "__main__":
    train()
