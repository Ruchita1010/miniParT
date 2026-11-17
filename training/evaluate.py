from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import MiniParT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.npz"
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "minipart_best.pt"
EVALUATION_PATH = PROJECT_ROOT / "artifacts" / "evaluation_results.npz"

BATCH_SIZE = 32
CLASS_NAMES = ["Hbb", "Hcc", "QCD"]


def load_test_data():
    """Load the untouched test dataset."""

    data = np.load(DATASET_PATH)

    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)

    test_dataset = TensorDataset(X_test, y_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return test_loader, y_test.numpy()


def load_model(device):
    """Load the best trained model checkpoint."""

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model = MiniParT(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Loaded checkpoint from epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["best_val_loss"])

    return model


def evaluate(model, test_loader, y_true, device):
    """Generate predictions and calculate evaluation metrics."""

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * batch_y.size(0)
            total += batch_y.size(0)

            all_labels.extend(batch_y.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)
    y_prob = np.asarray(all_probabilities)

    test_loss = total_loss / total
    accuracy = accuracy_score(y_true, y_pred)

    print("\nTest Results")
    print("-" * 40)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    print("\nClassification Report")
    print("-" * 40)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\nConfusion Matrix")
    print("-" * 40)
    print(cm)

    print("\nMulticlass ROC-AUC")
    print("-" * 40)

    auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro",
        labels=[0, 1, 2],
    )
    print(f"Macro OvR ROC-AUC: {auc:.4f}")

    print("\nOne-vs-Rest ROC-AUC per class")
    print("-" * 40)

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_auc = roc_auc_score(
            (y_true == class_index).astype(int),
            y_prob[:, class_index],
        )
        print(f"{class_name}: {class_auc:.4f}")

    np.savez_compressed(
        EVALUATION_PATH,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        confusion_matrix=cm,
        test_loss=np.asarray(test_loss),
        test_accuracy=np.asarray(accuracy),
    )

    print("\nEvaluation results saved to:", EVALUATION_PATH)

    return y_true, y_pred, y_prob, cm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_loader, y_true = load_test_data()
    model = load_model(device)

    evaluate(model, test_loader, y_true, device)


if __name__ == "__main__":
    main()
