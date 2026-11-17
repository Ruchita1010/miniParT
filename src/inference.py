from pathlib import Path

import joblib
import numpy as np
import torch

from src.model import MiniParT


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "minipart_best.pt"
SCALER_PATH = PROJECT_ROOT / "data" / "processed" / "scaler.joblib"

CLASS_NAMES = ["Hbb", "Hcc", "QCD"]


class MiniParTPredictor:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load the trained model checkpoint.
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=True,
        )

        self.model = MiniParT(
            input_dim=checkpoint["input_dim"],
            num_classes=checkpoint["num_classes"],
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load the scaler fitted during data preparation.
        self.scaler = joblib.load(SCALER_PATH)

        print("MiniParT predictor initialized.")
        print("Using device:", self.device)

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict the class of an event containing two jets.

        Expected input shape: (2, 10)
        """

        features = np.asarray(features, dtype=np.float64)

        if features.shape != (2, len(self.scaler.mean_)):
            raise ValueError(
                f"Expected input shape (2, {len(self.scaler.mean_)}), "
                f"received {features.shape}."
            )

        if not np.isfinite(features).all():
            raise ValueError("Input features must contain finite values.")

        # Apply the same scaling used during training.
        scaled_features = self.scaler.transform(features)

        # Add batch dimension: (2, 10) -> (1, 2, 10).
        input_tensor = torch.tensor(
            scaled_features,
            dtype=torch.float32,
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_index = int(probabilities.argmax().item())

        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "predicted_index": predicted_index,
            "probabilities": {
                name: float(probability)
                for name, probability in zip(
                    CLASS_NAMES,
                    probabilities.cpu().tolist(),
                )
            },
        }
