from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from app.cnn_model import (
    AssignmentCNN,
    CLASSES,
    get_transform,
)


class ImageClassifier:
    def __init__(self):
        model_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "cifar10_cnn.pth"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                "Train the classifier first: "
                "uv run python train_classifier.py"
            )

        self.model = AssignmentCNN()

        state_dict = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = get_transform()

    def predict(self, image_bytes):
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
                inputs = self.transform(image).unsqueeze(0)

        except (UnidentifiedImageError, OSError, ValueError):
            raise ValueError(
                "Please upload a valid image file."
            ) from None

        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=1)

            score, index = probabilities.max(dim=1)

        return {
            "class_id": index.item(),
            "class_name": CLASSES[index.item()],
            "confidence": score.item(),
        }