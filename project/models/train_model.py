"""Training pipeline for YOLOv12 + Transformer detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.yolo_transformer import create_model


@dataclass
class TrainConfig:
    data_dir: Path
    output_path: Path
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-3
    image_size: int = 256
    num_classes: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DetectionDataset(Dataset):
    """Dataset reading image files with YOLO-style text labels."""

    def __init__(self, data_dir: Path, image_size: int = 256) -> None:
        self.images_dir = data_dir / "images"
        self.labels_dir = data_dir / "labels"
        self.image_size = image_size
        self.image_paths = sorted(self.images_dir.glob("*"))

        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_path = self.image_paths[idx]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)

        class_id, bbox = self._read_label(label_path)

        return {
            "image": image_tensor,
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
        }

    @staticmethod
    def _read_label(label_path: Path) -> tuple[int, list[float]]:
        if not label_path.exists():
            return 0, [0.5, 0.5, 0.2, 0.2]

        line = label_path.read_text().strip()
        parts = line.split()

        if len(parts) != 5:
            return 0, [0.5, 0.5, 0.2, 0.2]

        class_id = int(parts[0])
        bbox = [float(v) for v in parts[1:]]
        return class_id, bbox


def train_model(config: TrainConfig) -> Path:
    """Train model and save a checkpoint."""
    dataset = DetectionDataset(config.data_dir, image_size=config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = create_model(num_classes=config.num_classes).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    classification_loss = nn.CrossEntropyLoss()
    box_loss = nn.SmoothL1Loss()

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            images = batch["image"].to(config.device)
            targets_class = batch["class_id"].to(config.device)
            targets_bbox = batch["bbox"].to(config.device)

            outputs = model(images)
            loss_cls = classification_loss(outputs["logits"], targets_class)
            loss_box = box_loss(outputs["boxes"], targets_bbox)
            loss = loss_cls + loss_box

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1}/{config.epochs} - loss: {avg_loss:.4f}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": config.num_classes,
            "image_size": config.image_size,
        },
        config.output_path,
    )
    print(f"Model saved to: {config.output_path}")
    return config.output_path
