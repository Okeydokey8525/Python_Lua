"""Entry point for training the YOLOv12 + Transformer detector."""

from __future__ import annotations

from pathlib import Path

from models.train_model import TrainConfig, train_model


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config = TrainConfig(
        data_dir=root / "data",
        output_path=root / "models" / "yolo_transformer.pt",
    )
    train_model(config)
