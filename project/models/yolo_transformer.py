"""YOLOv12 + Transformer inspired lightweight detector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Configuration for the detector architecture."""

    num_classes: int = 2
    image_size: int = 256
    hidden_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2


class TinyBackbone(nn.Module):
    """Small CNN backbone to simulate YOLO feature extraction."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class YOLOTransformerDetector(nn.Module):
    """Detection model combining CNN features with Transformer encoding."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        self.backbone = TinyBackbone(hidden_dim=self.config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.transformer_heads,
            dim_feedforward=self.config.hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.transformer_layers,
        )

        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.bbox_predictor = nn.Linear(self.config.hidden_dim, 4)
        self.class_predictor = nn.Linear(self.config.hidden_dim, self.config.num_classes)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Returns:
            dict with keys:
                - boxes: normalized [x_center, y_center, w, h]
                - logits: classification logits
        """
        features = self.backbone(images)
        batch_size, channels, height, width = features.shape

        tokens = features.view(batch_size, channels, height * width).transpose(1, 2)
        encoded = self.transformer(tokens)

        pooled = encoded.mean(dim=1)
        refined = self.head(pooled)

        boxes = torch.sigmoid(self.bbox_predictor(refined))
        logits = self.class_predictor(refined)

        return {"boxes": boxes, "logits": logits}


def create_model(num_classes: int = 2) -> YOLOTransformerDetector:
    """Factory function used by training and inference modules."""
    return YOLOTransformerDetector(ModelConfig(num_classes=num_classes))
