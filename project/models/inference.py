"""Inference utilities for YOLOv12 + Transformer detector."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from models.yolo_transformer import create_model


def load_model(weights_path: Path, device: str | None = None) -> tuple[torch.nn.Module, int]:
    """Load trained detector from checkpoint."""
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(weights_path, map_location=selected_device)
    num_classes = checkpoint.get("num_classes", 2)
    image_size = checkpoint.get("image_size", 256)

    model = create_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device)
    model.eval()

    return model, image_size


def _prepare_image(image_path: Path, image_size: int) -> tuple[np.ndarray, torch.Tensor]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Failed to read image at {image_path}")

    original = image_bgr.copy()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)

    return original, tensor


def run_inference(
    image_path: Path,
    weights_path: Path,
    output_path: Path,
    class_names: list[str] | None = None,
) -> dict[str, float | str | list[int]]:
    """Run prediction on one image and save visualization."""
    class_names = class_names or ["normal", "abnormal_region"]

    model, image_size = load_model(weights_path)
    device = next(model.parameters()).device

    image_bgr, image_tensor = _prepare_image(image_path, image_size)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    box = outputs["boxes"][0].cpu().numpy()
    logits = outputs["logits"][0]
    probs = torch.softmax(logits, dim=0)
    class_id = int(torch.argmax(probs).item())
    confidence = float(probs[class_id].item())

    height, width = image_bgr.shape[:2]
    x_center, y_center, box_w, box_h = box
    x1 = int((x_center - box_w / 2) * width)
    y1 = int((y_center - box_h / 2) * height)
    x2 = int((x_center + box_w / 2) * width)
    y2 = int((y_center + box_h / 2) * height)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)

    label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
    text = f"{label}: {confidence:.2f}"

    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image_bgr, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)

    return {
        "label": label,
        "confidence": confidence,
        "bbox": [x1, y1, x2, y2],
        "output_path": str(output_path),
    }
