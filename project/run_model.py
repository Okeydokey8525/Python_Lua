"""Run inference using a trained YOLOv12 + Transformer model."""

from __future__ import annotations

import argparse
from pathlib import Path

from models.inference import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv12 + Transformer inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--weights",
        default="models/yolo_transformer.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        default="models/example_inference.jpg",
        help="Path to save result image",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_inference(
        image_path=Path(args.image),
        weights_path=Path(args.weights),
        output_path=Path(args.output),
    )
    print("Inference complete")
    print(result)
