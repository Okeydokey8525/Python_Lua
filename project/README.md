# YOLOv12 + Transformer Image Detection Web App

Production-style, modular Python project for training, inference, and web deployment of a lightweight YOLOv12 + Transformer-inspired detector.

## Features

- Training pipeline (`train.py`) for dataset in `data/`
- Inference CLI (`run_model.py`) for single-image prediction
- FastAPI web app (`run_web.py`) with upload + result view
- Dummy sample dataset included under `data/images` and `data/labels`
- Basic tests with `pytest`

## Project Structure

```text
project/
├── data/
│   ├── images/
│   └── labels/
├── models/
│   ├── yolo_transformer.py
│   ├── train_model.py
│   ├── inference.py
├── src/
│   └── web/
│       ├── app.py
│       ├── templates/
│       │   └── index.html
│       └── static/
├── tests/
├── run_web.py
├── train.py
├── run_model.py
├── requirements.txt
├── AGENTS.md
└── README.md
```

## Installation

```bash
cd project
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

Outputs checkpoint:

- `models/yolo_transformer.pt`

## Inference

```bash
python run_model.py --image data/images/sample_1.ppm
```

Optional flags:

- `--weights models/yolo_transformer.pt`
- `--output models/example_inference.jpg`

## Web App

```bash
python run_web.py
```

Open:

- `http://127.0.0.1:8000`

Upload an image and the app will display the predicted image with bounding boxes and labels.

## Testing

```bash
pytest
```

## Improvement Suggestions

- Replace lightweight model with real YOLOv12 backbone/neck/head implementation.
- Add multi-object matching loss and NMS post-processing.
- Add model/version config management with Hydra or Pydantic Settings.
- Add CI pipeline for linting, unit tests, and Docker image build.
