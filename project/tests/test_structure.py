from pathlib import Path


def test_required_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "models" / "yolo_transformer.py",
        root / "models" / "train_model.py",
        root / "models" / "inference.py",
        root / "src" / "web" / "app.py",
        root / "src" / "web" / "templates" / "index.html",
        root / "train.py",
        root / "run_model.py",
        root / "run_web.py",
        root / "requirements.txt",
    ]
    for file_path in required:
        assert file_path.exists(), f"Missing required file: {file_path}"
