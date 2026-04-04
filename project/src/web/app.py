"""FastAPI web application for image upload and detection."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from models.inference import run_inference

BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "src" / "web" / "static"
TEMPLATES_DIR = BASE_DIR / "src" / "web" / "templates"
MODEL_WEIGHTS = BASE_DIR / "models" / "yolo_transformer.pt"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="YOLOv12 + Transformer Detection")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": None, "error": None},
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(...)) -> HTMLResponse:
    if not MODEL_WEIGHTS.exists():
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"result": None, "error": "Model not found. Please run training first."},
        )

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    file_name = f"{uuid4().hex}{suffix}"
    upload_path = UPLOAD_DIR / file_name

    content = await image.read()
    upload_path.write_bytes(content)

    result_path = RESULT_DIR / f"result_{file_name}"
    result = run_inference(upload_path, MODEL_WEIGHTS, result_path)

    web_result = {
        "label": result["label"],
        "confidence": f"{result['confidence']:.2f}",
        "bbox": result["bbox"],
        "image_url": f"/static/results/{result_path.name}",
    }

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": web_result, "error": None},
    )
