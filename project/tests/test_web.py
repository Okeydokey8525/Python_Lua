import pytest


def test_home_page() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    from fastapi.testclient import TestClient
    from src.web.app import app

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "YOLOv12 + Transformer" in response.text
