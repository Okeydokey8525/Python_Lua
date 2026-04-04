import pytest


def test_model_forward() -> None:
    torch = pytest.importorskip("torch")

    from models.yolo_transformer import create_model

    model = create_model(num_classes=2)
    model.eval()

    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy)

    assert "boxes" in output
    assert "logits" in output
    assert output["boxes"].shape == (1, 4)
    assert output["logits"].shape == (1, 2)
