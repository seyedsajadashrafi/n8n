import torch

from models.mobilenet_v1 import mobilenet_v1


def test_mobilenet_v1_forward_shape():
    model = mobilenet_v1(num_classes=1000, width_mult=1.0)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 1000), f"unexpected output shape: {out.shape}"


def test_mobilenet_v1_small_width_and_grad():
    model = mobilenet_v1(num_classes=10, width_mult=0.5)
    model.train()

    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 10)

    # basic backward check
    loss = out.mean()
    loss.backward()

    # ensure at least one parameter has gradient
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), "no gradients computed"
