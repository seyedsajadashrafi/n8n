import torch
from models.resnet import resnet18, resnet50, BasicBlock, Bottleneck


def test_resnet18_forward():
    model = resnet18(num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


def test_resnet50_forward_and_backward():
    model = resnet50(num_classes=5)
    model.train()
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, 5)

    # simple backward check
    loss = out.sum()
    loss.backward()
    # ensure gradients exist for some parameter
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_block_expansion_values():
    assert BasicBlock.expansion == 1
    assert Bottleneck.expansion == 4


def test_model_has_expected_submodules():
    model = resnet18()
    # check top-level submodules exist
    assert hasattr(model, 'conv1')
    assert hasattr(model, 'bn1')
    assert hasattr(model, 'layer1')
    assert hasattr(model, 'layer2')
    assert hasattr(model, 'layer3')
    assert hasattr(model, 'layer4')
    assert hasattr(model, 'fc')
