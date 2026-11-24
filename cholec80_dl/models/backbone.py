"""ImageNet backbone wrapper that works with any torchvision classification net."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvmodels


_HEAD_ATTRS = ("head", "classifier", "fc", "heads")


def _strip_head(net: nn.Module) -> nn.Module:
    """Replace the classification head with :class:`nn.Identity`.

    For ``alexnet``/``vgg`` only the last ``Linear`` of ``classifier`` is
    replaced, keeping the 4096-d penultimate features that EndoNet used.
    """
    for attr in _HEAD_ATTRS:
        head = getattr(net, attr, None)
        if not isinstance(head, nn.Module):
            continue
        if isinstance(head, nn.Sequential) and len(head) and isinstance(head[-1], nn.Linear):
            head[-1] = nn.Identity()
        else:
            setattr(net, attr, nn.Identity())
        return net
    raise ValueError(f"cannot locate a classification head in {type(net).__name__}")


class Backbone(nn.Module):
    """Torchvision model -> feature extractor (``out_dim`` exposed)."""

    def __init__(self, name: str = "resnet50", pretrained: bool = True, input_size: int = 224):
        super().__init__()
        self.name = name
        try:
            net = tvmodels.get_model(name, weights="DEFAULT" if pretrained else None)
        except Exception as exc:  # pragma: no cover - network / cache issues
            raise RuntimeError(
                f"could not build torchvision model {name!r} (pretrained={pretrained}): {exc}\n"
                "If the machine has no internet access, pass --no-pretrained."
            ) from exc

        self.net = _strip_head(net)
        self.out_dim = self._infer_out_dim(input_size)

    def _infer_out_dim(self, input_size: int) -> int:
        was_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            out = self.net(dummy)
        self.net.train(was_training)
        if out.dim() > 2:
            out = out.mean(dim=(-2, -1))
        return int(out.flatten(1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net(x)
        if feats.dim() > 2:                     # (B,C,H,W) -> global average pooling
            feats = feats.mean(dim=(-2, -1))
        return feats.flatten(1)

    def set_trainable(self, flag: bool) -> None:
        for p in self.net.parameters():
            p.requires_grad = flag


def build_backbone(name: str, pretrained: bool = True, input_size: int = 224) -> Backbone:
    return Backbone(name=name, pretrained=pretrained, input_size=input_size)
