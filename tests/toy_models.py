"""Tiny differentiable detector and detector-wrapper stubs shared by the v2 tests."""

from __future__ import annotations

import numpy as np
import torch

from src.detector import Detection


class ToyDetectionModel(torch.nn.Module):
    """One fixed box [16,16,48,48] of COCO class 0 whose score depends on its pixels."""

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
        )
        with torch.no_grad():
            for module in self.features:
                if isinstance(module, torch.nn.Conv2d):
                    module.weight.fill_(1.0 / module.weight.numel())

    def forward(self, images):
        x = torch.stack(images, dim=0) if isinstance(images, list) else images
        feature_map = self.features(x)
        outputs = []
        for batch_index in range(x.shape[0]):
            response = feature_map[batch_index, 0]
            score = torch.sigmoid(response[16:48, 16:48].mean() * 10.0)
            outputs.append(
                {
                    "boxes": torch.tensor([[16.0, 16.0, 48.0, 48.0]], device=x.device),
                    "labels": torch.tensor([1], device=x.device),
                    "scores": score.unsqueeze(0),
                }
            )
        return outputs


def toy_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = (rng.random((64, 64, 3)) * 80).astype(np.uint8)
    image[16:48, 16:48] = 255
    return image


def make_detection(bbox, class_id: int = 0, confidence: float = 0.9, detection_id: int = 0) -> Detection:
    x1, y1, x2, y2 = bbox
    return Detection(
        bbox=list(bbox),
        class_id=class_id,
        class_name="person" if class_id == 0 else "car",
        confidence=confidence,
        area=(x2 - x1) * (y2 - y1),
        relative_size="medium",
        detection_id=detection_id,
    )


class ToyDetectorWrapper:
    """DetectorWrapper stand-in: one true box and one spurious box."""

    model_name = "toy"

    def __init__(self) -> None:
        self.model = ToyDetectionModel().eval()

    def detect(self, image: np.ndarray, nms_thresh=None, conf_thresh=None) -> list[Detection]:
        del image, nms_thresh, conf_thresh
        return [
            make_detection([16, 16, 48, 48], class_id=0, confidence=0.9, detection_id=0),
            make_detection([0, 48, 14, 63], class_id=2, confidence=0.4, detection_id=1),
        ]

    def get_model(self):
        return self.model

    def get_target_layer(self):
        return self.model.features[0]

    def unload_model(self) -> None:
        return None
