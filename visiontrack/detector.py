"""Détecteurs interchangeables. Chacun renvoie des sv.Detections filtrées sur les classes voulues."""

from __future__ import annotations

import numpy as np
import supervision as sv
import torch


class RFDETRDetector:
    """RF-DETR (Roboflow, Apache 2.0)."""

    SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
             "base": "RFDETRBase", "large": "RFDETRLarge"}

    def __init__(self, model: str, classes: list[str], confidence: float, optimize: bool = True):
        import rfdetr
        from rfdetr.assets.coco_classes import COCO_CLASSES

        self.model = getattr(rfdetr, self.SIZES[model])()
        if optimize and torch.cuda.is_available():
            # TorchScript + fp16 : supprime l'essentiel du coût CPU de lancement des noyaux CUDA
            self.model.inference(compile=True, dtype=torch.float16)
        self.confidence = confidence
        self.class_ids = [i for i, name in COCO_CLASSES.items() if name in classes]

    def __call__(self, frame: torch.Tensor) -> sv.Detections:
        """frame : tenseur (3, H, W) RGB dans [0, 1], déjà sur le GPU."""
        det = self.model.predict(frame, threshold=self.confidence, include_source_image=False)
        return det[np.isin(det.class_id, self.class_ids)]


class YOLODetector:
    """Ultralytics YOLO (AGPL-3.0 : licence commerciale nécessaire pour un produit fermé)."""

    def __init__(self, model: str, classes: list[str], confidence: float, optimize: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.confidence = confidence
        self.class_ids = [i for i, name in self.model.names.items() if name in classes]

    def __call__(self, frame: torch.Tensor) -> sv.Detections:
        bgr = (frame.permute(1, 2, 0).flip(-1) * 255).byte().cpu().numpy()
        result = self.model(bgr, conf=self.confidence,
                            classes=self.class_ids, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)


def build_detector(cfg: dict):
    backends = {"rfdetr": RFDETRDetector, "yolo": YOLODetector}
    return backends[cfg["backend"]](cfg["model"], cfg["classes"], cfg["confidence"], cfg.get("optimize", True))
