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
        if optimize and (torch.cuda.is_available() and torch.cuda.device_count() > 0):
            # TorchScript + fp16 : supprime l'essentiel du coût CPU de lancement des noyaux CUDA
            self.model.inference(compile=True, dtype=torch.float16)
        self.confidence = confidence
        self.names = {i: name for i, name in COCO_CLASSES.items() if name in classes}
        self.class_ids = list(self.names)

    def __call__(self, frame: torch.Tensor) -> sv.Detections:
        """frame : tenseur (3, H, W) RGB dans [0, 1], déjà sur le GPU."""
        det = self.model.predict(frame, threshold=self.confidence, include_source_image=False)
        return with_names(det[np.isin(det.class_id, self.class_ids)], self.names)


class YOLODetector:
    """Ultralytics YOLO (AGPL-3.0 : licence commerciale nécessaire pour un produit fermé)."""

    def __init__(self, model: str, classes: list[str], confidence: float, optimize: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.confidence = confidence
        self.names = {i: name for i, name in self.model.names.items() if name in classes}
        self.class_ids = list(self.names)

    def __call__(self, frame: torch.Tensor) -> sv.Detections:
        bgr = (frame.permute(1, 2, 0).flip(-1) * 255).byte().cpu().numpy()
        result = self.model(bgr, conf=self.confidence,
                            classes=self.class_ids, verbose=False)[0]
        return with_names(sv.Detections.from_ultralytics(result), self.names)


def with_names(det: sv.Detections, names: dict[int, str]) -> sv.Detections:
    """Ajoute le nom de classe COCO (person, car, truck…) à chaque détection."""
    det.data["class_name"] = np.array([names.get(int(c), "") for c in det.class_id], dtype=object)
    return det


_CACHE: dict = {}


def build_detector(cfg: dict):
    """Les modèles restent en mémoire entre deux analyses (interface de démo) : seul le seuil change."""
    backends = {"rfdetr": RFDETRDetector, "yolo": YOLODetector}
    key = (cfg["backend"], cfg["model"], tuple(cfg["classes"]), cfg.get("optimize", True))
    if key not in _CACHE:
        _CACHE[key] = backends[cfg["backend"]](cfg["model"], cfg["classes"], cfg["confidence"],
                                               cfg.get("optimize", True))
    detector = _CACHE[key]
    detector.confidence = cfg["confidence"]
    return detector
