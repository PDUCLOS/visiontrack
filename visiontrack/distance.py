"""Estimation de distance caméra → personne.

Trois méthodes :
- pinhole : géométrie (taille moyenne d'une personne / hauteur de la boîte en pixels)
- depth   : carte de profondeur métrique (modèles « Metric » de Depth Anything)
- fusion  : profondeur relative (Apache 2.0) recalée en mètres grâce à la méthode pinhole.
            Plus stable que pinhole seul quand une personne est assise, coupée ou masquée.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import supervision as sv
import torch


class DepthEstimator:
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, model_id: str, device: str = "cuda", input_size: int = 392):
        from transformers import AutoModelForDepthEstimation

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id, dtype=self.dtype)
        self.model.to(device).eval()
        self.input_size = input_size  # petit côté de l'image envoyée au modèle (multiple de 14)
        self.mean, self.std = self.MEAN.to(device), self.STD.to(device)
        # Les modèles « Metric » sortent des mètres ; les autres une profondeur inverse relative.
        self.metric = "metric" in model_id.lower()

    @torch.inference_mode()
    def __call__(self, frame: torch.Tensor) -> np.ndarray:
        """frame : tenseur (3, H, W) RGB dans [0, 1], déjà sur le GPU. Renvoie une carte basse résolution."""
        _, h, w = frame.shape
        s = self.input_size / min(h, w)
        size = (max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14))
        x = torch.nn.functional.interpolate(frame[None], size=size, mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).to(self.dtype)
        return self.model(pixel_values=x).predicted_depth[0].float().cpu().numpy()

    def colorize(self, depth: np.ndarray) -> np.ndarray:
        """Carte couleur BGR : clair = proche."""
        d = -depth if self.metric else depth
        lo, hi = np.percentile(d[::4, ::4], (2, 98))
        norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def sample_boxes(depth: np.ndarray, boxes: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """Valeur médiane de profondeur au centre du torse de chaque boîte (carte à n'importe quelle résolution)."""
    w, h = frame_size
    sx, sy = depth.shape[1] / w, depth.shape[0] / h
    boxes = boxes * np.array([sx, sy, sx, sy])
    values = []
    for x1, y1, x2, y2 in boxes.astype(int):
        bw, bh = x2 - x1, y2 - y1
        patch = depth[y1 + bh // 5: y1 + bh // 2, x1 + bw // 3: x2 - bw // 3]
        values.append(float(np.median(patch)) if patch.size else np.nan)
    return np.array(values)


class DistanceEstimator:
    def __init__(self, cfg: dict, frame_size: tuple[int, int], device: str = "cuda"):
        self.method = cfg["method"]
        self.person_h = cfg["person_height_m"]
        self.alpha = cfg["smoothing"]
        w, h = frame_size
        self.frame_size = frame_size
        self.focal_px = (w / 2) / math.tan(math.radians(cfg["horizontal_fov_deg"]) / 2)
        self.depth = (DepthEstimator(cfg["depth_model"], device, cfg.get("depth_input_size", 392))
                      if self.method in ("depth", "fusion") else None)
        self.every_n = cfg.get("depth_every_n_frames", 1)
        self.scale = None          # facteur relatif → mètres (fusion), lissé dans le temps
        self.history: dict[int, float] = {}
        self.depth_map = None
        self.depth_color = None
        self._frame = 0

    def colorized(self) -> np.ndarray | None:
        if self.depth is None or self.depth_map is None:
            return None
        if self.depth_color is None:
            self.depth_color = self.depth.colorize(self.depth_map)
        return self.depth_color

    def pinhole(self, boxes: np.ndarray) -> np.ndarray:
        heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1)
        return self.focal_px * self.person_h / heights

    def _fully_visible(self, boxes: np.ndarray) -> np.ndarray:
        """Personne entière : boîte loin des bords et proportions d'une silhouette debout.
        Une boîte trop « carrée » signale une personne masquée (table, voiture…) : la hauteur ne vaut plus 1,70 m."""
        w, h = self.frame_size
        m = 4
        inside = (boxes[:, 1] > m) & (boxes[:, 3] < h - m) & (boxes[:, 0] > m) & (boxes[:, 2] < w - m)
        ratio = (boxes[:, 3] - boxes[:, 1]) / np.maximum(boxes[:, 2] - boxes[:, 0], 1)
        return inside & (ratio > 2.0)

    def __call__(self, frame: torch.Tensor, det: sv.Detections) -> np.ndarray:
        if self.depth is not None and self._frame % self.every_n == 0:
            self.depth_map = self.depth(frame)
            self.depth_color = None  # la carte couleur est recalculée à la demande
        self._frame += 1
        if self.method == "none" or len(det) == 0:
            return np.full(len(det), np.nan)

        boxes = det.xyxy
        geo = self.pinhole(boxes)
        if self.method == "pinhole":
            raw = np.where(self._fully_visible(boxes), geo, np.nan)
        else:
            vals = sample_boxes(self.depth_map, boxes, self.frame_size)
            if self.depth.metric:
                raw = vals
            else:
                # profondeur relative ≈ inverse de la distance : d = k / v
                ok = self._fully_visible(boxes) & (vals > 0)
                if ok.any():
                    k = float(np.median(geo[ok] * vals[ok]))
                    self.scale = k if self.scale is None else 0.9 * self.scale + 0.1 * k
                raw = self.scale / np.maximum(vals, 1e-6) if self.scale else np.full(len(boxes), np.nan)
            if self.method == "fusion":
                # moyenne des deux quand la personne est entièrement visible
                full = self._fully_visible(boxes)
                raw = np.where(full, np.where(np.isfinite(raw), 0.5 * raw + 0.5 * geo, geo), raw)

        return self._smooth(det.tracker_id, raw)

    def _smooth(self, track_ids, raw: np.ndarray) -> np.ndarray:
        if track_ids is None:
            return raw
        out = raw.copy()
        for i, tid in enumerate(track_ids):
            prev = self.history.get(int(tid))
            if not np.isfinite(raw[i]):
                out[i] = prev if prev is not None else np.nan  # garde la dernière valeur connue
                continue
            if prev is not None:
                out[i] = self.alpha * prev + (1 - self.alpha) * raw[i]
            self.history[int(tid)] = out[i]
        return out
