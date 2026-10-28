"""Rendu visuel : boîtes, ID, distances, trajectoires, zones, incrustation profondeur, FPS."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

LOOKUP = sv.ColorLookup.TRACK


class Renderer:
    def __init__(self, cfg: dict, zones: "Zones"):
        self.cfg = cfg
        self.zones = zones
        self.box = sv.BoxAnnotator(thickness=2, color_lookup=LOOKUP)
        self.label = sv.LabelAnnotator(text_scale=0.55, text_padding=6, color_lookup=LOOKUP,
                                       text_position=sv.Position.TOP_LEFT)
        self.trace = sv.TraceAnnotator(trace_length=cfg["trace_length"], thickness=2,
                                       color_lookup=LOOKUP, position=sv.Position.BOTTOM_CENTER)
        self.heatmap = sv.HeatMapAnnotator(position=sv.Position.BOTTOM_CENTER) if cfg["heatmap"] else None

    def __call__(self, frame_bgr: np.ndarray, det: sv.Detections, distances: np.ndarray,
                 depth_color: np.ndarray | None, fps: float | None) -> np.ndarray:
        out = frame_bgr.copy()
        if self.heatmap is not None:
            out = self.heatmap.annotate(out, det)
        out = self.zones.annotate(out)
        if self.cfg["traces"] and det.tracker_id is not None:
            out = self.trace.annotate(out, det)
        if self.cfg["boxes"]:
            out = self.box.annotate(out, det)
        labels = None
        if self.cfg["labels"] and det.tracker_id is not None:
            labels = [f"#{tid}  {d:.1f} m" if np.isfinite(d) else f"#{tid}"
                      for tid, d in zip(det.tracker_id, distances)]
            out = self.label.annotate(out, det, labels=labels)
        self._hud(out, len(det), fps)
        if self.side_by_side:
            return np.hstack([out, self._depth_panel(frame_bgr.shape, det, labels, depth_color)])
        if self.cfg["depth_inset"] and depth_color is not None:
            out = self._inset(out, depth_color)
        return out

    @property
    def side_by_side(self) -> bool:
        return self.cfg.get("layout", "inset") == "side_by_side"

    def output_size(self, w: int, h: int) -> tuple[int, int]:
        return (2 * w, h) if self.side_by_side else (w, h)

    def _depth_panel(self, shape, det: sv.Detections, labels, depth_color) -> np.ndarray:
        """Panneau droit du mode côte à côte : carte de profondeur pleine taille + boîtes et distances."""
        h, w = shape[:2]
        panel = cv2.resize(depth_color, (w, h)) if depth_color is not None else np.zeros(shape, np.uint8)
        panel = self.box.annotate(panel, det)
        if labels:
            panel = self.label.annotate(panel, det, labels=labels)
        _text_panel(panel, ["carte de profondeur (Depth Anything V2)"], anchor="bottom")
        return panel

    def _inset(self, frame: np.ndarray, depth_color: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        iw = int(w * self.cfg["inset_scale"])
        ih = int(iw * h / w)
        small = cv2.resize(depth_color, (iw, ih))
        x, y = w - iw - 16, h - ih - 16
        frame[y:y + ih, x:x + iw] = small
        cv2.rectangle(frame, (x, y), (x + iw, y + ih), (255, 255, 255), 1)
        cv2.putText(frame, "profondeur", (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _hud(self, frame: np.ndarray, n_people: int, fps: float | None):
        lines = [f"Personnes : {n_people}"]
        lines += self.zones.summary()
        if self.cfg["fps"] and fps:
            lines.append(f"{fps:.0f} FPS  -  100 % local")
        _text_panel(frame, lines)


def _text_panel(frame: np.ndarray, lines: list[str], anchor: str = "top"):
    """Bloc de texte lisible sur fond sombre semi-transparent."""
    font, scale, th, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 26
    width = max(cv2.getTextSize(t, font, scale, th)[0][0] for t in lines) + 24
    height = lh * len(lines) + 14
    x0 = 12
    y0 = 12 if anchor == "top" else frame.shape[0] - height - 12
    roi = frame[y0:y0 + height, x0:x0 + width]
    roi[:] = (roi * 0.35).astype(np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x0 + 12, y0 + 26 + i * lh), font, scale, (255, 255, 255), th, cv2.LINE_AA)
