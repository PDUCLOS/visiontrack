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
        self.box = sv.BoxAnnotator(thickness=3, color_lookup=LOOKUP)
        self.label = sv.LabelAnnotator(text_scale=0.6, text_padding=5, color_lookup=LOOKUP,
                                       text_position=sv.Position.TOP_LEFT)
        self.trace = sv.TraceAnnotator(trace_length=cfg["trace_length"], thickness=4, smooth=True,
                                       color_lookup=LOOKUP, position=sv.Position.BOTTOM_CENTER)
        self.heatmap = sv.HeatMapAnnotator(position=sv.Position.BOTTOM_CENTER, opacity=0.55, radius=45) if cfg["heatmap"] else None
        self.near, self.mid = cfg.get("distance_thresholds_m", [3.0, 8.0])

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
            labels = [f"#{tid}" for tid in det.tracker_id]
            out = self.label.annotate(out, det, labels=labels)
            self._distances(out, det.xyxy, distances)
        self._hud(out, len(det), fps, show_legend=np.isfinite(distances).any())
        if self.side_by_side:
            panel = self._depth_panel(frame_bgr.shape, det, labels, depth_color)
            self._distances(panel, det.xyxy, distances)
            return np.hstack([out, panel])
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
        top = self.cfg.get("inset_position", "bottom_right") == "top_right"
        x, y = w - iw - 16, 16 if top else h - ih - 16
        frame[y:y + ih, x:x + iw] = small
        cv2.rectangle(frame, (x, y), (x + iw, y + ih), (255, 255, 255), 1)
        cv2.putText(frame, "profondeur", (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _color(self, d: float) -> tuple[int, int, int]:
        """Couleur BGR selon la proximité : rouge (proche), orange, vert (loin)."""
        if d < self.near:
            return (40, 40, 230)
        return (0, 150, 255) if d < self.mid else (60, 170, 40)

    def _distances(self, frame: np.ndarray, boxes: np.ndarray, distances: np.ndarray):
        """Grande étiquette « 4.3 m » sur le torse, taille proportionnelle à la silhouette."""
        font = cv2.FONT_HERSHEY_DUPLEX
        placed: list[tuple[int, int, int, int]] = []
        # du plus proche au plus lointain : les grandes étiquettes passent en premier
        for i in np.argsort(-(boxes[:, 3] - boxes[:, 1])):
            (x1, y1, x2, y2), d = boxes[i], distances[i]
            if not np.isfinite(d):
                continue
            text = f"{d:.1f} m"
            scale = float(np.clip((y2 - y1) / 260, 0.45, 1.2))
            th = 2 if scale > 0.7 else 1
            (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
            cx, cy = int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.38)
            pad = max(4, int(6 * scale))
            # évite le chevauchement : on descend l'étiquette dans la silhouette, sinon on l'omet
            for _ in range(6):
                rect = (cx - tw // 2 - pad, cy - tht // 2 - pad, cx + tw // 2 + pad, cy + tht // 2 + pad)
                if not any(_overlap(rect, r) for r in placed):
                    break
                cy += tht + 2 * pad
            else:
                continue
            if rect[3] > y2:
                continue
            placed.append(rect)
            p1, p2 = rect[:2], rect[2:]
            cv2.rectangle(frame, p1, p2, self._color(d), -1)
            cv2.rectangle(frame, p1, p2, (255, 255, 255), 1)
            cv2.putText(frame, text, (cx - tw // 2, cy + tht // 2), font, scale, (255, 255, 255), th, cv2.LINE_AA)

    def _hud(self, frame: np.ndarray, n_people: int, fps: float | None, show_legend: bool = False):
        lines = [f"Personnes : {n_people}"]
        lines += self.zones.summary()
        if self.cfg["fps"] and fps:
            lines.append(f"{fps:.0f} FPS  -  100 % local")
        _text_panel(frame, lines)
        if show_legend:
            self._legend(frame, 12 + 26 * len(lines) + 14 + 8)

    def _legend(self, frame: np.ndarray, y0: int):
        items = [(f"< {self.near:g} m", self._color(0)), (f"< {self.mid:g} m", self._color(self.near)),
                 (f"> {self.mid:g} m", self._color(self.mid))]
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.55
        roi = frame[y0:y0 + 36, 12:12 + 330]
        roi[:] = (roi * 0.35).astype(np.uint8)
        cv2.putText(frame, "Distance", (24, y0 + 24), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        x = 112
        for text, color in items:
            cv2.rectangle(frame, (x, y0 + 10), (x + 16, y0 + 26), color, -1)
            cv2.putText(frame, text, (x + 22, y0 + 24), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
            x += 76


def _overlap(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


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
