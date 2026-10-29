"""Rendu visuel : boîtes, ID, distances, vitesses, excès de vitesse, contresens, zones, profondeur, FPS."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

LOOKUP = sv.ColorLookup.TRACK

# noms affichés (police OpenCV sans accents)
FR = {"person": "personne", "car": "voiture", "truck": "camion", "bus": "bus", "motorcycle": "moto",
      "bicycle": "velo"}
FR_PLURAL = {"person": "Personnes", "car": "Voitures", "truck": "Camions", "bus": "Bus", "motorcycle": "Motos",
             "bicycle": "Velos"}
RED, SPEED_OK = (40, 40, 230), (200, 110, 20)


class Renderer:
    def __init__(self, cfg: dict, zones: "Zones", speed=None, class_names: list[str] | None = None):
        self.cfg = cfg
        self.zones = zones
        self.speed = speed
        self.class_names = class_names or ["person"]
        self.multi = len(self.class_names) > 1
        self.box = sv.BoxAnnotator(thickness=3, color_lookup=LOOKUP)
        self.label = sv.LabelAnnotator(text_scale=0.6, text_padding=5, color_lookup=LOOKUP,
                                       text_position=sv.Position.TOP_LEFT)
        self.trace = sv.TraceAnnotator(trace_length=cfg["trace_length"], thickness=4, smooth=True,
                                       color_lookup=LOOKUP, position=sv.Position.BOTTOM_CENTER)
        self.heatmap = sv.HeatMapAnnotator(position=sv.Position.BOTTOM_CENTER, opacity=0.55, radius=45) if cfg["heatmap"] else None
        self.near, self.mid = cfg.get("distance_thresholds_m", [3.0, 8.0])

    def __call__(self, frame_bgr: np.ndarray, det: sv.Detections, distances: np.ndarray,
                 depth_color: np.ndarray | None, fps: float | None, speeds: np.ndarray | None = None,
                 wrong: np.ndarray | None = None, extra: list[str] | None = None) -> np.ndarray:
        out = frame_bgr.copy()
        n = len(det)
        speeds = speeds if speeds is not None else np.full(n, np.nan)
        wrong = wrong if wrong is not None else np.zeros(n, bool)
        names = det.data.get("class_name", np.array(["person"] * n, dtype=object))
        if self.heatmap is not None:
            out = self.heatmap.annotate(out, det)
        out = self.zones.annotate(out)
        if self.speed is not None:
            out = self.speed.annotate(out)
        if self.cfg["traces"] and det.tracker_id is not None:
            out = self.trace.annotate(out, det)
        if self.cfg["boxes"]:
            out = self.box.annotate(out, det)
        labels, big = None, None
        if self.cfg["labels"] and det.tracker_id is not None:
            labels = [f"#{tid} {FR.get(nm, nm)}" if self.multi else f"#{tid}"
                      for tid, nm in zip(det.tracker_id, names)]
            out = self.label.annotate(out, det, labels=labels)
            big = self._big_texts(names, distances, speeds)
            self._big_labels(out, det.xyxy, big)
        self._banners(out, det.xyxy, names, speeds, wrong)
        self._hud(out, det, names, speeds, fps, extra or [], big)
        if self.side_by_side:
            panel = self._depth_panel(frame_bgr.shape, det, labels, depth_color)
            if big:
                self._big_labels(panel, det.xyxy, big)
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

    @property
    def limit(self) -> float | None:
        return self.speed.limit if self.speed is not None else None

    def _speeding(self, name: str, v: float) -> bool:
        return bool(self.limit) and name != "person" and np.isfinite(v) and v > self.limit

    def _big_texts(self, names, distances, speeds) -> list:
        """Étiquette principale : vitesse pour un véhicule en mouvement, sinon distance. (texte, couleur, type)"""
        out = []
        for nm, d, v in zip(names, distances, speeds):
            if nm != "person" and np.isfinite(v):
                out.append((f"{v:.0f} km/h", RED if self._speeding(nm, v) else SPEED_OK, "speed"))
            elif nm != "person" and self.speed is not None:
                out.append(None)  # véhicule sans vitesse encore mesurée : pas de distance, l'image reste lisible
            elif np.isfinite(d):
                out.append((f"{d:.1f} m", self._color(d), "distance"))
            else:
                out.append(None)
        return out

    def _big_labels(self, frame: np.ndarray, boxes: np.ndarray, big: list):
        """Grande étiquette (« 4.3 m », « 52 km/h ») au centre de l'objet, taille proportionnelle à l'objet."""
        font = cv2.FONT_HERSHEY_DUPLEX
        placed: list[tuple[int, int, int, int]] = []
        # du plus grand au plus petit (≈ du plus proche au plus lointain) : les grandes étiquettes d'abord
        for i in np.argsort(-(boxes[:, 3] - boxes[:, 1])):
            (x1, y1, x2, y2), item = boxes[i], big[i]
            if item is None:
                continue
            text, color, _ = item
            scale = float(np.clip(max(y2 - y1, 0.6 * (x2 - x1)) / 260, 0.45, 1.2))
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
            cv2.rectangle(frame, p1, p2, color, -1)
            cv2.rectangle(frame, p1, p2, (255, 255, 255), 1)
            cv2.putText(frame, text, (cx - tw // 2, cy + tht // 2), font, scale, (255, 255, 255), th, cv2.LINE_AA)

    def _banners(self, frame: np.ndarray, boxes: np.ndarray, names, speeds, wrong: np.ndarray):
        """Cadre rouge épais et bandeau : « CONTRESENS » (prioritaire) ou « EXCES DE VITESSE »."""
        font = cv2.FONT_HERSHEY_DUPLEX
        for (x1, y1, x2, y2), nm, v, w in zip(boxes.astype(int), names, speeds, wrong):
            text = "CONTRESENS" if w else ("EXCES DE VITESSE" if self._speeding(nm, v) else None)
            if text is None:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 5)
            (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
            y = max(y1 - 10, th + 12)
            cv2.rectangle(frame, (x1, y - th - 10), (x1 + tw + 16, y + 6), RED, -1)
            cv2.putText(frame, text, (x1 + 8, y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def _hud(self, frame: np.ndarray, det: sv.Detections, names, speeds, fps: float | None,
             extra: list[str], big):
        if self.multi:
            counts = {c: int(sum(1 for nm in names if nm == c)) for c in self.class_names}
            lines = ["  ".join(f"{FR_PLURAL.get(c, c)} : {k}" for c, k in counts.items() if k) or "Aucun objet"]
        else:
            lines = [f"{FR_PLURAL.get(self.class_names[0], 'Objets')} : {len(det)}"]
        moving = [v for nm, v in zip(names, speeds) if nm != "person" and np.isfinite(v) and v > 0]
        if moving:
            lines.append(f"Vitesse moyenne : {np.mean(moving):.0f} km/h")
        lines += extra
        lines += self.zones.summary()
        if self.speed is not None:
            lines += self.speed.summary()
        if self.cfg["fps"] and fps:
            lines.append(f"{fps:.0f} FPS  -  100 % local")
        right = self.cfg.get("hud_position", "left") == "right"
        _text_panel(frame, lines, right=right)
        kinds = {b[2] for b in big or [] if b}
        y = 12 + 26 * len(lines) + 14 + 8
        if "distance" in kinds:
            self._legend(frame, y, "Distance", [(f"< {self.near:g} m", self._color(0)),
                                                (f"< {self.mid:g} m", self._color(self.near)),
                                                (f"> {self.mid:g} m", self._color(self.mid))], right)
            y += 44
        if "speed" in kinds and self.limit:
            self._legend(frame, y, "Vitesse", [(f"<= {self.limit:g} km/h", SPEED_OK), (f"> {self.limit:g} km/h", RED)],
                         right)

    def _legend(self, frame: np.ndarray, y0: int, title: str, items, right: bool = False):
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.55
        width = 112 + sum(cv2.getTextSize(t, font, scale, 1)[0][0] + 40 for t, _ in items)
        x0 = frame.shape[1] - width - 12 if right else 12
        roi = frame[y0:y0 + 36, x0:x0 + width]
        roi[:] = (roi * 0.35).astype(np.uint8)
        cv2.putText(frame, title, (x0 + 12, y0 + 24), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        x = x0 + 100
        for text, color in items:
            cv2.rectangle(frame, (x, y0 + 10), (x + 16, y0 + 26), color, -1)
            cv2.putText(frame, text, (x + 22, y0 + 24), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
            x += cv2.getTextSize(text, font, scale, 1)[0][0] + 40


def _overlap(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _text_panel(frame: np.ndarray, lines: list[str], anchor: str = "top", right: bool = False):
    """Bloc de texte lisible sur fond sombre semi-transparent (en haut ou en bas, à gauche ou à droite)."""
    font, scale, th, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 26
    width = max(cv2.getTextSize(t, font, scale, th)[0][0] for t in lines) + 24
    height = lh * len(lines) + 14
    x0 = frame.shape[1] - width - 12 if right else 12
    y0 = 12 if anchor == "top" else frame.shape[0] - height - 12
    roi = frame[y0:y0 + height, x0:x0 + width]
    roi[:] = (roi * 0.35).astype(np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x0 + 12, y0 + 26 + i * lh), font, scale, (255, 255, 255), th, cv2.LINE_AA)
