"""Vitesse et sens de circulation des objets suivis.

Deux façons d'obtenir la position au sol de chaque objet, en mètres :
- calibration (précise) : 4 points de l'image et la taille réelle du rectangle qu'ils forment au sol
  (ex. une voie de 3,5 m de large sur 20 m de long), d'où une homographie image → sol ;
- automatique (approchée) : position 3D reconstruite à partir de la distance estimée et de la focale.

Sens de circulation : chaque voie est un polygone avec une flèche (sens autorisé). Un objet qui s'y déplace
nettement dans le sens opposé pendant plus d'une demi-seconde déclenche une alerte « contresens ».
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import supervision as sv


class Lane:
    def __init__(self, cfg: dict, frame_size: tuple[int, int]):
        w, h = frame_size
        self.name = cfg["name"]
        self.polygon = (np.array(cfg["points"]) * [w, h]).astype(np.int32)
        d = np.array(cfg["direction"], dtype=float) * [w, h]
        self.direction = d / max(np.linalg.norm(d), 1e-6)
        self.wrong_ids: set[int] = set()  # objets surpris à contresens depuis le début

    def contains(self, point) -> bool:
        return cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False) >= 0


class SpeedEstimator:
    def __init__(self, cfg: dict | None, frame_size: tuple[int, int], focal_px: float):
        cfg = cfg or {}
        self.enabled = cfg.get("enabled", True)
        self.window = cfg.get("window_s", 1.5)         # durée sur laquelle la vitesse est mesurée
        self.min_kmh = cfg.get("min_kmh")              # en dessous : objet considéré à l'arrêt
        self.max_distance = cfg.get("max_distance_m", 60.0)  # au-delà, mesure automatique trop imprécise
        self.confirm_s = cfg.get("confirm_s", 0.7)     # durée au-dessus de la limite avant l'alerte
        self.max_kmh = cfg.get("max_kmh", 250.0)       # au-dessus : mesure aberrante, ignorée
        self.alpha = cfg.get("smoothing", 0.6)
        self.limit = cfg.get("limit_kmh")              # vitesse limite (affichage en rouge au-delà)
        w, h = frame_size
        self.frame_size = frame_size
        self.f, self.cx = focal_px, w / 2
        self.H = None
        cal = cfg.get("calibration")
        if cal:
            src = (np.array(cal["image_points"], dtype=np.float32) * [w, h]).astype(np.float32)
            W, L = cal["width_m"], cal["length_m"]
            dst = np.array([[0, 0], [W, 0], [W, L], [0, L]], dtype=np.float32)
            self.H = cv2.getPerspectiveTransform(src, dst)
        if self.min_kmh is None:  # la mesure automatique (via la distance) est plus bruitée
            self.min_kmh = 3.0 if self.H is not None else 8.0
        self.lanes = [Lane(l, frame_size) for l in cfg.get("lanes") or []]
        self.min_px = 0.015 * w                        # déplacement minimal (px) pour juger du sens
        self.ground: dict[int, deque] = {}             # id → (t, x_m, y_m)
        self.image: dict[int, deque] = {}              # id → (t, u, v)
        self.speed: dict[int, float] = {}
        self.against: dict[int, float] = {}            # id → depuis quand il roule à contresens
        self.current_wrong: set[int] = set()           # objets à contresens sur l'image courante
        self.speeding_ids: set[int] = set()            # véhicules ayant dépassé la limite depuis le début
        self.current_speeding: set[int] = set()
        self.over_since: dict[int, float] = {}         # id → depuis quand il dépasse la limite

    @property
    def mode(self) -> str:
        return "calibrée" if self.H is not None else "automatique"

    def _ground(self, anchors: np.ndarray, distances: np.ndarray) -> np.ndarray:
        if self.H is not None:
            return cv2.perspectiveTransform(anchors.reshape(-1, 1, 2).astype(np.float32), self.H).reshape(-1, 2)
        # sans calibration : x latéral = (u - cx) · d / f, profondeur = d
        return np.stack([(anchors[:, 0] - self.cx) * distances / self.f, distances], axis=1)

    def update(self, t: float, det: sv.Detections, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Renvoie (vitesses en km/h, nan si inconnue ; booléens « à contresens »)."""
        n = len(det)
        speeds, wrong = np.full(n, np.nan), np.zeros(n, bool)
        if not self.enabled or n == 0 or det.tracker_id is None:
            return speeds, wrong
        anchors = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        ground = self._ground(anchors, distances)
        for i, tid in enumerate(int(x) for x in det.tracker_id):
            img = self.image.setdefault(tid, deque())
            img.append((t, *anchors[i]))
            while img and t - img[0][0] > self.window:
                img.popleft()
            if np.all(np.isfinite(ground[i])) and (self.H is not None or distances[i] <= self.max_distance):
                g = self.ground.setdefault(tid, deque())
                g.append((t, *ground[i]))
                while g and t - g[0][0] > self.window:
                    g.popleft()
                speeds[i] = self._speed(tid, g)
            wrong[i] = self._wrong_way(tid, img, anchors[i], t)
        self.current_wrong = {int(tid) for tid, w in zip(det.tracker_id, wrong) if w}
        names = det.data.get("class_name", np.array(["person"] * n, dtype=object))
        self.current_speeding = set()
        if self.limit:
            for tid, nm, v in zip((int(x) for x in det.tracker_id), names, speeds):
                if nm != "person" and np.isfinite(v) and v > self.limit:
                    if t - self.over_since.setdefault(tid, t) >= self.confirm_s:  # excès confirmé
                        self.current_speeding.add(tid)
                else:
                    self.over_since.pop(tid, None)
            self.speeding_ids |= self.current_speeding
        return speeds, wrong

    def _speed(self, tid: int, g: deque) -> float:
        if len(g) < 4 or g[-1][0] - g[0][0] < 0.5 * self.window:
            return self.speed.get(tid, np.nan)
        pts = np.array(g)
        # pente d'une régression linéaire sur toute la fenêtre : bien moins sensible au bruit que 2 points
        t = pts[:, 0] - pts[0, 0]
        vx, vy = np.polyfit(t, pts[:, 1], 1)[0], np.polyfit(t, pts[:, 2], 1)[0]
        kmh = float(np.hypot(vx, vy) * 3.6)
        if kmh > self.max_kmh:
            return self.speed.get(tid, np.nan)
        prev = self.speed.get(tid)
        kmh = kmh if prev is None or not np.isfinite(prev) else self.alpha * prev + (1 - self.alpha) * kmh
        self.speed[tid] = kmh
        return kmh if kmh >= self.min_kmh else 0.0

    def _wrong_way(self, tid: int, img: deque, anchor, t: float) -> bool:
        lane = next((l for l in self.lanes if l.contains(anchor)), None)
        if lane is None or len(img) < 3:
            self.against.pop(tid, None)
            return False
        motion = np.array(img[-1][1:]) - np.array(img[0][1:])
        if np.linalg.norm(motion) < self.min_px:
            return tid in lane.wrong_ids and tid in self.against
        cos = float(motion @ lane.direction / np.linalg.norm(motion))
        if cos < -0.3:
            since = self.against.setdefault(tid, t)
            if t - since >= 0.5:
                lane.wrong_ids.add(tid)
                return True
            return False
        self.against.pop(tid, None)
        return False

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Dessine les voies et leur sens autorisé (flèche verte)."""
        for lane in self.lanes:
            alert = bool(lane.wrong_ids & self.current_wrong)
            color = (40, 40, 230) if alert else (80, 200, 80)
            cv2.polylines(frame, [lane.polygon], True, color, 2, cv2.LINE_AA)
            c = lane.polygon.mean(axis=0)
            length = 0.35 * float(np.ptp(lane.polygon[:, 1]) or np.ptp(lane.polygon[:, 0]))
            tail, head = c - lane.direction * length / 2, c + lane.direction * length / 2
            cv2.arrowedLine(frame, tuple(int(v) for v in tail), tuple(int(v) for v in head), color, 4,
                            cv2.LINE_AA, tipLength=0.3)
        return frame

    def summary(self) -> list[str]:
        out = []
        if self.limit:
            out.append(f"Exces de vitesse (> {self.limit:g} km/h) : {len(self.current_speeding)} "
                       f"(total {len(self.speeding_ids)})")
        for lane in self.lanes:
            active = len(lane.wrong_ids & self.current_wrong)
            out.append(f"ALERTE contresens {lane.name} : {active}" if active
                       else f"Voie {lane.name} : sens respecte ({len(lane.wrong_ids)} contresens)")
        return out
