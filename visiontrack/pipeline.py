"""Boucle principale : source → détection → tracking → distance → zones → rendu → export."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from trackers import ByteTrackTracker

from .detector import build_detector
from .distance import DistanceEstimator
from .render import Renderer
from .sources import open_source
from .zones import Zones


def run(cfg: dict) -> dict:
    """Traite la source et renvoie {"video": chemin MP4 ou None, "jsonl": chemin ou None, "fps": vitesse, "frames": n}."""
    cap, name = open_source(cfg["source"])
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source : {cfg['source']}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    start = int((cfg.get("start_seconds") or 0) * src_fps)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    max_frames = int(cfg["max_seconds"] * src_fps) if cfg.get("max_seconds") else None
    if total:
        max_frames = min(max_frames or total, total - start)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("La source ne renvoie aucune image.")
    frame = _resize(frame, cfg.get("resize_width"))
    h, w = frame.shape[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = build_detector(cfg["detector"])
    tracker_cfg = dict(cfg["tracker"])
    # le tracker ne doit pas rejeter des détections que le détecteur a volontairement gardées
    conf = cfg["detector"]["confidence"]
    tracker_cfg["track_activation_threshold"] = min(tracker_cfg["track_activation_threshold"], conf)
    tracker_cfg["high_conf_det_threshold"] = min(tracker_cfg.get("high_conf_det_threshold", 0.6), conf + 0.15)
    tracker = ByteTrackTracker(frame_rate=src_fps, **tracker_cfg)
    distance = DistanceEstimator(cfg["distance"], (w, h), device)
    zones = Zones(cfg.get("zones"), (w, h))
    renderer = Renderer(cfg["render"], zones)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = cfg["output"].get("name") or f"{name}_{cfg['detector']['backend']}_{cfg['distance']['method']}"
    writer = (cv2.VideoWriter(str(out_dir / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), src_fps,
                              renderer.output_size(w, h))
              if cfg["output"]["video"] else None)
    jsonl = open(out_dir / f"{stem}.jsonl", "w", encoding="utf-8") if cfg["output"]["json"] else None

    fps_ema, idx = None, 0
    bar = tqdm(total=max_frames, unit="img")
    try:
        while ok and (max_frames is None or idx < max_frames):
            t0 = time.perf_counter()
            # une seule conversion BGR → tenseur RGB GPU, partagée par le détecteur et la profondeur
            tensor = torch.from_numpy(frame).to(device, non_blocking=True).permute(2, 0, 1).flip(0).float() / 255

            det = tracker.update(detector(tensor))
            det = det[det.tracker_id >= 0] if det.tracker_id is not None else det
            dist = distance(tensor, det)
            inside = zones.update(det)

            depth_color = distance.colorized() if (cfg["render"]["depth_inset"] or renderer.side_by_side) else None
            out = renderer(frame, det, dist, depth_color, fps_ema)
            dt = time.perf_counter() - t0  # temps complet : détection, tracking, distance et rendu
            fps_ema = 1 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / dt

            if writer:
                writer.write(out)
            if jsonl:
                jsonl.write(json.dumps(_record(idx, idx / src_fps, det, dist, inside, zones)) + "\n")
            if cfg["output"]["show"]:
                cv2.imshow("VisionTrack", out)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            idx += 1
            bar.update(1)
            bar.set_postfix(fps=f"{fps_ema:.1f}", personnes=len(det))
            ok, frame = cap.read()
            if ok:
                frame = _resize(frame, cfg.get("resize_width"))
    finally:
        bar.close()
        cap.release()
        if writer:
            writer.release()
        if jsonl:
            jsonl.close()
        cv2.destroyAllWindows()

    print(f"{idx} images traitées — vitesse moyenne du pipeline ~ {fps_ema:.1f} FPS")
    return {"video": out_dir / f"{stem}.mp4" if writer else None,
            "jsonl": out_dir / f"{stem}.jsonl" if jsonl else None,
            "fps": fps_ema, "frames": idx}


def _resize(frame: np.ndarray, width: int | None) -> np.ndarray:
    if not width or frame.shape[1] == width:
        return frame
    h = int(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)


def _record(idx, t, det, dist, inside, zones) -> dict:
    tracks = []
    for i in range(len(det)):
        tracks.append({
            "id": int(det.tracker_id[i]) if det.tracker_id is not None else None,
            "bbox": [round(float(v), 1) for v in det.xyxy[i]],
            "conf": round(float(det.confidence[i]), 3) if det.confidence is not None else None,
            "distance_m": round(float(dist[i]), 2) if np.isfinite(dist[i]) else None,
        })
    return {"frame": idx, "t": round(t, 3), "tracks": tracks, "zones": inside, "lines": zones.counts()}
