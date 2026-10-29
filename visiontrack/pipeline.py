"""Boucle principale : source → détection → tracking → distance → vitesse et sens → zones → rendu → export."""

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
from .speed import SpeedEstimator
from .zones import Zones


def run(cfg: dict, on_frame=None) -> dict:
    """Traite la source et renvoie {"video": chemin MP4 ou None, "jsonl": chemin ou None, "fps": vitesse, "frames": n}.

    on_frame(image_bgr, infos) est appelé à chaque image (affichage en direct) ; s'il renvoie "stop", l'analyse s'arrête.
    """
    cap, name, start_s = open_source(cfg["source"], start=cfg.get("start_seconds") or 0,
                                     seconds=cfg.get("max_seconds"), youtube_mode=cfg.get("youtube_mode", "stream"))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source : {cfg['source']}")

    # une webcam ou un flux renvoient souvent -1 ou 0 : on suppose 30 i/s et une durée inconnue
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if src_fps and src_fps > 0 else 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = total if total > 0 else None
    start = int(start_s * src_fps)
    if start:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)  # fonctionne aussi sur un flux YouTube distant
    max_frames = int(cfg["max_seconds"] * src_fps) if cfg.get("max_seconds") else None
    if total:
        max_frames = min(max_frames or total, total - start)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("La source ne renvoie aucune image.")
    frame = _resize(frame, cfg.get("resize_width"))
    h, w = frame.shape[:2]

    device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    detector = build_detector(cfg["detector"])
    tracker_cfg = dict(cfg["tracker"])
    # le tracker ne doit pas rejeter des détections que le détecteur a volontairement gardées
    conf = cfg["detector"]["confidence"]
    tracker_cfg["track_activation_threshold"] = min(tracker_cfg["track_activation_threshold"], conf)
    tracker_cfg["high_conf_det_threshold"] = min(tracker_cfg.get("high_conf_det_threshold", 0.6), conf + 0.15)
    tracker = ByteTrackTracker(frame_rate=src_fps, **tracker_cfg)
    distance = DistanceEstimator(cfg["distance"], (w, h), device)
    speed_cfg = cfg.get("speed") or {}
    speed = SpeedEstimator(speed_cfg, (w, h), distance.focal_px) if speed_cfg.get("enabled") else None
    zones = Zones(cfg.get("zones"), (w, h))
    zones.class_names = detector.names
    classes = [c for c in cfg["detector"]["classes"] if c in detector.names.values()]
    renderer = Renderer(cfg["render"], zones, speed, classes)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = cfg["output"].get("name") or f"{name}_{cfg['detector']['backend']}_{cfg['distance']['method']}"
    writer = (cv2.VideoWriter(str(out_dir / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), src_fps,
                              renderer.output_size(w, h))
              if cfg["output"]["video"] else None)
    jsonl = open(out_dir / f"{stem}.jsonl", "w", encoding="utf-8") if cfg["output"]["json"] else None

    # temps réel (fichiers) : comme une vraie caméra, on saute les images qu'on n'a pas le temps de traiter
    realtime = bool(cfg.get("realtime")) and total is not None
    fps_ema, idx, pos = None, 0, 0  # idx = images traitées, pos = position dans l'extrait
    wall0 = time.perf_counter()
    bar = tqdm(total=max_frames, unit="img")
    try:
        while ok and (max_frames is None or pos < max_frames):
            t0 = time.perf_counter()
            # une seule conversion BGR → tenseur RGB GPU, partagée par le détecteur et la profondeur
            tensor = torch.from_numpy(frame).to(device, non_blocking=True).permute(2, 0, 1).flip(0).float() / 255

            det = tracker.update(detector(tensor))
            det = det[det.tracker_id >= 0] if det.tracker_id is not None else det
            dist = distance(tensor, det)
            t_video = pos / src_fps
            speeds, wrong = speed.update(t_video, det, dist) if speed is not None else (None, None)
            inside = zones.update(det)
            flow = _flow(zones, t_video)

            depth_color = distance.colorized() if (cfg["render"]["depth_inset"] or renderer.side_by_side) else None
            extra = [f"Debit : {flow:.0f} vehicules/min"] if flow is not None else []
            out = renderer(frame, det, dist, depth_color, fps_ema, speeds, wrong, extra)
            dt = time.perf_counter() - t0  # temps complet : détection, tracking, distance et rendu
            fps_ema = 1 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / dt

            if writer:
                writer.write(out)
            if jsonl:
                jsonl.write(json.dumps(_record(idx, t_video, det, dist, inside, zones, speeds, wrong, speed)) + "\n")
            if cfg["output"]["show"]:
                cv2.imshow("VisionTrack", out)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if on_frame is not None:
                finite = dist[np.isfinite(dist)]
                names = list(det.data.get("class_name", []))
                moving = ([v for nm, v in zip(names, speeds) if nm != "person" and np.isfinite(v) and v > 0]
                          if speeds is not None else [])
                infos = {"frame": pos + 1, "total": max_frames, "fps": fps_ema, "people": len(det),
                         "classes": {c: names.count(c) for c in classes if names.count(c)},
                         "closest_m": float(finite.min()) if finite.size else None,
                         "mean_speed_kmh": float(np.mean(moving)) if moving else None,
                         "flow_per_min": flow,
                         "speeding": sorted(speed.current_speeding) if speed is not None else [],
                         "speeding_total": len(speed.speeding_ids) if speed is not None else 0,
                         "wrong_way": sorted(speed.current_wrong) if speed is not None else [],
                         "speed_limit": speed.limit if speed is not None else None,
                         "lines": zones.counts(), "zones": inside}
                if on_frame(out, infos) == "stop":
                    break

            idx, pos = idx + 1, pos + 1
            skipped = 0
            if idx == 1:  # la 1re image inclut le préchauffage des modèles : on démarre l'horloge après
                wall0 = time.perf_counter() - pos / src_fps
            if realtime:
                target = int((time.perf_counter() - wall0) * src_fps)
                while pos < target and (max_frames is None or pos < max_frames) and cap.grab():
                    pos, skipped = pos + 1, skipped + 1
            bar.update(1 + skipped)
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
    # en temps réel, la vidéo annotée contient moins d'images que l'extrait : cadence réelle à appliquer à la lecture
    video_fps = idx / (pos / src_fps) if (realtime and pos) else src_fps
    return {"video": out_dir / f"{stem}.mp4" if writer else None, "video_fps": video_fps, "src_fps": src_fps,
            "jsonl": out_dir / f"{stem}.jsonl" if jsonl else None,
            "fps": fps_ema, "frames": idx}


def _resize(frame: np.ndarray, width: int | None) -> np.ndarray:
    if not width or frame.shape[1] == width:
        return frame
    h = int(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)


def _flow(zones, t_video: float) -> float | None:
    """Débit de véhicules sur les lignes de comptage (passages par minute), piétons exclus."""
    vehicles = [c for c, n in zones.class_names.items() if n != "person"]
    if not zones.lines or not vehicles or t_video < 5:
        return None
    crossings = sum(l.in_count_per_class.get(c, 0) + l.out_count_per_class.get(c, 0)
                    for _, l in zones.lines for c in vehicles)
    return crossings / (t_video / 60)


def _record(idx, t, det, dist, inside, zones, speeds=None, wrong=None, speed=None) -> dict:
    names = det.data.get("class_name")
    tracks = []
    for i in range(len(det)):
        v = speeds[i] if speeds is not None else np.nan
        track = {
            "id": int(det.tracker_id[i]) if det.tracker_id is not None else None,
            "class": str(names[i]) if names is not None else "person",
            "bbox": [round(float(x), 1) for x in det.xyxy[i]],
            "conf": round(float(det.confidence[i]), 3) if det.confidence is not None else None,
            "distance_m": round(float(dist[i]), 2) if np.isfinite(dist[i]) else None,
        }
        if speed is not None:
            track["speed_kmh"] = round(float(v), 1) if np.isfinite(v) else None
            track["speeding"] = bool(speed.limit and np.isfinite(v) and v > speed.limit and track["class"] != "person")
            track["wrong_way"] = bool(wrong[i])
        tracks.append(track)
    return {"frame": idx, "t": round(t, 3), "tracks": tracks, "zones": inside, "lines": zones.counts()}
