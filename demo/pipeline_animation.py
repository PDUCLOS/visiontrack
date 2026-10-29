"""Animation « le pipeline étape par étape », construite avec le vrai outil sur de vraies images.

Pour chaque exemple (rue piétonne, trafic routier), le pipeline tourne sur quelques secondes de vidéo
(pour que le suivi ait un historique), puis l'image finale est rendue étape par étape :
photo → détection → suivi → profondeur → distance (ou vitesse) → zones et alertes → résultat.

    python demo/pipeline_animation.py
Sorties : outputs/demo/pipeline_steps/*.png, puis pour chaque exemple pipeline_<nom>.mp4 (1920×1080) et .gif (site).
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from make_demo import BG, FG, GREEN, MUTED, blend, centered, font, to_bgr  # noqa: E402
from trackers import ByteTrackTracker  # noqa: E402
from visiontrack.detector import build_detector  # noqa: E402
from visiontrack.distance import DistanceEstimator  # noqa: E402
from visiontrack.render import Renderer  # noqa: E402
from visiontrack.speed import SpeedEstimator  # noqa: E402
from visiontrack.zones import Zones  # noqa: E402

OUT = ROOT / "outputs" / "demo"
STEPS_DIR = OUT / "pipeline_steps"
W, H, FPS = 1920, 1080, 30

EXAMPLES = [
    {"name": "rue", "config": "config.yaml", "video": "Zeew8NpJb50.mp4", "start": 14, "seconds": 3.0,
     "credit": "« Chelmsford High Street after Christmas 2022 » — MOAH CLUB, licence CC BY",
     "zones": {"lines": [{"name": "passage", "start": [0.05, 0.75], "end": [0.95, 0.75]}],
               "polygons": [{"name": "interdite", "points": [[0.62, 0.62], [0.95, 0.62], [0.95, 0.95], [0.62, 0.95]]}]},
     "speed": None},
    {"name": "trafic", "config": "config_trafic.yaml", "video": "MNn9qKG2UFI.mp4", "start": 20, "seconds": 3.0,
     "credit": "« 4K Road traffic video for object detection and tracking » — Karol Majek, licence CC BY",
     "zones": {"lines": [{"name": "comptage", "start": [0.0, 0.62], "end": [0.40, 0.62]}], "polygons": []},
     "render": {"hud_position": "right"},
     "speed": {"enabled": True, "limit_kmh": 50,
               "calibration": {"image_points": [[0.255, 0.10], [0.355, 0.10], [0.39, 0.45], [0.185, 0.45]],
                               "width_m": 7.0, "length_m": 28.0},
               "lanes": [{"name": "vers la camera", "points": [[0.25, 0.03], [0.36, 0.03], [0.345, 1.0], [0.0, 1.0]],
                          "direction": [0, 1]}]}},
]

STEPS = [
    ("photo", "1. Image brute", "La vidéo arrive : caméra IP, webcam, fichier ou YouTube"),
    ("detection", "2. Détection — RF-DETR", "Chaque objet est repéré et classé, image par image"),
    ("suivi", "3. Suivi — ByteTrack", "Un identifiant stable par objet, et sa trajectoire"),
    ("profondeur", "4. Profondeur — Depth Anything V2", "Une carte de profondeur estimée par IA (clair = proche)"),
    ("mesure", "5. Distance et vitesse", "Géométrie + profondeur → distance en mètres ; trajectoire → km/h"),
    ("zones", "6. Zones et alertes", "Lignes de comptage, zones interdites, excès de vitesse, contresens"),
    ("resultat", "7. Résultat", "Vidéo annotée en direct + données JSON — 100 % local"),
]


def compute_stages(ex: dict) -> dict[str, np.ndarray]:
    """Fait tourner le vrai pipeline jusqu'à l'image cible et rend chaque étape séparément."""
    cfg = yaml.safe_load(open(ROOT / ex["config"], encoding="utf-8"))
    cfg["zones"] = ex["zones"]
    if ex["speed"]:
        cfg["speed"] = {**cfg.get("speed", {}), **ex["speed"]}
    cfg["render"].update(ex.get("render", {}))
    cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / ex["video"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_MSEC, ex["start"] * 1000)
    device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"

    detector = build_detector(cfg["detector"])
    tracker = ByteTrackTracker(frame_rate=fps, lost_track_buffer=cfg["tracker"]["lost_track_buffer"],
                               track_activation_threshold=min(cfg["tracker"]["track_activation_threshold"],
                                                              cfg["detector"]["confidence"]),
                               minimum_consecutive_frames=cfg["tracker"]["minimum_consecutive_frames"])
    w, h = 1280, 720
    dist_cfg = {**cfg["distance"], "depth_every_n_frames": 1}
    distance = DistanceEstimator(dist_cfg, (w, h), device)
    speed = SpeedEstimator(cfg["speed"], (w, h), distance.focal_px) if (cfg.get("speed") or {}).get("enabled") else None
    zones = Zones(cfg["zones"], (w, h))
    zones.class_names = detector.names
    classes = [c for c in cfg["detector"]["classes"] if c in detector.names.values()]

    def renderer(**flags):
        rcfg = {**cfg["render"], "heatmap": False, "depth_inset": False, "layout": "inset", **flags}
        return Renderer(rcfg, zones if flags.get("_zones", False) else Zones(None, (w, h)),
                        speed if flags.get("_speed", False) else None, classes)

    full = renderer(_zones=True, _speed=True, depth_inset=True)
    trace_only = renderer(boxes=True, labels=True, traces=True, fps=False)
    trace_only.trace = full.trace  # partage l'historique des trajectoires

    n = int(ex["seconds"] * fps)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(frame).to(device).permute(2, 0, 1).flip(0).float() / 255
        raw = detector(tensor)
        det = tracker.update(raw)
        det = det[det.tracker_id >= 0] if det.tracker_id is not None else det
        dist = distance(tensor, det)
        speeds, wrong = speed.update(i / fps, det, dist) if speed else (None, None)
        zones.update(det)
        full.trace.annotate(frame.copy(), det)  # alimente les trajectoires à chaque image
    cap.release()

    names = det.data.get("class_name")
    stages = {"photo": frame.copy()}

    # 2. détection : boîtes et classe + score, sans identifiant
    img = sv.BoxAnnotator(thickness=3, color=sv.Color.from_hex("#3B82F6")).annotate(frame.copy(), raw)
    labels = [f"{n} {c:.2f}" for n, c in zip(raw.data.get("class_name"), raw.confidence)]
    stages["detection"] = sv.LabelAnnotator(text_scale=0.55, color=sv.Color.from_hex("#3B82F6")).annotate(
        img, raw, labels=labels)

    # 3. suivi : identifiants et trajectoires
    img = full.trace.annotate(frame.copy(), det)
    img = trace_only.box.annotate(img, det)
    stages["suivi"] = trace_only.label.annotate(img, det, labels=[f"#{t}" for t in det.tracker_id])

    # 4. profondeur : carte couleur pleine image, boîtes en surimpression
    depth = cv2.resize(distance.colorized(), (w, h))
    stages["profondeur"] = sv.BoxAnnotator(thickness=2, color=sv.Color.WHITE).annotate(depth, det)

    # 5. mesures : distance (personnes) ou vitesse (véhicules)
    meas = renderer(traces=False, fps=False, _speed=bool(speed))
    stages["mesure"] = meas(frame.copy(), det, dist, None, None, speeds, None if wrong is None else wrong * False)

    # 6. zones et alertes, 7. résultat complet
    zone_r = renderer(traces=False, labels=False, fps=False, _zones=True, _speed=True)
    stages["zones"] = zone_r(frame.copy(), det, dist, None, None, speeds, wrong)
    stages["resultat"] = full(frame.copy(), det, dist, distance.colorized(), 30.0, speeds, wrong)
    return stages


def stage_frame(img: np.ndarray, step: int, title: str, subtitle: str, credit: str, example: str) -> np.ndarray:
    """Une image d'animation : l'étape en grand, le titre, et la barre du pipeline avec l'étape en cours."""
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((80, 40), title, font=font(54, True), fill=FG)
    d.text((82, 112), subtitle, font=font(30), fill=MUTED)
    d.text((W - 80 - d.textlength(example, font=font(26, True)), 60), example, font=font(26, True), fill=GREEN)
    # barre du pipeline
    n = len(STEPS)
    bw, gap, y = 220, 26, H - 130
    x0 = (W - (n * bw + (n - 1) * gap)) // 2
    for i, (_, t, _) in enumerate(STEPS):
        x = x0 + i * (bw + gap)
        active, done = i == step, i < step
        fill = (34, 197, 94) if active else ((30, 64, 175) if done else (51, 65, 85))
        d.rounded_rectangle([x, y, x + bw, y + 64], 14, fill=fill)
        label = t.split(" — ")[0].split(". ", 1)[1]
        centered(d, y + 17, label, font(23, True), FG, x, x + bw)
        if i:
            d.line([x - gap + 4, y + 32, x - 6, y + 32], fill=MUTED, width=3)
    d.text((W - 80 - d.textlength(credit, font=font(18)), H - 40), "Vidéo : " + credit, font=font(18), fill=MUTED)
    out = to_bgr(canvas)
    # image de l'étape : 1280×720 centrée entre le titre et la barre
    ih = y - 180 - 20
    iw = int(ih * 16 / 9)
    pic = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA)
    x, yy = (W - iw) // 2, 180
    out[yy:yy + ih, x:x + iw] = pic
    cv2.rectangle(out, (x - 2, yy - 2), (x + iw + 1, yy + ih + 1), (94, 197, 34) if step == n - 1 else (85, 65, 51), 2)
    return out


def encode(frames: list[np.ndarray], mp4: Path, gif: Path, hold_s: float = 2.4, fade_s: float = 0.5):
    """Encode les étapes en vidéo (fondu enchaîné) puis en GIF léger pour le site."""
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "19",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
    hold, fade = int(hold_s * FPS), int(fade_s * FPS)
    prev = None
    for f in frames:
        if prev is not None:
            for k in range(fade):
                a = (k + 1) / fade
                ff.stdin.write(cv2.addWeighted(prev, 1 - a, f, a, 0).tobytes())
        for _ in range(hold):
            ff.stdin.write(f.tobytes())
        prev = f
    ff.stdin.close()
    ff.wait()
    # GIF : 960 px, 10 i/s, palette optimisée
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-vf",
                    "fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
                    str(gif)], check=True)


def main():
    STEPS_DIR.mkdir(parents=True, exist_ok=True)
    for ex in EXAMPLES:
        print(f"exemple {ex['name']} : calcul des étapes…")
        stages = compute_stages(ex)
        label = "Exemple : rue piétonne" if ex["name"] == "rue" else "Exemple : trafic routier"
        frames = []
        for i, (key, title, sub) in enumerate(STEPS):
            cv2.imwrite(str(STEPS_DIR / f"{ex['name']}_{i + 1}_{key}.png"), stages[key])
            if key == "mesure" and ex["name"] == "trafic":
                sub = "Trajectoire au sol (calibrée sur le marquage) → vitesse en km/h pour chaque véhicule"
            frames.append(stage_frame(stages[key], i, title, sub, ex["credit"], label))
        mp4, gif = OUT / f"pipeline_{ex['name']}.mp4", OUT / f"pipeline_{ex['name']}.gif"
        encode(frames, mp4, gif)
        print(f"  {mp4}  ·  {gif}")


if __name__ == "__main__":
    main()
