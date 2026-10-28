"""Génère la vidéo de démo VisionTrack pour le site freelance (1920×1080, 30 i/s, H.264).

Montage : titre → pipeline → 5 scènes (suivi, distance, comptage, zone interdite, heatmap)
→ réglages → comparatif réglage rapide / précision → conclusion « 100 % local ».

    python demo/make_demo.py            # réutilise les scènes déjà calculées
    python demo/make_demo.py --force    # recalcule toutes les scènes
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visiontrack.pipeline import run  # noqa: E402

W, H, FPS = 1920, 1080, 30
OUT = ROOT / "outputs" / "demo"

BG = (15, 23, 42)          # bleu nuit
FG = (241, 245, 249)
MUTED = (148, 163, 184)
BLUE = (59, 130, 246)
GREEN = (34, 197, 94)
RED = (239, 68, 68)

CHELMSFORD = "https://www.youtube.com/watch?v=Zeew8NpJb50"
MONTREAL = "https://www.youtube.com/watch?v=bwJ-TNu0hGM"
BUSY = "https://www.youtube.com/watch?v=tfrmUlY-aYQ"
CREDITS = {
    CHELMSFORD: "Vidéo : « Chelmsford High Street after Christmas 2022 » — MOAH CLUB, licence CC BY",
    MONTREAL: "Vidéo : « Free City Street Footage » — Montreal Walking Tours, licence CC BY",
    BUSY: "Vidéo : « Busy People on Street » — Free Footage, licence CC BY",
}
NO_ZONES = {"lines": [], "polygons": []}

SCENES = [
    {"kind": "title", "seconds": 4.5},
    {"kind": "pipeline", "seconds": 8},
    {"kind": "scene", "name": "01_suivi",
     "title": "Détection et suivi en temps réel",
     "subtitle": "Chaque personne reçoit un identifiant stable, sa trajectoire et sa distance",
     "cfg": {"source": CHELMSFORD, "start_seconds": 0, "max_seconds": 10, "zones": NO_ZONES,
             "render": {"inset_position": "top_right"}}},
    {"kind": "scene", "name": "02_distance",
     "title": "Estimation de distance",
     "subtitle": "Géométrie + carte de profondeur Depth Anything V2, fusionnées et lissées par personne",
     "cfg": {"source": CHELMSFORD, "start_seconds": 22, "max_seconds": 10, "zones": NO_ZONES,
             "render": {"layout": "side_by_side", "traces": False}}},
    {"kind": "scene", "name": "02b_foule",
     "title": "Foule dense, caméra en mouvement",
     "subtitle": "Times Square : des dizaines de personnes suivies en même temps, caméra tenue à la main",
     "cfg": {"source": BUSY, "start_seconds": 118, "max_seconds": 10, "zones": NO_ZONES,
             "render": {"inset_position": "top_right", "traces": False}}},
    {"kind": "scene", "name": "02c_nuit",
     "title": "De nuit, en éclairage urbain",
     "subtitle": "Détection et suivi robustes malgré la faible lumière et les reflets",
     "cfg": {"source": BUSY, "start_seconds": 147, "max_seconds": 10, "zones": NO_ZONES,
             "render": {"depth_inset": False}}},
    {"kind": "scene", "name": "03_comptage",
     "title": "Comptage entrées / sorties",
     "subtitle": "Une ligne virtuelle, définie en deux points dans config.yaml",
     "cfg": {"source": MONTREAL, "start_seconds": 14, "max_seconds": 12,
             "zones": {"lines": [{"name": "passage", "start": [0.33, 0.80], "end": [0.90, 0.80]}],
                       "polygons": []},
             "render": {"depth_inset": False}}},
    {"kind": "scene", "name": "03b_comptage_rue",
     "title": "Comptage sur rue piétonne",
     "subtitle": "Flux dans les deux sens : idéal pour mesurer la fréquentation d'une rue commerçante",
     "cfg": {"source": CHELMSFORD, "start_seconds": 43, "max_seconds": 9,
             "zones": {"lines": [{"name": "passage", "start": [0.05, 0.75], "end": [0.95, 0.75]}],
                       "polygons": []},
             "render": {"depth_inset": False}}},
    {"kind": "scene", "name": "04_zone",
     "title": "Alerte zone interdite",
     "subtitle": "Toute intrusion colore la zone en rouge et produit un événement JSON horodaté",
     "cfg": {"source": MONTREAL, "start_seconds": 55, "max_seconds": 10,
             "zones": {"lines": [],
                       "polygons": [{"name": "chaussee",
                                     "points": [[0.36, 0.68], [0.70, 0.68], [0.72, 0.83], [0.34, 0.83]]}]},
             "render": {"depth_inset": False}}},
    {"kind": "scene", "name": "05_heatmap",
     "title": "Carte de fréquentation",
     "subtitle": "Où passent les gens : aménagement, retail, flux piétons",
     "cfg": {"source": CHELMSFORD, "start_seconds": 30, "max_seconds": 10, "zones": NO_ZONES,
             "render": {"heatmap": True, "traces": False, "depth_inset": False}}},
    {"kind": "settings", "seconds": 9},
    {"kind": "compare", "name": "06_comparatif",
     "title": "Réglages : vitesse ou précision",
     "subtitle": "Même vidéo, deux configurations — on choisit selon le besoin du client",
     "left": {"label": "Réglage rapide",
              "cfg": {"source": MONTREAL, "start_seconds": 0, "max_seconds": 8, "zones": NO_ZONES,
                      "detector": {"model": "medium", "confidence": 0.45}, "resize_width": 1280,
                      "render": {"depth_inset": False}}},
     "right": {"label": "Réglage précision",
               "cfg": {"source": MONTREAL, "start_seconds": 0, "max_seconds": 8, "zones": NO_ZONES,
                       "detector": {"model": "large", "confidence": 0.30}, "resize_width": 1920,
                       "render": {"depth_inset": False}}}},
    {"kind": "outro", "seconds": 7},
]


# ---------------------------------------------------------------- outils d'image

def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"])
    for name in names:
        for folder in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu"):
            try:
                return ImageFont.truetype(f"{folder}/{name}", size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def centered(draw: ImageDraw.ImageDraw, y: int, text: str, f, fill, x0: int = 0, x1: int = W):
    tw = draw.textlength(text, font=f)
    draw.text(((x0 + x1 - tw) / 2, y), text, font=f, fill=fill)


def fade(frame: np.ndarray, i: int, n: int, length: int = 9) -> np.ndarray:
    k = min(1.0, (i + 1) / length, (n - i) / length)
    return frame if k >= 1 else (frame * k).astype(np.uint8)


def fit(frame: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    """Redimensionne en conservant les proportions pour tenir dans box_w × box_h."""
    h, w = frame.shape[:2]
    s = min(box_w / w, box_h / h)
    return cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def lower_third(title: str, subtitle: str, credit: str | None) -> np.ndarray:
    """Bandeau bas semi-transparent (RGBA) : titre, sous-titre et crédit de la vidéo source."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, H - 150, W, H], fill=(*BG, 215))
    d.rectangle([0, H - 150, 10, H], fill=(*GREEN, 255))
    d.text((48, H - 136), title, font=font(46, True), fill=FG)
    d.text((50, H - 72), subtitle, font=font(28), fill=MUTED)
    if credit:
        f = font(18)
        d.text((W - 40 - d.textlength(credit, font=f), H - 34), credit, font=f, fill=MUTED)
    return np.array(img)


def blend(frame: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    a = overlay_rgba[..., 3:4].astype(np.float32) / 255
    rgb = overlay_rgba[..., 2::-1].astype(np.float32)  # RGBA → BGR
    return (frame * (1 - a) + rgb * a).astype(np.uint8)


# ---------------------------------------------------------------- cartes animées

def card_title(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    centered(d, 330, "VisionTrack", font(130, True), FG)
    centered(d, 500, "Détection, suivi et distance de personnes en temps réel", font(44), MUTED)
    if t > 1.0:
        tag = "100 % local  ·  GPU NVIDIA grand public  ·  aucune image dans le cloud"
        f = font(34, True)
        tw = d.textlength(tag, font=f)
        d.rounded_rectangle([(W - tw) / 2 - 30, 610, (W + tw) / 2 + 30, 676], 33, fill=(20, 83, 45))
        centered(d, 618, tag, f, (187, 247, 208))
    if t > 1.8:
        centered(d, 900, "Patrice Duclos — ingénieur Data & IA, Lyon", font(30), MUTED)
    return img


PIPELINE_STEPS = [
    ("Source vidéo", "YouTube · fichier", "webcam · caméra IP"),
    ("Détection", "RF-DETR", "(Apache 2.0)"),
    ("Suivi", "ByteTrack", "ID stables"),
    ("Distance", "géométrie +", "Depth Anything V2"),
    ("Zones", "lignes · polygones", "heatmap"),
    ("Sorties", "vidéo annotée", "JSON · temps réel"),
]


def card_pipeline(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    centered(d, 110, "Le pipeline", font(72, True), FG)
    centered(d, 215, "Six briques interchangeables, pilotées par un seul fichier : config.yaml", font(34), MUTED)

    bw, bh, gap, y = 250, 190, 50, 430
    x0 = (W - (6 * bw + 5 * gap)) // 2
    # conteneur « PC local »
    d.rounded_rectangle([x0 + bw + gap - 25, y - 70, x0 + 6 * bw + 5 * gap + 25, y + bh + 40], 24,
                        outline=GREEN, width=3)
    d.text((x0 + bw + gap, y - 58), "PC local — GPU NVIDIA : aucune donnée ne sort du site",
           font=font(26, True), fill=GREEN)
    shown = int(t / 0.55) + 1
    for i, (name, l1, l2) in enumerate(PIPELINE_STEPS[:shown]):
        x = x0 + i * (bw + gap)
        color = (51, 65, 85) if i in (0, 5) else (30, 64, 175)
        d.rounded_rectangle([x, y, x + bw, y + bh], 18, fill=color)
        centered(d, y + 28, name, font(36, True), FG, x, x + bw)
        centered(d, y + 92, l1, font(25), (203, 213, 225), x, x + bw)
        centered(d, y + 128, l2, font(25), (203, 213, 225), x, x + bw)
        if i:
            ax = x - gap + 8
            d.line([ax, y + bh / 2, x - 10, y + bh / 2], fill=FG, width=4)
            d.polygon([(x - 8, y + bh / 2), (x - 22, y + bh / 2 - 10), (x - 22, y + bh / 2 + 10)], fill=FG)
    if t > 4.0:
        centered(d, 760, "≈ 28 à 30 images/s de bout en bout sur une RTX 4070 Ti SUPER (1280 × 720)",
                 font(34, True), (187, 247, 208))
        centered(d, 820, "Adapter à un client = modifier config.yaml : source, modèle, classes, zones, seuils",
                 font(30), MUTED)
    return img


SETTINGS = [
    ("Taille du modèle", "nano → large", "plus grand = plus précis, un peu plus lent"),
    ("Seuil de confiance", "0,30 – 0,45", "plus bas = plus de personnes lointaines détectées"),
    ("Résolution d'entrée", "1280 → 1920 px", "détecte les petites silhouettes au loin"),
    ("Champ de vision caméra", "horizontal_fov_deg", "le vrai FOV (ou une calibration) = distances justes"),
    ("Méthode de distance", "pinhole · depth · fusion", "fusion : la plus stable, même si la personne est masquée"),
    ("Fréquence profondeur", "1 image sur 1 ou 2", "compromis fluidité / vitesse"),
    ("Mémoire du suivi", "lost_track_buffer", "garde le même ID malgré les occultations"),
]


def card_settings(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    centered(d, 80, "Les réglages pour de meilleurs résultats", font(64, True), FG)
    centered(d, 175, "Tout est dans config.yaml — aucun code à modifier", font(32), MUTED)
    y = 270
    shown = int(t / 0.6) + 1
    for i, (name, value, effect) in enumerate(SETTINGS[:shown]):
        yy = y + i * 98
        d.rounded_rectangle([170, yy, W - 170, yy + 82], 14, fill=(30, 41, 59))
        d.text((205, yy + 20), name, font=font(32, True), fill=FG)
        d.text((720, yy + 22), value, font=font(30, True), fill=(147, 197, 253))
        d.text((1120, yy + 24), effect, font=font(26), fill=MUTED)
    return img


def card_outro(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    centered(d, 230, "100 % local", font(120, True), GREEN)
    centered(d, 400, "Vos images ne quittent jamais votre site", font(52), FG)
    if t > 1.0:
        centered(d, 520, "RGPD facilité  ·  pas d'abonnement par caméra  ·  fonctionne hors ligne", font(34), MUTED)
    if t > 2.0:
        centered(d, 640, "Audit  →  POC en 2 semaines  →  déploiement", font(40, True), (147, 197, 253))
    if t > 3.0:
        centered(d, 800, "github.com/PDUCLOS/visiontrack", font(36), FG)
        centered(d, 860, "Patrice Duclos — ingénieur Data & IA, Lyon", font(30), MUTED)
    return img


CARDS = {"title": card_title, "pipeline": card_pipeline, "settings": card_settings, "outro": card_outro}


# ---------------------------------------------------------------- scènes calculées

def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def compute(name: str, scene_cfg: dict, base: dict, force: bool) -> dict:
    """Lance le pipeline sur une scène (ou réutilise le résultat) et renvoie vidéo + statistiques."""
    stats_path = OUT / "scenes" / f"{name}.stats.json"
    video = OUT / "scenes" / f"{name}.mp4"
    if not force and video.exists() and stats_path.exists():
        return {**json.loads(stats_path.read_text(encoding="utf-8")), "video": video}
    cfg = merge(base, scene_cfg)
    cfg["output"].update({"dir": str(OUT / "scenes"), "name": name, "video": True, "json": True, "show": False})
    print(f"\n=== scène {name}")
    res = run(cfg)
    rows = [json.loads(line) for line in open(res["jsonl"], encoding="utf-8")]
    stats = {"fps": res["fps"], "people": float(np.mean([len(r["tracks"]) for r in rows])) if rows else 0.0}
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    return {**stats, "video": video}


def read_frames(path: Path, max_frames: int | None = None):
    cap = cv2.VideoCapture(str(path))
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or FPS) / FPS))
    i = n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        if i % step == 0:
            n += 1
            yield frame
        i += 1
    cap.release()


def count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, round((cap.get(cv2.CAP_PROP_FPS) or FPS) / FPS)))
    cap.release()
    return n


def scene_frames(scene: dict, res: dict):
    overlay = lower_third(scene["title"], scene["subtitle"], CREDITS.get(scene["cfg"]["source"]))
    n = count_frames(res["video"])
    for i, frame in enumerate(read_frames(res["video"])):
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:] = BG[::-1]
        img = fit(frame, W, H)
        y = (H - 150 - img.shape[0]) // 2 if img.shape[0] < H - 150 else 0
        x = (W - img.shape[1]) // 2
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        yield fade(blend(canvas, overlay), i, n)


def compare_frames(scene: dict, left: dict, right: dict):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pw = 944
    for x, side, res in ((16, scene["left"], left), (960, scene["right"], right)):
        det = side["cfg"]["detector"]
        d.text((x, 70), side["label"], font=font(40, True), fill=FG)
        d.text((x, 130), f"RF-DETR {det['model']} · seuil {det['confidence']:.2f} · {side['cfg']['resize_width']} px",
               font=font(26), fill=(147, 197, 253))
        d.text((x, 172), f"{res['people']:.1f} personnes détectées en moyenne  ·  ≈ {res['fps']:.0f} images/s",
               font=font(26, True), fill=(187, 247, 208))
    lt = lower_third(scene["title"], scene["subtitle"], CREDITS.get(scene["left"]["cfg"]["source"]))
    overlay = np.maximum(np.array(img), lt * (lt[..., 3:4] > 0))
    n = min(count_frames(left["video"]), count_frames(right["video"]))
    for i, (a, b) in enumerate(zip(read_frames(left["video"], n), read_frames(right["video"], n))):
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:] = BG[::-1]
        for x, f in ((16, a), (960, b)):
            p = cv2.resize(f, (pw, int(pw * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA)
            canvas[225:225 + p.shape[0], x:x + pw] = p
        yield fade(blend(canvas, overlay), i, n)


def card_frames(fn, seconds: float):
    n = int(seconds * FPS)
    last_t, last = None, None
    for i in range(n):
        t = i / FPS
        key = round(t, 1)  # l'animation change au dixième de seconde : on réutilise l'image sinon
        if key != last_t:
            last, last_t = to_bgr(fn(t)), key
        yield fade(last, i, n)


# ---------------------------------------------------------------- montage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recalcule toutes les scènes")
    ap.add_argument("--out", default=str(OUT / "visiontrack_demo.mp4"))
    args = ap.parse_args()

    base = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    (OUT / "scenes").mkdir(parents=True, exist_ok=True)

    # 1) calcul de toutes les scènes (pipeline réel, sur GPU)
    results = {}
    for s in SCENES:
        if s["kind"] == "scene":
            results[s["name"]] = compute(s["name"], s["cfg"], base, args.force)
        elif s["kind"] == "compare":
            results[s["name"]] = (compute(s["name"] + "_gauche", s["left"]["cfg"], base, args.force),
                                  compute(s["name"] + "_droite", s["right"]["cfg"], base, args.force))

    # 2) montage et encodage H.264
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE)
    total = 0
    for s in SCENES:
        if s["kind"] in CARDS:
            frames = card_frames(CARDS[s["kind"]], s["seconds"])
        elif s["kind"] == "scene":
            frames = scene_frames(s, results[s["name"]])
        else:
            frames = compare_frames(s, *results[s["name"]])
        for f in frames:
            ffmpeg.stdin.write(f.tobytes())
            total += 1
        print(f"montage : {s.get('name', s['kind'])} ok ({total / FPS:.1f} s)")
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"\nVidéo de démo : {args.out}  ({total / FPS:.0f} s)")


if __name__ == "__main__":
    main()
