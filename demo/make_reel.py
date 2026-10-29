"""Bande-démo courte et rythmée de VisionTrack (≈ 75 s, 1920×1080, 30 i/s, H.264) pour le site freelance.

Structure : accroche sur l'image → marque → 4 chapitres d'exemples courts (vitesse, excès de vitesse, flux,
personnes) → 3 bénéfices chiffrés → appel à l'action. Chaque extrait dure ≈ 3,4 s ; le passage le plus
parlant de chaque scène est choisi automatiquement à partir du JSON produit par le pipeline, et le chiffre
incrusté (vitesse max, alertes, véhicules à l'image…) vient de ces mêmes données.

    python demo/make_reel.py            # réutilise les scènes déjà calculées par make_demo.py
    python demo/make_reel.py --force    # recalcule les scènes propres à la bande-démo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_demo import (BG, BUSY, CHELMSFORD, CREDITS, FG, FPS, GREEN, H, MONTREAL, MUTED, OUT,  # noqa: E402
                       TRAFFIC_CREDITS, W, blend, centered, compute, fit, font, read_frames, to_bgr)

CLIP_S = 3.4
ACCENT = (187, 247, 208)
NO_ZONES = {"lines": [], "polygons": []}


# ---------------------------------------------------------------- scènes

def traffic(name, preset_idx, start, seconds, **over):
    """Scène véhicules construite comme un essai « Trafic » de l'interface, avec quelques réglages modifiés."""
    import app

    names = [n for n in app.PRESETS if n.startswith("Trafic")]
    p = {**app.PRESETS[names[preset_idx]], **over}
    source = app.preset_source(names[preset_idx])
    cfg = app.build_cfg(source, start, seconds, False, p.get("model", "medium"), p.get("conf", 0.45), 1280,
                        p.get("method", "fusion"), 65, 1.7, p["traces"], p["layout"], p["heatmap"],
                        p["line"][0], p["line"][1], p["zone"][0], p["zone"][1], p["profile"], p.get("limit", 0),
                        p.get("lanes", False), p.get("invert", False), p.get("speed"), p.get("render"))
    return {"name": name, "cfg": cfg, "full": True, "credit": TRAFFIC_CREDITS[p["yt"]]}


def people(name, source, start, seconds, **render):
    return {"name": name, "cfg": {"source": source, "start_seconds": start, "max_seconds": seconds,
                                  "zones": NO_ZONES, "render": render}, "credit": CREDITS[source]}


def build_reel() -> list[dict]:
    """Le storyboard : (type, contenu). Les scènes « t* » et « 0* » sont celles des démos longues."""
    lane_off = dict(lanes=False)
    T = {
        "t1": {"name": "t1_vitesse", "credit": TRAFFIC_CREDITS["MNn9qKG2UFI"]},
        "t2": {"name": "t2_exces", "credit": TRAFFIC_CREDITS["MNn9qKG2UFI"]},
        "t5": {"name": "t5_rondpoint", "credit": TRAFFIC_CREDITS["JSBJskCT_-w"]},
        "t6": {"name": "t6_voie_rapide", "credit": TRAFFIC_CREDITS["BjF9vlJA7CY"]},
    }
    new = {
        "var60": traffic("r_var_60", 0, 60, 8),
        "var150": traffic("r_var_150", 0, 150, 8),
        "exces200": traffic("r_exces_200", 1, 200, 10),
        "depth": traffic("r_var_profondeur", 0, 235, 8, layout="side_by_side", **lane_off),
        # vues du ciel : vitesse affichée, mais pas d'alerte d'excès (mesure zénithale trop approximative)
        "carrefour": traffic("r_carrefour", 3, 0, 12, limit=0),
        "heat": traffic("r_carrefour_heat", 3, 14, 10, heatmap=True, traces=False, limit=0),
        "oslo": traffic("r_oslo_22", 5, 22, 10),
    }
    P = {
        "foule": {"name": "02b_foule", "credit": CREDITS[BUSY]},
        "nuit": {"name": "02c_nuit", "credit": CREDITS[BUSY]},
        "zone": {"name": "04_zone", "credit": CREDITS[MONTREAL]},
        "comptage": {"name": "03_comptage", "credit": CREDITS[MONTREAL]},
        "distance": {"name": "02_distance", "credit": CREDITS[CHELMSFORD]},
    }

    def clip(scene, title, sub, chip, score="tracks"):
        return ("clip", {**scene, "title": title, "sub": sub, "chip": chip, "score": score})

    return [
        ("hook", {**new["carrefour"], "score": "tracks"}),
        ("brand", {"seconds": 1.8}),
        ("sting", {"text": "Mesurer la vitesse", "n": "01"}),
        clip(new["var60"], "Vitesse de chaque véhicule", "calibrée sur le marquage au sol", "maxspeed", "speed"),
        clip(new["oslo"], "Voie rapide", "sans calibration, à partir des distances", "maxspeed", "speed"),
        clip(new["carrefour"], "Carrefour vu du ciel", "des dizaines de véhicules à la fois", "inframe"),
        clip(new["depth"], "Distance de chaque véhicule", "image + carte de profondeur", "inframe"),
        ("sting", {"text": "Détecter les excès", "n": "02"}),
        clip(T["t2"], "Excès de vitesse", "limite réglable : ici 30 km/h", "alerts", "speeding"),
        clip(new["exces200"], "Alerte confirmée en 0,7 s", "pas de fausse alerte sur un pic de mesure", "alerts",
             "speeding"),
        ("sting", {"text": "Compter les flux", "n": "03"}),
        clip(T["t1"], "Comptage sur ligne virtuelle", "par type : voitures, camions, bus…", "inframe"),
        clip(new["heat"], "Carte de chaleur", "les trajectoires les plus empruntées", "inframe"),
        clip(T["t5"], "Rond-point", "fréquentation et trajectoires", "inframe"),
        clip(new["var150"], "Débit en temps réel", "véhicules par minute, en continu", "inframe"),
        ("sting", {"text": "Et les personnes", "n": "04"}),
        clip(P["foule"], "Foule dense", "caméra tenue à la main", "people"),
        clip(P["nuit"], "De nuit", "éclairage urbain", "people"),
        clip(P["zone"], "Zone interdite", "alerte à chaque intrusion", "people"),
        clip(P["distance"], "Distance de chaque personne", "géométrie + profondeur", "people"),
        clip(P["comptage"], "Entrées / sorties", "comptage de passage", "people"),
        ("benefit", {"big": "29 images/s", "sub": "de bout en bout sur un PC avec RTX 4070 Ti"}),
        ("benefit", {"big": "0 image dans le cloud", "sub": "tout tourne sur site : RGPD facilité, pas d'abonnement"}),
        ("benefit", {"big": "1 fichier de réglages", "sub": "adapté à votre caméra, vos zones et vos règles"}),
        ("cta", {"seconds": 4.5}),
    ], new


# ---------------------------------------------------------------- données des scènes

def load_rows(name: str) -> list[dict]:
    path = OUT / "scenes" / f"{name}.jsonl"
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def frame_scores(rows: list[dict], kind: str) -> np.ndarray:
    def score(r):
        t = r["tracks"]
        if kind == "speed":
            return sum(1 for x in t if x.get("speed_kmh")) + 0.02 * max([x.get("speed_kmh") or 0 for x in t] + [0])
        if kind == "speeding":
            return 5 * sum(x.get("speeding", False) for x in t) + 0.1 * len(t)
        if kind == "wrong":
            return 5 * sum(x.get("wrong_way", False) for x in t) + 0.1 * len(t)
        return len(t)
    return np.array([score(r) for r in rows], float)


def best_window(scores: np.ndarray, n: int, margin: int = 15) -> int:
    """Début de la fenêtre de n images au score cumulé maximal (on évite les toutes premières images,
    le temps que le suivi et la mesure de vitesse s'installent)."""
    if len(scores) <= n + margin:
        return max(0, len(scores) - n)
    c = np.concatenate([[0], np.cumsum(scores)])
    sums = c[n:] - c[:-n]
    sums[:margin] = -1
    return int(np.argmax(sums))


def chip_text(kind: str, rows: list[dict]) -> str:
    """Chiffre incrusté, calculé sur les images déjà montrées de l'extrait (il évolue en direct)."""
    cur = rows[-1]["tracks"]
    if kind == "maxspeed":
        v = max([x.get("speed_kmh") or 0 for r in rows for x in r["tracks"] if x.get("class", "person") != "person"] + [0])
        return f"max {v:.0f} km/h" if v else ""
    if kind == "alerts":
        n = len({x["id"] for r in rows for x in r["tracks"] if x.get("speeding")})
        return f"{n} excès détecté{'s' if n > 1 else ''}"
    if kind == "wrong":
        n = len({x["id"] for r in rows for x in r["tracks"] if x.get("wrong_way")})
        return f"{n} contresens"
    if kind == "people":
        return f"{sum(x.get('class', 'person') == 'person' for x in cur)} personnes à l'image"
    n = sum(x.get("class", "person") != "person" for x in cur)
    return f"{n} véhicules à l'image"


# ---------------------------------------------------------------- rendu

def ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def place(frame: np.ndarray, zoom: float = 1.0) -> np.ndarray:
    """Image plein cadre sur fond bleu nuit, avec un léger zoom d'entrée (« punch-in »)."""
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = BG[::-1]
    img = fit(frame, int(W * zoom), int(H * zoom))
    x, y = (W - img.shape[1]) // 2, (H - img.shape[0]) // 2
    sx, sy = max(0, -x), max(0, -y)
    img = img[sy:sy + H, sx:sx + W]
    x, y = max(0, x), max(0, y)
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


def caption(i: int, title: str, sub: str, chip: str, credit: str, chip_color=ACCENT) -> np.ndarray:
    """Bandeau en bas à gauche, qui glisse depuis la gauche au début de l'extrait."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ft, fs, fc = font(58, True), font(30), font(34, True)
    bw = int(max(d.textlength(title, font=ft), d.textlength(sub, font=fs)) + 90)
    cw = int(d.textlength(chip, font=fc) + 44) if chip else 0
    dx = int((1 - ease(i / 8)) * -(bw + cw + 60))
    y0 = H - 210
    d.rectangle([dx, y0, dx + bw, y0 + 150], fill=(*BG, 225))
    d.rectangle([dx, y0, dx + 10, y0 + 150], fill=(*GREEN, 255))
    d.text((dx + 44, y0 + 18), title, font=ft, fill=FG)
    d.text((dx + 46, y0 + 96), sub, font=fs, fill=MUTED)
    if chip:
        cx = dx + bw + 16
        d.rounded_rectangle([cx, y0 + 46, cx + cw, y0 + 106], 30, fill=(20, 83, 45, 240))
        d.text((cx + 22, y0 + 54), chip, font=fc, fill=chip_color)
    f = font(17)
    d.text((W - 30 - d.textlength(credit, font=f), H - 28), credit, font=f, fill=(*MUTED, 255))
    return np.array(img)


def progress(frame: np.ndarray, k: float) -> np.ndarray:
    cv2.rectangle(frame, (0, H - 6), (int(W * k), H), GREEN[::-1], -1)
    return frame


def flash(frame: np.ndarray, i: int, n: int = 4) -> np.ndarray:
    """Coupe franche avec un bref flash blanc (rythme)."""
    if i >= n:
        return frame
    a = 0.55 * (1 - i / n)
    return cv2.addWeighted(frame, 1 - a, np.full_like(frame, 255), a, 0)


def clip_frames(c: dict, frames_s: float = CLIP_S):
    video = OUT / "scenes" / f"{c['name']}.mp4"
    rows = load_rows(c["name"])
    cap = cv2.VideoCapture(str(video))
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or FPS) / FPS))
    cap.release()
    rows = rows[::step]
    n = int(frames_s * FPS)
    s = best_window(frame_scores(rows, c.get("score", "tracks")), n)
    frames = list(read_frames(video, s + n))[s:s + n]
    for i, frame in enumerate(frames):
        out = place(frame, 1.0 + 0.06 * (1 - ease(i / 12)))
        if c.get("title"):
            chip = chip_text(c["chip"], rows[s:s + i + 1])
            color = (254, 202, 202) if c["chip"] in ("alerts", "wrong") and chip[0] != "0" else ACCENT
            out = blend(out, caption(i, c["title"], c["sub"], chip, c["credit"], color))
        yield out if c.get("no_flash") else flash(out, i)


def hook_frames(c: dict):
    """Accroche : on ouvre directement sur l'image, trois mots apparaissent l'un après l'autre."""
    words = ["Chaque véhicule.", "Suivi.", "Mesuré."]
    for i, frame in enumerate(clip_frames({**c, "title": None, "no_flash": True}, 3.6)):
        t = i / FPS
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, H], fill=(0, 0, 0, 70))
        f = font(104, True)
        x = 110
        for k, w in enumerate(words):
            a = ease((t - 0.35 - 0.75 * k) / 0.25)
            if a <= 0:
                continue
            col = GREEN if k == 2 else FG
            d.text((x, 420 + int((1 - a) * 40)), w, font=f, fill=(*col, int(255 * a)))
            x += int(d.textlength(w + " ", font=f))
        yield blend(frame, np.array(img))


def card_brand(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    a = ease(t / 0.35)
    centered(d, 360 + int((1 - a) * 30), "VisionTrack", font(150, True), FG)
    centered(d, 560, "Vision par ordinateur en temps réel, 100 % locale", font(46), MUTED)
    if t > 0.5:
        centered(d, 660, "Véhicules  ·  vitesse  ·  excès  ·  flux  ·  personnes", font(36, True), ACCENT)
    return img


def card_sting(text: str, n: str):
    def fn(t: float) -> Image.Image:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        a = ease(t / 0.3)
        f = font(120, True)
        tw = d.textlength(text, font=f)
        x = (W - tw) / 2
        d.text((x, 470 + int((1 - a) * 60)), text, font=f, fill=FG)
        d.text((x, 400), n, font=font(48, True), fill=GREEN)
        d.rectangle([x, 640, x + tw * a, 650], fill=GREEN)
        return img
    return fn


def card_benefit(big: str, sub: str):
    def fn(t: float) -> Image.Image:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        a = ease(t / 0.3)
        centered(d, 380 + int((1 - a) * 50), big, font(140, True), GREEN)
        if t > 0.25:
            centered(d, 600, sub, font(46), FG)
        return img
    return fn


def card_cta(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    centered(d, 230, "Un projet de vision par ordinateur ?", font(80, True), FG)
    if t > 0.5:
        centered(d, 380, "Trafic, sécurité, retail, logistique, événementiel", font(42), MUTED)
    if t > 1.0:
        steps = "Audit   →   POC en 2 semaines   →   déploiement sur site"
        f = font(48, True)
        tw = d.textlength(steps, font=f)
        d.rounded_rectangle([(W - tw) / 2 - 40, 500, (W + tw) / 2 + 40, 590], 45, fill=(20, 83, 45))
        centered(d, 512, steps, f, ACCENT)
    if t > 1.6:
        centered(d, 720, "Parlons-en", font(64, True), GREEN)
        centered(d, 830, "Patrice Duclos — ingénieur Data & IA, Lyon", font(36), FG)
        centered(d, 890, "github.com/PDUCLOS/visiontrack", font(32), MUTED)
    return img


def card(fn, seconds: float):
    n = int(seconds * FPS)
    for i in range(n):
        img = to_bgr(fn(i / FPS))
        yield flash(img, i, 3) if i < 3 else img


# ---------------------------------------------------------------- montage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recalcule les scènes propres à la bande-démo")
    ap.add_argument("--out", default=str(OUT / "visiontrack_reel.mp4"))
    args = ap.parse_args()

    reel, new = build_reel()
    for s in new.values():
        compute(s["name"], s["cfg"], {}, args.force)
    for kind, c in reel:  # les scènes reprises des démos longues doivent exister
        if kind in ("clip", "hook") and not (OUT / "scenes" / f"{c['name']}.jsonl").exists():
            sys.exit(f"Scène manquante : {c['name']} — lancer d'abord make_demo.py (et --profile trafic)")

    durations = {"hook": 3.6, "brand": 1.8, "sting": 0.9, "clip": CLIP_S, "benefit": 1.5, "cta": 4.5}
    total_n = sum(int(durations[k] * FPS) for k, _ in reel)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE)
    done = 0
    for kind, c in reel:
        if kind == "hook":
            frames = hook_frames(c)
        elif kind == "clip":
            frames = clip_frames(c)
        elif kind == "brand":
            frames = card(card_brand, durations[kind])
        elif kind == "sting":
            frames = card(card_sting(c["text"], c["n"]), durations[kind])
        elif kind == "benefit":
            frames = card(card_benefit(c["big"], c["sub"]), durations[kind])
        else:
            frames = card(card_cta, durations[kind])
        for f in frames:
            ffmpeg.stdin.write(progress(f, done / total_n).tobytes())
            done += 1
        print(f"montage : {c.get('name') or c.get('text') or c.get('big') or kind} ({done / FPS:.1f} s)")
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"\nBande-démo : {args.out}  ({done / FPS:.0f} s)")


if __name__ == "__main__":
    main()
