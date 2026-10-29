"""Interface locale de démonstration (Gradio) : python app.py → http://127.0.0.1:7860

Analyse en direct : l'image annotée et les statistiques s'affichent pendant le traitement.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import yaml

from visiontrack.pipeline import run

ROOT = Path(__file__).parent
os.chdir(ROOT)  # les chemins relatifs (cache des vidéos, sorties) partent toujours du projet
OUT_DIR = ROOT / "outputs" / "app"
VIDEOS = ROOT / "data" / "videos"
FEEDBACK_FILE = ROOT / "feedback" / "feedback.jsonl"
BASE = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
STOP = threading.Event()  # bouton « Arrêter » : démo locale, une analyse à la fois

USE_CASES = ["Retail / fréquentation", "Sécurité / zones interdites", "Événementiel / foules",
             "Smart city / comptage", "Entrepôt / logistique", "Autre"]

NO_LINE = (False, [0.05, 0.75, 0.95, 0.75])
NO_ZONE = (False, [0.6, 0.5, 0.95, 0.5, 0.95, 0.95, 0.6, 0.95])

# Vidéos d'essai (CC BY) : chaque essai règle l'extrait, l'affichage, la ligne et la zone.
# Si la vidéo n'est pas encore en cache dans data/videos, elle est téléchargée depuis YouTube au premier lancement.
PRESETS = {
    "Rue piétonne — suivi et distances (Chelmsford)": dict(
        yt="Zeew8NpJb50", start=0, seconds=15, layout="inset", traces=True, heatmap=False,
        line=(True, [0.05, 0.75, 0.95, 0.75]), zone=NO_ZONE),
    "Distance — vidéo et carte de profondeur côte à côte (Chelmsford)": dict(
        yt="Zeew8NpJb50", start=22, seconds=12, layout="side_by_side", traces=False, heatmap=False,
        line=NO_LINE, zone=NO_ZONE),
    "Comptage entrées / sorties (Montréal)": dict(
        yt="bwJ-TNu0hGM", start=14, seconds=20, layout="inset", traces=True, heatmap=False,
        line=(True, [0.33, 0.80, 0.90, 0.80]), zone=NO_ZONE),
    "Alerte zone interdite (Montréal)": dict(
        yt="bwJ-TNu0hGM", start=55, seconds=15, layout="inset", traces=True, heatmap=False,
        line=NO_LINE, zone=(True, [0.36, 0.68, 0.70, 0.68, 0.72, 0.83, 0.34, 0.83])),
    "Personnes lointaines — rue de Montréal": dict(
        yt="bwJ-TNu0hGM", start=0, seconds=12, layout="inset", traces=True, heatmap=False,
        line=NO_LINE, zone=NO_ZONE),
    "Carte de fréquentation (Chelmsford)": dict(
        yt="Zeew8NpJb50", start=30, seconds=15, layout="inset", traces=False, heatmap=True,
        line=NO_LINE, zone=NO_ZONE),
    "Foule dense, caméra à la main (Times Square)": dict(
        yt="tfrmUlY-aYQ", start=103, seconds=20, layout="inset", traces=False, heatmap=False,
        line=NO_LINE, zone=NO_ZONE),
    "De nuit, éclairage urbain": dict(
        yt="tfrmUlY-aYQ", start=147, seconds=11, layout="inset", traces=True, heatmap=False,
        line=NO_LINE, zone=NO_ZONE),
    "Cas difficile — personnes masquées par une table (Porto)": dict(
        yt="XfdW6eRyaOI", start=40, seconds=15, layout="inset", traces=True, heatmap=False,
        line=NO_LINE, zone=NO_ZONE),
}

CSS = """
.help {font-size: 0.85em; opacity: 0.75; margin-top: -8px;}
#live img {object-fit: contain;}
"""


def help_text(text: str) -> gr.Markdown:
    return gr.Markdown(text, elem_classes="help")


def preset_source(name: str) -> str:
    """Fichier local s'il est déjà en cache, sinon l'URL YouTube (téléchargée une seule fois)."""
    yt = PRESETS[name]["yt"]
    cached = sorted(VIDEOS.glob(f"{yt}.*"))
    return str(cached[0]) if cached else f"https://www.youtube.com/watch?v={yt}"


# --------------------------------------------------------------------------- traitement

def build_cfg(source, start, seconds, realtime, model, conf, width, method, fov, height, traces, layout,
              heatmap, line_on, line_pts, zone_on, zone_pts) -> dict:
    """Fusionne les réglages de l'interface dans la config par défaut (config.yaml)."""
    cfg = copy.deepcopy(BASE)
    cfg["source"] = source
    cfg["start_seconds"] = float(start)
    cfg["max_seconds"] = float(seconds)
    cfg["realtime"] = bool(realtime)
    cfg["resize_width"] = int(width)
    cfg["output"].update(dir=str(OUT_DIR), name=f"run_{datetime.now():%Y%m%d_%H%M%S}", show=False,
                         video=True, json=True)
    cfg["detector"].update(model=model, confidence=float(conf))
    cfg["distance"].update(method=method, horizontal_fov_deg=float(fov), person_height_m=float(height))
    cfg["render"].update(traces=bool(traces), heatmap=bool(heatmap), layout=layout,
                         depth_inset=(layout == "inset" and method in ("depth", "fusion")))
    lines, polygons = [], []
    if line_on:
        x1, y1, x2, y2 = line_pts
        lines.append({"name": "passage", "start": [x1, y1], "end": [x2, y2]})
    if zone_on:
        polygons.append({"name": "zone_interdite",
                         "points": [[zone_pts[i], zone_pts[i + 1]] for i in range(0, 8, 2)]})
    cfg["zones"] = {"lines": lines, "polygons": polygons}
    return cfg


def to_h264(src: Path, written_fps: float, real_fps: float) -> Path:
    """Ré-encode le MP4 d'OpenCV (mp4v) en H.264 lisible dans un navigateur, à la cadence réelle de l'analyse."""
    dst = src.with_name(src.stem + "_web.mp4")
    retime = f"setpts=PTS*{written_fps / max(real_fps, 1e-3):.6f}"
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-vf", retime, "-r", f"{max(real_fps, 1):.3f}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
                   check=True, capture_output=True)
    return dst


def live_stats(infos: dict | None, status: str) -> str:
    """Bandeau de statistiques mis à jour pendant l'analyse."""
    if not infos:
        return f"**{status}**"
    total = f" / {infos['total']}" if infos.get("total") else ""
    md = [f"**{status}** — image {infos['frame']}{total} · **{infos['fps'] or 0:.0f} images/s**",
          f"- Personnes à l'image : **{infos['people']}**"]
    if infos["closest_m"] is not None:
        md.append(f"- Personne la plus proche : **{infos['closest_m']:.1f} m**")
    for name, c in infos["lines"].items():
        md.append(f"- Ligne « {name} » : **{c['in']} entrées / {c['out']} sorties**")
    for name, ids in infos["zones"].items():
        md.append(f"- Zone « {name} » : " + (f"**ALERTE, {len(ids)} personne(s)**" if ids else "libre"))
    return "\n".join(md)


def summarize(jsonl: Path, fps: float, frames: int) -> str:
    """Résumé en français calculé à partir du fichier JSONL."""
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    counts = [len(r["tracks"]) for r in rows] or [0]
    dists = [t["distance_m"] for r in rows for t in r["tracks"] if t["distance_m"] is not None]
    ids = {t["id"] for r in rows for t in r["tracks"]}
    md = ["### Résumé de l'analyse",
          f"- **Vitesse moyenne** : {fps:.1f} images/s",
          f"- **Images analysées** : {frames}",
          f"- **Personnes par image** : {sum(counts) / len(counts):.1f} en moyenne, {max(counts)} au maximum",
          f"- **Personnes différentes suivies** : {len(ids)}"]
    for c in (rows[-1]["lines"] if rows else {}).values():
        md.append(f"- **Ligne de comptage** : {c['in']} entrées / {c['out']} sorties")
    if rows and rows[0]["zones"]:
        alerts = sum(1 for r in rows if any(r["zones"].values()))
        md.append(f"- **Alertes zone interdite** : {alerts} image(s) avec au moins une personne dans la zone")
    md.append(f"- **Personne la plus proche** : {min(dists):.1f} m" if dists
              else "- **Personne la plus proche** : non disponible")
    return "\n".join(md)


def analyse(source_type, preset, url, upload, cam, start, seconds, realtime, model, conf, width, method, fov, height,
            traces, layout, heatmap, line_on, lx1, ly1, lx2, ly2, zone_on, *zone_pts):
    """Lance le vrai pipeline dans un fil séparé et diffuse l'image annotée en direct.

    Sorties successives : (image en direct, statistiques, vidéo finale, fichier JSONL, réglages utilisés).
    """
    if source_type == "Webcam":
        source, realtime = str(int(cam)), True
    elif source_type == "Vidéo d'essai":
        source = preset_source(preset)
    elif source_type == "Fichier vidéo":
        source = upload
    else:
        source = (url or "").strip()
    if not source:
        raise gr.Error("Indiquez une source : vidéo d'essai, URL YouTube, fichier vidéo ou webcam.")
    cfg = build_cfg(source, 0 if source_type == "Webcam" else start, seconds, realtime, model, conf, width,
                    method, fov, height, traces, layout, heatmap, line_on, (lx1, ly1, lx2, ly2), zone_on, zone_pts)
    used = {"source_type": source_type, "preset": preset if source_type == "Vidéo d'essai" else None,
            "start": start, "seconds": seconds, "realtime": realtime, "model": model, "confidence": conf,
            "width": width, "distance": method, "fov": fov, "person_height": height, "traces": traces,
            "layout": layout, "heatmap": heatmap, "lines": cfg["zones"]["lines"], "polygons": cfg["zones"]["polygons"]}

    state = {"frame": None, "infos": None, "result": None, "error": None, "done": False}
    stop = STOP
    stop.clear()

    def on_frame(frame, infos):
        state["frame"], state["infos"] = frame, infos
        return "stop" if stop.is_set() else None

    def worker():
        try:
            state["result"] = run(cfg, on_frame=on_frame)
        except Exception as e:  # remonté dans l'interface
            state["error"] = e
        finally:
            state["done"] = True

    t0 = time.time()
    threading.Thread(target=worker, daemon=True).start()
    status = "Chargement des modèles et de la vidéo…"
    yield None, live_stats(None, status), None, None, used
    try:
        shown = None
        while not state["done"]:
            time.sleep(0.08)
            frame = state["frame"]
            if frame is not None and frame is not shown:
                shown = frame
                yield (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), live_stats(state["infos"], "Analyse en direct"),
                       gr.skip(), gr.skip(), used)
    finally:
        stop.set()  # page fermée en cours d'analyse : on arrête aussi le pipeline

    if state["error"]:
        raise gr.Error(f"L'analyse a échoué : {state['error']}")
    out = state["result"]
    last = cv2.cvtColor(state["frame"], cv2.COLOR_BGR2RGB) if state["frame"] is not None else None
    yield last, live_stats(state["infos"], "Encodage de la vidéo…"), gr.skip(), gr.skip(), used
    video = to_h264(Path(out["video"]), out["src_fps"], out["video_fps"])
    summary = summarize(Path(out["jsonl"]), out["fps"], out["frames"])
    summary += f"\n- **Durée totale** : {time.time() - t0:.0f} s"
    yield last, summary, str(video), str(out["jsonl"]), used


# --------------------------------------------------------------------------- avis

def save_feedback(rating, cases, comment, email, last_settings) -> str:
    """Ajoute l'avis (et les réglages du dernier essai) à feedback/feedback.jsonl, en local uniquement."""
    if not rating:
        return "Merci de donner une note de 1 à 5 avant d'envoyer."
    entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), "rating": int(rating),
             "use_cases": cases or [], "comment": (comment or "").strip(), "email": (email or "").strip(),
             "settings": last_settings}
    FEEDBACK_FILE.parent.mkdir(exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return "Merci pour votre retour ! Il a été enregistré localement sur ce PC."


# --------------------------------------------------------------------------- interface

def num(value, label):
    return gr.Number(value, label=label, minimum=0, maximum=1, step=0.05)


def build_ui() -> gr.Blocks:
    d, dist, r = BASE["detector"], BASE["distance"], BASE["render"]
    first = next(iter(PRESETS))
    with gr.Blocks(title="VisionTrack — démo locale") as ui:
        gr.Markdown("# VisionTrack — démo locale\n"
                    "Détection, suivi et distance des personnes **en direct** — **100 % local, "
                    "aucune image envoyée dans le cloud.**")
        last = gr.State({})

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 1. Source vidéo")
                source_type = gr.Radio(["Vidéo d'essai", "Webcam", "YouTube", "Fichier vidéo"],
                                       value="Vidéo d'essai", label="Type de source")
                preset = gr.Dropdown(list(PRESETS), value=first, label="Vidéo d'essai")
                preset_help = help_text("Chaque essai règle automatiquement l'extrait, l'affichage, la ligne "
                                        "de comptage et la zone. Vidéos sous licence CC BY.")
                url = gr.Textbox(label="URL YouTube", value=BASE["source"], visible=False)
                upload = gr.File(label="Fichier vidéo", file_types=["video"], type="filepath", visible=False)
                cam = gr.Dropdown([0, 1, 2, 3], value=0, label="Caméra", visible=False)
                webcam_help = gr.Markdown("La caméra choisie est analysée en direct, sur ce PC. Si l'image reste "
                                          "noire ou fixe, essayez une autre caméra. Cliquez sur **Arrêter** "
                                          "pour terminer.", visible=False, elem_classes="help")
                start = gr.Slider(0, 600, value=PRESETS[first]["start"], step=1, label="Début de l'extrait (s)")
                seconds = gr.Slider(2, 300, value=PRESETS[first]["seconds"], step=1, label="Durée maximale (s)")
                realtime = gr.Checkbox(value=True, label="Temps réel (comme une caméra en direct)")
                realtime_help = help_text("Coché : la vidéo défile à sa vitesse réelle, les images qu'on n'a pas "
                                          "le temps d'analyser sont sautées. Décoché : chaque image est analysée "
                                          "(ralenti).")

                gr.Markdown("### 2. Réglages")
                with gr.Accordion("Détection", open=False):
                    model = gr.Dropdown(["nano", "small", "medium", "base", "large"], value=d["model"],
                                        label="Modèle de détection")
                    help_text("nano = très rapide mais moins précis ; large = très précis mais plus lent.")
                    conf = gr.Slider(0.1, 0.9, value=d["confidence"], step=0.05, label="Seuil de confiance")
                    help_text("Plus bas = plus de personnes lointaines détectées, mais risque de faux positifs.")
                    width = gr.Dropdown([960, 1280, 1920], value=BASE["resize_width"],
                                        label="Largeur d'image traitée (px)")
                    help_text("Plus large = meilleure détection des petites silhouettes, mais plus lent.")
                with gr.Accordion("Distance", open=False):
                    method = gr.Dropdown(["pinhole", "depth", "fusion", "none"], value=dist["method"],
                                         label="Méthode de distance")
                    help_text("pinhole = géométrique, rapide ; depth = carte de profondeur (IA) ; "
                              "fusion = les deux, la plus fiable ; none = pas de distance.")
                    fov = gr.Slider(30, 120, value=dist["horizontal_fov_deg"], step=1,
                                    label="Champ de vision horizontal de la caméra (°)")
                    help_text("Une valeur erronée fausse les distances. Smartphone ≈ 65–70°, webcam ≈ 60–80°, "
                              "grand-angle ≈ 90°+.")
                    height = gr.Slider(1.4, 2.0, value=dist["person_height_m"], step=0.01,
                                       label="Taille moyenne d'une personne (m)")
                    help_text("Sert au calcul géométrique : une taille supposée plus petite rapproche les estimations.")
                with gr.Accordion("Affichage", open=False):
                    traces = gr.Checkbox(value=PRESETS[first]["traces"], label="Afficher les trajectoires")
                    help_text("Montre le chemin de chaque personne ; peut surcharger l'image dans une foule.")
                    layout = gr.Radio([("Incrustation de la profondeur", "inset"), ("Côte à côte", "side_by_side")],
                                      value=PRESETS[first]["layout"], label="Disposition")
                    help_text("Côte à côte = vidéo et carte de profondeur, image deux fois plus large.")
                    heatmap = gr.Checkbox(value=PRESETS[first]["heatmap"], label="Carte de fréquentation (heatmap)")
                    help_text("Superpose les zones les plus fréquentées (plus lent).")
                with gr.Accordion("Ligne de comptage et zone interdite", open=False):
                    lp = PRESETS[first]["line"]
                    line_on = gr.Checkbox(value=lp[0], label="Ligne de comptage")
                    help_text("Compte les entrées / sorties au franchissement de la ligne. Coordonnées "
                              "relatives : 0 = gauche / haut, 1 = droite / bas.")
                    with gr.Row():
                        lx1, ly1 = num(lp[1][0], "Départ x"), num(lp[1][1], "Départ y")
                        lx2, ly2 = num(lp[1][2], "Arrivée x"), num(lp[1][3], "Arrivée y")
                    zp = PRESETS[first]["zone"]
                    zone_on = gr.Checkbox(value=zp[0], label="Zone interdite")
                    help_text("Alerte quand une personne entre dans le quadrilatère (4 points, "
                              "dans l'ordre autour de la zone).")
                    zone_pts = []
                    for i in range(4):
                        with gr.Row():
                            zone_pts += [num(zp[1][2 * i], f"Point {i + 1} x"),
                                         num(zp[1][2 * i + 1], f"Point {i + 1} y")]

                with gr.Row():
                    go = gr.Button("▶ Lancer l'analyse en direct", variant="primary")
                    halt = gr.Button("■ Arrêter", variant="stop")

            with gr.Column(scale=3):
                gr.Markdown("### 3. En direct")
                live = gr.Image(label="Analyse en direct", elem_id="live", format="jpeg", interactive=False,
                                height=560)
                stats = gr.Markdown("*Choisissez une vidéo d'essai ou la webcam, puis lancez l'analyse.*")
                with gr.Accordion("Vidéo annotée complète et données", open=False):
                    video = gr.Video(label="Vidéo annotée", autoplay=False)
                    jsonl = gr.File(label="Télécharger les données (JSONL)")

                gr.Markdown("### 4. Votre avis")
                gr.Markdown("**Vos retours restent sur ce PC** : rien n'est envoyé sur le réseau.")
                rating = gr.Radio([1, 2, 3, 4, 5], label="Note (1 = décevant, 5 = excellent)")
                cases = gr.CheckboxGroup(USE_CASES, label="Cas d'usage qui vous intéresse")
                comment = gr.Textbox(label="Commentaire", lines=3)
                email = gr.Textbox(label="E-mail (facultatif, si vous souhaitez être recontacté)")
                send = gr.Button("Envoyer mon avis")
                confirm = gr.Markdown()

        def on_source(s):
            essai, webcam = s == "Vidéo d'essai", s == "Webcam"
            return (gr.update(visible=essai), gr.update(visible=essai), gr.update(visible=s == "YouTube"),
                    gr.update(visible=s == "Fichier vidéo"), gr.update(visible=webcam), gr.update(visible=webcam),
                    gr.update(visible=not webcam), gr.update(visible=not webcam), gr.update(visible=not webcam),
                    gr.update(value=60) if webcam else gr.update())

        source_type.change(on_source, source_type, [preset, preset_help, url, upload, cam, webcam_help, start,
                                                    realtime, realtime_help, seconds])

        def on_preset(name):
            p = PRESETS[name]
            return [p["start"], p["seconds"], p["layout"], p["traces"], p["heatmap"], p["line"][0],
                    *p["line"][1], p["zone"][0], *p["zone"][1]]

        preset.change(on_preset, preset, [start, seconds, layout, traces, heatmap, line_on, lx1, ly1, lx2, ly2,
                                          zone_on, *zone_pts])

        inputs = [source_type, preset, url, upload, cam, start, seconds, realtime, model, conf, width, method, fov,
                  height, traces, layout, heatmap, line_on, lx1, ly1, lx2, ly2, zone_on, *zone_pts]
        go.click(analyse, inputs, [live, stats, video, jsonl, last])
        halt.click(lambda: STOP.set(), None, None, queue=False)
        send.click(save_feedback, [rating, cases, comment, email, last], confirm)
    return ui


def preload():
    """Charge les modèles par défaut au démarrage : la première analyse démarre alors tout de suite."""
    from visiontrack.detector import build_detector
    from visiontrack.distance import _depth_model
    build_detector(BASE["detector"])
    _depth_model(BASE["distance"]["depth_model"], "cuda" if __import__("torch").cuda.is_available() else "cpu",
                 BASE["distance"].get("depth_input_size", 392))
    print("Modèles chargés : prêt pour la démo.")


if __name__ == "__main__":
    threading.Thread(target=preload, daemon=True).start()
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860, share=False,
                              theme=gr.themes.Soft(), css=CSS)
