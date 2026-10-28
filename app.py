"""Interface locale de démonstration (Gradio) : python app.py → http://127.0.0.1:7860"""

from __future__ import annotations

import copy
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import yaml

from visiontrack.pipeline import run

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "outputs" / "app"
FEEDBACK_FILE = ROOT / "feedback" / "feedback.jsonl"
BASE = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

USE_CASES = ["Retail / fréquentation", "Sécurité / zones interdites", "Événementiel / foules",
             "Smart city / comptage", "Entrepôt / logistique", "Autre"]

CSS = ".help {font-size: 0.85em; opacity: 0.75; margin-top: -8px;}"


def help_text(text: str) -> gr.Markdown:
    return gr.Markdown(text, elem_classes="help")


# --------------------------------------------------------------------------- traitement

def build_cfg(source, start, seconds, model, conf, width, method, fov, height, traces, layout, heatmap,
              line_on, line_pts, zone_on, zone_pts) -> dict:
    """Fusionne les réglages de l'interface dans la config par défaut (config.yaml)."""
    cfg = copy.deepcopy(BASE)
    cfg["source"] = source
    cfg["start_seconds"] = float(start)
    cfg["max_seconds"] = float(seconds)
    cfg["resize_width"] = int(width)
    cfg["output"].update(dir=str(OUT_DIR), name=f"run_{datetime.now():%Y%m%d_%H%M%S}", show=False,
                         video=True, json=True)
    cfg["detector"].update(model=model, confidence=float(conf))
    cfg["tracker"]["track_activation_threshold"] = float(conf)  # sinon le tracker écarte les détections faibles
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


def to_h264(src: Path) -> Path:
    """Ré-encode le MP4 d'OpenCV (mp4v) en H.264, lisible dans un navigateur."""
    dst = src.with_name(src.stem + "_web.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(dst)], check=True, capture_output=True)
    return dst


def summarize(jsonl: Path, fps: float, frames: int) -> str:
    """Résumé en français calculé à partir du fichier JSONL."""
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    counts = [len(r["tracks"]) for r in rows] or [0]
    dists = [t["distance_m"] for r in rows for t in r["tracks"] if t["distance_m"] is not None]
    md = [f"- **Vitesse moyenne** : {fps:.1f} images/s",
          f"- **Images traitées** : {frames}",
          f"- **Personnes par image** : {sum(counts) / len(counts):.1f} en moyenne, {max(counts)} au maximum"]
    for c in (rows[-1]["lines"] if rows else {}).values():
        md.append(f"- **Ligne de comptage** : {c['in']} entrées / {c['out']} sorties")
    if rows and rows[0]["zones"]:
        alerts = sum(1 for r in rows if any(r["zones"].values()))
        md.append(f"- **Alertes zone interdite** : {alerts} image(s) avec au moins une personne dans la zone")
    md.append(f"- **Personne la plus proche** : {min(dists):.1f} m" if dists
              else "- **Personne la plus proche** : non disponible")
    return "\n".join(md)


def analyse(source_type, url, upload, start, seconds, model, conf, width, method, fov, height, traces, layout,
            heatmap, line_on, lx1, ly1, lx2, ly2, zone_on, *zone_pts, progress=gr.Progress(track_tqdm=True)):
    """Lance le vrai pipeline et renvoie (vidéo, résumé, fichier JSONL, réglages utilisés)."""
    if source_type == "Webcam":
        source = "0"
    elif source_type == "Fichier vidéo":
        source = upload
    else:
        source = (url or "").strip()
    if not source:
        raise gr.Error("Indiquez une source : URL YouTube, fichier vidéo ou webcam.")
    cfg = build_cfg(source, start, min(seconds, 60), model, conf, width, method, fov, height, traces, layout,
                    heatmap, line_on, (lx1, ly1, lx2, ly2), zone_on, zone_pts)
    progress(0, desc="Chargement des modèles…")
    t0 = time.time()
    try:
        out = run(cfg)
        video = to_h264(Path(out["video"]))
    except Exception as e:  # message lisible dans l'interface
        raise gr.Error(f"L'analyse a échoué : {e}") from e
    summary = summarize(Path(out["jsonl"]), out["fps"], out["frames"])
    summary += f"\n- **Durée totale du calcul** : {time.time() - t0:.0f} s"
    used = {"source_type": source_type, "start": start, "seconds": seconds, "model": model, "confidence": conf,
            "width": width, "distance": method, "fov": fov, "person_height": height, "traces": traces,
            "layout": layout, "heatmap": heatmap, "lines": cfg["zones"]["lines"],
            "polygons": cfg["zones"]["polygons"]}
    return str(video), summary, str(out["jsonl"]), used


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
    with gr.Blocks(title="VisionTrack — démo locale") as ui:
        gr.Markdown("# VisionTrack — démo locale\n"
                    "Détection, tracking et distance des personnes en temps réel — **100 % local, "
                    "aucune image envoyée dans le cloud.**")
        last = gr.State({})

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Source vidéo")
                source_type = gr.Radio(["YouTube", "Fichier vidéo", "Webcam"], value="YouTube",
                                       label="Type de source")
                url = gr.Textbox(label="URL YouTube", value=BASE["source"])
                upload = gr.File(label="Fichier vidéo", file_types=["video"], type="filepath", visible=False)
                help_text("La webcam utilise la caméra d'index 0 de ce PC.")
                start = gr.Slider(0, 3600, value=BASE["start_seconds"], step=1, label="Début de l'extrait (s)")
                seconds = gr.Slider(1, 60, value=10, step=1, label="Durée (s, 60 max)")
                help_text("Plus la durée est longue, plus le calcul est long.")

                gr.Markdown("### 2. Réglages")
                with gr.Accordion("Détection", open=True):
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
                    help_text("Une valeur erronée fausse les distances. Smartphone ≈ 65–70°, grand-angle ≈ 90°+.")
                    height = gr.Slider(1.4, 2.0, value=dist["person_height_m"], step=0.01,
                                       label="Taille moyenne d'une personne (m)")
                    help_text("Sert au calcul géométrique : une taille supposée plus petite rapproche les estimations.")
                with gr.Accordion("Affichage", open=False):
                    traces = gr.Checkbox(value=r["traces"], label="Afficher les trajectoires")
                    help_text("Montre le chemin de chaque personne ; peut surcharger l'image dans une foule.")
                    layout = gr.Radio([("Incrustation de la profondeur", "inset"), ("Côte à côte", "side_by_side")],
                                      value=r["layout"], label="Disposition")
                    help_text("Côte à côte = vidéo et carte de profondeur, image deux fois plus large.")
                    heatmap = gr.Checkbox(value=r["heatmap"], label="Carte de fréquentation (heatmap)")
                    help_text("Superpose les zones les plus fréquentées sur l'image.")
                with gr.Accordion("Ligne de comptage et zone interdite", open=False):
                    zl = BASE["zones"]["lines"][0]
                    line_on = gr.Checkbox(value=True, label="Ligne de comptage")
                    help_text("Compte les entrées / sorties au franchissement de la ligne. Coordonnées "
                              "relatives : 0 = gauche / haut, 1 = droite / bas.")
                    with gr.Row():
                        lx1, ly1 = num(zl["start"][0], "Départ x"), num(zl["start"][1], "Départ y")
                        lx2, ly2 = num(zl["end"][0], "Arrivée x"), num(zl["end"][1], "Arrivée y")
                    zone_on = gr.Checkbox(value=False, label="Zone interdite")
                    help_text("Alerte quand une personne entre dans le quadrilatère (4 points, "
                              "dans l'ordre autour de la zone).")
                    default_pts = [0.6, 0.5, 0.95, 0.5, 0.95, 0.95, 0.6, 0.95]
                    zone_pts = []
                    for i in range(4):
                        with gr.Row():
                            zone_pts += [num(default_pts[2 * i], f"Point {i + 1} x"),
                                         num(default_pts[2 * i + 1], f"Point {i + 1} y")]

                go = gr.Button("Lancer l'analyse", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 3. Résultats")
                video = gr.Video(label="Vidéo annotée", autoplay=True)
                summary = gr.Markdown("*Lancez une analyse pour voir les résultats ici.*")
                jsonl = gr.File(label="Télécharger les données (JSONL)")

                gr.Markdown("### 4. Votre avis")
                gr.Markdown("**Vos retours restent sur ce PC** : rien n'est envoyé sur le réseau.")
                rating = gr.Radio([1, 2, 3, 4, 5], label="Note (1 = décevant, 5 = excellent)")
                cases = gr.CheckboxGroup(USE_CASES, label="Cas d'usage qui vous intéresse")
                comment = gr.Textbox(label="Commentaire", lines=3)
                email = gr.Textbox(label="E-mail (facultatif, si vous souhaitez être recontacté)")
                send = gr.Button("Envoyer mon avis")
                confirm = gr.Markdown()

        source_type.change(lambda s: (gr.update(visible=s == "YouTube"), gr.update(visible=s == "Fichier vidéo")),
                           source_type, [url, upload])
        inputs = [source_type, url, upload, start, seconds, model, conf, width, method, fov, height, traces,
                  layout, heatmap, line_on, lx1, ly1, lx2, ly2, zone_on, *zone_pts]
        go.click(analyse, inputs, [video, summary, jsonl, last])
        send.click(save_feedback, [rating, cases, comment, email, last], confirm)
    return ui


if __name__ == "__main__":
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860, share=False,
                              theme=gr.themes.Soft(), css=CSS)
