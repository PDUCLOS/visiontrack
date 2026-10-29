"""Point d'entrée : python run.py [--config config.yaml] [--source URL] [--show] …"""

import argparse

import yaml

from visiontrack.pipeline import run


def main():
    p = argparse.ArgumentParser(description="VisionTrack — détection, tracking et distance de personnes")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", help="URL YouTube, fichier, index webcam ou rtsp://")
    p.add_argument("--start", type=float, help="début de l'extrait en secondes")
    p.add_argument("--seconds", type=float, help="durée à traiter en secondes")
    p.add_argument("--detector", choices=["rfdetr", "yolo"])
    p.add_argument("--model", help="taille / poids du détecteur")
    p.add_argument("--distance", choices=["pinhole", "depth", "fusion", "none"])
    p.add_argument("--show", action="store_true", help="affiche la vidéo en direct")
    p.add_argument("--realtime", action="store_true", help="vitesse réelle : saute des images si besoin")
    a = p.parse_args()

    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if a.source is not None:
        cfg["source"] = a.source
    if a.start is not None:
        cfg["start_seconds"] = a.start
    if a.seconds is not None:
        cfg["max_seconds"] = a.seconds
    if a.detector:
        cfg["detector"]["backend"] = a.detector
    if a.model:
        cfg["detector"]["model"] = a.model
    if a.distance:
        cfg["distance"]["method"] = a.distance
    if a.show:
        cfg["output"]["show"] = True
    if a.realtime:
        cfg["realtime"] = True

    out = run(cfg)
    if out["video"]:
        print(f"Vidéo annotée : {out['video']}")


if __name__ == "__main__":
    main()
