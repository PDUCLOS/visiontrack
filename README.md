# VisionTrack

Détection, tracking et estimation de distance de personnes en temps réel, sur GPU NVIDIA grand public.
**100 % local** : aucune image ne quitte le site.

[![Vidéo de démo VisionTrack — cliquer pour la regarder](docs/images/demo_poster.jpg)](https://github.com/PDUCLOS/visiontrack/releases/download/v0.1.0/visiontrack_demo_web.mp4)

▶ **[Voir la vidéo de démo (1 min 57)](https://github.com/PDUCLOS/visiontrack/releases/download/v0.1.0/visiontrack_demo_web.mp4)** : pipeline, 8 exemples (suivi, distance, foule dense, nuit, comptage, zone interdite, heatmap), réglages et comparatif rapide / précision. [Version 1080p](https://github.com/PDUCLOS/visiontrack/releases/download/v0.1.0/visiontrack_demo.mp4) · [Release v0.1.0](https://github.com/PDUCLOS/visiontrack/releases/tag/v0.1.0)

![Démo VisionTrack — rue piétonne de Chelmsford](docs/images/demo_chelmsford.png)

*Mesuré sur une RTX 4070 Ti SUPER, en 1280×720 : environ 28 à 30 images/s pour l'ensemble du traitement (détection RF-DETR medium, tracking, profondeur, distances, zones et rendu).*
*Vidéo source : « Chelmsford High Street after Christmas 2022 », licence CC BY.*

- **Détection** : [RF-DETR](https://huggingface.co/Roboflow/rf-detr-medium) (Apache 2.0) ou YOLO (Ultralytics, AGPL-3.0)
- **Tracking** : ByteTrack (via [supervision](https://github.com/roboflow/supervision)), avec des identifiants stables
- **Distance** : méthode géométrique, profondeur [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), ou fusion des deux
- **Zones** : lignes de comptage entrées/sorties, zones interdites, heatmap de fréquentation
- **Sorties** : vidéo MP4 annotée et un fichier JSON Lines (une ligne par image : pistes, distances, zones)

Toute l'adaptation à un nouveau besoin se fait dans `config.yaml` : source vidéo, modèle, classes, zones, seuils.

## Installation (Windows / Linux, x86 + GPU NVIDIA)

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux : source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Utilisation

```bash
# configuration par défaut
python run.py

# une vidéo YouTube, 20 s à partir de la 10e seconde, avec aperçu en direct
python run.py --source "https://www.youtube.com/watch?v=..." --start 10 --seconds 20 --show

# webcam, méthode de distance géométrique seule
python run.py --source 0 --distance pinhole --show
```

Les résultats sont écrits dans `outputs/`, et les vidéos YouTube sont mises en cache dans `data/videos/`.

## Interface de démonstration (locale)

```bash
python app.py   # puis ouvrir http://127.0.0.1:7860
```

Interface Gradio pour montrer l'outil en rendez-vous : choix de la source (YouTube, fichier, webcam), tous les réglages avec leur explication, vidéo annotée et statistiques (personnes, comptages, alertes, distance minimale), export JSON, et formulaire « Votre avis » enregistré localement dans `feedback/feedback.jsonl`. Rien n'est envoyé sur le réseau.

## Estimation de distance

| Méthode | Principe | Quand l'utiliser |
|---|---|---|
| `pinhole` | distance = focale × 1,70 m / hauteur de la boîte | Rapide, personnes debout et entières |
| `depth` | carte de profondeur (métrique si le modèle est un « Metric ») | Scènes encombrées |
| `fusion` | profondeur relative recalée en mètres par la méthode géométrique | Par défaut, le plus stable |

Sans calibration de la caméra, on obtient une **estimation** (environ ±15–20 %). Pour une mission client, on renseigne le vrai champ de vision (`horizontal_fov_deg`) ou on calibre la caméra.

## Réglages pour de meilleurs résultats

Tous ces réglages se trouvent dans `config.yaml`.

| Objectif | Réglage | Effet / coût |
|---|---|---|
| Détecter les personnes lointaines | `detector.confidence: 0.30` | Plus de détections ; risque de quelques faux positifs |
| Plus de précision | `detector.model: large` | Meilleure détection des petites silhouettes, un peu plus lent |
| Petites silhouettes au loin | `resize_width: 1920` | Plus de pixels par personne, coût GPU plus élevé |
| Plus de vitesse | `detector.model: nano` ou `small`, `resize_width: 960` | Plus rapide, moins de détections lointaines |
| Distances justes | `distance.horizontal_fov_deg` = vrai champ de vision de la caméra | Corrige l'échelle de toutes les distances |
| Population particulière (enfants…) | `distance.person_height_m` | Taille de référence de la méthode géométrique |
| Personnes masquées (tables, voitures) | `distance.method: fusion` | La carte de profondeur prend le relais quand la silhouette est coupée |
| Distances moins « nerveuses » | `distance.smoothing: 0.4` | Lissage plus fort, réaction plus lente |
| Carte de profondeur plus fine | `distance.depth_input_size: 518`, `depth_every_n_frames: 1` | Profondeur plus détaillée et plus fluide, plus lente |
| ID stables malgré les occultations | `tracker.lost_track_buffer: 60` | Garde l'identité plus longtemps (≈ 2 s à 30 i/s) |
| Moins d'ID fantômes | `tracker.minimum_consecutive_frames: 5` | Une piste n'est créée qu'après 5 images |

**Mesure sur la même vidéo (Montréal)** : la vidéo de démo montre côte à côte le réglage rapide (`medium`, seuil 0,45, 1280 px) et le réglage précision (`large`, seuil 0,30, 1920 px), avec le nombre moyen de personnes détectées et la vitesse obtenue.

## Vidéo de démo

```bash
python demo/make_demo.py          # environ 2 minutes, en 1920×1080 H.264
python demo/make_demo.py --force  # recalcule toutes les scènes
```

Le script fait tourner le vrai pipeline sur chaque scène, puis monte une vidéo de présentation dans `outputs/demo/visiontrack_demo.mp4` : titre, schéma du pipeline, 8 exemples (suivi, distance, foule dense, nuit, comptage ×2, zone interdite, heatmap), réglages, comparatif rapide / précision, et conclusion. Les vidéos sources sont sous licence CC BY et créditées à l'écran.

## Documentation

- [Argumentaire et storyboard de la vidéo de démo](docs/MARKETING.md)
- Schémas draw.io : [architecture](docs/diagrams/architecture.drawio), [local vs cloud](docs/diagrams/local_vs_cloud.drawio), [cas d'usage](docs/diagrams/use_cases.drawio)

## Licences des modèles

- RF-DETR, Depth Anything V2 Small : Apache 2.0, utilisables commercialement.
- Depth Anything V2 Base/Large et modèles « Metric » : vérifier la licence avant tout usage commercial.
- YOLO Ultralytics : AGPL-3.0, licence commerciale nécessaire pour un produit fermé.
