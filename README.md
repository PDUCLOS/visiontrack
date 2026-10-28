# VisionTrack

Détection, tracking et estimation de distance de personnes en temps réel, sur GPU NVIDIA grand public.
**100 % local** : aucune image ne quitte le site.

![Démo VisionTrack — rue piétonne de Chelmsford](docs/images/demo_chelmsford.png)

*Mesuré sur une RTX 4070 Ti SUPER, en 1280×720 : environ 32 images/s pour l'ensemble du traitement (détection RF-DETR medium, tracking, profondeur, distances, zones et rendu).*
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

## Estimation de distance

| Méthode | Principe | Quand l'utiliser |
|---|---|---|
| `pinhole` | distance = focale × 1,70 m / hauteur de la boîte | Rapide, personnes debout et entières |
| `depth` | carte de profondeur (métrique si le modèle est un « Metric ») | Scènes encombrées |
| `fusion` | profondeur relative recalée en mètres par la méthode géométrique | Par défaut, le plus stable |

Sans calibration de la caméra, on obtient une **estimation** (environ ±15–20 %). Pour une mission client, on renseigne le vrai champ de vision (`horizontal_fov_deg`) ou on calibre la caméra.

## Documentation

- [Argumentaire et storyboard de la vidéo de démo](docs/MARKETING.md)
- Schémas draw.io : [architecture](docs/diagrams/architecture.drawio), [local vs cloud](docs/diagrams/local_vs_cloud.drawio), [cas d'usage](docs/diagrams/use_cases.drawio)

## Licences des modèles

- RF-DETR, Depth Anything V2 Small : Apache 2.0, utilisables commercialement.
- Depth Anything V2 Base/Large et modèles « Metric » : vérifier la licence avant tout usage commercial.
- YOLO Ultralytics : AGPL-3.0, licence commerciale nécessaire pour un produit fermé.
