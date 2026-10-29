# Dimensionnement matériel

Ce guide aide à choisir le matériel d'un déploiement VisionTrack. Il s'appuie sur des **mesures réelles** sur le PC de développement, puis sur des **estimations** pour les autres configurations (signalées comme telles).

## 1. Mesures sur le PC de développement

- **Matériel** : AMD Ryzen 9 3900X (12 cœurs / 24 threads), 32 Go de RAM, NVIDIA RTX 4070 Ti SUPER (16 Go), Windows 11.
- **Vidéo de test** : rue piétonne de Chelmsford, environ 12 personnes à l'image, traitée en 1280×720 sauf indication contraire.
- **Pour reproduire** : `python demo/benchmark.py`, résultats dans `outputs/benchmark/benchmark.json`.

| Configuration | Modèle | Image | Distance | Images/s | Mémoire GPU* | RAM | CPU |
|---|---|---|---|---:|---:|---:|---:|
| Complet précision | RF-DETR large | 1280 px | fusion, profondeur à chaque image | **18,8** | 0,6 Go | 2,6 Go | 24 % |
| **Complet (défaut)** | RF-DETR medium | 1280 px | fusion, profondeur 1 image sur 2 | **29,1** | 0,7 Go | 3,1 Go | 22 % |
| Complet haute définition | RF-DETR medium | 1920 px | fusion, 1 image sur 2 | **26,2** | 0,6 Go | 3,1 Go | 25 % |
| Sans profondeur | RF-DETR medium | 1280 px | géométrique | **36,1** | 0,5 Go | 3,1 Go | 23 % |
| Léger | RF-DETR nano | 960 px | géométrique | **40,4** | 0,8 Go | 3,3 Go | 21 % |
| Léger + profondeur | RF-DETR nano | 960 px | fusion, 1 image sur 3 | **36,9** | 0,8 Go | 3,2 Go | 20 % |
| **CPU seul, sans GPU** | RF-DETR nano | 960 px | géométrique | **26,4** | — | 2,2 Go | 37 % |

\* Mémoire réservée par les modèles (PyTorch). En pratique, il faut compter environ 1,5 à 2 Go de mémoire GPU par flux, contexte CUDA compris.

Les chiffres couvrent tout le traitement d'une image : détection, suivi, profondeur, distances, zones et rendu. Le décodage de la vidéo n'y est pas inclus.

**Ce que montrent ces mesures**

- **La mémoire GPU n'est pas le facteur limitant.** Moins de 1 Go par flux pour les modèles : une carte de 8 Go suffit largement pour un flux, et 12 à 16 Go permettent plusieurs flux.
- **La profondeur est l'étape la plus coûteuse.** Sur GPU, elle prend environ 17 ms par image. Sur CPU, elle monte à **349 ms par image**, ce qui la rend inutilisable en temps réel sans carte graphique.
- **Le détecteur seul tourne bien sur un CPU puissant** : 28 ms par image en nano, 32 ms en medium, sur 12 cœurs. Sans GPU, le comptage et le suivi restent donc possibles en temps réel, avec la seule distance géométrique.
- **Sur GPU, le détecteur est limité par le CPU**, à cause du coût de lancement des opérations GPU. Le nano n'est pas beaucoup plus rapide que le medium, et un CPU correct reste important même avec une bonne carte.

## 2. Configurations recommandées

Toutes les configurations sont en x86-64, sous Windows ou Linux, avec un GPU NVIDIA (CUDA 12). Les images/s indiquées hors PC de développement sont des **estimations**.

### Minimum : démonstration ou un point de comptage, sans GPU

- **Processeur** : 8 cœurs récents (Ryzen 7 5700X, Core i7-12700 ou équivalent).
- **Mémoire** : 16 Go de RAM.
- **Stockage** : SSD avec 20 Go libres (modèles environ 1,5 Go, plus les vidéos).
- **Ce que ça permet** : détection, suivi, comptage, zones et distance géométrique en 960 px. Pas de carte de profondeur.
- **Vitesse estimée** : 10 à 25 images/s selon le processeur (26 mesurées avec 12 cœurs).

### Minimum avec toutes les fonctions : un flux temps réel

- **GPU** : NVIDIA RTX 3060 (12 Go) ou RTX 4060 / 4060 Ti (8 Go).
- **Processeur** : 6 à 8 cœurs (Ryzen 5 5600, Core i5-12400 ou plus).
- **Mémoire** : 16 Go de RAM.
- **Stockage** : SSD avec 50 Go libres.
- **Vitesse estimée** : 15 à 22 images/s en configuration complète, 25 à 30 images/s sans profondeur. C'est assez pour une caméra de surveillance, qui filme généralement entre 15 et 25 images/s.

### Recommandé : 1 à 2 caméras, toutes fonctions, démo fluide

- **GPU** : RTX 4070, 4070 Ti ou 4070 Ti SUPER (12 à 16 Go).
- **Processeur** : 8 à 12 cœurs (Ryzen 7 / Ryzen 9, Core i7).
- **Mémoire** : 32 Go de RAM.
- **Vitesse** : **29 images/s mesurées** en configuration complète. C'est le PC de développement.

### Multi-caméras : 4 à 8 flux sur site

- **GPU** : RTX 4090 (24 Go), RTX 5000 Ada ou NVIDIA L4 en serveur.
- **Processeur** : 16 cœurs.
- **Mémoire** : 64 Go de RAM.
- **Configuration** :
  - modèles exportés en TensorRT ;
  - profondeur calculée 1 image sur 3 ;
  - 10 à 15 images/s par caméra, ce qui suffit pour compter et pour déclencher des alertes ;
  - décodage vidéo matériel par NVDEC.
- **Vitesse estimée** : 60 à 120 images/s au total, à répartir entre les caméras. Le chiffre est à valider par une mesure sur la machine cible.

### Embarqué : boîtier sur site, faible consommation

- **Matériel** : NVIDIA Jetson Orin NX 16 Go ou AGX Orin (Linux ARM).
- **Ce qu'il faut prévoir** : un portage TensorRT des modèles, avec la configuration légère (nano, distance géométrique ou profondeur peu fréquente).
- **Vitesse estimée** : 10 à 20 images/s. Le Jetson Orin Nano est trop juste pour la profondeur.

## 3. Autres points à prévoir

- **Caméras** : flux RTSP en H.264 ou H.265, entre 1080p et 4K selon la distance à couvrir. Une résolution supérieure aide à détecter les silhouettes lointaines.
- **Réseau local** : compter environ 4 à 8 Mbit/s par caméra 1080p. Rien ne sort du site : seuls les comptages et les événements JSON peuvent être exportés, si le client le souhaite.
- **Système** : un pilote NVIDIA récent (CUDA 12.x) et Python 3.12. L'installation est décrite dans le README.
- **Stockage des vidéos annotées** : en 1280×720, environ 0,3 Go par heure et par caméra, plus ou moins selon la qualité choisie. Sans vidéo enregistrée, le JSON seul ne pèse que quelques Mo par heure.
