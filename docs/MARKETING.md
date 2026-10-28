# VisionTrack — vision par ordinateur 100 % locale

## Comptez, suivez et mesurez les personnes sur votre site, sans qu'une seule image ne le quitte

VisionTrack détecte, suit et mesure la distance des personnes en temps réel, sur un simple PC équipé d'une carte NVIDIA, directement chez vous. Vos vidéos restent sur votre réseau : pas de cloud, pas d'abonnement par caméra, et des résultats exploitables tout de suite (comptages, alertes de zone, cartes de fréquentation).

*Patrice Duclos, ingénieur data / IA freelance, Lyon.*

---

## 5 bénéfices clés (le local d'abord)

1. **Vos images ne quittent jamais votre site.** Aucune vidéo n'est envoyée à un tiers : la conformité RGPD est pensée dès la conception, et vous restez maître de vos données. Si vous le souhaitez, seules des métriques anonymisées (comptages, JSON) sortent du site.
2. **Aucun abonnement par caméra ni facturation à l'image.** Le traitement tourne sur votre matériel : le coût ne grandit pas avec le volume de vidéo analysé.
3. **Faible latence, et fonctionne hors ligne.** Pas d'aller-retour Internet : l'analyse se fait sur le réseau local, même si la connexion est coupée.
4. **Adaptable en modifiant un seul fichier.** Source vidéo, modèle, classes, zones et seuils se règlent dans `config.yaml`, sans toucher au code.
5. **Modèles à licence ouverte, utilisables commercialement.** RF-DETR et Depth Anything V2 Small sont sous licence Apache 2.0 : pas de mauvaise surprise de licence pour un usage professionnel.

> Honnêteté technique : sans calibration de la caméra, la distance est une estimation (environ ±15–20 %). En mission, on renseigne le vrai champ de vision ou on calibre la caméra.

---

## Comment ça marche

Schéma complet : [`diagrams/architecture.drawio`](diagrams/architecture.drawio) · comparaison : [`diagrams/local_vs_cloud.drawio`](diagrams/local_vs_cloud.drawio)

1. **Branchez votre source vidéo.** Caméra IP (RTSP), webcam, fichier ou vidéo YouTube : on indique la source dans `config.yaml`.
2. **Le PC local analyse en temps réel.** RF-DETR détecte les personnes, ByteTrack leur donne des identifiants stables, puis la distance est estimée (géométrie + carte de profondeur Depth Anything V2) et croisée avec vos zones.
3. **Vous récupérez des résultats exploitables.** Vidéo MP4 annotée, fichier JSON Lines (une ligne par image : pistes, distances, zones) et aperçu en direct, le tout stocké chez vous.

---

## Cas d'usage

Vue d'ensemble : [`diagrams/use_cases.drawio`](diagrams/use_cases.drawio)

| Cas d'usage | Ce que VisionTrack apporte | Bénéfice |
|---|---|---|
| **Commerce de détail** | Comptage entrées/sorties, heatmap de fréquentation | Fréquentation et heatmap |
| **Sécurité bâtiment / site** | Zone interdite, distance aux machines | Alerte zone interdite, distances |
| **Événementiel et foules** | Comptage et densité par zone | Densité maîtrisée en direct |
| **Smart city** | Comptage de passages sur voirie, places, transports | Comptage sans cloud |
| **Entrepôt logistique** | Personnes à proximité des chariots | Piétons proches des chariots |
| **Analyse sportive** | Trajectoires et déplacements des joueurs | Trajectoires et déplacements |

---

## Mon offre : de l'audit au déploiement

| Étape | Contenu | Délai |
|---|---|---|
| **1. Audit** | Cadrage du besoin, analyse des caméras et du site, choix du matériel, points RGPD | Quelques jours |
| **2. POC** | Prototype sur vos vidéos, zones et seuils réglés, rapport de résultats | **2 semaines** |
| **3. Déploiement** | Installation sur votre PC GPU, réglages, documentation, formation de vos équipes | Selon le périmètre |

Prestation en freelance, basée à Lyon, à distance ou sur site. Le code et la configuration restent chez vous.

**Contact** : à compléter (Malt / LinkedIn / site) · GitHub : PDUCLOS

---

## Storyboard de la vidéo de démo (75 s)

Format : capture de l'aperçu VisionTrack, voix off ou sous-titres. Durée totale : 75 s (cible 60–90 s).

| # | Durée | Plan | Texte à l'écran |
|---|---|---|---|
| 1 | 0–6 s (6 s) | Accroche : vidéo brute d'une scène animée, sans annotation | « Et si votre vidéosurveillance comptait, sans rien envoyer dans le cloud ? » |
| 2 | 6–16 s (10 s) | Détection puis suivi : les boîtes apparaissent, chaque personne reçoit un **identifiant** et une **traînée** (trajectoire) | « Détection et suivi en temps réel — identifiants stables » |
| 3 | 16–28 s (12 s) | Incrustation de la **carte de profondeur** ; les étiquettes affichent les **distances** en mètres | « Distance estimée par fusion géométrie + profondeur » |
| 4 | 28–40 s (12 s) | Une **ligne de comptage** : le compteur entrées/sorties s'incrémente à chaque passage | « Comptage des entrées / sorties » |
| 5 | 40–52 s (12 s) | Une personne entre dans la **zone interdite** : la zone passe en rouge, **alerte** affichée | « Zone interdite : alerte immédiate » |
| 6 | 52–60 s (8 s) | Gros plan sur le **compteur FPS** ; option heatmap de fréquentation en fondu | « 30+ images/s sur une RTX 4070 Ti SUPER — détection, suivi et distance » |
| 7 | 60–68 s (8 s) | Schéma `local_vs_cloud` : flèche d'upload barrée à gauche, PC local sur le réseau à droite | « Pas de cloud, pas d'abonnement par caméra, conforme RGPD par conception » |
| 8 | 68–75 s (7 s) | **Carte de fin** sur fond uni, logo et contact | « **100 % local — vos images ne quittent jamais votre site** » · Patrice Duclos, Lyon |

Conseils de tournage : utiliser une vidéo dont les droits permettent la diffusion (ou vos propres images), activer `render.depth_inset`, `render.traces`, `render.fps` et, pour le plan 6, `render.heatmap` dans `config.yaml`, et définir la ligne `passage` et une zone interdite dans la section `zones`.
