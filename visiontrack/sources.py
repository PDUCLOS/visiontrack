"""Résolution de la source vidéo : YouTube, fichier, webcam ou flux RTSP."""

from __future__ import annotations

import re
from pathlib import Path

import cv2

YOUTUBE_RE = re.compile(r"(youtube\.com|youtu\.be)", re.IGNORECASE)


def download_youtube(url: str, cache_dir: Path) -> Path:
    """Télécharge la vidéo (sans audio, ≤1080p) une seule fois et renvoie son chemin."""
    import yt_dlp

    cache_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bv*[height<=1080][ext=mp4]/bv*[height<=1080]/b[height<=1080]",
        "outtmpl": str(cache_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        existing = list(cache_dir.glob(f"{info['id']}.*"))
        if existing:
            return existing[0]
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info))


def open_source(source, cache_dir: Path = Path("data/videos")) -> tuple[cv2.VideoCapture, str]:
    """Ouvre la source et renvoie (capture, nom lisible)."""
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return cv2.VideoCapture(int(source)), f"webcam{source}"
    if YOUTUBE_RE.search(str(source)):
        path = download_youtube(str(source), cache_dir)
        return cv2.VideoCapture(str(path)), path.stem
    cap = cv2.VideoCapture(str(source))
    return cap, Path(str(source)).stem or "stream"
