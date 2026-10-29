"""Résolution de la source vidéo : YouTube, fichier, webcam ou flux RTSP."""

from __future__ import annotations

import re
from pathlib import Path

import cv2

YOUTUBE_RE = re.compile(r"(youtube\.com|youtu\.be)", re.IGNORECASE)


FULL_DOWNLOAD_MAX_S = 600  # au-delà de 10 min, on ne télécharge que l'extrait demandé
VIDEO_FORMAT = "bv*[height<=1080][ext=mp4]/bv*[height<=1080]/b[height<=1080]"


def cached_video(cache_dir: Path, vid: str) -> Path | None:
    """Vidéo complète déjà téléchargée (on ignore les téléchargements inachevés)."""
    files = [f for f in sorted(cache_dir.glob(f"{vid}.*")) if f.suffix not in (".part", ".ytdl")]
    return files[0] if files else None


def youtube_stream_url(url: str) -> str:
    """Adresse directe du flux vidéo YouTube (≤1080p) : lecture immédiate, sans téléchargement."""
    import yt_dlp

    fmt = "bv*[height<=1080][ext=mp4][protocol=https]/bv*[height<=1080][protocol=https]/b[height<=1080]"
    with yt_dlp.YoutubeDL({"format": fmt, "quiet": True, "noplaylist": True}) as ydl:
        return ydl.extract_info(url, download=False)["url"]


def download_youtube(url: str, cache_dir: Path, start: float = 0, seconds: float | None = None
                     ) -> tuple[Path, float]:
    """Télécharge la vidéo (sans audio, ≤1080p) une seule fois et renvoie (chemin, début à appliquer).

    Vidéo courte : téléchargée en entier (réutilisable pour d'autres extraits).
    Vidéo longue (direct de plusieurs heures…) : seul l'extrait [start, start + seconds] est téléchargé,
    le début à appliquer dans le fichier devient alors 0.
    """
    import yt_dlp

    cache_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": VIDEO_FORMAT,
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    vid, duration = info["id"], info.get("duration") or 0

    full = cached_video(cache_dir, vid)
    if full:
        return full, start
    if duration and duration <= FULL_DOWNLOAD_MAX_S or not seconds:
        with yt_dlp.YoutubeDL({**opts, "outtmpl": str(cache_dir / "%(id)s.%(ext)s")}) as ydl:
            info = ydl.extract_info(url, download=True)
            return Path(ydl.prepare_filename(info)), start

    s, e = int(start), int(start + seconds) + 1
    clip = sorted(cache_dir.glob(f"{vid}_{s}-{e}.*"))
    if clip:
        return clip[0], 0
    opts.update(outtmpl=str(cache_dir / f"{vid}_{s}-{e}.%(ext)s"),
                download_ranges=yt_dlp.utils.download_range_func(None, [(s, e)]))
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return sorted(cache_dir.glob(f"{vid}_{s}-{e}.*"))[0], 0


def youtube_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([\w-]{11})", url)
    return m.group(1) if m else None


def open_source(source, cache_dir: Path = Path("data/videos"), start: float = 0,
                seconds: float | None = None, youtube_mode: str = "stream") -> tuple[cv2.VideoCapture, str, float]:
    """Ouvre la source et renvoie (capture, nom lisible, début de l'extrait à appliquer en secondes).

    YouTube : vidéo du cache si elle y est déjà, sinon lecture directe du flux (youtube_mode="stream",
    démarrage immédiat) ou téléchargement préalable (youtube_mode="download", utile hors ligne).
    """
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return cv2.VideoCapture(int(source)), f"webcam{source}", 0
    if YOUTUBE_RE.search(str(source)):
        vid = youtube_id(str(source))
        cached = cached_video(cache_dir, vid) if vid else None
        if cached:
            return cv2.VideoCapture(str(cached)), cached.stem, start
        if youtube_mode == "stream":
            return cv2.VideoCapture(youtube_stream_url(str(source)), cv2.CAP_FFMPEG), vid or "youtube", start
        path, start = download_youtube(str(source), cache_dir, start, seconds)
        return cv2.VideoCapture(str(path)), path.stem, start
    cap = cv2.VideoCapture(str(source))
    return cap, Path(str(source)).stem or "stream", start
