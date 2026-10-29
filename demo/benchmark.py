"""Mesure des performances du pipeline selon la configuration (sert au dimensionnement matériel).

    python demo/benchmark.py            # toutes les configurations GPU + une configuration CPU
    python demo/benchmark.py --cpu      # (interne) une configuration lancée sans GPU

Résultat : outputs/benchmark/benchmark.json et un tableau dans la console.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "benchmark"
VIDEO = ROOT / "data" / "videos" / "Zeew8NpJb50.mp4"  # rue piétonne de Chelmsford, ~12 personnes à l'image

CONFIGS = [
    # nom, modèle, largeur, distance, profondeur 1 image sur n
    ("Complet précision", "large", 1280, "fusion", 1),
    ("Complet (défaut)", "medium", 1280, "fusion", 2),
    ("Complet 1920 px", "medium", 1920, "fusion", 2),
    ("Sans profondeur", "medium", 1280, "pinhole", 1),
    ("Léger", "nano", 960, "pinhole", 1),
    ("Léger + profondeur", "nano", 960, "fusion", 3),
]


def bench(name, model, width, method, every_n, seconds=10.0) -> dict:
    import psutil
    import torch
    import yaml

    from visiontrack.pipeline import run

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg.update(source=str(VIDEO), start_seconds=0, max_seconds=seconds, resize_width=width, realtime=False)
    cfg["output"].update(dir=str(OUT), name="bench", video=False, json=False, show=False)
    cfg["detector"]["model"] = model
    cfg["distance"].update(method=method, depth_every_n_frames=every_n)
    cfg["render"]["depth_inset"] = method in ("depth", "fusion")

    gpu = (torch.cuda.is_available() and torch.cuda.device_count() > 0)
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    proc = psutil.Process()
    cpu_samples, stop = [], threading.Event()

    def sample():
        psutil.cpu_percent(None)
        while not stop.wait(0.5):
            cpu_samples.append(psutil.cpu_percent(None))

    threading.Thread(target=sample, daemon=True).start()
    t0 = time.perf_counter()
    res = run(cfg)
    wall = time.perf_counter() - t0
    stop.set()
    return {
        "config": name, "model": model, "width": width, "distance": method, "depth_every_n": every_n,
        "device": torch.cuda.get_device_name(0) if gpu else "CPU",
        "fps_pipeline": round(res["fps"], 1), "frames": res["frames"], "wall_s": round(wall, 1),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if gpu else None,
        "ram_gb": round(proc.memory_info().rss / 1e9, 2),
        "cpu_percent": round(sum(cpu_samples) / max(len(cpu_samples), 1), 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.cpu:  # lancé dans un sous-processus sans GPU visible
        print(json.dumps(bench("CPU seul — léger", "nano", 960, "pinhole", 1, seconds=3.0)))
        return

    results = [bench(*c) for c in CONFIGS]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    out = subprocess.run([sys.executable, __file__, "--cpu"], env=env, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    last = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if last:
        results.append(json.loads(last[-1]))
    else:
        print("Mesure CPU impossible :", out.stderr[-500:])

    (OUT / "benchmark.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'Configuration':24s} {'Matériel':28s} {'i/s':>6s} {'VRAM':>6s} {'RAM':>6s} {'CPU%':>5s}")
    for r in results:
        vram = f"{r['vram_peak_gb']:.1f}" if r["vram_peak_gb"] is not None else "-"
        print(f"{r['config']:24s} {r['device'][:28]:28s} {r['fps_pipeline']:6.1f} {vram:>6s} "
              f"{r['ram_gb']:6.1f} {r['cpu_percent']:5.0f}")


if __name__ == "__main__":
    main()
