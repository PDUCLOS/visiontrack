"""Lignes de comptage et zones polygonales, définies en coordonnées relatives dans config.yaml."""

from __future__ import annotations

import numpy as np
import supervision as sv


class Zones:
    def __init__(self, cfg: dict, frame_size: tuple[int, int]):
        w, h = frame_size
        cfg = cfg or {}
        self.lines = []
        for line in cfg.get("lines") or []:
            start = sv.Point(line["start"][0] * w, line["start"][1] * h)
            end = sv.Point(line["end"][0] * w, line["end"][1] * h)
            self.lines.append((line["name"], sv.LineZone(start=start, end=end,
                                                         triggering_anchors=[sv.Position.BOTTOM_CENTER])))
        self.polygons = []
        for poly in cfg.get("polygons") or []:
            pts = (np.array(poly["points"]) * [w, h]).astype(int)
            zone = sv.PolygonZone(polygon=pts, triggering_anchors=[sv.Position.BOTTOM_CENTER])
            self.polygons.append((poly["name"], zone))
        self.line_annot = sv.LineZoneAnnotator(thickness=2, text_scale=0.6, display_in_count=False,
                                               display_out_count=False)
        self.poly_annots = [sv.PolygonZoneAnnotator(zone=z, color=sv.Color.RED, thickness=2,
                                                    display_in_zone_count=False)
                            for _, z in self.polygons]

    def update(self, det: sv.Detections) -> dict:
        """Met à jour les compteurs et renvoie, par zone, les ID présents."""
        for _, line in self.lines:
            line.trigger(det)
        inside = {}
        for name, zone in self.polygons:
            mask = zone.trigger(det)
            ids = det.tracker_id[mask] if det.tracker_id is not None else []
            inside[name] = [int(i) for i in ids]
        return inside

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        for _, line in self.lines:
            frame = self.line_annot.annotate(frame, line_counter=line)
        for annot in self.poly_annots:
            frame = annot.annotate(frame)
        return frame

    def summary(self) -> list[str]:
        out = [f"Ligne {name} : {l.in_count} entrees / {l.out_count} sorties" for name, l in self.lines]
        out += [f"Zone {name} : {z.current_count} personne(s)" for name, z in self.polygons]
        return out

    def counts(self) -> dict:
        return {name: {"in": l.in_count, "out": l.out_count} for name, l in self.lines}
