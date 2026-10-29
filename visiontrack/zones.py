"""Lignes de comptage et zones polygonales, définies en coordonnées relatives dans config.yaml."""

from __future__ import annotations

import numpy as np
import supervision as sv


class Zones:
    def __init__(self, cfg: dict, frame_size: tuple[int, int]):
        w, h = frame_size
        cfg = cfg or {}
        self.class_names: dict[int, str] = {}  # id de classe COCO → nom, fourni par le pipeline
        self.lines = []
        for line in cfg.get("lines") or []:
            start = sv.Point(line["start"][0] * w, line["start"][1] * h)
            end = sv.Point(line["end"][0] * w, line["end"][1] * h)
            # franchissement validé sur plusieurs images : une boîte qui « tremble » sur la ligne ne compte pas
            self.lines.append((line["name"], sv.LineZone(start=start, end=end,
                                                         triggering_anchors=[sv.Position.BOTTOM_CENTER],
                                                         minimum_crossing_threshold=line.get("min_frames", 4))))
        self.polygons = []
        for poly in cfg.get("polygons") or []:
            pts = (np.array(poly["points"]) * [w, h]).astype(int)
            zone = sv.PolygonZone(polygon=pts, triggering_anchors=[sv.Position.BOTTOM_CENTER])
            self.polygons.append((poly["name"], zone))
        self.line_annot = sv.LineZoneAnnotator(thickness=2, text_scale=0.6, display_in_count=False,
                                               display_out_count=False)
        # zone au repos : contour orange ; zone occupée : remplissage rouge (alerte)
        self.poly_annots = [(sv.PolygonZoneAnnotator(zone=z, color=sv.Color.from_hex("#FFA500"), thickness=2,
                                                     display_in_zone_count=False),
                             sv.PolygonZoneAnnotator(zone=z, color=sv.Color.RED, thickness=3,
                                                     display_in_zone_count=False, opacity=0.35))
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
        for (_, zone), (idle, alert) in zip(self.polygons, self.poly_annots):
            frame = (alert if zone.current_count else idle).annotate(frame)
        return frame

    def summary(self) -> list[str]:
        from .render import FR_PLURAL

        out = []
        for name, l in self.lines:
            out.append(f"Ligne {name} : {l.in_count} entrees / {l.out_count} sorties")
            if len(self.class_names) > 1 and (l.in_count or l.out_count):
                per = {c: l.in_count_per_class.get(c, 0) + l.out_count_per_class.get(c, 0)
                       for c in set(l.in_count_per_class) | set(l.out_count_per_class)}
                out.append("   " + "  ".join(f"{FR_PLURAL.get(self.class_names.get(c, ''), c)} {k}"
                                             for c, k in sorted(per.items(), key=lambda x: -x[1])))
        out += [f"ALERTE zone {name} : {z.current_count} personne(s)" if z.current_count
                else f"Zone {name} : libre" for name, z in self.polygons]
        return out

    def counts(self) -> dict:
        out = {}
        for name, l in self.lines:
            out[name] = {"in": l.in_count, "out": l.out_count}
            if len(self.class_names) > 1:
                out[name]["par_classe"] = {
                    self.class_names.get(c, str(c)): {"in": l.in_count_per_class.get(c, 0),
                                                      "out": l.out_count_per_class.get(c, 0)}
                    for c in set(l.in_count_per_class) | set(l.out_count_per_class)}
        return out
