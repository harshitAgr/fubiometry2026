"""Derive clinical parameters from predicted landmark points."""
from __future__ import annotations
import numpy as np
from scoring import geometry, param_specs


def derive_from_specs(specs, points: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in specs:
        if s.kind in ("distance", "diameter"):
            i, j = s.indices
            out[s.name] = geometry.euclidean(points[i], points[j])
        elif s.kind == "angle":
            a, v, b = s.indices
            out[s.name] = geometry.angle_deg(points[a], points[v], points[b])
        elif s.kind == "ellipse_perimeter":
            i, j, k, l = s.indices
            a = geometry.euclidean(points[i], points[j]) / 2.0
            b = geometry.euclidean(points[k], points[l]) / 2.0
            out[s.name] = geometry.ellipse_perimeter(a, b)
        else:
            raise ValueError(f"unknown param kind {s.kind!r} for {s.name!r}")
    return out


def derive_parameters(task_id: str, points: np.ndarray) -> dict[str, float]:
    return derive_from_specs(param_specs.PARAM_SPECS.get(task_id, []), points)
