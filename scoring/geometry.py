"""Pure geometric primitives for deriving clinical parameters from landmarks."""
from __future__ import annotations
import numpy as np


def euclidean(p, q) -> float:
    p = np.asarray(p, float); q = np.asarray(q, float)
    return float(np.linalg.norm(p - q))


def angle_deg(a, vertex, b) -> float:
    """Angle in degrees of the rays vertex->a and vertex->b, in [0, 180]."""
    a = np.asarray(a, float); v = np.asarray(vertex, float); b = np.asarray(b, float)
    u = a - v; w = b - v
    nu = np.linalg.norm(u); nw = np.linalg.norm(w)
    if nu == 0 or nw == 0:
        raise ValueError("degenerate angle: coincident points")
    cos = np.clip(np.dot(u, w) / (nu * nw), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def ellipse_perimeter(a: float, b: float) -> float:
    """Ramanujan II approximation of an ellipse perimeter from semi-axes a, b."""
    a = float(a); b = float(b)
    h = ((a - b) ** 2) / ((a + b) ** 2) if (a + b) else 0.0
    return float(np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h))))
