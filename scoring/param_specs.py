"""Per-task parameter definitions: how each clinical parameter is derived from landmarks.

kind:
  'distance'          indices=(i, j)            -> euclidean(p_i, p_j)
  'angle'             indices=(a, vertex, b)    -> angle_deg(p_a, p_vertex, p_b)
  'ellipse_perimeter' indices=(i, j, k, l)      -> ellipse_perimeter(|p_i-p_j|/2, |p_k-p_l|/2)
  'diameter'          indices=(i, j)            -> euclidean(p_i, p_j)  (alias of distance)

PARAM_SPECS is populated from source docs in Task 3 Step 6 (deferred).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str            # 'distance' | 'angle' | 'ellipse_perimeter' | 'diameter'
    indices: tuple[int, ...]


def _pairs(names: list[str]) -> list[ParamSpec]:
    """Consecutive landmark pairs -> distance params (cardiac convention)."""
    return [ParamSpec(n, "distance", (2 * i, 2 * i + 1)) for i, n in enumerate(names)]


# Populated from the baseline's per-task CSV column order + Cardiac Parameter
# Description docs (verified 2026-06-15). Cardiac tasks: each named parameter is the
# euclidean distance between a consecutive landmark pair, in CSV column order.
PARAM_SPECS: dict[str, list[ParamSpec]] = {
    # A4C: 16 landmarks -> 8 chamber dimensions (LV/RV/LA/RA x up-down/left-right).
    "A4C": _pairs(["LV_ud", "LV_lr", "RV_ud", "RV_lr", "LA_ud", "LA_lr", "RA_ud", "RA_lr"]),
    # PLAX: 22 landmarks -> 11 long-axis distances.
    "PLAX": _pairs(["LV", "RV", "IVS", "LVPW", "VAO", "STJ", "AAO", "AV", "LVOT", "LA", "RVOT"]),
    # PSAX: 4 landmarks -> 2 distances.
    "PSAX": _pairs(["RVOT", "PA"]),
    # IVC: 2 landmarks -> 1 diameter.
    "IVC": _pairs(["IVC"]),
    # FUGC: 2 landmarks -> cervical length.
    "FUGC": [ParamSpec("cervical_length", "distance", (0, 1))],
    # fetal_femur: 2 landmarks -> femur length.
    "fetal_femur": [ParamSpec("femur_length", "distance", (0, 1))],
    # HC: 4 landmarks = two perpendicular diameters through a shared centre -> ellipse.
    "HC": [ParamSpec("head_circumference", "ellipse_perimeter", (0, 1, 2, 3))],
    # FA (fetal abdomen): 4 landmarks = two perpendicular diameters -> ellipse (AC).
    "FA": [ParamSpec("abdominal_circumference", "ellipse_perimeter", (0, 1, 2, 3))],
    # AOP: Angle of Progression = angle at the symphysis apex between the symphysis axis and
    # the line to the fetal-head leading edge; HSD = symphysis->head distance. Triple chosen
    # EMPIRICALLY, gated to a clinical median of 90-180 deg (exact IUGC landmark roles still
    # unverified against the source paper) — see
    # tests/test_param_specs_clinical.py). (0,2,3) ~118 deg; switch to (1,2,3) ~164 deg only if
    # the source convention says so.
    "AOP": [ParamSpec("aop", "angle", (0, 2, 3)),
            ParamSpec("hsd", "distance", (2, 3))],
}
