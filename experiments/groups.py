"""Leak-free group keys: frames/planes from the same clip/patient must share a group so
GroupKFold never splits near-duplicates across train/val. AOP has no clip id in filenames,
so it is grouped separately (contiguous frame blocks) in folds.py; here we only expose its
frame index."""
from __future__ import annotations
import os
import re

CARDIAC = {"A4C", "PLAX", "PSAX", "IVC"}
_CLIP = re.compile(r"(?:DCM|PNG)_IM_(\d+)")     # clip number for both filename families
_SERIES = re.compile(r"(?:_s(\d+)|-(\d+))")      # optional series/sub-acquisition id
_PATIENT = re.compile(r"Patient(\d+)")
_FIRST_INT = re.compile(r"(\d+)")
_EXT = re.compile(r"^ext_([A-Za-z]+)_(.+)$")   # ingested external: ext_<SRC>_<origname>


def group_key(task_id: str, image_path: str) -> str:
    base = os.path.basename(image_path)
    m = _EXT.match(base)
    if m:
        # External (FP/UCL) images: group by source+subject so same-fetus scans
        # (e.g. FP PatientNNNNN_Plane*, UCL 002_AC/002_2AC) never split across folds.
        src, rest = m.group(1), m.group(2)
        pat = _PATIENT.search(rest)
        subj = pat.group(1) if pat else (_FIRST_INT.search(rest).group(1)
                                         if _FIRST_INT.search(rest) else rest)
        return f"{task_id}:ext:{src}:{subj}"
    if task_id in CARDIAC:
        clip = _CLIP.search(base)
        if clip:
            ser = _SERIES.search(base)
            series = (ser.group(1) or ser.group(2)) if ser else "0"
            return f"{task_id}:clip:{clip.group(1)}:{series}"
        return f"{task_id}:img:{base}"
    if task_id == "fetal_femur":
        p = _PATIENT.search(base)
        return f"femur:patient:{p.group(1)}" if p else f"femur:img:{base}"
    return f"{task_id}:img:{base}"


def aop_frame_index(image_path: str) -> int:
    m = _FIRST_INT.search(os.path.basename(image_path))
    return int(m.group(1)) if m else -1
