"""
predictor.py — Converts raw YOLO results into structured DataFrames.

This module handles all post-processing: parsing bounding boxes,
computing optional length estimates, and assigning life-stage labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Column names kept in a single place so the rest of the app never
# hard-codes strings.
DETECTION_COLUMNS = [
    "fish_id",
    "class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_width_px",
    "bbox_height_px",
    "estimated_length_cm",
    "life_stage",
]


def detections_to_dataframe(
    result,
    cm_per_pixel: float = 0.0,
    adult_threshold_cm: float = 10.0,
) -> pd.DataFrame:
    """Parse a YOLO result object into a tidy DataFrame.

    Args:
        result: A single ``ultralytics.engine.results.Results`` object.
        cm_per_pixel: Scale factor for length estimation.
            Pass ``0`` (default) to skip length / life-stage columns.
        adult_threshold_cm: Fish longer than this value (cm) are
            classified as adults. Only used when *cm_per_pixel* > 0.

    Returns:
        A :class:`pandas.DataFrame` with one row per detected fish.
        Returns an empty DataFrame (with correct columns) when no
        detections are present.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return pd.DataFrame(columns=DETECTION_COLUMNS)

    names = result.names
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    rows = []
    for idx, (box, class_id, conf) in enumerate(
        zip(boxes, classes, confidences), start=1
    ):
        x1, y1, x2, y2 = box
        width_px = x2 - x1
        height_px = y2 - y1

        if cm_per_pixel > 0:
            estimated_length_cm = round(float(width_px * cm_per_pixel), 2)
            life_stage = "Adult" if estimated_length_cm >= adult_threshold_cm else "Juvenile"
        else:
            estimated_length_cm = np.nan
            life_stage = ""

        rows.append(
            {
                "fish_id": idx,
                "class": names.get(class_id, str(class_id)),
                "confidence": round(float(conf), 3),
                "x1": round(float(x1), 1),
                "y1": round(float(y1), 1),
                "x2": round(float(x2), 1),
                "y2": round(float(y2), 1),
                "bbox_width_px": round(float(width_px), 1),
                "bbox_height_px": round(float(height_px), 1),
                "estimated_length_cm": estimated_length_cm,
                "life_stage": life_stage,
            }
        )

    return pd.DataFrame(rows)


def summary_stats(df: pd.DataFrame) -> dict:
    """Return high-level summary statistics for a detections DataFrame.

    Args:
        df: Output of :func:`detections_to_dataframe`.

    Returns:
        A dict with keys ``total``, ``species``, ``avg_confidence``.
    """
    if df.empty:
        return {"total": 0, "species": 0, "avg_confidence": 0.0}

    return {
        "total": len(df),
        "species": df["class"].nunique(),
        "avg_confidence": round(float(df["confidence"].mean()), 3),
    }
