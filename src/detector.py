"""
detector.py — Wraps YOLOv8 model loading and inference.

Keeps all ultralytics-specific code in one place so the rest of
the codebase never imports from ultralytics directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from ultralytics import YOLO


def load_model(model_path: Union[str, Path]) -> YOLO:
    """Load a YOLOv8 model from *model_path*.

    Args:
        model_path: Absolute or relative path to a ``.pt`` weights file.

    Returns:
        A loaded :class:`ultralytics.YOLO` instance.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
        RuntimeError: If ultralytics fails to load the model.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at: {model_path}\n"
            "Place best.pt inside the models/ directory."
        )
    try:
        return YOLO(str(model_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLO model: {exc}") from exc


def run_inference(
    model: YOLO,
    image: Image.Image,
    conf: float = 0.25,
    iou: float = 0.45,
):
    """Run YOLOv8 inference on a PIL image.

    Args:
        model: A loaded :class:`ultralytics.YOLO` model.
        image: Input image (RGB).
        conf: Minimum confidence threshold (0–1).
        iou: IoU threshold for NMS (0–1).

    Returns:
        The first :class:`ultralytics.engine.results.Results` object.
    """
    image_rgb = image.convert("RGB")
    results = model.predict(
        source=np.array(image_rgb),
        conf=conf,
        iou=iou,
        verbose=False,
    )
    return results[0]
