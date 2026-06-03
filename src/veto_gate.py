# src/veto_gate.py — Non-fish object detection veto gate
#
# Uses a lightweight COCO-pretrained YOLOv8n model to detect humans,
# hands, and common objects. If a "fish" detection overlaps with a
# person / non-fish object, it is vetoed (blocked).
#
# This prevents the fish model from misclassifying body parts,
# electronics, furniture, etc. as fish species.

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ── COCO classes to veto ──────────────────────────────────────────────────────
# If any of these overlap with a fish bbox, the fish detection is rejected.
VETO_CLASSES = {
    0: "person",
    15: "cat",
    16: "dog",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    73: "book",
    56: "chair",
    57: "couch",
    58: "potted plant",
    60: "dining table",
    62: "tv",
    24: "backpack",
    26: "handbag",
}

# Minimum confidence for the veto model
VETO_MIN_CONF = 0.35


def _iou(box_a: tuple, box_b: tuple) -> float:
    """Compute IoU between two (x1, y1, x2, y2) bounding boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def _overlap_ratio(fish_box: tuple, veto_box: tuple) -> float:
    """Fraction of the fish box that is covered by the veto box.

    Even if IoU is low (e.g. person is much larger), we should still
    veto if the fish bbox sits *inside* the person bbox.
    """
    xa = max(fish_box[0], veto_box[0])
    ya = max(fish_box[1], veto_box[1])
    xb = min(fish_box[2], veto_box[2])
    yb = min(fish_box[3], veto_box[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    fish_area = (fish_box[2] - fish_box[0]) * (fish_box[3] - fish_box[1])
    if fish_area <= 0:
        return 0.0
    return inter / fish_area


class VetoGate:
    """
    Lightweight veto gate that detects non-fish objects and blocks
    overlapping fish detections.

    Usage::

        gate = VetoGate()
        veto_boxes = gate.detect_veto_objects(frame)
        if not gate.is_vetoed(fish_bbox, veto_boxes):
            # Accept fish detection
    """

    def __init__(self, veto_iou: float = 0.15, veto_overlap: float = 0.40):
        """
        Parameters
        ----------
        veto_iou : float
            Minimum IoU between fish bbox and veto bbox to trigger a veto.
        veto_overlap : float
            Minimum fraction of the fish bbox covered by a veto bbox
            to trigger a veto (catches fish boxes inside person boxes).
        """
        self.veto_iou = veto_iou
        self.veto_overlap = veto_overlap

        # Load the lightweight COCO-pretrained model
        print("  Loading veto gate model (yolov8n.pt) ...")
        self._model = YOLO("yolov8n.pt")
        self._veto_class_ids = set(VETO_CLASSES.keys())
        print(f"  Veto gate ready — blocking: "
              f"{', '.join(VETO_CLASSES.values())}")

    def detect_veto_objects(
        self,
        frame: np.ndarray,
    ) -> list[tuple[int, int, int, int, str]]:
        """
        Run the COCO model on the frame and return bounding boxes
        for any non-fish objects detected.

        Returns
        -------
        list of (x1, y1, x2, y2, class_name) tuples
        """
        results = self._model(
            frame,
            imgsz=320,          # small size for speed
            verbose=False,
            conf=VETO_MIN_CONF,
            classes=list(self._veto_class_ids),
        )[0]

        veto_boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = VETO_CLASSES.get(cls_id, "unknown")
                veto_boxes.append((x1, y1, x2, y2, cls_name))

        return veto_boxes

    def is_vetoed(
        self,
        fish_box: tuple[int, int, int, int],
        veto_boxes: list[tuple],
    ) -> str | None:
        """
        Check if a fish detection should be vetoed.

        Returns
        -------
        str or None
            The name of the veto class if vetoed, or None if the
            fish detection is accepted.
        """
        for vb in veto_boxes:
            veto_bbox = vb[:4]
            veto_name = vb[4]

            iou = _iou(fish_box, veto_bbox)
            overlap = _overlap_ratio(fish_box, veto_bbox)

            if iou >= self.veto_iou or overlap >= self.veto_overlap:
                return veto_name

        return None
