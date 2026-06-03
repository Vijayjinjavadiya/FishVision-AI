# src/tracker.py — IoU-based fish tracker for deduplication
#
# Tracks detected fish across frames using bounding-box IoU overlap.
# Each unique fish gets a persistent track_id. A fish is only logged
# (and its frame saved) on the FIRST frame it appears — never again.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackedFish:
    """Internal state for a single tracked fish."""
    track_id: int
    species: str
    best_conf: float
    bbox: tuple[int, int, int, int]      # (x1, y1, x2, y2)
    last_seen: float                      # time.monotonic()
    logged: bool = False                  # True once written to CSV
    frames_seen: int = 1

    # Optional biometric data captured at first detection
    length_cm: Optional[float] = None
    width_cm: Optional[float] = None
    depth_mm: Optional[float] = None
    weight_g: Optional[float] = None
    maturity: str = "—"


def _iou(box_a: tuple, box_b: tuple) -> float:
    """Compute Intersection-over-Union between two (x1, y1, x2, y2) boxes."""
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


def _box_center(box: tuple) -> tuple[float, float]:
    """Return (cx, cy) center of a bounding box."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _center_distance(box_a: tuple, box_b: tuple) -> float:
    """Euclidean distance between the centers of two boxes."""
    ca = _box_center(box_a)
    cb = _box_center(box_b)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


class FishTracker:
    """
    Simple IoU + center-distance tracker that assigns persistent IDs
    to detected fish across frames.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to consider two detections the same fish (0.2–0.4 works
        well for slow-moving fish in an aquarium / pond).
    max_distance_px : float
        Maximum center-to-center pixel distance to match even when IoU is low
        (handles cases where the fish moves fast between frames).
    max_age_s : float
        Seconds a track is kept alive without any matching detection before
        being retired.
    min_bbox_area_px : int
        Minimum bounding-box area (in px²).  Detections smaller than this are
        discarded as noise.
    max_aspect_ratio : float
        Maximum bbox width / height (or height / width) ratio.  Real fish
        rarely exceed ~5:1; extreme ratios usually indicate hands / arms.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_distance_px: float = 120,
        max_age_s: float = 3.0,
        min_bbox_area_px: int = 1500,
        max_aspect_ratio: float = 5.0,
    ):
        self.iou_threshold = iou_threshold
        self.max_distance_px = max_distance_px
        self.max_age_s = max_age_s
        self.min_bbox_area_px = min_bbox_area_px
        self.max_aspect_ratio = max_aspect_ratio

        self._tracks: dict[int, TrackedFish] = {}
        self._next_id: int = 1
        self._total_unique: int = 0      # cumulative unique fish count

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def total_unique(self) -> int:
        """Total number of unique fish detected since start."""
        return self._total_unique

    @property
    def active_tracks(self) -> list[TrackedFish]:
        """Currently active (alive) tracks."""
        return list(self._tracks.values())

    def is_valid_detection(
        self,
        x1: int, y1: int, x2: int, y2: int,
    ) -> bool:
        """Return True if the bounding box passes sanity checks."""
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return False
        area = w * h
        if area < self.min_bbox_area_px:
            return False
        aspect = max(w / h, h / w)
        if aspect > self.max_aspect_ratio:
            return False
        return True

    def update(
        self,
        detections: list[dict],
    ) -> list[tuple[TrackedFish, bool]]:
        """
        Update tracker with this frame's detections.

        Parameters
        ----------
        detections : list[dict]
            Each dict must contain keys:
                species, conf, x1, y1, x2, y2
            and optionally:
                length_cm, width_cm, depth_mm, weight_g, maturity

        Returns
        -------
        list of (TrackedFish, is_new)
            ``is_new`` is True only the *first* time this fish is seen —
            meaning it should be logged to CSV and its frame saved.
        """
        now = time.monotonic()
        results: list[tuple[TrackedFish, bool]] = []

        # ── Try to match each detection to an existing track ──────────────
        unmatched_dets = list(range(len(detections)))
        matched_track_ids: set[int] = set()

        # Build candidate pairs sorted by IoU (greedy matching)
        pairs: list[tuple[float, int, int]] = []   # (iou, det_idx, track_id)
        for di in unmatched_dets:
            det = detections[di]
            det_box = (det["x1"], det["y1"], det["x2"], det["y2"])
            for tid, trk in self._tracks.items():
                # Species must match for a valid assignment
                if det["species"] != trk.species:
                    continue
                iou = _iou(det_box, trk.bbox)
                dist = _center_distance(det_box, trk.bbox)
                # Accept match if IoU is above threshold OR center is close
                if iou >= self.iou_threshold or dist <= self.max_distance_px:
                    score = iou + (1.0 - dist / (self.max_distance_px * 2))
                    pairs.append((score, di, tid))

        # Sort descending by score (best matches first)
        pairs.sort(key=lambda x: x[0], reverse=True)

        remaining_dets = set(unmatched_dets)
        for _score, di, tid in pairs:
            if di not in remaining_dets or tid in matched_track_ids:
                continue
            # ── Update existing track ─────────────────────────────────────
            det = detections[di]
            trk = self._tracks[tid]
            trk.bbox = (det["x1"], det["y1"], det["x2"], det["y2"])
            trk.last_seen = now
            trk.frames_seen += 1
            if det["conf"] > trk.best_conf:
                trk.best_conf = det["conf"]

            remaining_dets.discard(di)
            matched_track_ids.add(tid)
            results.append((trk, False))  # not new

        # ── Create new tracks for unmatched detections ────────────────────
        for di in remaining_dets:
            det = detections[di]
            trk = TrackedFish(
                track_id=self._next_id,
                species=det["species"],
                best_conf=det["conf"],
                bbox=(det["x1"], det["y1"], det["x2"], det["y2"]),
                last_seen=now,
                length_cm=det.get("length_cm"),
                width_cm=det.get("width_cm"),
                depth_mm=det.get("depth_mm"),
                weight_g=det.get("weight_g"),
                maturity=det.get("maturity", "—"),
            )
            self._tracks[self._next_id] = trk
            self._next_id += 1
            self._total_unique += 1
            results.append((trk, True))   # NEW fish!

        # ── Expire old tracks ─────────────────────────────────────────────
        expired = [
            tid for tid, trk in self._tracks.items()
            if (now - trk.last_seen) > self.max_age_s
        ]
        for tid in expired:
            del self._tracks[tid]

        return results
