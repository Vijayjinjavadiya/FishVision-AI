# src/oak_runner.py — FishVision-AI OAK-D Pro live detection (depthai v3.6.x)
#
# Run from project root:
#   python -m src.oak_runner
#
# Press Q or Esc to quit.  Detections are auto-saved to logs/oak_detections.csv

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── project imports ────────────────────────────────────────────────────────────
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.biometrics import classify_maturity, predict_weight
from src.oak_detector import (
    IMAGE_WIDTH_PX,
    build_pipeline,
    connect_device,
    model,
    pixel_to_real_mm,
)

# ── constants ──────────────────────────────────────────────────────────────────
MIN_CONF     = 0.50
MIN_DEPTH_MM = 100       # < 10 cm → skip (sensor noise)
MAX_DEPTH_MM = 5_000     # > 5 m  → skip (unreliable range)
LOG_DIR      = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE     = LOG_DIR / "oak_detections.csv"
CSV_HEADER   = ["timestamp", "species", "confidence",
                "length_cm", "width_cm", "depth_mm",
                "weight_g", "maturity",
                "x1", "y1", "x2", "y2"]

# ── colour palette per class (BGR) ────────────────────────────────────────────
_PALETTE = [
    (0, 200, 100), (255, 100,   0), (0, 100, 255), (200, 200,   0),
    (180,   0, 180), (0, 220, 220), (255, 140,   0), (50, 205,  50),
    (0, 165, 255), (138,  43, 226), (255,  20, 147), (64, 224, 208),
    (255, 215,   0),
]


def _box_colour(cls_idx: int):
    return _PALETTE[cls_idx % len(_PALETTE)]


# ── depth sampling ─────────────────────────────────────────────────────────────
def get_depth_at_box(depth_frame: np.ndarray,
                     x1: int, y1: int, x2: int, y2: int) -> float | None:
    """Return median depth (mm) sampled from central 30 % of the bounding box."""
    cx1 = int(x1 + (x2 - x1) * 0.35)
    cx2 = int(x1 + (x2 - x1) * 0.65)
    cy1 = int(y1 + (y2 - y1) * 0.35)
    cy2 = int(y1 + (y2 - y1) * 0.65)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    roi   = depth_frame[cy1:cy2, cx1:cx2]
    valid = roi[(roi > MIN_DEPTH_MM) & (roi < MAX_DEPTH_MM)]
    return float(np.median(valid)) if len(valid) > 0 else None


# ── annotation ────────────────────────────────────────────────────────────────
def _draw_detection(frame, x1, y1, x2, y2, colour, lines):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness  = 2
    line_h     = 18

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    sq = 8
    for px, py in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.rectangle(frame,
                      (px - sq // 2, py - sq // 2),
                      (px + sq // 2, py + sq // 2),
                      colour, -1)

    for i, line in enumerate(lines):
        y_pos = y1 - 10 - (len(lines) - 1 - i) * line_h
        if y_pos < 15:
            y_pos = y1 + 15 + i * line_h
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(frame,
                      (x1 - 1, y_pos - th - 3),
                      (x1 + tw + 2, y_pos + 3),
                      (0, 0, 0), -1)
        cv2.putText(frame, line, (x1, y_pos),
                    font, font_scale, colour, thickness, cv2.LINE_AA)


def _draw_hud(frame, fps: float, n_fish: int):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (260, 60), (0, 0, 0), -1)
    cv2.putText(frame, "FishVision-AI  OAK-D Pro",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 120), 2)
    cv2.putText(frame, f"FPS: {fps:5.1f}   Fish: {n_fish}",
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, "Press Q to quit",
                (w - 160, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (120, 120, 120), 1)


# ── CSV logger ────────────────────────────────────────────────────────────────
def _open_csv():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    already_exists = LOG_FILE.exists()
    fh = LOG_FILE.open("a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if not already_exists:
        writer.writerow(CSV_HEADER)
    return fh, writer


# ── main ──────────────────────────────────────────────────────────────────────
def run():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_fh, csv_writer = _open_csv()

    print("=" * 60)
    print("  FishVision-AI  —  OAK-D Pro Live Detection")
    print("=" * 60)
    print(f"  Classes      : {len(model.names)}")
    print(f"  Min conf     : {MIN_CONF:.0%}")
    print(f"  Depth range  : {MIN_DEPTH_MM}–{MAX_DEPTH_MM} mm")
    print(f"  CSV log      : {LOG_FILE}")
    print("  Press Q or Esc to stop.")
    print("=" * 60)

    # ── Step 1: Connect device ────────────────────────────────────────────────
    print("\n  Scanning for OAK-D Pro ...")
    try:
        device = connect_device()
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        csv_fh.close()
        sys.exit(1)

    # ── Step 2: Build pipeline on device ──────────────────────────────────────
    print("  Building pipeline ...")
    try:
        pipeline, q_rgb, q_depth = build_pipeline(device)
    except Exception as e:
        print(f"\n[ERROR] Pipeline build failed: {e}")
        device.close()
        csv_fh.close()
        sys.exit(1)

    print("  Pipeline running. Detection loop started.\n")

    fps_timer   = time.perf_counter()
    frame_count = 0
    fps         = 0.0

    try:
        while True:
            # ── grab frames ───────────────────────────────────────────────────
            in_rgb   = q_rgb.get()
            in_depth = q_depth.get()

            frame       = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame()    # uint16, mm

            # ensure depth map matches RGB frame size
            fh, fw = frame.shape[:2]
            dh, dw = depth_frame.shape[:2]
            if (dh, dw) != (fh, fw):
                depth_frame = cv2.resize(
                    depth_frame, (fw, fh), interpolation=cv2.INTER_NEAREST
                )

            # ── YOLO inference ────────────────────────────────────────────────
            results = model(frame, imgsz=640, verbose=False)[0]
            n_fish  = 0
            ts      = datetime.now().isoformat(timespec="seconds")

            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < MIN_CONF:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_idx = int(box.cls[0])
                species = model.names[cls_idx]
                colour  = _box_colour(cls_idx)

                # ── depth ─────────────────────────────────────────────────────
                Z = get_depth_at_box(depth_frame, x1, y1, x2, y2)
                length_cm = width_cm = weight_g = None
                maturity  = "—"
                depth_label = "no depth"

                if Z is not None:
                    length_mm = pixel_to_real_mm(x2 - x1, Z, image_width_px=fw)
                    width_mm  = pixel_to_real_mm(y2 - y1, Z, image_width_px=fw)
                    length_cm = round(length_mm / 10, 1)
                    width_cm  = round(width_mm  / 10, 1)
                    weight_g  = predict_weight(species, length_cm)
                    maturity  = classify_maturity(species, length_cm)
                    depth_label = f"D:{Z/10:.0f}cm"

                # ── annotation ────────────────────────────────────────────────
                if length_cm is not None:
                    lines = [
                        f"{species}  {conf:.0%}",
                        f"L:{length_cm}cm  W:{width_cm}cm  {depth_label}",
                        f"{weight_g}g  {maturity}" if weight_g else maturity,
                    ]
                else:
                    lines = [f"{species}  {conf:.0%}", depth_label]

                _draw_detection(frame, x1, y1, x2, y2, colour, lines)
                n_fish += 1

                # ── log ───────────────────────────────────────────────────────
                csv_writer.writerow([
                    ts, species, f"{conf:.3f}",
                    length_cm, width_cm, Z,
                    weight_g, maturity,
                    x1, y1, x2, y2,
                ])

            csv_fh.flush()

            # ── FPS ───────────────────────────────────────────────────────────
            frame_count += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                fps        = frame_count / elapsed
                frame_count = 0
                fps_timer   = time.perf_counter()

            _draw_hud(frame, fps, n_fish)
            cv2.imshow("FishVision-AI  |  OAK-D Pro", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        csv_fh.close()
        cv2.destroyAllWindows()
        device.close()
        print(f"\n[INFO] Detections saved to {LOG_FILE}")
        print("[INFO] Done.")


if __name__ == "__main__":
    run()