# src/oak_detector.py
# DepthAI v3 (3.6.x)  —  confirmed working pattern
#
# Correct v3 lifecycle:
#   1. connect_device()                — find & boot OAK-D Pro
#   2. build_pipeline(device)          — create nodes → .build(socket) → requestOutput → createOutputQueue → pipeline.start()
#   3. Caller reads from returned q_rgb / q_depth queues
#   4. device.close() when done

import sys
import time
import warnings

import depthai as dai
import numpy as np
from ultralytics import YOLO
from src.utils import resolve_model_path

# Suppress deprecation warnings (Camera node emits some internally)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── OAK-D Pro camera intrinsics ───────────────────────────────────────────────
FOCAL_LENGTH_MM = 3.37   # mm
SENSOR_WIDTH_MM = 5.37   # mm
IMAGE_WIDTH_PX  = 640    # preview width
IMAGE_HEIGHT_PX = 640    # preview height

# ── Load YOLO model ───────────────────────────────────────────────────────────
model = YOLO(str(resolve_model_path("models/best.pt")))


# ── depthai v3 preset shim ────────────────────────────────────────────────────
def _stereo_preset():
    """Return the best available StereoDepth PresetMode."""
    PM = dai.node.StereoDepth.PresetMode
    for name in ("DENSITY", "HIGH_DENSITY", "HIGH_DETAIL", "DEFAULT"):
        if hasattr(PM, name):
            return getattr(PM, name)
    raise AttributeError(
        f"No recognised StereoDepth PresetMode. Available: {dir(PM)}"
    )


# ── Math ──────────────────────────────────────────────────────────────────────
def pixel_to_real_mm(pixel_span: float,
                     depth_z_mm: float,
                     image_width_px: int = IMAGE_WIDTH_PX) -> float:
    """
    Convert a pixel span to real-world mm via the pinhole camera formula.
    No dataset required — pure geometry using known OAK-D Pro intrinsics.
    """
    return (pixel_span / image_width_px) * (SENSOR_WIDTH_MM / FOCAL_LENGTH_MM) * depth_z_mm


# ── Device connection (with retry) ────────────────────────────────────────────
def connect_device(max_retries: int = 5, retry_delay: float = 3.0) -> dai.Device:
    """
    Scan for an OAK-D device and connect to it.

    Returns a connected dai.Device.  Retries several times because in
    depthai v3 the device may take a few seconds to re-enumerate on USB
    after a previous session closes.
    """
    for attempt in range(1, max_retries + 1):
        devices = dai.Device.getAllAvailableDevices()
        if devices:
            dev = dai.Device(devices[0])
            print(f"  Connected to: {dev.getDeviceName()}  "
                  f"(USB {dev.getUsbSpeed().name})")
            return dev
        if attempt < max_retries:
            print(f"  No device found (attempt {attempt}/{max_retries}), "
                  f"retrying in {retry_delay:.0f}s ...")
            time.sleep(retry_delay)

    raise RuntimeError(
        "No OAK-D device found after several retries.\n"
        "  -> Check USB cable (must be USB 3)\n"
        "  -> Unplug and re-plug the device\n"
        "  -> Run:  python check_oak.py"
    )


# ── Pipeline builder ──────────────────────────────────────────────────────────
def build_pipeline(device: dai.Device):
    """
    Build and start a depthai v3 pipeline on the given device.

    Uses the v3 Camera.build(socket) → requestOutput() pattern.

    Returns:
        (pipeline, q_rgb, q_depth)
        — q_rgb:   queue delivering BGR frames  (640x640x3)
        — q_depth: queue delivering depth maps  (640x640, uint16, mm)
    """
    pipeline = dai.Pipeline(device)

    # ── Camera nodes: build with explicit board socket ────────────────────────
    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # ── Request output streams ────────────────────────────────────────────────
    rgb_out   = cam_rgb.requestOutput((IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX),
                                      dai.ImgFrame.Type.BGR888p)
    left_out  = cam_left.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8)
    right_out = cam_right.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8)

    # ── StereoDepth ───────────────────────────────────────────────────────────
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(_stereo_preset())
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)   # align to RGB
    stereo.setOutputSize(IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)

    left_out.link(stereo.left)
    right_out.link(stereo.right)

    # ── Create output queues (BEFORE start) ───────────────────────────────────
    q_rgb   = rgb_out.createOutputQueue(maxSize=4, blocking=False)
    q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

    # ── Start the pipeline on the device ──────────────────────────────────────
    pipeline.start()

    return pipeline, q_rgb, q_depth