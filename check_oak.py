#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_oak.py  -  FishVision-AI  System Diagnostic
===================================================
Run this BEFORE running oak_runner.py to verify that everything is in order.

Usage:
    python check_oak.py            # full check, camera optional
    python check_oak.py --camera   # also smoke-tests the live pipeline

Exit code 0  = all required checks passed
Exit code 1  = one or more required checks FAILED
"""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colours (work on Windows 10+ with ANSI enabled)
# ──────────────────────────────────────────────────────────────────────────────
import io
import os
# Force UTF-8 on Windows consoles
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    os.system("")   # enable ANSI escape codes on Windows 10+

try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(
        ctypes.windll.kernel32.GetStdHandle(-11), 7
    )
except Exception:
    pass   # non-Windows or already enabled

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

OK   = f"{GREEN}  [OK]{RESET}"
WARN = f"{YELLOW}  [WARN]{RESET}"
FAIL = f"{RED}  [FAIL]{RESET}"

PROJECT_ROOT = Path(__file__).resolve().parent
_failures    = []
_warnings    = []


def _header(title: str):
    print(f"\n{CYAN}{BOLD}{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}{RESET}")


def _pass(msg: str):
    print(f"{OK}  {msg}")


def _warn(msg: str):
    print(f"{WARN}  {msg}")
    _warnings.append(msg)


def _fail(msg: str):
    print(f"{FAIL}  {msg}")
    _failures.append(msg)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Python version
# ──────────────────────────────────────────────────────────────────────────────
def check_python():
    _header("1 · Python version")
    major, minor = sys.version_info[:2]
    ver = f"{major}.{minor}"
    if major == 3 and minor >= 10:
        _pass(f"Python {ver}  (3.10+ required)")
    elif major == 3 and minor >= 8:
        _warn(f"Python {ver} — depthai recommends 3.10+, but may still work")
    else:
        _fail(f"Python {ver} — 3.10+ is required")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Required packages
# ──────────────────────────────────────────────────────────────────────────────
REQUIRED_PACKAGES = {
    "depthai":      ("depthai",        "2.0"),
    "cv2":          ("opencv-python",  "4.5"),
    "numpy":        ("numpy",          "1.24"),
    "ultralytics":  ("ultralytics",    "8.2"),
    "streamlit":    ("streamlit",      "1.35"),
    "PIL":          ("Pillow",         "10.0"),
    "pandas":       ("pandas",         "2.0"),
}


def check_packages():
    _header("2 · Python packages")
    for import_name, (pip_name, min_ver) in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            _pass(f"{pip_name:<24} {ver}")
        except ImportError:
            if pip_name == "depthai":
                _fail(f"{pip_name} not installed  →  pip install depthai")
            else:
                _fail(f"{pip_name} not installed  →  pip install {pip_name}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Model weights
# ──────────────────────────────────────────────────────────────────────────────
def check_model():
    _header("3 · Model weights")
    model_path = PROJECT_ROOT / "models" / "best.pt"
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1_048_576
        _pass(f"models/best.pt  found  ({size_mb:.1f} MB)")
    else:
        _fail("models/best.pt  NOT FOUND — download from project Releases")

    # try loading the model (requires ultralytics)
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        _pass(f"YOLO model loaded  ({len(model.names)} classes)")
    except Exception as exc:
        _fail(f"Failed to load YOLO model: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Biometrics math sanity check
# ──────────────────────────────────────────────────────────────────────────────
def check_biometrics():
    _header("4 · Biometrics math (no camera needed)")
    try:
        from src.biometrics import classify_maturity, predict_weight

        # weight prediction
        w = predict_weight("GoldFish", 12.0)
        _pass(f"predict_weight('GoldFish', 12cm) = {w} g")

        # maturity
        m = classify_maturity("GoldFish", 12.0)
        _pass(f"classify_maturity('GoldFish', 12cm) = {m}  (expected Adult)")
        if m != "Adult":
            _warn("classify_maturity returned unexpected result — check MATURITY_LENGTH_CM")

        # pinhole formula — pure Python math, no depthai needed
        FOCAL_LENGTH_MM = 3.37
        SENSOR_WIDTH_MM = 5.37
        IMAGE_WIDTH_PX  = 640
        pixel_span, depth_z_mm = 200, 500
        mm = (pixel_span / IMAGE_WIDTH_PX) * (SENSOR_WIDTH_MM / FOCAL_LENGTH_MM) * depth_z_mm
        _pass(f"pixel_to_real_mm(200px, 500mm, 640w) = {mm:.1f} mm  "
              f"({'OK' if 700 < mm < 1000 else 'check formula'})")

    except Exception as exc:
        _fail(f"Biometrics check failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. OAK-D device detection
# ──────────────────────────────────────────────────────────────────────────────
def check_device_present():
    _header("5 · OAK-D hardware detection")
    try:
        import depthai as dai

        devices = dai.Device.getAllAvailableDevices()
        if devices:
            for d in devices:
                # depthai v3 uses d.name; v2 used d.getMxId()
                dev_id = getattr(d, "name", None) or getattr(d, "getMxId", lambda: str(d))()
                state  = d.state.name if hasattr(d, "state") else "AVAILABLE"
                _pass(f"OAK-D device found:  {dev_id}  [{state}]")
        else:
            _warn("No OAK-D device detected via USB (camera may not be connected yet).\n"
                  "       Connect the OAK-D Pro and run:  python check_oak.py --camera")
    except ImportError:
        _fail("depthai not installed — cannot scan for devices")
    except Exception as exc:
        _fail(f"Device scan error: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Live pipeline smoke-test  (only with --camera flag)
# ──────────────────────────────────────────────────────────────────────────────
def check_pipeline_smoke():
    _header("6 · Live pipeline smoke-test (3 seconds)")
    device = None
    try:
        import time
        from src.oak_detector import build_pipeline, connect_device

        print("     Connecting to OAK-D Pro ...")
        device = connect_device(max_retries=3, retry_delay=2.0)

        print("     Building pipeline ...")
        pipeline, q_rgb, q_depth = build_pipeline(device)

        print("     Waiting for pipeline warmup (2 s) ...")
        time.sleep(2)

        print("     Capturing frames for 5 seconds ...")
        frames_received = 0
        t0 = time.time()
        while time.time() - t0 < 5.0:
            try:
                in_rgb   = q_rgb.get()
                in_depth = q_depth.get()
                _ = in_rgb.getCvFrame()
                _ = in_depth.getFrame()
                frames_received += 1
            except Exception:
                time.sleep(0.1)

        if frames_received > 0:
            elapsed = time.time() - t0
            fps_approx = frames_received / elapsed
            _pass(f"Smoke-test passed  ({frames_received} frame pairs in "
                  f"{elapsed:.0f} s, ~{fps_approx:.0f} fps)")
        else:
            _fail("Pipeline connected but 0 frame pairs received — "
                  "check USB 3.x port or try a different cable")

    except RuntimeError as e:
        _fail(f"Could not connect to OAK-D: {e}")
    except Exception as exc:
        _fail(f"Pipeline smoke-test failed: {exc}")
    finally:
        if device is not None:
            try:
                device.close()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# 7. USB host controller check (Windows)
# ──────────────────────────────────────────────────────────────────────────────
def check_usb():
    _header("7 · USB host controller (Windows)")
    if sys.platform != "win32":
        print("     (Skipped — not Windows)")
        return
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-PnpDevice -PresentOnly | "
             "Where-Object { $_.FriendlyName -like '*USB*' } | "
             "Select-Object -First 6 FriendlyName, Status | Format-Table"],
            capture_output=True, text=True, timeout=10
        )
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        for line in lines[:8]:
            print(f"     {line}")
        if "USB 3" in result.stdout:
            _pass("USB 3.x host controller found")
        else:
            _warn("No USB 3.x controller detected — OAK-D requires USB 3 for full bandwidth")
    except Exception as exc:
        _warn(f"Could not query USB controllers: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Connection guide
# ──────────────────────────────────────────────────────────────────────────────
OAK_CONNECTION_GUIDE = f"""
{CYAN}{BOLD}{'='*60}
  OAK-D Pro  --  Connection & Setup Guide
{'='*60}{RESET}

{BOLD}Step 1 — Physical connection{RESET}
  • Use the USB-C cable supplied with the OAK-D Pro.
  • Plug into a {BOLD}USB 3.x (blue) port{RESET} on your PC (NOT a hub).
  • USB 2.0 ports work but will limit depth bandwidth.

{BOLD}Step 2 — Windows driver{RESET}
  • Windows should auto-install the Myriad X driver.
  • If not: open Device Manager → "Luxonis Device" → Update driver
    → search automatically.
  • Alternatively, run:  pip install --upgrade depthai
    (includes the driver installer on first run).

{BOLD}Step 3 — Verify in Device Manager{RESET}
  • Start → Device Manager → look for "Myriad X" or "Luxonis Device"
    under Universal Serial Bus controllers.
  • If you see a yellow ⚠  → right-click → Update driver.

{BOLD}Step 4 — Power{RESET}
  • Some high-power USB-C ports (laptops with charging via USB-C) work fine.
  • If the device disconnects: use a {BOLD}powered USB hub (USB 3, ≥ 1.5 A){RESET}.

{BOLD}Step 5 — Run diagnostics again{RESET}
    python check_oak.py --camera

{BOLD}Step 6 — Run the detector{RESET}
    python -m src.oak_runner

{BOLD}Useful links{RESET}
  Docs    → https://docs.luxonis.com/en/latest/
  Drivers → https://docs.luxonis.com/en/latest/pages/troubleshooting/
  Forum   → https://discuss.luxonis.com/
"""

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
def _print_summary():
    print(f"\n{CYAN}{BOLD}{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}{RESET}")

    if not _failures and not _warnings:
        print(f"{GREEN}{BOLD}  All checks passed — you are ready to run the detector!{RESET}")
        print(f"\n  {BOLD}Launch:{RESET}  python -m src.oak_runner\n")
    else:
        if _failures:
            print(f"{RED}{BOLD}  ✖ {len(_failures)} check(s) FAILED:{RESET}")
            for f in _failures:
                print(f"    • {f}")
        if _warnings:
            print(f"{YELLOW}  ⚠ {len(_warnings)} warning(s):{RESET}")
            for w in _warnings:
                print(f"    • {w}")
        print()

    print(OAK_CONNECTION_GUIDE)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FishVision-AI — OAK-D Pro system diagnostic"
    )
    parser.add_argument(
        "--camera", action="store_true",
        help="Also run a 3-second live pipeline smoke-test (requires OAK-D connected)"
    )
    args = parser.parse_args()

    print(f"\n{CYAN}{BOLD}  FishVision-AI  —  System Diagnostic{RESET}")
    print(f"  Project root: {PROJECT_ROOT}\n")

    check_python()
    check_packages()
    check_model()
    check_biometrics()
    check_device_present()
    check_usb()

    if args.camera:
        check_pipeline_smoke()
    else:
        print(f"\n{YELLOW}  Tip: run  python check_oak.py --camera  "
              f"to also smoke-test the live pipeline.{RESET}")

    _print_summary()
    sys.exit(1 if _failures else 0)


if __name__ == "__main__":
    main()
