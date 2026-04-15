#!/usr/bin/env python3
"""
detect_colors.py — Live HSV color detection tuning tool.

Automatically detects whether OpenCV has display support (GTK/X11).
  - With display:   shows a live split-view window (raw + mask side by side)
  - Without display: terminal output every second + JPEG snapshots saved to disk

Run this BEFORE sort_controller.py to verify HSV ranges work under
your real workspace lighting. No robot arm needed — front camera only.

Terminal output (every second):
    GREEN:  1247 px   BLUE:  12 px   YELLOW:  0 px   → DETECTED: GREEN

Snapshots (headless mode, or press 's' in GUI mode):
    outputs/captured_images/snap_raw.jpg
    outputs/captured_images/snap_mask_green.jpg
    outputs/captured_images/snap_mask_blue.jpg
    outputs/captured_images/snap_mask_yellow.jpg

GUI controls:   g=green mask   b=blue mask   y=yellow mask   a=all   s=snapshot   q=quit
Headless input: type 's' + Enter to snapshot,  'q' + Enter to quit

Usage:
    conda activate lerobot
    python scripts/cylinder_sorting/detect_colors.py

    # Force a specific camera index:
    python scripts/cylinder_sorting/detect_colors.py --cam_front 2
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# HSV ranges — tune here, then paste final values into sort_controller.py
# ─────────────────────────────────────────────────────────────────────────────
COLOR_HSV = {
    "green":  {"lower": np.array([35,  80,  50],  dtype=np.uint8),
               "upper": np.array([85,  255, 255], dtype=np.uint8)},
    "blue":   {"lower": np.array([100, 80,  50],  dtype=np.uint8),
               "upper": np.array([130, 255, 255], dtype=np.uint8)},
    "yellow": {"lower": np.array([20,  100, 100], dtype=np.uint8),
               "upper": np.array([35,  255, 255], dtype=np.uint8)},
}

MIN_BLOB_PIXELS = 1500  # must match sort_controller.py

SNAPSHOT_DIR = Path("/home/jetson23/lerobot/outputs/captured_images")

# BGR overlay colors for the "all masks" view
OVERLAY_COLORS = {
    "green":  (0,   200, 0),
    "blue":   (200, 0,   0),
    "yellow": (0,   200, 200),
}


# ─────────────────────────────────────────────────────────────────────────────
# Display support detection
# ─────────────────────────────────────────────────────────────────────────────

def has_display_support() -> bool:
    """
    Check if this OpenCV build has GUI display support (GTK / X11 / Cocoa).
    Returns True if cv2.imshow will work, False if it will crash.
    """
    build_info = cv2.getBuildInformation()
    gui_lines  = [l for l in build_info.splitlines() if "GUI:" in l or "GTK" in l or "Cocoa" in l or "Win32" in l]
    for line in gui_lines:
        if "YES" in line or "ON" in line:
            return True
    # Fallback: try creating a tiny test window
    try:
        test = np.zeros((4, 4, 3), dtype=np.uint8)
        cv2.imshow("__test__", test)
        cv2.destroyWindow("__test__")
        cv2.waitKey(1)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Vision utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_pixel_counts(frame_bgr: np.ndarray) -> dict[str, int]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return {
        color: int(np.count_nonzero(cv2.inRange(hsv, cfg["lower"], cfg["upper"])))
        for color, cfg in COLOR_HSV.items()
    }


def get_mask_bgr(frame_bgr: np.ndarray, color: str) -> np.ndarray:
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_HSV[color]["lower"], COLOR_HSV[color]["upper"])
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def get_all_overlay(frame_bgr: np.ndarray) -> np.ndarray:
    overlay = frame_bgr.copy()
    hsv     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    for color, cfg in COLOR_HSV.items():
        mask = cv2.inRange(hsv, cfg["lower"], cfg["upper"])
        overlay[mask > 0] = OVERLAY_COLORS[color]
    return overlay


def draw_contours(frame_bgr: np.ndarray) -> np.ndarray:
    """Draw detection outlines on the raw frame for each active color."""
    out = frame_bgr.copy()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    for color, cfg in COLOR_HSV.items():
        mask       = cv2.inRange(hsv, cfg["lower"], cfg["upper"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if int(np.count_nonzero(mask)) >= MIN_BLOB_PIXELS:
            cv2.drawContours(out, contours, -1, OVERLAY_COLORS[color], 2)
    return out


def save_snapshots(frame_bgr: np.ndarray) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SNAPSHOT_DIR / "snap_raw.jpg"), frame_bgr)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    for color, cfg in COLOR_HSV.items():
        mask = cv2.inRange(hsv, cfg["lower"], cfg["upper"])
        cv2.imwrite(str(SNAPSHOT_DIR / f"snap_mask_{color}.jpg"), mask)
    print(f"  → snapshots saved to {SNAPSHOT_DIR}/snap_*.jpg")


def print_counts(counts: dict[str, int]) -> None:
    detections = [c for c, n in counts.items() if n >= MIN_BLOB_PIXELS]
    detected   = ("DETECTED: " + ", ".join(d.upper() for d in detections)) if detections else "nothing detected"
    print(
        f"  GREEN:{counts['green']:6d}px  "
        f"BLUE:{counts['blue']:6d}px  "
        f"YELLOW:{counts['yellow']:6d}px  "
        f"→  {detected}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Camera discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_front_camera() -> int:
    found = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
            cap.release()
        if len(found) == 2:
            break
    if not found:
        print("ERROR: No cameras found.")
        sys.exit(1)
    front = found[1] if len(found) >= 2 else found[0]
    print(f"Cameras found: {found}  →  using index {front} as front camera")
    return front


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HSV color detection test — auto GUI/headless")
    parser.add_argument("--cam_front", type=int, default=None)
    args = parser.parse_args()

    cam_idx = args.cam_front if args.cam_front is not None else find_front_camera()

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {cam_idx}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    gui = has_display_support()

    print(f"\nCamera {cam_idx} opened at 640×480.")
    print(f"Mode: {'GUI (live window)' if gui else 'HEADLESS (terminal + snapshots)'}")
    print(f"MIN_BLOB_PIXELS = {MIN_BLOB_PIXELS}  (same as sort_controller.py)\n")

    if gui:
        print("Controls: g=green  b=blue  y=yellow  a=all masks  s=snapshot  q=quit")
    else:
        print("Controls (type + Enter): s=snapshot   q=quit")
        print(f"Snapshots → {SNAPSHOT_DIR}/snap_*.jpg\n")

    active      = "green"
    last_log    = 0.0
    last_snap   = 0.0
    snap_interval = 5.0   # headless: auto-save snapshot every 5s
    quit_flag   = threading.Event()

    # Headless stdin thread
    def _stdin():
        while not quit_flag.is_set():
            try:
                line = sys.stdin.readline().strip().lower()
                if line == "q":
                    quit_flag.set()
                elif line == "s":
                    # flag for main loop to save on next frame
                    _stdin.save_now = True
            except Exception:
                break
    _stdin.save_now = False

    if not gui:
        t = threading.Thread(target=_stdin, daemon=True)
        t.start()

    while not quit_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed — exiting")
            break

        counts = get_pixel_counts(frame)
        now    = time.time()

        # Terminal log every second
        if now - last_log >= 1.0:
            print_counts(counts)
            last_log = now

        if gui:
            # Build side-by-side display
            left  = draw_contours(frame)
            right = get_all_overlay(frame) if active == "all" else get_mask_bgr(frame, active)

            # Status bar
            detections = [c for c, n in counts.items() if n >= MIN_BLOB_PIXELS]
            status = "DETECTED: " + ", ".join(d.upper() for d in detections) if detections else "none"
            bar    = np.zeros((40, 1280, 3), dtype=np.uint8)
            cv2.putText(
                bar,
                f"  G:{counts['green']:5d}px  B:{counts['blue']:5d}px  "
                f"Y:{counts['yellow']:5d}px  |  {status}  |  [{active}]  g/b/y/a/s/q",
                (4, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
            )
            combined = np.vstack([np.hstack([left, right]), bar])
            cv2.imshow("detect_colors  |  left=raw  right=mask", combined)

            key = cv2.waitKey(1) & 0xFF
            if   key == ord("q"): break
            elif key == ord("g"): active = "green"
            elif key == ord("b"): active = "blue"
            elif key == ord("y"): active = "yellow"
            elif key == ord("a"): active = "all"
            elif key == ord("s"): save_snapshots(frame)

        else:
            # Headless: auto-snapshot every 5s, or on demand
            if _stdin.save_now or (now - last_snap >= snap_interval):
                save_snapshots(frame)
                last_snap       = now
                _stdin.save_now = False

    cap.release()
    if gui:
        cv2.destroyAllWindows()

    print("\nFinal HSV ranges — paste into sort_controller.py COLOR_HSV if changed:")
    for color, cfg in COLOR_HSV.items():
        print(f"  {color:8s}: lower={cfg['lower'].tolist()}  upper={cfg['upper'].tolist()}")


if __name__ == "__main__":
    main()
