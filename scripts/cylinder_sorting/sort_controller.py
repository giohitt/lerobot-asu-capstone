#!/usr/bin/env python3
"""
sort_controller.py — Autonomous cylinder sorting via direct policy inference.

No lerobot-record. No dataset. No recording. Pure robot control.

State machine:
    IDLE → DETECTING → RUNNING → SETTLING → DETECTING → ...
                          ↓ (on error)
                        ERROR → EXIT

Usage:
    # Autonomous loop — sorts any detected color indefinitely
    python sort_controller.py \
        --model_green  ~/lerobot/outputs/train/act_green_v1_laptop_100k/checkpoints/last/pretrained_model \
        --model_blue   ~/lerobot/outputs/train/act_blue_v1_laptop_100k/checkpoints/last/pretrained_model

    # Single episode — run one sort for a specific color then exit
    python sort_controller.py \
        --model_green  ~/lerobot/outputs/train/act_green_v1_laptop_100k/checkpoints/last/pretrained_model \
        --color green
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HSV color ranges — tune these with detect_colors.py before running
# ─────────────────────────────────────────────────────────────────────────────
COLOR_HSV = {
    "green":  {"lower": np.array([35,  80,  50],  dtype=np.uint8),
               "upper": np.array([85,  255, 255], dtype=np.uint8)},
    "blue":   {"lower": np.array([100, 80,  50],  dtype=np.uint8),
               "upper": np.array([130, 255, 255], dtype=np.uint8)},
    "yellow": {"lower": np.array([20,  100, 100], dtype=np.uint8),
               "upper": np.array([35,  255, 255], dtype=np.uint8)},
}

# Minimum pixel blob to count as a real detection (not a reflection or noise)
MIN_BLOB_PIXELS = 1500

# Settling — how still the arm must be to count as "home"
# Units match the robot's position units (degrees when use_degrees=True, else normalized)
SETTLE_THRESHOLD  = 1.0   # max joint delta across all motors between two readings
SETTLE_INTERVAL_S = 0.5   # seconds between position samples during settling
SETTLE_TIMEOUT_S  = 8.0   # if arm hasn't settled by this point, something is wrong


# ─────────────────────────────────────────────────────────────────────────────
# State labels — printed to terminal so you can see exactly what's happening
# ─────────────────────────────────────────────────────────────────────────────
class State:
    IDLE      = "IDLE"
    DETECTING = "DETECTING"
    RUNNING   = "RUNNING"
    SETTLING  = "SETTLING"
    ERROR     = "ERROR"


def log_state(state: str, msg: str = "") -> None:
    label = f"[{state}]"
    ts = time.strftime("%H:%M:%S")
    print(f"{ts} {label:<12} {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Color detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_color(frame_bgr: np.ndarray, enabled_colors: list[str]) -> str | None:
    """
    Return the enabled color with the largest blob in the frame, or None.
    Uses HSV thresholding — tune COLOR_HSV constants with detect_colors.py first.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    best_color, best_count = None, 0

    for color in enabled_colors:
        cfg = COLOR_HSV.get(color)
        if cfg is None:
            continue
        mask  = cv2.inRange(hsv, cfg["lower"], cfg["upper"])
        count = int(np.count_nonzero(mask))
        if count > MIN_BLOB_PIXELS and count > best_count:
            best_count, best_color = count, color

    return best_color


# ─────────────────────────────────────────────────────────────────────────────
# Policy loading
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(checkpoint_path: str | Path, device: torch.device):
    """
    Load ACT policy + pre/post-processors from a pretrained_model/ checkpoint dir.
    Normalization stats are read from the checkpoint — no dataset object needed.
    """
    path = str(Path(checkpoint_path).expanduser().resolve())
    log.info(f"  Loading: {path}")
    policy = ACTPolicy.from_pretrained(path)
    policy = policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=path,
    )
    return policy, preprocessor, postprocessor


# ─────────────────────────────────────────────────────────────────────────────
# HOME POSITION — saved neutral pose used to return arm after each episode
# ─────────────────────────────────────────────────────────────────────────────

HOME_POS_FILE = Path(__file__).parent / "home_position.json"

# Homing motion parameters
HOME_STEPS    = 80    # interpolation steps (~2.7s at 30ms each)
HOME_STEP_S   = 0.033 # seconds between each interpolation step


def load_home_position() -> dict | None:
    """Load saved home position from JSON, or return None if not yet captured."""
    if HOME_POS_FILE.exists():
        data = json.loads(HOME_POS_FILE.read_text())
        log_state(State.IDLE, f"loaded home position from {HOME_POS_FILE.name}")
        return data
    log_state(State.IDLE,
        "WARNING: no home_position.json found — run with --capture_home first. "
        "Arm will rely on policy return behavior only.")
    return None


def save_home_position(pos: dict) -> None:
    """Persist motor positions to JSON for reuse across runs."""
    HOME_POS_FILE.write_text(json.dumps(pos, indent=2))
    log_state(State.SETTLING, f"home position saved → {HOME_POS_FILE.name}")


def get_motor_positions(robot: SO101Follower) -> dict:
    """Read current motor positions (float values only, no camera frames)."""
    obs = robot.get_observation()
    return {k: float(v) for k, v in obs.items() if isinstance(v, (int, float))}


def smooth_home(robot: SO101Follower, home_pos: dict) -> None:
    """
    Smoothly interpolate from current arm position to home_pos over HOME_STEPS steps.
    Sends position commands at ~30Hz so the arm moves steadily rather than snapping.
    """
    log_state(State.SETTLING, "returning arm to home position...")
    current = get_motor_positions(robot)

    for i in range(1, HOME_STEPS + 1):
        t      = i / HOME_STEPS                          # 0 → 1
        # ease-in-out: smooth start and end, faster in the middle
        t_ease = t * t * (3.0 - 2.0 * t)
        target = {k: current[k] + t_ease * (home_pos[k] - current[k]) for k in home_pos}
        robot.send_action(target)
        time.sleep(HOME_STEP_S)

    log_state(State.SETTLING, "homing motion complete")


# ─────────────────────────────────────────────────────────────────────────────
# SETTLING — confirm arm has stopped moving, optionally capture home position
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_settle(robot: SO101Follower, home_pos: dict | None) -> bool:
    """
    Verify the arm has come to rest after homing.

    run_sort_episode already called smooth_home() before returning, so the arm
    should be near home_pos. This function confirms it is stationary by comparing
    consecutive motor position readings. While polling it keeps sending home_pos
    so Feetech servos don't lose torque and drop.

    Returns True if settled within SETTLE_TIMEOUT_S, False otherwise.
    """
    log_state(State.SETTLING, "verifying arm is at rest...")
    start    = time.perf_counter()
    prev_pos = None
    poll     = 0

    while time.perf_counter() - start < SETTLE_TIMEOUT_S:
        curr_pos = get_motor_positions(robot)

        # Keep commanding home so servos don't go limp while we poll
        if home_pos is not None:
            robot.send_action(home_pos)

        if prev_pos is not None:
            max_delta = max(abs(curr_pos[k] - prev_pos[k]) for k in curr_pos)
            if poll % 4 == 0:
                log_state(State.SETTLING, f"max joint delta: {max_delta:.2f}")
            if max_delta < SETTLE_THRESHOLD:
                elapsed = time.perf_counter() - start
                log_state(State.SETTLING, f"✓ arm settled in {elapsed:.1f}s")
                return True

        prev_pos = curr_pos
        poll    += 1
        time.sleep(SETTLE_INTERVAL_S)

    log_state(State.ERROR, f"arm did not settle within {SETTLE_TIMEOUT_S}s — aborting")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# RUNNING — inference loop for one sort cycle
# ─────────────────────────────────────────────────────────────────────────────

def run_sort_episode(
    robot:          SO101Follower,
    policy:         ACTPolicy,
    preprocessor,
    postprocessor,
    ds_features:    dict,
    device:         torch.device,
    episode_time_s: float,
    fps:            int,
    task:           str,
    home_pos:       dict | None = None,
) -> None:
    """
    Drive the robot through one sort cycle at `fps` Hz for up to `episode_time_s`.

    After the loop (normal or error), smoothly returns the arm to home_pos so
    servos never go limp and the arm doesn't fall onto objects.

    Loop:
        robot.get_observation()
            → build_dataset_frame()     map obs dict to policy-expected keys
            → predict_action()          run the neural net
            → make_robot_action()       tensor → named motor targets
            → robot.send_action()       move the arm
            → precise_sleep()           maintain Hz
    """
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    start_t   = time.perf_counter()
    step      = 0
    timestamp = 0.0

    try:
        while timestamp < episode_time_s:
            loop_t = time.perf_counter()

            obs       = robot.get_observation()
            obs_frame = build_dataset_frame(ds_features, obs, prefix=OBS_STR)

            action_tensor = predict_action(
                observation=obs_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=task,
                robot_type=robot.robot_type,
            )

            action = make_robot_action(action_tensor, ds_features)
            robot.send_action(action)

            dt        = time.perf_counter() - loop_t
            precise_sleep(max(0.0, 1.0 / fps - dt))
            timestamp = time.perf_counter() - start_t
            step     += 1

        print("─" * 60, flush=True)
        log_state(State.RUNNING, f"episode done — {step} steps in {timestamp:.1f}s")
        print("─" * 60, flush=True)

    except Exception as e:
        print("─" * 60, flush=True)
        log_state(State.ERROR, f"episode failed at step {step} ({timestamp:.1f}s): {e}")
        print("─" * 60, flush=True)
        if home_pos is not None:
            log_state(State.SETTLING, "emergency home — bringing arm back after error...")
            try:
                smooth_home(robot, home_pos)
            except Exception as home_err:
                log_state(State.ERROR, f"emergency home also failed: {home_err}")
        raise  # re-raise so the main loop can catch and handle it

    # ── Normal end: return to home so servos don't go limp ───────────────────
    if home_pos is not None:
        smooth_home(robot, home_pos)


# ─────────────────────────────────────────────────────────────────────────────
# Camera discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_cameras() -> tuple[int, int]:
    """
    Return the indices of the first two capture-capable cameras.

    On Jetson/V4L2 each USB camera creates two device nodes:
      /dev/video0 (capture)  /dev/video1 (metadata-only)
      /dev/video2 (capture)  /dev/video3 (metadata-only)
    Probing the metadata nodes causes 'ioctl VIDIOC_QBUF: Bad file descriptor'
    spam. We suppress stderr at the OS level during probing so that noise stays
    out of the terminal, then restore it before returning.
    """
    import os

    # Silence V4L2 / OpenCV low-level error output during probing
    devnull  = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)

    found = []
    try:
        for i in range(12):
            if not os.path.exists(f"/dev/video{i}"):
                continue
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Read one frame to confirm it's a real capture node
                ok, _ = cap.read()
                if ok:
                    found.append(i)
                cap.release()
            if len(found) == 2:
                break
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)

    if len(found) < 2:
        raise RuntimeError(
            f"Found only {len(found)} capture-capable camera(s). Need 2 (handeye + front). "
            "Check USB connections."
        )
    return found[0], found[1]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous cylinder sorting — direct policy inference, no recording"
    )

    # Model paths — provide any combination of colors
    parser.add_argument("--model_green",  type=str, default=None)
    parser.add_argument("--model_blue",   type=str, default=None)
    parser.add_argument("--model_yellow", type=str, default=None)

    # Capture home position: connect, read motors, save, exit — no sort
    parser.add_argument(
        "--capture_home", action="store_true",
        help="Save current arm position as home_position.json then exit"
    )

    # Force a single episode for one color then exit (CLI trigger / Chunk 2)
    parser.add_argument(
        "--color", type=str, default=None,
        choices=["green", "blue", "yellow"],
        help="Run one episode for this color and exit (skips detection loop)"
    )

    # Timing
    parser.add_argument("--episode_time",       type=float, default=30.0,
                        help="Max seconds the policy runs per sort cycle (default: 30)")
    parser.add_argument("--fps",                type=int,   default=30,
                        help="Inference loop frequency (default: 30)")
    parser.add_argument("--detect_pause",       type=float, default=0.3,
                        help="Seconds between detection checks when no cylinder found (default: 0.3)")
    parser.add_argument("--post_settle_pause",  type=float, default=1.0,
                        help="Extra pause after settling before next detection (default: 1.0)")

    # Robot
    parser.add_argument("--robot_port", type=str, default="/dev/ttyFOLLOWER")
    parser.add_argument("--robot_id",   type=str, default="my_follower")

    args = parser.parse_args()

    # ── Build model map ───────────────────────────────────────────────────────
    model_paths: dict[str, str] = {}
    if args.model_green:  model_paths["green"]  = args.model_green
    if args.model_blue:   model_paths["blue"]   = args.model_blue
    if args.model_yellow: model_paths["yellow"] = args.model_yellow

    if not args.capture_home:
        if not model_paths:
            parser.error(
                "Provide at least one model path.\n"
                "  --model_green  <path/to/pretrained_model>\n"
                "  --model_blue   <path/to/pretrained_model>\n"
                "  --model_yellow <path/to/pretrained_model>"
            )
        # If --color is given, that color must have a model
        if args.color and args.color not in model_paths:
            parser.error(
                f"--color {args.color} requires --model_{args.color} to be set"
            )

    enabled_colors = list(model_paths.keys())

    # ── Device ───────────────────────────────────────────────────────────────
    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    log_state(State.IDLE, f"device: {device}")
    log_state(State.IDLE, f"enabled colors: {enabled_colors}")

    # ── Find cameras ─────────────────────────────────────────────────────────
    cam_handeye, cam_front = find_cameras()
    log_state(State.IDLE, f"cameras: handeye={cam_handeye}, front={cam_front}")

    # ── Connect robot ─────────────────────────────────────────────────────────
    robot_cfg = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras={
            "handeye": OpenCVCameraConfig(
                index_or_path=cam_handeye, width=640, height=480, fps=args.fps
            ),
            "front": OpenCVCameraConfig(
                index_or_path=cam_front, width=640, height=480, fps=args.fps
            ),
        },
    )
    robot = SO101Follower(robot_cfg)
    # Suppress V4L2 ioctl noise that some camera nodes emit during connect
    import os as _os
    _devnull  = _os.open(_os.devnull, _os.O_WRONLY)
    _saved_fd = _os.dup(2)
    _os.dup2(_devnull, 2)
    _os.close(_devnull)
    try:
        robot.connect()
    finally:
        _os.dup2(_saved_fd, 2)
        _os.close(_saved_fd)
    log_state(State.IDLE, "robot connected")

    # ── Capture home position and exit (skip policy loading entirely) ────────
    if args.capture_home:
        pos = get_motor_positions(robot)
        save_home_position(pos)
        log_state(State.IDLE, "motor positions:")
        for k, v in pos.items():
            log_state(State.IDLE, f"  {k}: {v:.2f}")
        robot.disconnect()
        log_state(State.IDLE, "robot disconnected — goodbye")
        return

    # ── Load all policies once at startup ────────────────────────────────────
    log_state(State.IDLE, "loading policies...")
    policies: dict[str, tuple] = {}
    for color, path in model_paths.items():
        policies[color] = load_policy(path, device)
    log_state(State.IDLE, "all policies loaded and in GPU memory")

    # ── Build feature map from robot specs (no dataset object needed) ─────────
    # This tells build_dataset_frame and make_robot_action which keys map where.
    ds_features = {
        **hw_to_dataset_features(robot.action_features, ACTION),
        **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────────
    log_state(State.IDLE, "="*50)
    if args.color:
        log_state(State.IDLE, f"single-episode mode: will sort one {args.color} cylinder then exit")
    else:
        log_state(State.IDLE, "autonomous loop — Ctrl+C to stop")
    log_state(State.IDLE, "="*50)

    home_pos       = load_home_position()
    cycles         = 0
    last_heartbeat = time.perf_counter()
    HEARTBEAT_S    = 10.0  # print "watching..." every N seconds when idle
    try:
        while True:

            # ── DETECTING ────────────────────────────────────────────────────
            if args.color:
                # --color flag: skip detection, force the specified color
                detected = args.color
                log_state(State.DETECTING, f"CLI override: {detected}")
            else:
                obs          = robot.get_observation()
                front_frame  = obs.get("front")

                if front_frame is None:
                    log_state(State.DETECTING, "no frame from front camera — check USB")
                    time.sleep(args.detect_pause)
                    continue

                detected = detect_color(front_frame, enabled_colors)

                if not detected:
                    now = time.perf_counter()
                    if now - last_heartbeat >= HEARTBEAT_S:
                        log_state(State.DETECTING, "watching... (no cylinder detected)")
                        last_heartbeat = now
                    time.sleep(args.detect_pause)
                    continue

                last_heartbeat = time.perf_counter()  # reset on detection

                log_state(State.DETECTING, f"detected: {detected}")

            # ── RUNNING ──────────────────────────────────────────────────────
            cycles += 1
            policy, preprocessor, postprocessor = policies[detected]
            task = f"Pick {detected} cylinder and place in bin"

            log_state(State.RUNNING, f"cycle {cycles} — {detected} — starting")

            run_sort_episode(
                robot=robot,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                ds_features=ds_features,
                device=device,
                episode_time_s=args.episode_time,
                fps=args.fps,
                task=task,
                home_pos=home_pos,
            )

            # ── SETTLING ─────────────────────────────────────────────────────
            settled = wait_for_settle(robot, home_pos)
            if not settled:
                log_state(State.ERROR,
                    "arm failed to settle — stopping to avoid unsafe next episode")
                break

            time.sleep(args.post_settle_pause)

            # ── Exit after one cycle if --color was given ─────────────────────
            if args.color:
                print("=" * 60, flush=True)
                log_state(State.IDLE, "single-episode complete — exiting")
                print("=" * 60, flush=True)
                break

            log_state(State.DETECTING, f"cycle {cycles} complete — watching for next cylinder")

    except KeyboardInterrupt:
        log_state(State.IDLE, f"stopped by user — {cycles} cycle(s) completed")
    except Exception as e:
        log_state(State.ERROR, f"unhandled exception: {e}")
        raise
    finally:
        import os as _os
        _devnull  = _os.open(_os.devnull, _os.O_WRONLY)
        _saved_fd = _os.dup(2)
        _os.dup2(_devnull, 2)
        _os.close(_devnull)
        try:
            robot.disconnect()
        finally:
            _os.dup2(_saved_fd, 2)
            _os.close(_saved_fd)
        log_state(State.IDLE, "robot disconnected — goodbye")


if __name__ == "__main__":
    main()
