# Cylinder Sorting — SO101 Training Guide

ACT policy trained to sort colored cylinders into matching bins using the SO101 arm.

**Task:** Green cylinder → left bin | Blue cylinder → right bin | Yellow → ignore (distractor)

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Hardware Setup](#2-hardware-setup)
3. [Recording Tips](#3-recording-tips)
4. [Commands Reference](#4-commands-reference)
5. [Autonomous Sorting — sort_controller.py](#5-autonomous-sorting--sort_controllerpy)
6. [Training on Laptop / Windows](#6-training-on-laptop--windows)
7. [Model Map & Checkpoints](#7-model-map--checkpoints)
8. [Recommended Pipeline](#8-recommended-pipeline)
9. [Monitoring & Maintenance](#9-monitoring--maintenance)
10. [Common Issues](#10-common-issues)

---

## 1. Quick Start

### Record → Train → Run

```bash
# 1. Record green episodes
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green record 30

# 2. Train (runs in background)
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green train 20000

# 3. Run autonomous sorting
cd /home/jetson23/lerobot
python scripts/cylinder_sorting/sort_controller.py \
    --model_green outputs/train/act_green_v1/checkpoints/last/pretrained_model \
    --model_blue  outputs/train/act_blue_v1/checkpoints/last/pretrained_model
```

### Kill all lerobot processes
```bash
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh stop
```

### Check training progress
```bash
tail -f /home/jetson23/lerobot/outputs/train/logs/act_green_v1.log
```

---

## 2. Hardware Setup

| Component | Port | Notes |
|---|---|---|
| Follower arm | `/dev/ttyFOLLOWER` | Sorts cylinders autonomously |
| Leader arm | `/dev/ttyLEADER` | Used during recording only |
| Handeye camera | auto-detected | Wrist-mounted |
| Front camera | auto-detected | Scene overview, used for color detection |

> Ports are pinned by udev rules — won't shift on replug. Camera indices are auto-detected each run.

Verify arms are connected:
```bash
ls -la /dev/ttyLEADER /dev/ttyFOLLOWER
```

### Verify Camera Framing

```bash
/home/jetson23/miniforge3/envs/lerobot/bin/python -c "
import cv2
working = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        working.append((i, cap))
    if len(working) == 2:
        break
for idx, (i, cap) in enumerate(working):
    ret, frame = cap.read()
    if ret:
        path = f'/home/jetson23/lerobot/cam_{idx}_index{i}.jpg'
        cv2.imwrite(path, frame)
        print(f'Saved: {path}')
    cap.release()
"
```

Open the saved `cam_*.jpg` files in Cursor to verify framing.

### Warm Up with Teleoperation

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyFOLLOWER \
  --robot.id=my_follower \
  --robot.cameras='{"handeye":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30},"front":{"type":"opencv","index_or_path":4,"width":640,"height":480,"fps":30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyLEADER \
  --teleop.id=my_leader
```

### Maximize Performance Before Training/Eval

```bash
sudo jetson_clocks        # lock clocks to max
sudo nvpmodel -m 2        # 30W mode (safe with USB-C power supply)
```

> If you have the official 180W barrel jack adapter, use `sudo nvpmodel -m 0` (MAXN) for full performance.

---

## 3. Recording Tips

### Scene Setup

- **One cylinder per episode** — only the target color on the table (or all 3 if training with distractors)
- **Fixed positions** — green always left zone, blue always right zone, yellow always center
- **All 3 cylinders visible** if you want the model to learn color recognition with distractors present
- **Fixed bucket positions** — left bucket for green, right bucket for blue

### Motion Tips

- **10–20 seconds per episode** — short, clean, deliberate motions
- **Go straight to the cylinder** — no searching, no hesitation
- **Hold over bucket for 1–2 seconds** before releasing — gives ACT a clear place signal
- **Consistent home position** — start every episode from the same arm pose
- **Return to home** at the end of every episode

### SSH Controls During Recording

| Input | When | What it does |
|---|---|---|
| `Enter` | Before episode | Start recording |
| `Enter` | During episode | End episode early and save |
| `r` + `Enter` | Anytime | Discard episode, re-record |
| `q` + `Enter` | Anytime | Stop recording, save all completed episodes |

---

## 4. Commands Reference

All commands run through `sort.sh`. Run interactively or pass arguments directly.

### record

```bash
# Interactive
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green record

# Direct — 30 episodes
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green record 30
```

Sessions are **additive** — each run appends to the same dataset.

---

### train

```bash
# Interactive
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green train

# Direct — 20k steps
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green train 20000
```

- Runs in background — `Ctrl+C` detaches, training keeps going
- Checkpoints saved every `steps ÷ 5`
- Re-attach: `tail -f ~/lerobot/outputs/train/logs/act_green_v1.log`
- Stop: `kill <PID printed at startup>`

**Step count guide:**

| Episodes | Recommended steps |
|---|---|
| 25 | 8 000–10 000 |
| 50 | 15 000–20 000 |
| 100 | 20 000–30 000 |

**Reading the log:**
```
INFO 14:30:02 ot_train.py:458 step:2K smpl:26K ep:38 epch:1.89 loss:0.302 grdn:15.310 lr:1.0e-04 updt_s:2.087 data_s:0.009
```

| Field | Meaning |
|---|---|
| `step` | Current training step |
| `loss` | Target: below **0.1** for reliable sorting |
| `updt_s` | Seconds per step |
| `epch` | How many times through the full dataset |

---

### finetune

Continue training from the last checkpoint after adding more data.

```bash
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green finetune 10000
```

---

### eval

Run the trained policy and record what it does (for review).

```bash
# Interactive — asks which model, which checkpoint, how many episodes
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green eval

# Direct — 5 episodes
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green eval 5
```

> For continuous autonomous sorting without recording, use `sort_controller.py` instead (see section 5).

---

### clean

```bash
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green clean all    # everything
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green clean data   # dataset only
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green clean model  # model only
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green clean eval   # eval dataset only
```

---

### stop

Kill all running lerobot processes instantly.

```bash
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh stop
```

---

## 5. Autonomous Sorting — sort_controller.py

This is the **production demo script**. No recording, no dataset — pure robot control. The front camera detects which color cylinder is present and runs the matching model automatically.

### Run — continuous autonomous loop

The controller runs indefinitely until Ctrl+C. Each detected cylinder triggers a full sort cycle; the robot automatically resets to home between cycles and resumes watching for the next one.

```bash
cd /home/jetson23/lerobot
python scripts/cylinder_sorting/sort_controller.py \
    --model_green outputs/train/act_green_v2/checkpoints/last/pretrained_model \
    --model_blue  outputs/train/act_blue_v1_laptop_100k/checkpoints/last/pretrained_model
```

Load only the colors you have models for. The system will only act on cylinders whose color has a corresponding model.

### Run — single episode then exit

```bash
python scripts/cylinder_sorting/sort_controller.py \
    --model_green outputs/train/act_green_v1/checkpoints/last/pretrained_model \
    --color green
```

Runs one green sort cycle and exits. Useful for testing a model without starting the full autonomous loop.

### Save home position (first-time setup)

Before running demos, capture the neutral arm pose so the robot can reliably return between cycles:

```bash
# 1. Manually position the arm at its neutral/collapsed pose
# 2. Run:
python scripts/cylinder_sorting/sort_controller.py --capture_home
```

This saves `home_position.json` next to the script. The controller loads it automatically every run.

### Continuous loop — what actually happens

```
Start
  ↓
IDLE          Load both policies into GPU memory. Connect robot.
  ↓
DETECTING     Front camera → HSV mask → is there a green or blue blob?
              If no: print "watching..." every 10s, loop.
              If yes: log the detected color and proceed.
  ↓
RUNNING       policy.reset() → run at 30Hz for up to 30s
              → get_observation() → predict_action() → send_action()
              Early exit: cylinder leaves the ROI after 8s → end episode
  ↓
HOMING        Smooth 2-phase interpolation to home_position.json (~2.7s)
              Gripper opens during motion to release the object
  ↓
SETTLING      Poll joint positions every 0.5s
              All joints < 1° delta → proceed
              Timeout 8s → ERROR + disconnect
  ↓
DETECTING     Ready for next cylinder — loop repeats
```

Ctrl+C at any point: arm returns to home cleanly, then robot disconnects.

### Terminal output during a demo

```
21:55:10 [IDLE]        device: cuda
21:55:10 [IDLE]        enabled colors: ['green', 'blue']
21:55:10 [IDLE]        cameras: handeye=/dev/video_handeye, front=/dev/video_front
21:55:10 [IDLE]        robot connected
21:55:10 [IDLE]        loaded home position from home_position.json
21:55:10 [IDLE]        all policies loaded and in GPU memory
21:55:10 [IDLE]        autonomous loop — Ctrl+C to stop
21:55:10 [DETECTING]   watching... (no cylinder detected)
21:55:30 [DETECTING]   detected: green
21:55:30 [RUNNING]     cycle 1 — green — starting
────────────────────────────────────────────────────────────
21:56:01 [RUNNING]     target cleared from ROI — ending episode early at 21.3s
────────────────────────────────────────────────────────────
21:56:01 [SETTLING]    returning arm to home position...
21:56:04 [SETTLING]    homing motion complete
21:56:04 [SETTLING]    ✓ arm settled in 1.8s
21:56:05 [DETECTING]   cycle 1 complete — watching for next cylinder
```

### Operator controls

| Action | How |
|---|---|
| Start | `python sort_controller.py --model_green <path> --model_blue <path>` |
| Force one color | `--color green` or `--color blue` |
| Pause / stop | `Ctrl+C` — arm homes cleanly then exits |
| Check what is running | `ps aux | grep sort_controller` |

> **Note — GPIO keypad not implemented:** The system specification included a Phase 3 physical keypad (6-button GPIO interface for color override, pause, and stop). This was designed and specified but not implemented before the project deadline. The CLI `--color` flag provides equivalent operator control for the demo. See `SYSTEM_REQUIREMENTS.md` Phase 3 for full design details.

### Tuning color detection

If the controller misidentifies colors, run the tuning tool:

```bash
python scripts/cylinder_sorting/detect_colors.py
```

This shows a live split view of the camera alongside the HSV mask for each color and prints pixel counts every second. Adjust the `COLOR_HSV` ranges at the top of `sort_controller.py` until each cylinder reads ≥600px when present and ≤50px when absent.

---

## 6. Training on Laptop / Windows

Training on a laptop with a dedicated GPU is **2–4x faster** than the Jetson in 30W mode. See `LAPTOP_TRAINING.md` for the full setup guide.

### Transfer trained model to Jetson

```bash
# From laptop — copy model to Jetson
scp -r outputs/train/act_green_v1 jetson23@<JETSON_IP>:/home/jetson23/lerobot/outputs/train/act_green_v1_laptop_100k
```

### Use laptop model in sort_controller.py

```bash
python scripts/cylinder_sorting/sort_controller.py \
    --model_green outputs/train/act_green_v1_laptop_100k/checkpoints/last/pretrained_model \
    --model_blue  outputs/train/act_blue_v1_laptop_100k/checkpoints/last/pretrained_model
```

---

## 7. Model Map & Checkpoints

| Color | Zone | Bin | Dataset | Model folder |
|---|---|---|---|---|
| `green` | Left | Left | `cylinder_sorting_green_v1` | `outputs/train/act_green_v1` |
| `blue` | Right | Right | `cylinder_sorting_blue_v1` | `outputs/train/act_blue_v1` |
| `yellow` | Center | — (ignore) | `cylinder_sorting_yellow_v1` | `outputs/train/act_yellow_v1` |
| `mixed` | Both | Left / Right | `cylinder_sorting_mixed_v1` | `outputs/train/act_mixed_v1` |

**Checkpoint structure:**
```
outputs/train/act_green_v1/checkpoints/
├── 008000/
├── 016000/
├── 024000/
└── last/                 ← always most recent
    └── pretrained_model/
        ├── config.json
        └── model.safetensors
```

Datasets stored at: `~/.cache/huggingface/lerobot/local/`

---

## 8. Recommended Pipeline

### Record and train one color at a time

```bash
# Green — record then train
bash sort.sh green record 30
bash sort.sh green train 20000

# Blue — record then train
bash sort.sh blue record 30
bash sort.sh blue train 20000
```

### Run the autonomous demo

```bash
cd /home/jetson23/lerobot
python scripts/cylinder_sorting/sort_controller.py \
    --model_green outputs/train/act_green_v1/checkpoints/last/pretrained_model \
    --model_blue  outputs/train/act_blue_v1/checkpoints/last/pretrained_model
```

### Model misses the grasp → add data and finetune

```bash
bash sort.sh green record 20          # more episodes
bash sort.sh green finetune 10000     # continue from checkpoint
```

### Start completely over

```bash
bash sort.sh green clean all
bash sort.sh green record 30
bash sort.sh green train 20000
```

---

## 9. Monitoring & Maintenance

### Check memory
```bash
free -h
```

### Check temps and GPU usage
```bash
tegrastats
```

### Kill all lerobot processes
```bash
bash sort.sh stop
# or manually:
ps aux | grep "lerobot.scripts" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
```

### Prevent Jetson from sleeping
```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0
gsettings set org.gnome.desktop.session idle-delay 0
```

---

## 10. Common Issues

| Error | Cause | Fix |
|---|---|---|
| `FileExistsError` on dataset | Prior run left partial data | `sort.sh <color> clean data` |
| `FileExistsError` on eval | Prior eval still on disk | `sort.sh <color> clean eval` |
| `ERROR: No trained model found` | Train hasn't run yet | `sort.sh <color> train` first |
| `Missing motor IDs: 6` | Gripper motor cable loose | Power off, reseat TTL cable at wrist, power on |
| `ConnectionError: Could not connect on port` | Arm USB disconnected | `ls -la /dev/ttyLEADER /dev/ttyFOLLOWER` — replug if missing |
| `ERROR: Could not find 2 working cameras` | Camera unplugged | Check USB, retry |
| Robot moves toward cylinder but misses | Needs more data | Record more episodes, finetune |
| Robot barely moves | Model didn't converge | More steps or better quality data |
| Training very slow (~2s/step) | Clocks not maxed | Run `sudo jetson_clocks` |
| OOM during training | Too many background processes | Run `sort.sh stop`, kill browser/IDE |
