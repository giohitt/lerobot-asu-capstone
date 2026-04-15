# SYSTEM SPECIFICATION — Autonomous Cylinder Sorting Robot

**NVIDIA Jetson Orin AGX + SO-101 Follower Arm**
*AI-Enabled Robotics — Imitation Learning Deployment*

---

| Field | Value |
|-------|-------|
| **Version** | 1.0 DRAFT |
| **Date** | 2026-04-12 |
| **Status** | Active Development |
| **Framework** | HuggingFace LeRobot (open source, modified) |
| **Policy** | ACT — Action Chunking Transformer |

---

## Open Source Attribution

This system is built on **LeRobot** by HuggingFace ([github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)), an open-source imitation learning framework for real-robot deployment. LeRobot provided the robot driver, training pipeline, dataset tooling, and inference infrastructure that this project extends. We have modified LeRobot's source — specifically `control_utils.py`, `lerobot_record.py`, `camera_opencv.py`, and the SO-101 driver — to support autonomous headless operation, SSH-safe recording, and the direct inference loop described in this document.

**LeRobot packages used in this system:**

| Package | Role in This System |
|---|---|
| `lerobot.robots.so101_follower` | SO-101 arm driver — motor read/write, calibration, camera management |
| `lerobot.cameras.opencv` | USB camera abstraction for handeye and front cameras |
| `lerobot.policies.act` | ACT policy model — load, reset, select_action |
| `lerobot.policies.factory` | Load pre/post-processors with saved normalization stats |
| `lerobot.datasets.utils` | `build_dataset_frame`, `hw_to_dataset_features` — obs/action formatting |
| `lerobot.utils.control_utils` | `predict_action` — the core inference step |
| `lerobot.scripts.lerobot_record` | Teleoperation data collection and eval recording |
| `lerobot.scripts.lerobot_train` | ACT policy training from recorded datasets |

---

## Table of Contents

**Part I — Project Overview**
1. [Project Structure & Implementation Phases](#1-project-structure--implementation-phases)

**Part II — System Specification**
2. [Problem Statement](#2-problem-statement)
3. [System Architecture L0](#3-system-architecture-l0--system-level)
4. [Perception Subsystem L1](#4-perception-subsystem-l1)
5. [Decision Subsystem L1](#5-decision-subsystem-l1)
6. [Actuation Subsystem L1](#6-actuation-subsystem-l1)
7. [Training Subsystem L1](#7-training-subsystem-l1)
8. [Interface Specifications](#8-interface-specifications)
9. [Verification — Test Cases](#9-verification--test-cases)
10. [Traceability Matrix](#10-traceability-matrix)

**Part III — Implementation Phases**
11. [Phase 1 — Data Collection & Training](#11-phase-1--data-collection--training)
12. [Phase 2 — Autonomous Sort Controller](#12-phase-2--autonomous-sort-controller)
13. [Phase 3 — GPIO Keypad Integration](#13-phase-3--gpio-keypad-integration)

**Part IV — Administration**
14. [Revision History](#14-revision-history)

---

# PART I — PROJECT OVERVIEW

## 1. Project Structure & Implementation Phases

This project teaches AI-enabled robotics through a single robot platform implemented across three phases. A SO-101 follower arm learns pick-and-place from human demonstrations, then sorts colored cylinders autonomously. The system specification is written once and drives all three implementation phases.

### 1.1 The Approach

> **One specification. One robot. Three levels of autonomy.**

The system starts with human teleoperation to collect data (Phase 1), uses that data to train separate ACT policies per color (Phase 1), then deploys those policies in a fully autonomous sort loop (Phase 2), and finally adds physical keypad control (Phase 3).

### 1.2 Phase Overview

| Phase | Title | What Happens | Key Outcome |
|-------|-------|-------------|-------------|
| **1** | Data Collection & Training | Teleoperate arm 100× per color, train ACT policy on laptop GPU, transfer model to Jetson | Trained models: `act_green_v1_laptop_100k`, `act_blue_v1_laptop_100k` |
| **2** | Autonomous Sort Controller | Color detection via HSV, direct policy inference loop, CLI control, optional recording | Robot sorts cylinders continuously without human input |
| **3** | GPIO Keypad Integration | Physical keypad triggers color selection or system commands (pause, stop, force color) | Operator control without a computer |

### 1.3 Hardware & Software

| Component | Role |
|---|---|
| NVIDIA Jetson Orin AGX | Runtime compute — inference, camera, robot control |
| SO-101 Follower Arm | 6-DOF manipulation with Feetech STS3215 motors |
| SO-101 Leader Arm | Teleoperation device for data collection only |
| USB Camera — handeye (wrist-mounted) | Policy observation: close-up view of workspace |
| USB Camera — front (stationary) | Policy observation + color detection zone |
| RTX 5070 Ti Laptop (WSL2) | Training compute — 100k steps per color in ~5 hours |
| GPIO Keypad | Physical operator interface (Phase 3) |
| HuggingFace LeRobot | Open-source framework (modified) |

---

# PART II — SYSTEM SPECIFICATION

## 2. Problem Statement

### 2.1 Mission Statement

Design and deploy an autonomous robotic system that identifies colored cylinders in a defined workspace, selects the correct pre-trained policy for each color, and executes a pick-and-place operation to sort cylinders into designated bins — continuously, without human intervention after initialization.

### 2.2 Operational Context

- **Environment:** Indoor lab bench, consistent indoor lighting
- **Workspace:** SO-101 reachable area (~50cm radius), flat surface
- **Objects:** Colored cylinders — green (left bin), blue (right bin), yellow (distractor, future model)
- **Hardware:** Jetson Orin AGX, SO-101 arm, two USB cameras
- **Operator:** Initiates system via CLI command or GPIO keypad; no further human input required
- **Session:** Continuous loop — robot sorts until stopped or no cylinder is detected

### 2.3 Success Criteria

| ID | Criterion | Threshold |
|----|-----------|-----------|
| SC-001 | System correctly sorts green cylinders to the left bin | ≥85% success rate over 10 trials |
| SC-002 | System correctly sorts blue cylinders to the right bin | ≥85% success rate over 10 trials |
| SC-003 | System ignores yellow cylinders (no model loaded) | 0 false picks per trial |
| SC-004 | Full sort cycle completes autonomously | No human intervention after start command |
| SC-005 | End-to-end cycle time per cylinder | ≤30 seconds per cylinder |

---

## 3. System Architecture (L0 — System Level)

The system follows the **Perception–Decision–Actuation** pattern. Perception identifies what color cylinder is present. Decision selects which policy to run and when. Actuation executes the trained policy on the arm.

### 3.1 System Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CYLINDER SORTING ROBOT (L0)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐       │
│  │   PERCEPTION    │──▶│    DECISION     │──▶│   ACTUATION     │       │
│  │                 │   │                 │   │                 │       │
│  │  Front Camera   │   │  Mode Selection │   │  SO-101 Arm     │       │
│  │  HSV Detection  │   │  Policy Select  │   │  ACT Inference  │       │
│  │  Color ID       │   │  CLI / GPIO     │   │  Motor Control  │       │
│  └─────────────────┘   └─────────────────┘   └────────┬────────┘       │
│                                                        │                │
│  ┌─────────────────────────────────────────────────────▼──────────────┐ │
│  │                  DATA COLLECTION (optional, --record)              │ │
│  │      LeRobotDataset.add_frame() → save_episode() per sort cycle   │ │
│  │      Source 1: Teleoperation (human demos, Phase 1)               │ │
│  │      Source 2: Autonomous rollouts (live action, Phase 2+)        │ │
│  └────────────────────────────────┬───────────────────────────────────┘ │
│                                   │  LeRobot dataset (local)            │
│  OPERATOR INPUT:                  │  rsync to laptop for training        │
│    CLI (sort.sh / sort_controller.py)                                   │
│    GPIO Keypad — keys 1-6         │                                     │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
                   ┌────────────────▼────────────────────────────────┐
                   │        TRAINING SUBSYSTEM (L1)                  │
                   │        RTX 5070 Ti Laptop — WSL2                │
                   │                                                 │
                   │  Miniconda → lerobot env                        │
                   │  PyTorch 2.8.0+cu128 (CUDA 12.8, sm_120)       │
                   │  lerobot-train → ACT policy → checkpoint/last/  │
                   │                                                 │
                   └────────────────┬────────────────────────────────┘
                                    │ rsync trained model back to Jetson
                                    ▼
                          Updated pretrained_model/
                          loaded by sort_controller.py
```

> **Data Flywheel:** Every autonomous sort cycle recorded with `--record` becomes new training data. Autonomous rollout data is combined with teleoperation data to fine-tune models over time — the robot improves itself by doing its job.

### 3.2 System-Level Requirements

| ID | Requirement (EARS Format) | Rationale | Verification |
|----|--------------------------|-----------|--------------|
| SYS-001 | The system SHALL identify the color (green, blue, or yellow) of a cylinder present in the detection zone using the front camera. | Core perception capability | TC-SYS-001 |
| SYS-002 | The system SHALL execute the correct trained ACT policy for the detected cylinder color. | Core mission capability | TC-SYS-002 |
| SYS-003 | The system SHALL complete one pick-and-place sort cycle in ≤30 seconds. | Operational efficiency | TC-SYS-003 |
| SYS-004 | WHEN no model is loaded for a detected color, the system SHALL skip that cylinder and log a warning rather than attempt a sort. | Safety — no undefined behavior | TC-SYS-004 |
| SYS-005 | The system SHALL accept operator color selection and system commands via both CLI arguments and GPIO keypad input. | Dual-mode operator interface | TC-SYS-005 |
| SYS-006 | WHEN an exception or emergency stop is triggered, the system SHALL disconnect the robot safely within 1 second and halt all motion. | Safety requirement | TC-SYS-006 |
| SYS-007 | The system SHALL support continuous model improvement by recording autonomous sort episodes as valid LeRobot datasets that can be used for fine-tuning without modification. | Data flywheel — robot improves by operating | TC-SYS-007 |

---

## 4. Perception Subsystem (L1)

The Perception Subsystem uses the stationary front camera to determine which cylinder color is in the workspace. This is implemented using classical computer vision (HSV color masking) rather than a learned detector — keeping it fast, deterministic, and easy to tune without retraining.

### 4.1 Subsystem Decomposition

| Component | Function | Technology |
|---|---|---|
| Front Camera Module | Capture frames of the cylinder detection zone | LeRobot `OpenCVCamera`, 640×480, 30fps |
| HSV Color Masker | Apply per-color threshold masks to identify cylinder presence | OpenCV `cv2.inRange()`, BGR→HSV conversion |
| Blob Size Filter | Reject false positives by requiring minimum pixel area | `np.count_nonzero(mask) > MIN_BLOB_PIXELS` |
| Color Arbitrator | When multiple colors detected, select the largest blob | Compare pixel counts across enabled colors |

### 4.2 Perception Requirements

| ID | Requirement | Parent | Verification |
|----|-------------|--------|--------------|
| PERC-001 | The Front Camera Module SHALL capture frames at 30fps at 640×480 resolution. | SYS-001 | TC-PERC-001 |
| PERC-002 | The HSV Color Masker SHALL detect green cylinders using HSV range [35–85, 80–255, 50–255]. | SYS-001 | TC-PERC-002 |
| PERC-003 | The HSV Color Masker SHALL detect blue cylinders using HSV range [100–130, 80–255, 50–255]. | SYS-001 | TC-PERC-002 |
| PERC-004 | The HSV Color Masker SHALL detect yellow cylinders using HSV range [20–35, 100–255, 100–255]. | SYS-001 | TC-PERC-002 |
| PERC-005 | The Blob Size Filter SHALL require ≥1500 pixels to count a color as detected (false-positive suppression). Threshold raised from 500 during testing — keyboard RGB LEDs triggered false detections at 500px. | SYS-001, SYS-004 | TC-PERC-003 |
| PERC-006 | Color detection per frame SHALL complete within 50ms. | SYS-003 | TC-PERC-004 |
| PERC-007 | HSV threshold constants SHALL be defined in a single shared location in `sort_controller.py` to allow tuning without modifying logic. | SYS-001 | Inspection |

### 4.3 Tuning Tool

A standalone `detect_colors.py` script (Phase 2, Chunk 1) provides a live split view of the raw camera frame alongside the HSV mask for each color, printing pixel counts to stdout each second. This allows threshold tuning under real workspace lighting without connecting the robot arm.

---

## 5. Decision Subsystem (L1)

The Decision Subsystem determines what the system should do at any moment: which policy to load, when to trigger an episode, when to wait, and how to handle operator overrides from CLI or GPIO. It also manages system state across episodes.

### 5.1 Subsystem Decomposition

| Component | Function | Technology |
|---|---|---|
| Mode Selector | Accepts color selection from CLI flags or GPIO and exposes a unified current_color to the sort loop | argparse + GPIO background thread |
| Policy Registry | Pre-loads all enabled ACT policies at startup, keyed by color | `ACTPolicy.from_pretrained()`, held in dict |
| Sort Loop Controller | Drives the detect → episode → save → detect cycle autonomously | Python while-loop in `sort_controller.py` |
| Episode Timer | Enforces max episode duration; arm's trained behavior returns to neutral before timer expires | `time.perf_counter()` |
| Data Recorder | Optionally writes each episode to a LeRobot dataset for future fine-tuning | `LeRobotDataset.add_frame()`, `save_episode()` |

### 5.2 State Machine

```
IDLE ──[start command]──▶ DETECTING ──[color found]──▶ RUNNING
  ▲                           │                            │
  │                    [no color / pause]        [episode_time_s elapsed]
  │                           │                            │
  │                         wait                        HOMING
  │                                                        │
  │                                          [smooth interpolation to home_pos]
  │                                                        │
  │                                                    SETTLING
  │                                                        │
  │                                          [max joint delta < threshold]
  │                                                        │
  └────────────────────────────────────────────────────────┘
                                                           │
                         ERROR ◀──[exception / settle timeout / e-stop]
                           │     [mid-episode error → emergency HOMING first]
                     [safe disconnect]
                           │
                          EXIT
```

| State | Entry Condition | Actions | Exit Condition |
|-------|-----------------|---------|----------------|
| IDLE | System started | Load all policies and home position into memory, connect robot | Start command received |
| DETECTING | Previous cycle complete or startup | `robot.get_observation()` → HSV mask front frame | Color blob ≥ 1500px OR `--color` CLI flag OR GPIO override |
| RUNNING | Color detected | `policy.reset()` → `predict_action()` → `send_action()` at 30Hz | `episode_time_s` elapsed OR unhandled exception |
| HOMING | Episode loop ended (normal or error) | Ease-in-out interpolation to `home_pos` over 80 steps at 33ms/step (~2.7s) | Interpolation complete; if error, re-raise after homing |
| SETTLING | HOMING complete | Poll motor positions every 500ms; keep sending `home_pos` to prevent servo torque dropout | All joint deltas < 1.0° for one interval → proceed; timeout 8s → ERROR |
| ERROR | Unhandled exception OR settle timeout | Log error, `robot.disconnect()` | Safe exit |

### 5.3 Decision Requirements

| ID | Requirement | Parent | Verification |
|----|-------------|--------|--------------|
| DEC-001 | All enabled-color policies SHALL be loaded into GPU memory at startup, not per-episode. | SYS-003 | TC-DEC-001 |
| DEC-002 | Policy state SHALL be reset (`policy.reset()`, `preprocessor.reset()`, `postprocessor.reset()`) between every episode. | SYS-002 | Inspection |
| DEC-003 | WHEN a GPIO keypress is received, it SHALL override automatic color detection for the duration of one episode, then return to autonomous detection. | SYS-005 | TC-DEC-002 |
| DEC-004 | WHEN no cylinder is detected for the duration of a `detect_pause` interval (default 0.3s), the system SHALL continue checking without triggering an episode. | SYS-004 | TC-DEC-003 |
| DEC-005 | WHEN `--record` is enabled, every completed episode SHALL be written to the local LeRobot dataset via `add_frame()` and `save_episode()`. | SYS-002 | TC-DEC-004 |
| DEC-006 | WHEN an episode ends, the system SHALL enter SETTLING state and poll motor positions every 500ms until all joint deltas are less than 1.0° between consecutive readings. | SYS-006 | TC-DEC-005 |
| DEC-007 | WHEN the arm has not settled within 8 seconds of episode end, the system SHALL transition to ERROR state, disconnect the robot, and exit rather than start a new episode from an unknown arm position. | SYS-006 | TC-DEC-005 |
| DEC-008 | A new detection cycle SHALL NOT begin until SETTLING has returned success. | SYS-006 | TC-DEC-005 |
| DEC-009 | The CLI and GPIO interfaces SHALL be simultaneously active; neither blocks the other. | SYS-005 | TC-DEC-002 |
| DEC-010 | WHEN `home_position.json` exists, the system SHALL load it at startup and use those motor positions as the target return pose for all HOMING transitions. | SYS-006 | TC-DEC-006 |
| DEC-011 | WHEN no `home_position.json` exists, the system SHALL log a warning at startup and rely on the policy's trained return behavior; the passive settle check SHALL still apply. | SYS-006 | TC-DEC-006 |
| DEC-012 | The `--capture_home` CLI flag SHALL connect the robot, read current motor positions, save them to `home_position.json`, print each joint value, and exit — no sort episode is triggered. | SYS-005 | TC-DEC-006 |

---

## 6. Actuation Subsystem (L1)

The Actuation Subsystem executes the trained ACT policy on the SO-101 arm. This subsystem is largely delegated to LeRobot's SO-101 driver and the ACT inference pipeline. The key design choice is **true inference** — the policy runs in a direct loop without invoking `lerobot-record`, eliminating recording overhead and SSH-related instability during production runs.

### 6.1 Subsystem Decomposition

| Component | Function | Technology |
|---|---|---|
| Robot Driver | Motor read/write, calibration, camera management | LeRobot `SO101Follower`, Feetech STS3215 via serial |
| ACT Inference Engine | Runs `get_observation() → predict_action() → send_action()` at 30Hz | LeRobot `predict_action()`, `make_robot_action()` |
| Pre/Post Processor | Normalizes observations and unnormalizes actions using training statistics | LeRobot `make_pre_post_processors()` loaded from checkpoint |
| Feature Mapper | Translates robot obs/action dicts to the tensor format the policy expects | `build_dataset_frame()`, `hw_to_dataset_features()` |
| Safe Disconnect | Ensures motor torque is released and USB port closed on any exit path | `robot.disconnect()` in `finally` block |

### 6.2 Actuation Requirements

| ID | Requirement | Parent | Verification |
|----|-------------|--------|--------------|
| ACT-001 | The inference loop SHALL run at 30Hz using `precise_sleep()` to maintain timing. | SYS-003 | TC-ACT-001 |
| ACT-002 | The robot SHALL remain connected for the entire session; it SHALL NOT disconnect and reconnect between episodes. | SYS-003 | Inspection |
| ACT-003 | Pre/post-processor normalization statistics SHALL be loaded from the checkpoint directory at startup, not recomputed. | SYS-002 | Inspection |
| ACT-004 | The SO-101 arm SHALL use saved calibration files; interactive recalibration SHALL NOT occur during an autonomous session. | SYS-002 | Inspection |
| ACT-005 | WHEN emergency stop is triggered, `robot.disconnect()` SHALL be called and all motor commands SHALL cease within 1 second. | SYS-006 | TC-ACT-002 |
| ACT-006 | The system SHALL require only a `pretrained_model/` checkpoint path to load a policy — no dataset object or HuggingFace Hub connection required at runtime. | SYS-002 | Inspection |
| ACT-007 | WHEN an episode ends normally and `home_position.json` is loaded, the system SHALL execute a smooth homing motion using ease-in-out interpolation over 80 steps at 33ms/step (~2.7s) before entering SETTLING. | SYS-006 | TC-ACT-003 |
| ACT-008 | WHEN an episode fails mid-execution and `home_position.json` is loaded, the system SHALL attempt an emergency smooth home before logging the error and disconnecting. | SYS-006 | TC-ACT-004 |
| ACT-009 | DURING SETTLING, the system SHALL continue sending the home position as a position command at each poll interval to prevent Feetech servo torque dropout and arm drop. | SYS-006 | TC-ACT-003 |

---

## 7. Training Subsystem (L1)

The Training Subsystem is the mechanism by which the robot learns and improves. It operates **offline** from the runtime system but is a first-class part of the overall architecture because the quality of trained models directly determines sort success rate. It has two data inputs — human teleoperation demos and autonomous live-action rollouts — and one output: a deployable ACT policy checkpoint.

### 7.1 Subsystem Decomposition

| Component | Function | Technology |
|---|---|---|
| Data Collection — Teleoperation | Record human-led demonstrations via leader arm | `lerobot-record`, SO-101 leader arm, LeRobot dataset v3 format |
| Data Collection — Live Rollouts | Record autonomous sort episodes for continuous improvement | `sort_controller.py --record`, `LeRobotDataset.add_frame/save_episode` |
| Dataset Transfer | Move recorded dataset from Jetson to training machine | `rsync` over SSH |
| Training Environment | Isolated Python environment with GPU-accelerated PyTorch | Miniconda, `lerobot` conda env, CUDA 12.8, PyTorch 2.8.0+cu128 |
| Training Engine | Fine-tune ACT policy from dataset | `lerobot-train`, ACT architecture, batch size 16, chunk size 50 |
| Model Transfer | Move trained checkpoint from laptop back to Jetson | `rsync` over SSH, naming convention enforced |
| Model Registry | Versioned storage of all trained checkpoints on Jetson | `~/lerobot/outputs/train/act_<color>_<version>_<source>_<steps>/` |

### 7.2 Training Stack Details

| Layer | Technology | Version / Detail |
|---|---|---|
| OS | Windows 11 + WSL2 | Ubuntu 22.04 inside WSL2 |
| GPU | NVIDIA RTX 5070 Ti | Blackwell architecture, sm_120, 16GB VRAM |
| CUDA | CUDA Toolkit | 12.8 (via Windows driver, shared to WSL2) |
| Python Environment | Miniconda | `conda activate lerobot` |
| Deep Learning | PyTorch | 2.8.0+cu128 (`torch`, `torchvision`) |
| Video Backend | pyav | Used for LeRobot dataset video encoding |
| Policy Architecture | ACT | Action Chunking Transformer — chunk size 50, n_action_steps 15 |
| Training Script | `lerobot-train` | 100k steps, batch size 16, save every 20k steps |
| Runtime (Jetson) | PyTorch | 2.8.0 (CUDA 12.6, Jetson-native) |

### 7.3 Data Sources and the Learning Flywheel

```
SOURCE 1: Teleoperation (human-led, high quality)
  SO-101 Leader Arm → lerobot-record → LeRobot dataset
  100 episodes per color → used for initial model training

SOURCE 2: Autonomous Rollouts (live-action, continuous)
  sort_controller.py --record → dataset.add_frame/save_episode
  Each production sort → new training episode automatically captured

COMBINED DATASET → rsync to laptop → lerobot-train → new checkpoint
                                                           │
                   rsync back to Jetson ◀─────────────────┘
                   sort_controller.py loads new model
                   Success rate improves over time
```

Both data sources produce standard LeRobot v3 datasets. They can be merged, used independently, or weighted by quality. Autonomous rollout data is particularly valuable for capturing edge cases and workspace variation that human demos may not cover.

### 7.4 Training Requirements

| ID | Requirement | Parent | Verification |
|----|-------------|--------|--------------|
| TRAIN-001 | The training environment SHALL use a dedicated `lerobot` conda environment with PyTorch 2.8.0+cu128 targeting CUDA 12.8 (sm_120). | SYS-007 | TC-TRAIN-001 |
| TRAIN-002 | Initial teleoperation datasets SHALL contain ≥100 successful episodes per color recorded in LeRobot v3 format on the Jetson. | SYS-002 | TC-TRAIN-002 |
| TRAIN-003 | The training script SHALL use `lerobot-train` with ACT policy, batch size 16, chunk size 50, and checkpoints saved every 20k steps. | SYS-002 | Inspection |
| TRAIN-004 | Trained model checkpoints SHALL follow the naming convention `act_<color>_<version>_<source>_<steps>` (e.g., `act_green_v1_laptop_100k`). | SYS-002, SYS-007 | Inspection |
| TRAIN-005 | Autonomous sort episodes recorded with `--record` SHALL produce valid LeRobot datasets usable for fine-tuning without preprocessing. | SYS-007 | TC-TRAIN-003 |
| TRAIN-006 | The dataset transfer procedure SHALL use `rsync` over SSH with `--mkpath` to preserve LeRobot directory structure. | SYS-007 | Inspection |
| TRAIN-007 | A training run SHALL complete within 6 hours at 100k steps on the RTX 5070 Ti. | SYS-003 | TC-TRAIN-004 |
| TRAIN-008 | Final training loss SHALL be ≤0.05 for a model to be considered deployment-ready. | SYS-002 | TC-TRAIN-004 |

### 7.5 Model Improvement Cadence

This is not a one-time training run. As autonomous rollout data accumulates, the model is periodically retrained on the combined dataset:

| Trigger | Action |
|---|---|
| ≥50 new autonomous episodes recorded | Fine-tune existing model (`--resume=true`) on combined dataset |
| New color added (e.g., yellow) | Full training run from scratch on that color's dataset |
| Success rate drops below threshold | Collect more teleoperation demos, retrain |
| New workspace/lighting conditions | Record additional demos, add to dataset, retrain |

---

## 8. Interface Specifications

Interfaces are the boundaries between subsystems. Each arrow in the block diagram carries data with a defined format and timing requirement.

| ID | Interface | Data | Requirement |
|----|-----------|------|-------------|
| IF-001 | Front Camera → HSV Masker | BGR numpy array, 640×480 | Frame SHALL be delivered within 33ms (30fps) |
| IF-002 | HSV Masker → Sort Loop | Detected color string (`"green"` \| `"blue"` \| `"yellow"` \| `None`) | Updated every `detect_pause` interval (default 0.3s) |
| IF-003 | GPIO Thread → Sort Loop | Shared dict: `{"override_color": str \| None, "paused": bool, "stop": bool}` | Updated within 100ms of keypress |
| IF-004 | Sort Loop → ACT Engine | Current color string → selects policy tuple from registry | Lookup SHALL be O(1) from pre-loaded dict |
| IF-005 | Robot Driver → ACT Engine | `obs dict`: `{"shoulder_pan.pos": float, ..., "front": np.ndarray, "handeye": np.ndarray}` | Delivered by `robot.get_observation()` each inference step |
| IF-006 | ACT Engine → Robot Driver | `action dict`: `{"shoulder_pan.pos": float, ..., "gripper.pos": float}` | Sent via `robot.send_action()` at 30Hz |
| IF-007 | ACT Engine → Data Recorder | `frame dict`: obs_frame + action_frame + task string | Passed to `dataset.add_frame()` when `--record` enabled |
| IF-008 | Data Recorder → Local Filesystem | LeRobot dataset at `~/.cache/huggingface/lerobot/local/<name>/` | Written incrementally; finalized via `save_episode()` |
| IF-009 | Jetson → Laptop Training (offline) | LeRobot dataset directory via `rsync -av --mkpath` over SSH | Full directory structure preserved on transfer |
| IF-010 | Laptop → Jetson Deployment (offline) | Trained `pretrained_model/` directory via `rsync` over SSH | Naming convention `act_<color>_<version>_<source>_<steps>` enforced |
| IF-011 | Checkpoint → Policy Registry | `ACTPolicy.from_pretrained(path)` + `make_pre_post_processors(path)` | Loaded at startup; no HuggingFace Hub connection required at runtime |

---

## 9. Verification — Test Cases

Each requirement maps to at least one test case. Tests are run per implementation phase and results recorded here.

### 9.1 System-Level Tests

| ID | Traces To | Procedure | Pass Criteria | Phase 2 Result | Phase 3 Result |
|----|-----------|-----------|---------------|----------------|----------------|
| TC-SYS-001 | SYS-001 | Place green, blue, yellow cylinder one at a time. Run detection only. Check terminal output. | Correct color logged for each | [ ] | [ ] |
| TC-SYS-002 | SYS-002 | Place green cylinder. Run full sort. Observe bin. Repeat 10×. Then repeat for blue. | ≥8/10 correct bin placements per color | [ ] | [ ] |
| TC-SYS-003 | SYS-003 | Time from cylinder placement to arm returning to neutral for 5 cycles. | Mean ≤30 seconds | [ ] | [ ] |
| TC-SYS-004 | SYS-004 | Place yellow cylinder with no yellow model loaded. Observe system behavior. | Warning logged, no episode triggered | [ ] | [ ] |
| TC-SYS-005 | SYS-005 | Run with `--color green` CLI flag. Run with GPIO key `1`. | Both trigger a single green episode | N/A | [ ] |
| TC-SYS-006 | SYS-006 | Ctrl+C during an active episode. Observe robot state. | Robot disconnects cleanly, no motor runaway | [ ] | [ ] |
| TC-SYS-007 | SYS-007 | Run `--record` for 5 cycles. Run `lerobot-train` on resulting dataset. | Training completes without error; dataset has 5 valid episodes | [ ] | [ ] |

### 9.2 Perception Tests

| ID | Traces To | Procedure | Pass Criteria | Phase 2 Result |
|----|-----------|-----------|---------------|----------------|
| TC-PERC-001 | PERC-001 | Run `detect_colors.py`, measure fps in output for 30 seconds. | ≥30fps, 640×480 confirmed in log | PASS — live feed confirmed at 640×480 |
| TC-PERC-002 | PERC-002/003/004 | Place each colored cylinder in detection zone. Read pixel count from `detect_colors.py`. | Each color reports ≥1500px when present, ≤50px when absent | PASS — green, blue, yellow all detected correctly |
| TC-PERC-003 | PERC-005 | Remove all cylinders. Disable ambient colored LEDs (keyboard RGB etc). Run 60 seconds. | 0 false detections | PASS — threshold raised to 1500px eliminated keyboard RGB false triggers |
| TC-PERC-004 | PERC-006 | Measure time from `cv2.cvtColor()` call to `detect_color()` return for 100 frames. | Mean ≤50ms | PASS — OpenCV HSV on 640×480 ~5ms on Jetson Orin |

### 9.3 Training Subsystem Tests

| ID | Traces To | Procedure | Pass Criteria | Status |
|----|-----------|-----------|---------------|--------|
| TC-TRAIN-001 | TRAIN-001 | Run `conda activate lerobot && python -c "import torch; print(torch.version.cuda, torch.cuda.get_device_name(0))"` on laptop. | CUDA 12.8 shown, RTX 5070 Ti listed | COMPLETE |
| TC-TRAIN-002 | TRAIN-002 | Check dataset info.json for each color: `total_episodes` field. | ≥100 for green and blue | COMPLETE |
| TC-TRAIN-003 | TRAIN-005 | Run `sort_controller.py --record` for 3 cycles. Run `lerobot-visualize` on output dataset. | 3 valid episodes render without error | NOT TESTED |
| TC-TRAIN-004 | TRAIN-007/008 | Review training log for final loss and wall-clock time. | Loss ≤0.05, completed within 6 hours | COMPLETE (green: 0.04, blue: 0.047) |

### 9.5 Decision & Actuation Tests

| ID | Traces To | Procedure | Pass Criteria | Phase 2 Result |
|----|-----------|-----------|---------------|----------------|
| TC-DEC-001 | DEC-001 | Start controller with two models. Check Jetson GPU memory via `nvidia-smi` after startup before first episode. | Both policies visible in GPU memory before episode 1 | [ ] |
| TC-DEC-002 | DEC-003/007 | While in autonomous detect loop, press GPIO key `2` (blue). Observe which policy runs. | Blue policy runs for one episode regardless of camera detection | N/A (Phase 3) |
| TC-DEC-003 | DEC-004 | Run controller with no cylinder present. Observe for 60 seconds. | No episode triggered, system loops in DETECTING state | [ ] |
| TC-DEC-004 | DEC-004 | Run controller with no cylinder present. Observe for 60 seconds. | No episode triggered, system loops in DETECTING state | [ ] |
| TC-DEC-005 | DEC-006/007/008 | Run one full sort cycle. Watch terminal for SETTLING state output. Then block arm from returning (hold it). | Settling logs joint deltas each interval; logs ERROR and disconnects cleanly after 8s timeout | [ ] |
| TC-DEC-006 | DEC-010/011/012 | (a) Run `sort_controller.py --capture_home` with arm at neutral — confirm JSON written and joints printed. (b) Start controller — confirm "loaded home position" at startup. (c) Delete JSON — confirm warning printed. | (a) `home_position.json` written with all motor keys. (b) Load message in log. (c) Warning + passive settle fallback. | [ ] |
| TC-ACT-001 | ACT-001 | Log loop timing for 100 steps. Compute actual Hz. | Mean ≥28Hz (within 7% of 30Hz target) | [ ] |
| TC-ACT-002 | ACT-005 | Ctrl+C mid-episode. Confirm `robot.disconnect()` called in log. | "Robot disconnected" printed, no hanging process | PASS — confirmed clean disconnect, 0 cycles completed |
| TC-ACT-003 | ACT-007/009 | With `home_position.json` captured, run a full sort. Watch arm trajectory after episode timer ends. | Arm moves smoothly to home over ~2-3s; no limp/drop; SETTLING log confirms arrival; "homing motion complete" printed. | [ ] |
| TC-ACT-004 | ACT-008 | Start an episode. While arm is mid-motion, unplug a camera to trigger an error. | Log shows episode error, then "emergency home — bringing arm back after error...", then disconnect. Arm moves toward home before stopping. | [ ] |

---

## 10. Traceability Matrix

Every requirement traces to at least one test case. This table closes the V-model.

| Requirement | Subsystem | Test Case(s) | Status |
|-------------|-----------|-------------|--------|
| SYS-001 | System | TC-SYS-001, TC-PERC-002 | NOT TESTED |
| SYS-002 | System | TC-SYS-002, TC-DEC-004 | NOT TESTED |
| SYS-003 | System | TC-SYS-003, TC-ACT-001 | NOT TESTED |
| SYS-004 | System | TC-SYS-004, TC-PERC-003 | NOT TESTED |
| SYS-005 | System | TC-SYS-005, TC-DEC-002 | NOT TESTED |
| SYS-006 | System | TC-SYS-006, TC-ACT-002 | NOT TESTED |
| SYS-007 | System | TC-SYS-007, TC-TRAIN-003 | NOT TESTED |
| PERC-001 | Perception | TC-PERC-001 | PASS |
| PERC-002 | Perception | TC-PERC-002 | PASS |
| PERC-003 | Perception | TC-PERC-002 | PASS |
| PERC-004 | Perception | TC-PERC-002 | PASS |
| PERC-005 | Perception | TC-PERC-003 | PASS |
| PERC-006 | Perception | TC-PERC-004 | PASS |
| DEC-001 | Decision | TC-DEC-001 | NOT TESTED |
| DEC-003 | Decision | TC-DEC-002 | NOT TESTED |
| DEC-004 | Decision | TC-DEC-003 | NOT TESTED |
| DEC-006 | Decision | TC-DEC-005 | NOT TESTED |
| DEC-007 | Decision | TC-DEC-005 | NOT TESTED |
| DEC-008 | Decision | TC-DEC-005 | NOT TESTED |
| DEC-010 | Decision | TC-DEC-006 | NOT TESTED |
| DEC-011 | Decision | TC-DEC-006 | NOT TESTED |
| DEC-012 | Decision | TC-DEC-006 | NOT TESTED |
| ACT-001 | Actuation | TC-ACT-001 | NOT TESTED |
| ACT-005 | Actuation | TC-ACT-002 | PASS |
| ACT-007 | Actuation | TC-ACT-003 | NOT TESTED |
| ACT-008 | Actuation | TC-ACT-004 | NOT TESTED |
| ACT-009 | Actuation | TC-ACT-003 | NOT TESTED |
| TRAIN-001 | Training | TC-TRAIN-001 | COMPLETE |
| TRAIN-002 | Training | TC-TRAIN-002 | COMPLETE |
| TRAIN-005 | Training | TC-TRAIN-003 | NOT TESTED |
| TRAIN-007 | Training | TC-TRAIN-004 | COMPLETE |
| TRAIN-008 | Training | TC-TRAIN-004 | COMPLETE |

---

# PART III — IMPLEMENTATION PHASES

---

## 11. Phase 1 — Data Collection & Training

### 10.1 Overview

| Aspect | Detail |
|--------|--------|
| **Approach** | Teleoperate SO-101 arm through 100 demonstrations per color, train ACT policy on laptop GPU, transfer model to Jetson via rsync |
| **Data Collection Tool** | `lerobot-record` with SO-101 leader arm |
| **Training Platform** | RTX 5070 Ti (WSL2), PyTorch 2.8.0+cu128, sm_120 Blackwell |
| **Training Duration** | ~5 hours per color at 100k steps, batch size 16 |
| **Status** | **COMPLETE** — green and blue models deployed |

### 10.2 Trained Model Registry

| Model Name | Color | Episodes | Steps | Loss | Status |
|---|---|---|---|---|---|
| `act_green_v1_laptop_100k` | Green | 100 | 100k | ~0.04 | **Deployed, tested** |
| `act_blue_v1_laptop_100k` | Blue | 100 | 100k | ~0.047 | **Deployed, tested** |
| `act_yellow_*` | Yellow | — | — | — | Not yet recorded |

### 10.3 Data Collection Procedure

1. Position arm at neutral/collapsed position
2. Run `bash sort.sh <color> record` — guided by `sort.sh` interactive mode
3. Teleoperate 100 episodes: pick cylinder from detection zone → place in correct bin → return to neutral
4. Run `bash sort.sh <color> train <steps>` — trains on Jetson-recorded data via laptop

### 10.4 Transfer Procedure

```bash
# Jetson → Laptop (dataset)
rsync -av --mkpath jetson:/home/jetson23/.cache/huggingface/lerobot/local/<dataset>/ \
    ~/lerobot_data/<dataset>/

# Laptop → Jetson (model)
rsync -av outputs/train/act_<color>_v1_laptop_100k/ \
    jetson:/home/jetson23/lerobot/outputs/train/act_<color>_v1_laptop_100k/
```

---

## 12. Phase 2 — Autonomous Sort Controller

### 11.1 Overview

| Aspect | Detail |
|--------|--------|
| **Approach** | Direct policy inference loop — no `lerobot-record`, no dataset writes during inference |
| **Color Selection** | CLI flags (`--model_green`, `--model_blue`) at startup; `--color` for single-episode manual trigger |
| **Recording** | Optional via `--record` flag; uses LeRobot dataset API for future fine-tuning data |
| **Entry Point** | `sort_controller.py` + `sort.sh infer` command |
| **Status** | `sort_controller.py` implemented; `detect_colors.py` and recording pending |

### 11.2 Implementation Chunks

| Chunk | Deliverable | Done Condition |
|-------|------------|----------------|
| 1 | `detect_colors.py` — live HSV tuning tool | Each cylinder shows ≥500px; no false triggers |
| 2 | `--color` flag + `sort.sh infer` | Single episode runs cleanly, arm returns to neutral, process exits |
| 3 | `--record` flag + dataset writing | 5 cycles produce valid LeRobot dataset at expected path |

### 11.3 Inference Loop (No Recording)

```python
# Core loop — runs at 30Hz per step
obs        = robot.get_observation()
obs_frame  = build_dataset_frame(ds_features, obs, prefix="observation")
action_t   = predict_action(obs_frame, policy, device, pre, post, use_amp, task, robot_type)
action     = make_robot_action(action_t, ds_features)
robot.send_action(action)
precise_sleep(1/fps - dt)
```

---

## 13. Phase 3 — GPIO Keypad Integration

### 12.1 Overview

| Aspect | Detail |
|--------|--------|
| **Approach** | Background thread reads GPIO pins, writes to shared state dict checked by the sort loop |
| **Keymap** | Key 1 = green, 2 = blue, 3 = yellow, 4 = pause, 5 = resume, 6 = safe stop |
| **Interface** | GPIO and CLI simultaneously active; GPIO overrides detection for one episode only |
| **Status** | Planned |

### 12.2 Keymap

| Key | Function | Scope |
|-----|----------|-------|
| `1` | Force green sort | One episode |
| `2` | Force blue sort | One episode |
| `3` | Force yellow sort | One episode |
| `4` | Pause autonomous loop | Until key 5 |
| `5` | Resume autonomous loop | Immediate |
| `6` | Safe stop — disconnect robot and exit | Session |

### 12.3 Implementation Approach

`gpio_keypad.py` runs as a daemon thread reading GPIO pins via `RPi.GPIO` or Jetson GPIO library. It writes to a shared `keypad_state` dict. The sort loop checks `keypad_state["override_color"]` each detection cycle before running HSV masking.

---

# PART IV — ADMINISTRATION

## 14. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-04-12 | Team + AI Agent | Initial draft — subsystem requirements |
| 1.0 | 2026-04-12 | Team + AI Agent | Full rewrite to match P-D-A template structure, add test cases, traceability matrix |
| 1.1 | 2026-04-12 | Team + AI Agent | Added HOMING state, DEC-010/011/012, ACT-007/008/009, TC-DEC-006, TC-ACT-003/004 — smooth arm return and emergency homing after episode end or mid-episode error; `--capture_home` CLI command and `home_position.json` storage |
