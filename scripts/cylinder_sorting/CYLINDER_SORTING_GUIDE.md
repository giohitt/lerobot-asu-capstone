# Cylinder Sorting — SO101 Training Guide

```bash


```

ACT policy trained to sort colored cylinders into matching bins using the SO101 arm.

**Task:** Green cylinder → left bin &nbsp;|&nbsp; Blue cylinder → right bin &nbsp;|&nbsp; Yellow → ignore (distractor)

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [What Interactive Mode Asks You](#2-what-interactive-mode-asks-you)
3. [Before You Start](#3-before-you-start)
   - Hardware Setup
   - Verify Camera Framing
   - Warm Up with Teleoperation
4. [Recording Tips & SSH Controls](#4-recording-tips--ssh-controls)
5. [Commands Reference](#5-commands-reference)
   - record
   - train
   - finetune
   - eval
   - clean
6. [Model Map & Checkpoints](#6-model-map--checkpoints)
7. [Recommended Pipeline](#7-recommended-pipeline)
8. [Common Issues](#8-common-issues)

---

## 1. Quick Start

Everything runs through one script.

### Interactive mode — guided prompts for every decision

Run with no arguments for the full experience. The script asks what color, what command, and walks through options step by step:

```bash
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh
```

Provide just the color to skip the color menu:

```bash
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed
```

### Direct mode — skip all prompts

Pass all arguments to run immediately with no questions asked:

```bash
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh <color> <command> [options]
```

| Color | Command | Options |
|---|---|---|
| `green` `blue` `yellow` `mixed` | `record` `train` `finetune` `eval` `clean` | episode count, step count, or clean target |

---

## 2. What Interactive Mode Asks You

**record**
- If dataset exists: Append / Fresh start / Save under a new custom name?
- How many episodes this session? (default: 10)
- Confirm before connecting to robot

**train**
- If model exists: Train from scratch (overwrites) / Continue from last checkpoint?
- How many steps? (auto-suggested based on episode count)
- Confirm before starting

**eval**
- If eval dataset exists: Overwrite / Use a custom run name?
- How many eval episodes? (default: 5)
- Confirm before connecting to robot

**clean**
- Shows what currently exists — data, model, eval — for that color
- What to delete: data / model / eval / all?
- Confirm before deleting (defaults to No to prevent accidents)

---

## 3. Before You Start

### Hardware Setup

| Component | Port | Notes |
|---|---|---|
| Follower arm | `/dev/ttyFOLLOWER` | Sorts cylinders autonomously during eval |
| Leader arm | `/dev/ttyLEADER` | Used during recording only |
| Handeye camera | auto-detected | Wrist-mounted |
| Front camera | auto-detected | Scene overview |

> Ports are pinned by udev rules — won't shift on replug. Camera indices are auto-detected each run.

Verify both arms are connected:
```bash
ls -la /dev/ttyLEADER /dev/ttyFOLLOWER
```

---

### Verify Camera Framing

Saves a snapshot from each camera — open the images in the IDE to check framing:

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

Open `cam_0_index*.jpg` and `cam_1_index*.jpg` in Cursor's file explorer.

---

### Warm Up with Teleoperation

Connects both arms and activates cameras without recording anything. Use to verify everything is live and move the arm around before starting a recording session:

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

> `Ctrl+C` to stop. Update `index_or_path` if camera indices have shifted.

---

## 4. Recording Tips & SSH Controls

### Tips for Good Data

- **One cylinder per episode** — pick up, place in bin, return arm to home
- **Green always left zone, blue always right zone** — never swap
- **Vary position within the zone** — slightly different spot each episode so the model generalizes
- **Go straight to the cylinder** — no searching or hesitation before grabbing
- **Consistent home position** — start every episode from the same arm pose
- **Alternate colors** — G, B, G, B... for `mixed` to keep 50/50 balance
- **5–10 seconds per episode** — short, clean, deliberate motions work best for ACT

### SSH Headless Controls During Recording

| Input | When | What it does |
|---|---|---|
| `Enter` | Before episode | Start recording |
| `Enter` | During episode | Save and end episode early |
| `Enter` | During reset | Skip remaining reset time |
| `r` + `Enter` | During episode | Discard episode, re-record |
| `q` + `Enter` | Anytime | Stop recording, save completed episodes |

---

## 5. Commands Reference

### record

Collect teleoperation episodes. Each session appends to the same dataset — take breaks freely.

```bash
# Interactive (asks append/fresh, how many episodes, confirms)
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed record

# Direct — 20 episodes, no questions
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed record 20
```

> Sessions are **additive** — each run appends to the same dataset automatically.

---

### train

Train an ACT policy from scratch on the recorded data. Runs in the background — `Ctrl+C` detaches from the log but training keeps going.

```bash
# Interactive (asks scratch vs finetune, how many steps, confirms)
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed train

# Direct
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed train 20000
```

- Checkpoints saved every **steps ÷ 5** automatically
- Stop training: `kill <PID printed at startup>`

**Re-attach to a running training session:**

```bash
# Follow live log — Ctrl+C detaches without stopping training
tail -f /home/jetson23/lerobot/outputs/train/logs/act_mixed_v1.log
```

**Reading the log line:**
```
INFO 2026-04-12 13:39:16 ot_train.py:458 step:400 smpl:6K ep:10 epch:0.47 loss:1.823 grdn:45.2 lr:1.0e-04 updt_s:1.95 data_s:0.02
```
| Field | Meaning |
|---|---|
| `step` | Current training step |
| `loss` | Training loss — target is below 0.1 for reliable sorting |
| `updt_s` | Seconds per step — lower is faster |
| `epch` | How many times through the full dataset |

**Step count guide:**

| Episodes | Recommended steps |
|---|---|
| 10 | 3 000–5 000 |
| 25 | 8 000–10 000 |
| 50 | 15 000–20 000 |
| 100 | 20 000–30 000 |

---

### finetune

Continue training an existing model after adding more episodes. Resumes from the last checkpoint — no wasted work.

```bash
# Interactive (guided through steps, confirms)
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed finetune

# Direct — 10k more steps
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed finetune 10000
```

> Use `train` when starting fresh. Use `finetune` when the model is already good and you just added more data.

---

### eval

Run the trained policy on the robot. No leader arm needed — follower moves autonomously.

```bash
# Interactive (asks overwrite vs custom name, how many episodes, confirms)
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed eval

# Direct — 5 eval episodes
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed eval 5
```

> Manually return the arm to home position between episodes during eval.

---

### clean

Delete recording data, trained models, and/or eval datasets.

```bash
# Interactive (shows what exists, asks what to delete, confirms)
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed clean

# Direct
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed clean all    # everything
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed clean data   # recording dataset only
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed clean model  # trained model only
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed clean eval   # eval dataset only
```

| Target | When to use |
|---|---|
| `all` | Starting completely fresh — bad data or wrong setup |
| `data` | Data quality was poor — re-record then retrain |
| `model` | Data is fine — retrain with different step count |
| `eval` | `FileExistsError` when running eval — required before every re-eval |

> `clean data` and `clean eval` use `sudo` — the HuggingFace cache is sometimes owned by root.

---

## 6. Model Map & Checkpoints

| Color | Zone | Bin | Dataset | Model folder |
|---|---|---|---|---|
| `green` | Left | Left | `cylinder_sorting_green_v1` | `outputs/train/act_green_v1` |
| `blue` | Right | Right | `cylinder_sorting_blue_v1` | `outputs/train/act_blue_v1` |
| `yellow` | Center | — (ignore) | `cylinder_sorting_yellow_v1` | `outputs/train/act_yellow_v1` |
| `mixed` | Both | Left / Right | `cylinder_sorting_mixed_v1` | `outputs/train/act_mixed_v1` |

**Checkpoint structure:**
```
outputs/train/act_mixed_v1/checkpoints/
├── 004000/               ← saved every steps÷5
├── 008000/
├── 012000/
├── 016000/
├── 020000/
└── last/                 ← always the most recent
    └── pretrained_model/ ← eval uses this path automatically
        ├── config.json
        └── model.safetensors
```

Datasets stored at: `~/.cache/huggingface/lerobot/local/`

---

## 7. Recommended Pipeline

### First run — build and test the mixed model

```bash
# Start fresh
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed clean all

# Record in sessions — build up to 100 episodes (50 green + 50 blue, alternated)
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed record 20   # session 1
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed record 20   # session 2
# ... keep going until you have ~100 total

# Train
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed train 20000

# Eval
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed eval
```

### Model moves toward cylinder but misses → add data and finetune

```bash
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed record 20      # more episodes
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed finetune 10000 # continue from checkpoint
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed clean eval
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed eval
```

### Model is completely wrong / bad data → start over

```bash
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed clean all
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed record 20 # do this 5 times to get 100 more episodes
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh mixed train 20000

# Repeat above collecting more than 100 episodes up to 200 and then try to fine tune the model towards correctness.

```

---

## 8. Common Issues

| Error | Cause | Fix |
|---|---|---|
| `FileExistsError` on dataset | Prior run left partial data | `sort.sh <color> clean data` |
| `FileExistsError` on eval | Prior eval still on disk | `sort.sh <color> clean eval` |
| `ERROR: No trained model found` | Train hasn't run yet | `sort.sh <color> train` first |
| `Missing motor IDs: 6` | Gripper motor cable loose | Power off, reseat TTL cable at wrist, power on |
| `Failed to sync read 'Present_Position'` / `There is no status packet!` | Follower arm lost serial communication or a motor is not responding | Check `/dev/ttyFOLLOWER`, verify follower power, reseat TTL cables, then rerun `lerobot-find-port` and retry |
| `ConnectionError: Could not connect on port` | Arm USB disconnected | `ls -la /dev/ttyLEADER /dev/ttyFOLLOWER` — replug if missing |
| `ERROR: Could not find 2 working cameras` | Camera unplugged | Check USB, retry |
| Robot moves toward cylinder but misses | Needs more data | Record more episodes, finetune |
| Robot barely moves | Model didn't converge | More steps or better quality data |


jetson23@ubuntu:~/lerobot$ bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green record 10
Detecting cameras...
[ WARN:0@0.020] global cap_v4l.cpp:914 open VIDEOIO(V4L2:/dev/video0): can't open camera by index
[ERROR:0@0.166] global obsensor_uvc_stream_channel.cpp:163 getStreamChannelGroup Camera index out of range
[ WARN:0@0.167] global cap_v4l.cpp:914 open VIDEOIO(V4L2:/dev/video1): can't open camera by index
[ERROR:0@0.168] global obsensor_uvc_stream_channel.cpp:163 getStreamChannelGroup Camera index out of range
[ WARN:0@0.196] global cap_v4l.cpp:914 open VIDEOIO(V4L2:/dev/video3): can't open camera by index
[ERROR:0@0.197] global obsensor_uvc_stream_channel.cpp:163 getStreamChannelGroup Camera index out of range
[ WARN:0@0.197] global cap_v4l.cpp:914 open VIDEOIO(V4L2:/dev/video4): can't open camera by index
[ERROR:0@0.198] global obsensor_uvc_stream_channel.cpp:163 getStreamChannelGroup Camera index out of range
[ WARN:0@0.198] global cap_v4l.cpp:914 open VIDEOIO(V4L2:/dev/video5): can't open camera by index
[ERROR:0@0.200] global obsensor_uvc_stream_channel.cpp:163 getStreamChannelGroup Camera index out of range
Detected cameras: handeye=2, front=6
INFO 2026-04-12 15:38:16 t_record.py:383 {'dataset': {'episode_time_s': 60,
             'fps': 30,
             'num_episodes': 10,
             'num_image_writer_processes': 0,
             'num_image_writer_threads_per_camera': 4,
             'private': False,
             'push_to_hub': False,
             'rename_map': {},
             'repo_id': 'local/cylinder_sorting_green_v1',
             'reset_time_s': 60,
             'root': None,
             'single_task': 'Pick green cylinder and place in left bin',
             'tags': None,
             'video': True,
             'video_encoding_batch_size': 1},
 'display_data': False,
 'play_sounds': True,
 'policy': None,
 'resume': True,
 'robot': {'calibration_dir': None,
           'cameras': {'front': {'color_mode': <ColorMode.RGB: 'rgb'>,
                                 'fourcc': None,
                                 'fps': 30,
                                 'height': 480,
                                 'index_or_path': 6,
                                 'rotation': <Cv2Rotation.NO_ROTATION: 0>,
                                 'warmup_s': 1,
                                 'width': 640},
                       'handeye': {'color_mode': <ColorMode.RGB: 'rgb'>,
                                   'fourcc': None,
                                   'fps': 30,
                                   'height': 480,
                                   'index_or_path': 2,
                                   'rotation': <Cv2Rotation.NO_ROTATION: 0>,
                                   'warmup_s': 1,
                                   'width': 640}},
           'disable_torque_on_disconnect': True,
           'id': 'my_follower',
           'max_relative_target': None,
           'port': '/dev/ttyFOLLOWER',
           'use_degrees': False},
 'teleop': {'calibration_dir': None,
            'id': 'my_leader',
            'port': '/dev/ttyLEADER',
            'use_degrees': False}}
WARNING 2026-04-12 15:38:16 deo_utils.py:40 'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder
INFO 2026-04-12 15:38:16 _client.py:1025 HTTP Request: GET https://huggingface.co/api/datasets/local/cylinder_sorting_green_v1/refs "HTTP/1.1 401 Unauthorized"
INFO 2026-04-12 15:38:16 ls/utils.py:227 Stop recording
INFO 2026-04-12 15:38:18 ls/utils.py:227 Exiting
Traceback (most recent call last):
  File "/home/jetson23/lerobot/src/lerobot/datasets/lerobot_dataset.py", line 105, in __init__
    self.load_metadata()
  File "/home/jetson23/lerobot/src/lerobot/datasets/lerobot_dataset.py", line 165, in load_metadata
    self.tasks = load_tasks(self.root)
  File "/home/jetson23/lerobot/src/lerobot/datasets/utils.py", line 352, in load_tasks
    tasks = pd.read_parquet(local_dir / DEFAULT_TASKS_PATH)
  File "/home/jetson23/miniforge3/envs/lerobot/lib/python3.10/site-packages/pandas/io/parquet.py", line 669, in read_parquet
    return impl.read(
  File "/home/jetson23/miniforge3/envs/lerobot/lib/python3.10/site-packages/pandas/io/parquet.py", line 258, in read
    path_or_handle, handles, filesystem = _get_path_or_handle(
  File "/home/jetson23/miniforge3/envs/lerobot/lib/python3.10/site-packages/pandas/io/parquet.py", line 141, in _get_path_or_handle
    handles = get_handle(
  File "/home/jetson23/miniforge3/envs/lerobot/lib/python3.10/site-packages/pandas/io/common.py", line 882, in get_handle
    handle = open(handle, ioargs.mode)
FileNotFoundError: [Errno 2] No such file or directory: '/home/jetson23/.cache/huggingface/lerobot/local/cylinder_sorting_green_v1/meta/tasks.parquet'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/jetson23/miniforge3/envs/lerobot/lib/python3.10/site-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
    response.raise_for_status()
  File "/home/jetson23/miniforge3/envs/lerobot/lib/python3.10/site-packages/httpx/_models.py", line 829, in raise_for_status
    raise HTTPStatusError(message, request=request, response=self)
httpx.HTTPStatusError: Client error '401 Unauthorized' for url 'https://huggingface.co/api/datasets/local/cylinder_sorting_green_v1/refs'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/jetson23/miniforge3/envs/lerobot/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/jetson23/miniforge3/envs/lerobot/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/jetson23/lerobot/src/lerobot/scripts/lerobot_record.py", line 556, in <module>
    main()
  File "/home/jetson23/lerobot/src/lerobot/scripts/lerobot_record.py", line 552, in main
    record()
  File "/home/jetson23/lerobot/src/lerobot/configs/parser.py", line 233, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
  File "/home/jetson23/lerobot/src/lerobot/scripts/lerobot_record.py", line 414, in record
    dataset = LeRobotDataset(
  File "/home/jetson23/lerobot/src/lerobot/datasets/lerobot_dataset.py", line 704, in __init__
    self.meta = LeRobotDatasetMetadata(
  File "/home/jetson23/lerobot/src/lerobot/datasets/lerobot_dataset.py", line 108, in __init__
    self.revision = get_safe_version(self.repo_id, self.revision)
  File "/home/jetson23/lerobot/src/lerobot/datasets/utils.py", line 529, in get_safe_version
    hub_versions = get_repo_versions(repo_id)
  File "/home/jetson23/lerobot/src/lerobot/datasets/utils.py", line 498, in get_repo_versions
    repo_refs = api.list_repo_refs(repo_id, repo_type="dataset")
  File "/home/jetson23/miniforge3/envs/lerobot/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "/home/jetson23/miniforge3/envs/lerobot/lib/python3.10/site-packages/huggingface_hub/hf_api.py", line 3738, in list_repo_refs
    hf_raise_for_status(response)
  File "/home/jetson23/miniforge3/envs/lerobot/lib/python3.10/site-packages/huggingface_hub/utils/_http.py", line 847, in hf_raise_for_status
    raise repo_err from e
huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error. (Request ID: Root=1-69dc1ed8-2b6f876f654799204eb5140a;a2c7b958-e6c7-4c19-a1df-7ae7620a55bf)

Repository Not Found for url: https://huggingface.co/api/datasets/local/cylinder_sorting_green_v1/refs.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated and your token has the required permissions.
For more details, see https://huggingface.co/docs/huggingface_hub/authentication
Invalid username or password.