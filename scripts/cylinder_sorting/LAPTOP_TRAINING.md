# Training on the RTX 5070 Laptop

Use the laptop for fast training (~1–2 hrs per run) and copy the model back to the Jetson for eval.

---

## Overview

| Machine | Role |
|---|---|
| Jetson | Record data · Run eval · Control robot |
| Laptop (RTX 5070) | Train the ACT policy |

---

## 1. One-Time Laptop Setup

### Install lerobot (Linux / WSL2)

Clone **our fork** (not the upstream HF repo — ours has custom fixes for relative actions, data loading, etc.):

```bash
git clone git@github.com:giohitt/lerobot-asu-capstone.git lerobot
cd lerobot
conda create -n lerobot python=3.10 -y
conda activate lerobot
pip install -e ".[act]"
```

> If using WSL2 on Windows, make sure your NVIDIA drivers are installed on Windows and CUDA is visible in WSL2:
> `nvidia-smi` should return your GPU info.
>
> You'll also need SSH set up for GitHub in WSL2. Run `ssh-keygen -t ed25519 -C "your-email"` and add the key to [github.com/settings/keys](https://github.com/settings/keys). Or clone via HTTPS: `https://github.com/giohitt/lerobot-asu-capstone.git`

### Verify CUDA is available

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Should print `True  NVIDIA GeForce RTX 5070` (or similar).

---

## 2. Copy Dataset from Jetson to Laptop

Run this **on the laptop**:

```bash
# SSH into the Jetson directly
ssh jetson23@192.168.0.165

JETSON_IP=192.168.0.165

# Copy the green dataset
rsync -avz --progress \
  jetson23@$JETSON_IP:~/.cache/huggingface/lerobot/local/cylinder_sorting_green_v1/ \
  ~/.cache/huggingface/lerobot/local/cylinder_sorting_green_v1/

# Or copy the mixed dataset
rsync -avz --progress \
  jetson23@$JETSON_IP:~/.cache/huggingface/lerobot/local/cylinder_sorting_mixed_v1/ \
  ~/.cache/huggingface/lerobot/local/cylinder_sorting_mixed_v1/
```

---

## 3. Train on the Laptop

From the `lerobot` directory on the laptop:

### Green model (100 episodes → 40,000 steps recommended)

```bash
DATASET="local/cylinder_sorting_green_v1"
MODEL="act_green_v1"
STEPS=40000
SAVE_FREQ=$(( STEPS / 5 ))

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --output_dir="outputs/train/$MODEL" \
  --job_name="$MODEL" \
  --steps=$STEPS \
  --save_freq=$SAVE_FREQ \
  --batch_size=32 \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_lr_backbone=1e-5 \
  --policy.use_amp=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=15
```

> `batch_size=32` instead of 16 — the 5070 has more VRAM so we can double it for faster training.
> Checkpoints saved every 8,000 steps (5 total).

### Mixed model (retraining with more data)

Same command but change:
```bash
DATASET="local/cylinder_sorting_mixed_v1"
MODEL="act_mixed_v1"
STEPS=20000
```

### Watch training progress

```bash
tail -f outputs/train/logs/act_green_v1.log
```

Target loss: **below 0.05** for reliable grasping. The curve should keep dropping through 40K steps with 100 clean episodes.

---

## 4. Copy Model Back to Jetson

Run this **on the laptop** when training is done:

```bash
JETSON_IP=192.168.0.165

rsync -avz --progress \
  outputs/train/act_green_v1/checkpoints/last/pretrained_model/ \
  jetson23@$JETSON_IP:~/lerobot/outputs/train/act_green_v1/checkpoints/last/pretrained_model/
```

The Jetson eval script always looks at `checkpoints/last/pretrained_model/` — no config changes needed.

---

## 5. Eval on the Jetson

Back on the Jetson:

```bash
bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green eval
```

---

## Checkpoint Structure (for reference)

```
outputs/train/act_green_v1/checkpoints/
├── 008000/
├── 016000/
├── 024000/
├── 032000/
├── 040000/
└── last/                        ← eval uses this
    └── pretrained_model/
        ├── config.json
        └── model.safetensors    ← the actual weights (~200MB)
```

Only the `last/pretrained_model/` folder needs to be copied back to the Jetson.

---

## Step Count Reference

| Episodes | Recommended steps | Laptop time (RTX 5070) | Jetson time |
|---|---|---|---|
| 20 | 8,000 | ~20 min | ~5 hrs |
| 50 | 15,000–20,000 | ~45 min | ~12 hrs |
| 100 | 30,000–40,000 | ~1.5–2 hrs | ~23 hrs |

---

## Common Issues

| Error | Fix |
|---|---|
| `CUDA not available` | Check `nvidia-smi` in WSL2; reinstall CUDA toolkit |
| `FileExistsError` on output dir | Delete `outputs/train/act_green_v1/` and retry |
| Loss stuck above 0.1 after 30K steps | Data too varied or too few episodes — add more data |
| Loss plateaus at 0.07–0.09 | Normal for mixed/varied data; try finetuning with 10K more steps |
