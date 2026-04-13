# Training on the RTX 5070 Ti Laptop (WSL2)

Use the laptop for fast training (~5 hrs for 100k steps) and copy the model back to the Jetson for eval.

---

## Overview

| Machine | Role |
|---|---|
| Jetson | Record data · Run eval · Control robot |
| Laptop (RTX 5070 Ti) | Train the ACT policy |

---

## 1. One-Time Laptop Setup

### Prerequisites

- Windows with WSL2 installed (`wsl --install` in PowerShell as admin, then restart)
- NVIDIA drivers installed on **Windows** (not inside WSL2 — WSL2 shares the Windows driver)
- Verify GPU is visible in WSL2: `nvidia-smi` should show your GPU

### Install Miniconda in WSL2

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc
```

### Accept Conda Terms of Service (required once)

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### Clone our repo and create the environment

Clone **our fork** (not upstream HF — ours has custom fixes for the SO-101, relative actions, data loading, etc.):

```bash
git clone https://github.com/giohitt/lerobot-asu-capstone.git lerobot
cd lerobot
conda create -n lerobot python=3.10 -y
conda activate lerobot
pip install -e ".[act]"
```

### Install PyTorch with Blackwell (sm_120) support

The default pip torch does **not** include kernels for the RTX 5070 Ti (sm_120). Force install the cu128 build:

```bash
pip install "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --force-reinstall
```

> This matches the Jetson's torch 2.8.0 exactly. Ignore the lerobot `<2.8.0` constraint warning — it works fine.

### Verify everything

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Should print: `2.8.0+cu128  True  NVIDIA GeForce RTX 5070 Ti Laptop GPU` — no warnings.

---

## 2. Copy Datasets from Jetson to Laptop

Run this **in WSL2** (datasets are ~500MB total, fast on local network):

```bash
JETSON_IP=192.168.0.165
mkdir -p ~/.cache/huggingface/lerobot/local/

rsync -avz --progress \
  jetson23@$JETSON_IP:~/.cache/huggingface/lerobot/local/cylinder_sorting_green_v1/ \
  ~/.cache/huggingface/lerobot/local/cylinder_sorting_green_v1/

rsync -avz --progress \
  jetson23@$JETSON_IP:~/.cache/huggingface/lerobot/local/cylinder_sorting_mixed_v1/ \
  ~/.cache/huggingface/lerobot/local/cylinder_sorting_mixed_v1/
```

rsync is incremental — safe to re-run if you add more episodes.

---

## 3. Train on the Laptop

From the `~/lerobot` directory with `conda activate lerobot`:

### Green model (100 episodes → 100,000 steps)

```bash
DATASET="local/cylinder_sorting_green_v1"
MODEL="act_green_v1"
STEPS=100000
SAVE_FREQ=10000

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
  --policy.n_action_steps=15 \
  --dataset.video_backend=pyav
```

> `--dataset.video_backend=pyav` — matches the Jetson. torchcodec is not used on either machine.
> `batch_size=32` — 5070 Ti has enough VRAM to double the Jetson's batch size of 16.
> Checkpoints saved every 10,000 steps (10 total).
> First batch is slow (~1-3 min) while CUDA JIT-compiles kernels for sm_120. Normal — speeds up after.
> 100k steps ≈ 5 hrs on RTX 5070 Ti.

### Mixed model

Same command, change:
```bash
DATASET="local/cylinder_sorting_mixed_v1"
MODEL="act_mixed_v1"
STEPS=100000
```

### Watch loss during training

Open a second WSL2 terminal and run:
```bash
# Loss prints every 200 steps to stdout
# Target: loss below 0.05 for reliable grasping
# If still dropping at 100k, resume for 20k more
```

---

## 4. Copy Model Back to Jetson

Run this **in WSL2** when training is done:

```bash
JETSON_IP=192.168.0.165

rsync -avz --progress \
  ~/lerobot/outputs/train/act_green_v1/checkpoints/last/pretrained_model/ \
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
├── 010000/
├── 020000/
├── 030000/
├── ...
└── last/                        ← eval uses this
    └── pretrained_model/
        ├── config.json
        ├── model.safetensors    ← the actual weights (~200MB)
        ├── train_config.json
        └── policy_*.safetensors
```

Only the `last/pretrained_model/` folder needs to be copied back to the Jetson.

---

## Step Count Reference

| Episodes | Recommended steps | Laptop time (RTX 5070 Ti) | Jetson time |
|---|---|---|---|
| 20 | 8,000 | ~20 min | ~5 hrs |
| 50 | 50,000–100,000 | ~2.5–5 hrs | ~30–60 hrs |
| 100 | 100,000 | ~5 hrs | ~60 hrs |

---

## Common Issues

| Error | Fix |
|---|---|
| `CUDA not available` | Check `nvidia-smi` in WSL2; update Windows NVIDIA driver |
| `no kernel image available` | Wrong torch build — reinstall with cu128 (see step 1) |
| `undefined symbol: ncclCommWindowRegister` | NCCL mismatch — reinstall torch **without** `--no-deps` |
| `Unsupported video backend: av` | Use `--dataset.video_backend=pyav` (not `av`) |
| `FileExistsError` on output dir | Delete `outputs/train/act_green_v1/` and retry |
| First batch takes 1-3 min | Normal — CUDA JIT-compiling sm_120 kernels, only happens once |
| Loss stuck above 0.1 after 50K steps | Data too varied or too few episodes — add more data |
| Loss plateaus at 0.07–0.09 | Normal for mixed data; try finetuning 10K more steps |
