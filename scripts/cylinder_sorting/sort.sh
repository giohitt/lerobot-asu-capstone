#!/usr/bin/env bash
# ============================================================
# CYLINDER SORTING — Unified Pipeline
# ============================================================
# Usage:
#   bash sort.sh                      full interactive mode
#   bash sort.sh mixed                pick command interactively
#   bash sort.sh mixed record 20      direct — 20 episodes
#   bash sort.sh mixed train 5000     direct — 5k steps
#   bash sort.sh mixed clean eval     direct — delete eval only
# ============================================================

PYTHON=/home/jetson23/miniforge3/envs/lerobot/bin/python
LEROBOT_DIR=/home/jetson23/lerobot

# Suppress noisy Python/HuggingFace warnings — show INFO and above only
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export DIFFUSERS_VERBOSITY=error
export HF_HUB_VERBOSITY=warning
export TOKENIZERS_PARALLELISM=false

# ── UI helpers ─────────────────────────────────────────────
top()     { echo "╔══════════════════════════════════════════╗"; }
divider() { echo "╠══════════════════════════════════════════╣"; }
bot()     { echo "╚══════════════════════════════════════════╝"; }
row()     { printf "║  %-40s║\n" "$1"; }
ask()     { read -p "$1 [${2}]: " REPLY; REPLY="${REPLY:-$2}"; }

# ── State helpers ──────────────────────────────────────────
dataset_dir()  { echo "$HOME/.cache/huggingface/lerobot/local/$(echo "$DATASET" | sed 's|local/||')"; }
eval_dir()     { echo "$HOME/.cache/huggingface/lerobot/local/eval_$(echo "$DATASET" | sed 's|local/||')_run1"; }
dataset_exists() { [ -d "$(dataset_dir)" ]; }
model_exists()   { [ -d "$LEROBOT_DIR/outputs/train/$MODEL/checkpoints/last" ]; }
eval_exists()    { [ -d "$(eval_dir)" ]; }
episode_count()  {
  local info="$(dataset_dir)/meta/info.json"
  if [ -f "$info" ]; then
    python3 -c "import json; d=json.load(open('$info')); print(d.get('total_episodes', d.get('num_episodes', 0)))" 2>/dev/null || echo "0"
  else
    echo "0"
  fi
}

# ── Color config ───────────────────────────────────────────
set_color_config() {
  case "$1" in
    green)
      DATASET="local/cylinder_sorting_green_v1"
      MODEL="act_green_v1"
      TASK="Pick green cylinder and place in left bin"
      ;;
    blue)
      DATASET="local/cylinder_sorting_blue_v1"
      MODEL="act_blue_v1"
      TASK="Pick blue cylinder and place in right bin"
      ;;
    yellow)
      DATASET="local/cylinder_sorting_yellow_v1"
      MODEL="act_yellow_v1"
      TASK="Pick yellow cylinder and place in center bin"
      ;;
    mixed|green-blue)
      COLOR="mixed"
      DATASET="local/cylinder_sorting_mixed_v1"
      MODEL="act_mixed_v1"
      TASK="Pick cylinder and place in correct bin"
      ;;
    *) echo "Unknown color: $1. Valid: green, blue, yellow, mixed"; exit 1 ;;
  esac
}

# ── Camera detection ───────────────────────────────────────
detect_cameras() {
  echo "Detecting cameras..."
  INDICES=$($PYTHON -c "
import cv2
working = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        working.append(i)
        cap.release()
    if len(working) == 2:
        break
print('ERROR' if len(working) < 2 else f'{working[0]} {working[1]}', end='')
")
  [ "$INDICES" = "ERROR" ] && echo "ERROR: Could not find 2 cameras. Check USB." && exit 1
  CAM0=$(echo $INDICES | cut -d' ' -f1)
  CAM1=$(echo $INDICES | cut -d' ' -f2)
  echo "Detected cameras: handeye=$CAM0, front=$CAM1"
  CAMERAS="{\"handeye\":{\"type\":\"opencv\",\"index_or_path\":$CAM0,\"width\":640,\"height\":480,\"fps\":30},\"front\":{\"type\":\"opencv\",\"index_or_path\":$CAM1,\"width\":640,\"height\":480,\"fps\":30}}"
}

# ══════════════════════════════════════════════════════════
# EXECUTORS — called by both direct and interactive modes
# ══════════════════════════════════════════════════════════

_run_record() {
  # $1=episodes  $2=dataset(optional override)  $3=resume flag(optional)
  local eps="${1:-10}"
  local repo="${2:-$DATASET}"
  local resume="${3:-}"
  detect_cameras
  PYTHONUNBUFFERED=1 $PYTHON -u -m lerobot.scripts.lerobot_record \
    --robot.type=so101_follower --robot.port=/dev/ttyFOLLOWER \
    --robot.id=my_follower --robot.cameras="$CAMERAS" \
    --teleop.type=so101_leader --teleop.port=/dev/ttyLEADER \
    --teleop.id=my_leader --dataset.repo_id="$repo" \
    --dataset.single_task="$TASK" --dataset.num_episodes="$eps" \
    --dataset.push_to_hub=false $resume
}

_run_train() {
  # $1=steps  $2=extra flags (e.g. --resume=true)
  local steps="${1:-20000}"
  local extra="${2:-}"
  local save_freq=$(( steps / 5 ))
  local log="$LEROBOT_DIR/outputs/train/logs/$MODEL.log"
  local model_dir="$LEROBOT_DIR/outputs/train/$MODEL"
  mkdir -p "$LEROBOT_DIR/outputs/train/logs"
  # If training fresh (no --resume), delete existing model dir to avoid FileExistsError
  if [[ "$extra" != *"--resume=true"* ]] && [ -d "$model_dir" ]; then
    echo "Removing existing model directory for fresh train..."
    rm -rf "$model_dir"
  fi
  echo "  Steps: $steps  |  Checkpoints every: $save_freq  |  Log: $log"
  nohup $PYTHON -u -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$DATASET" --policy.type=act \
    --policy.device=cuda --policy.push_to_hub=false \
    --wandb.enable=false \
    --output_dir="$LEROBOT_DIR/outputs/train/$MODEL" \
    --job_name="$MODEL" --steps="$steps" --save_freq="$save_freq" \
    --batch_size=16 \
    --policy.optimizer_lr=1e-4 \
    --policy.optimizer_lr_backbone=1e-5 \
    --policy.use_amp=true \
    --policy.chunk_size=50 \
    --policy.n_action_steps=15 \
    $extra > "$log" 2>&1 &
  echo "Training PID $! — Ctrl+C detaches (training keeps running)"
  echo "Re-attach: tail -f $log  |  Stop: kill $!"
  tail -f "$log"
}

_run_eval() {
  # $1=eval_repo  $2=episodes  $3=checkpoint path (default: last)
  local repo="${1:-local/eval_$(echo "$DATASET" | sed 's|local/||')_run1}"
  local eps="${2:-5}"
  local checkpoint="${3:-$LEROBOT_DIR/outputs/train/$MODEL/checkpoints/last/pretrained_model}"
  local dir="$HOME/.cache/huggingface/lerobot/local/$(echo "$repo" | sed 's|local/||')"
  if [ -d "$dir" ]; then
    sudo rm -rf "$dir"
    [ -d "$dir" ] && echo "ERROR: Could not delete $dir — try: sudo rm -rf $dir" && exit 1
    echo "Deleted existing eval dataset."
  fi
  detect_cameras
  PYTHONUNBUFFERED=1 $PYTHON -u -m lerobot.scripts.lerobot_record \
    --robot.type=so101_follower --robot.port=/dev/ttyFOLLOWER \
    --robot.id=my_follower --robot.cameras="$CAMERAS" \
    --dataset.repo_id="$repo" --dataset.single_task="$TASK" \
    --dataset.num_episodes="$eps" --dataset.push_to_hub=false \
    --dataset.reset_time_s=0 \
    --policy.path="$checkpoint"
}

_run_clean() {
  # $1=target (data|model|eval|all)
  local target="${1:-all}"
  local base=$(echo "$DATASET" | sed 's|local/||')
  case "$target" in
    data)
      sudo rm -rf "$(dataset_dir)" && echo "Deleted dataset: $base" ;;
    model)
      rm -rf "$LEROBOT_DIR/outputs/train/$MODEL" && echo "Deleted model: $MODEL" ;;
    eval)
      sudo rm -rf "$(eval_dir)" && echo "Deleted eval: eval_${base}_run1" ;;
    all)
      sudo rm -rf "$(dataset_dir)" "$(eval_dir)"
      rm -rf "$LEROBOT_DIR/outputs/train/$MODEL"
      echo "Deleted all for $COLOR." ;;
    *) echo "Unknown target: $target. Valid: data, model, eval, all" ;;
  esac
}

# ══════════════════════════════════════════════════════════
# INTERACTIVE — asks questions then calls executors
# ══════════════════════════════════════════════════════════

interactive_record() {
  local default_name=$(echo "$DATASET" | sed 's|local/||')
  local repo="" resume=""

  top; row "RECORD — $COLOR"; divider
  row "Default dataset name: $default_name"; bot
  echo ""
  echo "Press Enter to use the default name, or type a custom name."
  echo "Example custom name: cylinder_sorting_green_demo"
  ask "Dataset name" "$default_name"
  repo="local/$REPLY"

  # Check if this dataset already exists
  local existing_dir="$HOME/.cache/huggingface/lerobot/local/$REPLY"
  if [ -d "$existing_dir" ]; then
    local n=$(ls "$existing_dir/episodes" 2>/dev/null | wc -l | tr -d ' ')
    echo ""
    echo "This dataset already exists with $n episodes."
    echo "  1) Add more episodes to it"
    echo "  2) Delete it and start over"
    echo ""
    ask "Choice" "1"
    if [[ "$REPLY" =~ ^[2] ]]; then
      sudo rm -rf "$existing_dir"; echo "Old data deleted."
    else
      resume="--resume=true"
    fi
  fi

  echo ""
  ask "How many episodes do you want to record this session?" "10"
  local eps="$REPLY"
  echo ""
  echo "  Dataset:  $repo"
  echo "  Episodes: $eps"
  echo "  Action:   $([ -n "$resume" ] && echo 'Adding to existing data' || echo 'Starting fresh')"
  ask "Ready to start recording? (y/n)" "y"
  [[ "$REPLY" =~ ^[Nn] ]] && echo "Cancelled." && exit 0
  _run_record "$eps" "$repo" "$resume"
}

interactive_train() {
  top; row "TRAIN — $COLOR"; divider

  # List available datasets (exclude eval datasets)
  local cache_dir="$HOME/.cache/huggingface/lerobot/local"
  local datasets=()
  if [ -d "$cache_dir" ]; then
    while IFS= read -r d; do
      [[ "$d" == eval_* ]] && continue
      datasets+=("$d")
    done < <(ls "$cache_dir" 2>/dev/null)
  fi

  local chosen_dataset="$DATASET"
  if [ ${#datasets[@]} -eq 0 ]; then
    row "No recorded datasets found."; bot
    echo "Please run 'record' first."
    exit 1
  elif [ ${#datasets[@]} -eq 1 ]; then
    chosen_dataset="local/${datasets[0]}"
    local n=$(python3 -c "import json; d=json.load(open('$cache_dir/${datasets[0]}/meta/info.json')); print(d.get('total_episodes', d.get('num_episodes', 0)))" 2>/dev/null || echo "0")
    row "Dataset: ${datasets[0]} ($n episodes)"; bot
  else
    row "Available datasets:"; bot; echo ""
    local i=1
    for d in "${datasets[@]}"; do
      local n=$(python3 -c "import json; d=json.load(open('$cache_dir/$d/meta/info.json')); print(d.get('total_episodes', d.get('num_episodes', 0)))" 2>/dev/null || echo "0")
      echo "  $i) $d  ($n episodes)"
      (( i++ ))
    done
    # Find default index matching current color's dataset
    local default_name=$(echo "$DATASET" | sed 's|local/||')
    local default_idx=1
    for j in "${!datasets[@]}"; do
      [[ "${datasets[$j]}" == "$default_name" ]] && default_idx=$(( j + 1 ))
    done
    echo ""
    ask "Which dataset do you want to train on?" "$default_idx"
    local idx=$(( REPLY - 1 ))
    chosen_dataset="local/${datasets[$idx]}"
    local n=$(python3 -c "import json; d=json.load(open('$cache_dir/${datasets[$idx]}/meta/info.json')); print(d.get('total_episodes', d.get('num_episodes', 0)))" 2>/dev/null || echo "0")
    echo "Selected: ${datasets[$idx]} ($n episodes)"
    DATASET="$chosen_dataset"
  fi

  local existing=$(python3 -c "import json; d=json.load(open('$cache_dir/$(echo $chosen_dataset | sed 's|local/||')/meta/info.json')); print(d.get('total_episodes', d.get('num_episodes', 0)))" 2>/dev/null || echo "0")
  local suggested=$(( existing * 400 )); [ "$suggested" -lt 3000 ] && suggested=3000
  local extra=""

  if model_exists; then
    echo ""
    echo "A trained model already exists for $COLOR."
    echo "  1) Keep training — add more learning to the existing model"
    echo "  2) Start over   — delete the old model and train fresh"
    echo ""
    ask "Choice" "1"
    if [[ "$REPLY" =~ ^[2] ]]; then
      echo "Will delete existing model and train from scratch."
    else
      echo "Will continue training from the last checkpoint."
      extra="--resume=true"
    fi
  fi

  echo ""
  echo "How many steps to train?"
  echo "  (Suggested based on $existing episodes: $suggested)"
  ask "Steps" "$suggested"
  local steps="$REPLY"

  echo ""
  echo "  Dataset: $chosen_dataset"
  echo "  Model:   $MODEL"
  echo "  Steps:   $steps"
  echo "  Mode:    $([ -n "$extra" ] && echo 'Continue from checkpoint' || echo 'Train from scratch')"
  ask "Start training? (y/n)" "y"
  [[ "$REPLY" =~ ^[Nn] ]] && echo "Cancelled." && exit 0
  _run_train "$steps" "$extra"
}

interactive_eval() {
  top; row "TEST MODEL — $COLOR"; divider
  if ! model_exists; then
    row "No trained model found for $COLOR."
    row "Please run 'train' first, then come back."
    bot; exit 1
  fi

  # List available checkpoints
  local ckpt_dir="$LEROBOT_DIR/outputs/train/$MODEL/checkpoints"
  local checkpoints=()
  while IFS= read -r d; do
    checkpoints+=("$d")
  done < <(ls "$ckpt_dir" 2>/dev/null | sort)

  row "Available checkpoints:"; bot; echo ""
  local i=1
  for c in "${checkpoints[@]}"; do
    if [ "$c" = "last" ]; then
      echo "  $i) last  ← most recent (recommended)"
    else
      echo "  $i) step $c"
    fi
    (( i++ ))
  done
  echo ""
  ask "Which checkpoint to use?" "1"
  local idx=$(( REPLY - 1 ))
  local chosen_ckpt="${checkpoints[$idx]}"
  local checkpoint="$ckpt_dir/$chosen_ckpt/pretrained_model"
  echo "Using checkpoint: $chosen_ckpt"

  local base=$(echo "$DATASET" | sed 's|local/||')
  local repo="local/eval_${base}_run1"

  if eval_exists; then
    echo ""
    echo "A previous test run already exists."
    echo "  1) Replace it with a new test run"
    echo "  2) Save this run under a different name"
    echo ""
    ask "Choice" "1"
    if [[ "$REPLY" =~ ^[2] ]]; then
      ask "Enter a name for this run (example: run2)" "run2"
      repo="local/eval_${base}_${REPLY}"
    fi
  fi

  echo ""
  ask "How many episodes do you want to test?" "5"
  local eps="$REPLY"

  echo ""
  echo "  Color:      $COLOR"
  echo "  Checkpoint: $chosen_ckpt"
  echo "  Episodes:   $eps"
  echo ""
  echo "Before starting:"
  echo "  - Place a cylinder in the correct zone"
  echo "  - Make sure the arm is in its home position"
  echo "  - The arm will move on its own — stand clear"
  ask "Ready to start the test? (y/n)" "y"
  [[ "$REPLY" =~ ^[Nn] ]] && echo "Cancelled." && exit 0
  _run_eval "$repo" "$eps" "$checkpoint"
}

interactive_clean() {
  local base=$(echo "$DATASET" | sed 's|local/||')
  top; row "DELETE DATA — $COLOR"; divider
  row "Recording data : $(dataset_exists && echo 'exists' || echo 'nothing saved')"
  row "Trained model  : $(model_exists   && echo 'exists' || echo 'nothing saved')"
  row "Test run data  : $(eval_exists    && echo 'exists' || echo 'nothing saved')"
  bot; echo ""
  echo "What do you want to delete?"
  echo "  1) Recording data only   (re-record from scratch)"
  echo "  2) Trained model only    (retrain without re-recording)"
  echo "  3) Test run data only    (required before running a new test)"
  echo "  4) Everything            (full reset)"
  echo ""
  ask "Choice" "3"
  local t="$REPLY"
  case "$t" in 1) t=data;; 2) t=model;; 3) t=eval;; 4) t=all;; esac
  echo ""
  ask "Are you sure you want to delete '$t' for $COLOR? This cannot be undone. (y/n)" "n"
  [[ ! "$REPLY" =~ ^[Yy] ]] && echo "Cancelled. Nothing was deleted." && exit 0
  _run_clean "$t"
}

# ══════════════════════════════════════════════════════════
# MENUS
# ══════════════════════════════════════════════════════════

pick_color() {
  top; row "CYLINDER SORTING — Select Color"; divider
  row "1) green   left zone  → left bin"
  row "2) blue    right zone → right bin"
  row "3) yellow  center zone (distractor)"
  row "4) mixed   green + blue (two bins)"
  bot
  ask "Color (name or number)" ""
  case "$REPLY" in
    1|green) COLOR="green" ;; 2|blue) COLOR="blue" ;;
    3|yellow) COLOR="yellow" ;; 4|mixed|green-blue) COLOR="mixed" ;;
    *) echo "Invalid: $REPLY"; exit 1 ;;
  esac
}

pick_command() {
  top; printf "║  %-40s║\n" "Color: $COLOR"; divider
  row "1) record    collect teleoperation episodes"
  row "2) train     train ACT policy"
  row "3) eval      run trained policy on robot"
  row "4) clean     delete data / model"
  bot
  ask "Command (name or number)" ""
  case "$REPLY" in
    1|record) CMD="record" ;; 2|train) CMD="train" ;;
    3|eval)   CMD="eval"   ;; 4|clean) CMD="clean" ;;
    *) echo "Invalid: $REPLY"; exit 1 ;;
  esac
}

# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════

COLOR="${1:-}"; CMD="${2:-}"; ARG3="${3:-}"
[ -z "$COLOR" ] && pick_color
[ -z "$CMD" ]   && pick_command
set_color_config "$COLOR"

case "$CMD" in
  record)
    if [[ "$ARG3" =~ ^[0-9]+$ ]]; then
      # Direct: auto-detect append vs fresh
      RESUME=""; dataset_exists && RESUME="--resume=true"
      _run_record "$ARG3" "$DATASET" "$RESUME"
    else
      interactive_record
    fi ;;
  train)
    [[ "$ARG3" =~ ^[0-9]+$ ]] && _run_train "$ARG3" || interactive_train ;;
  finetune)
    ! model_exists && echo "ERROR: No checkpoint found. Run 'train' first." && exit 1
    [[ "$ARG3" =~ ^[0-9]+$ ]] && _run_train "$ARG3" "--resume=true" || interactive_train ;;
  eval)
    [[ "$ARG3" =~ ^[0-9]+$ ]] && _run_eval "" "$ARG3" || interactive_eval ;;
  clean)
    [ -n "$ARG3" ] && _run_clean "$ARG3" || interactive_clean ;;
  *)
    echo "Unknown command: $CMD. Valid: record, train, finetune, eval, clean"; exit 1 ;;
esac
