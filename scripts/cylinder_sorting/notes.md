# NOTES:
```
lerobot) jetson23@ubuntu:~/lerobot$ bash /home/jetson23/lerobot/scripts/cylinder_sorting/sort.sh green eval
Dataset/model version (e.g. v1, v2, v3) [v1]: v2
╔══════════════════════════════════════════╗
║  TEST MODEL — green                    ║
╠══════════════════════════════════════════╣
║  Available models:                       ║
╚══════════════════════════════════════════╝

  1) act_green_v1_60k  (last checkpoint: unknown)
  2) act_green_v1_laptop_100k  (last checkpoint: unknown)
  3) act_green_v2  (last checkpoint: 060000)
```

  ## Commands for testing
  
  ```bash
  python -m lerobot.scripts.lerobot_edit_dataset \
  --repo_id local/cylinder_sorting_green_v1 \
  --operation.type delete_episodes \
  --operation.episode_indices "[100]"
  ```

  ## Test Blue Model Only
  ```bash
  python scripts/cylinder_sorting/sort_controller.py \
    --model_blue outputs/train/act_blue_v1_laptop_100k/checkpoints/last/pretrained_model \
    --color blue \
    --episode_time 30
  ```

  ## Test Green Model Only
  ```bash
  python scripts/cylinder_sorting/sort_controller.py \
    --model_green outputs/train/act_green_v1_laptop_100k/checkpoints/last/pretrained_model \
    --color green \
    --episode_time 30
  ```

  ## Start Controller With Both Models Loaded
  ```bash
  python scripts/cylinder_sorting/sort_controller.py \
    --model_green outputs/train/act_green_v1_laptop_100k/checkpoints/last/pretrained_model \
    --model_blue outputs/train/act_blue_v1_laptop_100k/checkpoints/last/pretrained_model \
    --episode_time 30
  ```
