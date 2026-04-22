# NOTES:
```
19:54:14 [DETECTING]  ATOMS_REST http=200 rows=1
19:54:14 [DETECTING]  ATOMS_REST body=[{"id":"REQ-052","title":"IF-008 \u2014 sort_config.json to Color Enable State","type":"requirement","data":{"id":"REQ-052","body":"sort_config.json \u2192 Color Enable State: enabled_colors array: subset of [\"green\", \"blue\"]. Sort controller SHALL activate only the policies listed; colors absent from the array SHALL be ignored at detection time.","tags":{"domains":["interfaces"]},"type":"requirement","links":[],"title":"IF-008 \u2014 sort_config.json to Color Enable State","metadata":{"created_at":"2026-04-22T01:09:27.548Z","created_by":"7c28db98-d494-4c55-a0cf-a79e3ea5df9c","updated_at":"2026-04-22T01:09:27.548Z","updated_by":"7c28db98-d494-4c55-a0cf-a79e3ea5df9c"},"ownership":{"primary":null,"additional":[]},"relationships":{"parents":["REQ-005"],"related":[],"children":[],"verified_by":["TC-025","TC-005"]}}}]
19:54:14 [DETECTING]  BRIDGE skip: no ATOMS_BRIDGE_MODEL_* for ['green', 'blue']; sort_config.json untouched
19:54:14 [DETECTING]  vision_color=none
19:54:18 [DETECTING]  watching... (no cylinder detected)
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
