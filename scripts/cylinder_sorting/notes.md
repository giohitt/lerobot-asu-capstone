# NOTES:
```
0:56:44 [IDLE]       autonomous loop — Ctrl+C to stop
00:56:44 [IDLE]       ==================================================
00:56:44 [IDLE]       loaded home position from home_position.json
00:56:55 [DETECTING]  watching... (no cylinder detected)
00:57:05 [DETECTING]  watching... (no cylinder detected)
00:57:09 [DETECTING]  detected: green
00:57:09 [RUNNING]    cycle 1 — green — starting
────────────────────────────────────────────────────────────
00:57:39 [RUNNING]    episode done — 500 steps in 30.2s
────────────────────────────────────────────────────────────
00:57:39 [SETTLING]   returning arm to home position...
00:57:42 [SETTLING]   homing motion complete
00:57:42 [SETTLING]   verifying arm is at rest...
00:57:42 [SETTLING]   ✓ arm settled in 0.5s
00:57:43 [DETECTING]  cycle 1 complete — watching for next cylinder
00:57:43 [DETECTING]  detected: blue
00:57:43 [RUNNING]    cycle 2 — blue — starting
────────────────────────────────────────────────────────────
00:58:13 [RUNNING]    episode done — 469 steps in 30.0s













 front led:
  GREEN:  1590blob  BLUE:  1424blob  YELLOW:  1559blob  →  DETECTED: GREEN, BLUE, YELLOW
  GREEN:  1569blob  BLUE:  1421blob  YELLOW:  1566blob  →  DETECTED: GREEN, BLUE, YELLOW
  GREEN:  1590blob  BLUE:  1418blob  YELLOW:  1561blob  →  DETECTED: GREEN, BLUE, YELLOW
  GREEN:  1570blob  BLUE:  1429blob  YELLOW:  1562blob  →  DETECTED: GREEN, BLUE, YELLOW
  GREEN:  1580blob  BLUE:  1418blob  YELLOW:  1569blob  →  DETECTED: GREEN, BLUE, YELLOW
  GREEN:  1577blob  BLUE:  1418blob  YELLOW:  1569blob  →  DETECTED: GREEN, BLUE, YELLOW
  GREEN:  1572blob  BLUE:  1423blob  YELLOW:  1561blob  →  DETECTED: GREEN, BLUE, YELLOW
  GREEN:  1580blob  BLUE:  1422blob  YELLOW:  1574blob  →  DETECTED: GREEN, BLUE, YELLOW

  blob bright light behind no fornt led
  
  GREEN:  1516blob  BLUE:  1374blob  YELLOW:  1504blob  →  DETECTED: GREEN, BLUE, YELLOW
  GREEN:  1537blob  BLUE:  1369blob  YELLOW:  1505blob  →  DETECTED: GREEN, BLUE, YELLOW
  GREEN:  1515blob  BLUE:  1379blob  YELLOW:  1505blob  →  DETECTED: GREEN, BLUE, YELLOW
  GREEN:  1517blob  BLUE:  1372blob  YELLOW:  1500blob  →  DETECTED: GREEN, BLUE, YELLOW
  GREEN:  1517blob  BLUE:  1377blob  YELLOW:  1503blob  →  DETECTED: GREEN, BLUE, YELLOW
  GREEN:  1514blob  BLUE:  1377blob  YELLOW:  1500blob  →  DETECTED: GREEN, BLUE, YELLOW
  GREEN:  1573blob  BLUE:  1375blob  YELLOW:  1502blob  →  DETECTED: GREEN, BLUE, YELLOW
  GREEN:  1523blob  BLUE:  1377blob  YELLOW:  1503blob  →  DETECTED: GREEN, BLUE, YELLOW

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
