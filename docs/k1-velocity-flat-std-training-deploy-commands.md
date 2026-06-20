# K1 Velocity Flat Std Training And Deploy Commands

このメモは、`Booster-Velocity-Flat-K1-Std-v0` の学習と、MuJoCo sim2sim deploy に使った最終コマンドをまとめたものです。

## 対象

| 項目 | 値 |
| --- | --- |
| train worktree | `/mnt/ssd2/booster/booster_train_k1_velocity_flat_standard` |
| deploy repo | `/mnt/ssd2/booster/booster_deploy` |
| 学習 task | `Booster-Velocity-Flat-K1-Std-v0` |
| play/export task | `Booster-Velocity-Flat-K1-Std-Play-v0` |
| deploy task | `k1_velocity_flat_std` |
| run dir | `logs/rsl_rl/k1_flat_std/2026-06-08_04-53-01_std_g1like_no_norm` |
| 確認 checkpoint | `model_8200.pt` |
| exported policy | `exported/k1_flat_std_2026-06-08_04-53-01_std_g1like_no_norm.pt` |

## 学習

実際に使った学習コマンドです。`max_iterations` は task 側の default のままです。

```bash
cd /mnt/ssd2/booster/booster_train_k1_velocity_flat_standard
source /mnt/ssd2/booster/env_isaaclab/bin/activate
unset PYTHONPATH
export PYTHONPATH=/mnt/ssd2/booster/booster_train_k1_velocity_flat_standard/source/booster_train

python -u scripts/rsl_rl/train.py \
  --task=Booster-Velocity-Flat-K1-Std-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 10240 \
  --run_name std_g1like_no_norm | tee /tmp/k1_std_g1like_no_norm.log
```

`model_8200.pt` 時点の学習ログ抜粋:

```text
Learning iteration 8274/50000
Mean reward: 32.92
Mean episode length: 1000.00
Episode_Reward/track_lin_vel_xy_exp: 0.9563
Episode_Reward/track_ang_vel_z_exp: 0.8311
Metrics/base_velocity/error_vel_xy: 0.1756
Metrics/base_velocity/error_vel_yaw: 0.3792
Episode_Termination/base_contact: 0.0000
Episode_Termination/bad_orientation: 0.0000
```

## Play And Export

`scripts/rsl_rl/play.py` は checkpoint を load したあと、JIT / ONNX policy を `run_dir/exported/` に出力します。GUI 確認も兼ねて、以下で `model_8200.pt` を確認しました。

```bash
cd /mnt/ssd2/booster/booster_train_k1_velocity_flat_standard
source /mnt/ssd2/booster/env_isaaclab/bin/activate
unset PYTHONPATH
export PYTHONPATH=/mnt/ssd2/booster/booster_train_k1_velocity_flat_standard/source/booster_train
export DISPLAY=:1

python -u scripts/rsl_rl/play.py \
  --task=Booster-Velocity-Flat-K1-Std-Play-v0 \
  --device cuda:0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/k1_flat_std/2026-06-08_04-53-01_std_g1like_no_norm/model_8200.pt \
  --real-time
```

出力された TorchScript policy:

```text
/mnt/ssd2/booster/booster_train_k1_velocity_flat_standard/logs/rsl_rl/k1_flat_std/2026-06-08_04-53-01_std_g1like_no_norm/exported/k1_flat_std_2026-06-08_04-53-01_std_g1like_no_norm.pt
```

## Deploy Model Placement

最終的には、既存 `k1_walk` へ checkpoint を差し替えるのではなく、deploy 側に専用 task `k1_velocity_flat_std` を追加して使います。

deploy task のデフォルト checkpoint として、exported policy を `tasks/locomotion/models/k1_velocity_flat_std.pt` に置きました。このディレクトリは `.gitignore` 対象です。

```bash
cp -p \
  /mnt/ssd2/booster/booster_train_k1_velocity_flat_standard/logs/rsl_rl/k1_flat_std/2026-06-08_04-53-01_std_g1like_no_norm/exported/k1_flat_std_2026-06-08_04-53-01_std_g1like_no_norm.pt \
  /mnt/ssd2/booster/booster_deploy/tasks/locomotion/models/k1_velocity_flat_std.pt
```

## MuJoCo Sim2Sim Deploy

最終的に使う deploy コマンドです。

```bash
cd /mnt/ssd2/booster/booster_deploy
source /mnt/ssd2/booster/env_isaaclab/bin/activate
export DISPLAY=:1

python -u scripts/deploy.py \
  --task k1_velocity_flat_std \
  --mujoco \
  --device cpu
```

tmux で起動する場合:

```bash
tmux new-session -d -s k1_velocity_flat_std_sim2sim \
  'cd /mnt/ssd2/booster/booster_deploy && \
   source /mnt/ssd2/booster/env_isaaclab/bin/activate && \
   export DISPLAY=:1 && \
   python -u scripts/deploy.py --task k1_velocity_flat_std --mujoco --device cpu 2>&1 | tee /tmp/k1_velocity_flat_std_sim2sim.log'
```

前進 command は、MuJoCo deploy の標準入力に以下を送ります。

```bash
tmux send-keys -t k1_velocity_flat_std_sim2sim '0.5 0 0' Enter
```

停止:

```bash
tmux kill-session -t k1_velocity_flat_std_sim2sim
```

## 確認結果

`k1_velocity_flat_std` のデフォルト checkpoint で、viewer なしの 10 秒前進確認を行いました。

```text
command: 0.5 0 0
success: True
duration_s: 10.0
xy_displacement_m: [3.2565, 0.022]
distance_m: 3.2566
min_root_z_m: 0.5228
max_abs_roll_pitch_deg: 3.38
```

task 登録と action scale の確認:

```bash
cd /mnt/ssd2/booster/booster_deploy
source /mnt/ssd2/booster/env_isaaclab/bin/activate
python scripts/deploy.py --list
```

確認した状態:

```text
k1_walk action_scale: float 0.25
k1_velocity_flat_std action_scale: list 20
k1_beyondmimic_joystick action_scale: float 0.25
```

