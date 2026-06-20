# K1 Locomotion 既存Checkpoint評価

## 目的

既存の K1 locomotion checkpoint が、どの iteration 付近から Isaac Lab 側で安定して歩行候補になっていたかを確認する。

今回の評価では、新しい固定 command や `IdealFlatForward` 系 task は使わない。学習時の本命 task に対応する Play task をそのまま使う。

## 対象

対象 run:

```text
/mnt/ssd2/booster/booster_train/logs/rsl_rl/k1_locomotion/2026-05-21_02-09-01
```

この run は、腕を横に伸ばして上下に振る挙動を抑えるために `arm_joint_deviation_l1` / `arm_action_l2` / `arm_action_rate_l2` を追加した後の主参照 run。

学習 task:

```text
Booster-K1-Locomotion-v0
```

評価 task:

```text
Booster-K1-Locomotion-v0-Play
```

使わない task:

```text
Booster-K1-Locomotion-IdealFlatForward-v0-Play
Booster-K1-Locomotion-IdealFlatCommandRandom-v0-Play
```

これらは後から作った別 experiment 用の確認 variant であり、今回の `k1_locomotion` checkpoint 群の評価 task ではない。

## 根拠確認

task 登録確認:

```bash
cd /mnt/ssd2/booster/booster_train
rg -n 'Booster-K1-Locomotion-v0|Booster-K1-Locomotion-v0-Play|IdealFlat' \
  source/booster_train/booster_train/tasks/manager_based/locomotion/robots/k1/walk/__init__.py
```

確認結果:

```text
5:    id="Booster-K1-Locomotion-v0",
15:    id="Booster-K1-Locomotion-v0-Play",
25:    id="Booster-K1-Locomotion-IdealFlatForward-v0",
35:    id="Booster-K1-Locomotion-IdealFlatForward-v0-Play",
45:    id="Booster-K1-Locomotion-IdealFlatCommandRandom-v0",
55:    id="Booster-K1-Locomotion-IdealFlatCommandRandom-v0-Play",
```

既存手順書の確認コマンド:

```bash
rg -n '2026-05-21_02-09-01|model_16900|Booster-K1-Locomotion-v0-Play|GUI確認コマンド' \
  /mnt/ssd2/booster/booster-k1-training-performance-notes.md
```

確認内容:

```text
2026-05-21_02-09-01/model_16900.pt は
python scripts/rsl_rl/play.py --task=Booster-K1-Locomotion-v0-Play
で GUI 確認する手順になっている。
```

保存間隔:

```bash
rg -n 'save_interval|experiment_name' \
  logs/rsl_rl/k1_locomotion/2026-05-21_02-09-01/params/agent.yaml
```

確認結果:

```text
save_interval: 100
experiment_name: k1_locomotion
```

`model_10.pt` は存在しないため、今回の対象外。

## 評価条件

評価条件:

| 項目 | 値 |
| --- | --- |
| task | `Booster-K1-Locomotion-v0-Play` |
| run | `2026-05-21_02-09-01` |
| steps | `1000` |
| num_envs | Play cfg default |
| device | `cuda:0` |
| command override | なし |
| command range | Play cfg / base cfg のまま `lin_vel_x/y/ang_vel_z=(-1.0, 1.0)` |
| standing envs | Play cfg / base cfg のまま `rel_standing_envs=0.2` |
| 外乱 | Play cfg 通り無効 |
| reset randomization | Play cfg 通り無効 |

評価対象 checkpoint:

```text
model_16900.pt
model_16000.pt
model_13000.pt
model_10000.pt
```

歩行開始点を探す目的なので、評価は後ろのcheckpointから順に実行する。`13000` で未安定、`16000` で安定候補が見えたため、`10000` 未満は今回の判定には使わない。

判定は以下の目安で行う。

| 判定 | 条件 |
| --- | --- |
| `未安定` | non-timeout done が発生 |
| `安定候補` | non-timeout done がなく、episode length が timeout 近辺 |

## 実行コマンド

評価 script は repo には追加せず、実行時に `/tmp/k1_locomotion_eval_play_task.py` として作成する。
この script は checkpoint ごとに CSV / JSON を逐次保存するため、途中で止めても完了済みcheckpointの結果は残る。

実行前確認:

```bash
cd /mnt/ssd2/booster/booster_train
source /mnt/ssd2/booster/env_isaaclab/bin/activate
unset PYTHONPATH

find logs/rsl_rl/k1_locomotion/2026-05-21_02-09-01 -maxdepth 1 -type f \
  \( -name 'model_10000.pt' -o -name 'model_13000.pt' \
  -o -name 'model_16000.pt' -o -name 'model_16900.pt' \
  -o -name 'model_10.pt' \) -printf '%f\n' | sort -V
```

一時 script 確認:

```bash
python -m py_compile /tmp/k1_locomotion_eval_play_task.py
sha256sum /tmp/k1_locomotion_eval_play_task.py
```

確認結果:

```text
5d03287c5a409ba9f0fab53f3391832c7d7beaccf544216de1f9eeddeab3e339  /tmp/k1_locomotion_eval_play_task.py
```

評価実行:

```bash
cd /mnt/ssd2/booster/booster_train
source /mnt/ssd2/booster/env_isaaclab/bin/activate
unset PYTHONPATH

python /tmp/k1_locomotion_eval_play_task.py \
  --task Booster-K1-Locomotion-v0-Play \
  --headless \
  --device cuda:0 \
  --steps 1000 \
  --seed 42 \
  --csv /tmp/k1_locomotion_checkpoint_eval.csv \
  --json /tmp/k1_locomotion_checkpoint_eval.json \
  --checkpoints \
    /mnt/ssd2/booster/booster_train/logs/rsl_rl/k1_locomotion/2026-05-21_02-09-01/model_16900.pt \
    /mnt/ssd2/booster/booster_train/logs/rsl_rl/k1_locomotion/2026-05-21_02-09-01/model_16000.pt \
    /mnt/ssd2/booster/booster_train/logs/rsl_rl/k1_locomotion/2026-05-21_02-09-01/model_13000.pt \
    /mnt/ssd2/booster/booster_train/logs/rsl_rl/k1_locomotion/2026-05-21_02-09-01/model_10000.pt
```

出力:

```text
/tmp/k1_locomotion_checkpoint_eval.csv
/tmp/k1_locomotion_checkpoint_eval.json
```

出力ファイル hash:

```text
d20823a5b255b4938c763de71a51b1b0848008ee3d0966a83429964f43fccbda  /tmp/k1_locomotion_checkpoint_eval.csv
18d3b2ac7ba119079c0ab8638b2cf922dbfa84efbf51242106f4db6ff86dc0c0  /tmp/k1_locomotion_checkpoint_eval.json
```

## 結果

実行日時:

```text
2026-06-08 02:15 JST
```

実行環境:

```text
GPU: NVIDIA GeForce RTX 4070 Ti
device: cuda:0
num_envs: 50
steps: 1000
seed: 42
```

実行時に `Warp CUDA error: Failed to get driver entry point 'cuDeviceGetUuid'` と headless の GLFW / display warning が出たが、Isaac Sim は `cuda:0` で起動し、評価は完了した。

| iter | 判定 | mean episode length | non-timeout done | timeout | mean reward/step | track lin | track yaw | arm dev | arm action | arm rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16900 | 安定候補 | 1000 | 0 | 50 | 0.0369224 | 0.000947786 | 0.000965615 | -2.79675e-05 | -1.03996e-05 | -2.97388e-07 |
| 16000 | 安定候補 | 1000 | 0 | 50 | 0.0368433 | 0.000941317 | 0.00096261 | -2.3429e-05 | -1.04321e-05 | -3.46459e-07 |
| 13000 | 未安定 | 49.1159 | 969 | 49 | -0.0552405 | 0.00127716 | 0.00148627 | -0.000687745 | -0.00233624 | -0.00154835 |
| 10000 | 未安定 | 47.6644 | 1000 | 49 | -0.0516533 | 0.00129802 | 0.00150498 | -0.000400644 | -0.00125142 | -0.000827867 |

補足:

- `track lin` / `track yaw` などの `Episode_Reward/*` は Isaac Lab の episode log 由来で、`dt` 等が反映された小さい値として出る。このため、絶対値 `0.5` のような閾値では判定しない。
- 主判定は `non-timeout done` と `mean episode length` を見る。
- `10000` 未満は、`13000` で未安定、`16000` で安定候補が確認できたため今回の判定対象から外した。

## 結論

この評価条件では、主要checkpointの範囲で最初の安定候補は `model_16000.pt`。

`model_13000.pt` は non-timeout done が 969 回あり、まだ未安定。したがって、歩行が安定候補になる境界は `13000` と `16000` の間にある。

より細かく歩行開始時点を知るなら、次は `14000`, `15000`, `15500`, `16000` のように、この範囲だけを追加評価する。
