# K1 Flat Velocity と Isaac Lab G1/H1 Velocity タスクの設計差分

このメモは、K1 の flat velocity 学習タスクを Isaac Lab 標準の manager-based velocity task 形式へ移植するときに、どこを標準へ合わせ、どこを K1 用の設計差分として残すかを確認するためのものです。

結論として、K1 flat velocity は Isaac Lab の G1/H1 velocity task と同じ task family / manager 構造 / reward family を使っています。ただし、単に G1/H1 標準 task の robot asset を K1 に差し替えただけではありません。特に actor observation、action 順序、critic privileged observation、domain randomization、reward の追加項目、PPO 設定は K1 deploy と既存 K1 walk の学習結果を維持するために意図的に違います。なお Isaac Lab の `base_lin_vel` helper は内部的に `asset.data.root_lin_vel_b` を返すため、このメモでは標準 API 名として `base_lin_vel`、実体を示す必要がある場合だけ `root_lin_vel_b` と書きます。

## 比較対象

主比較対象は Isaac Lab の G1/H1 manager-based velocity task です。

| 種別 | 根拠 |
| --- | --- |
| Isaac Lab 共通 velocity cfg | `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` |
| Isaac Lab G1 cfg | `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/{rough_env_cfg.py,flat_env_cfg.py,agents/rsl_rl_ppo_cfg.py,__init__.py}` |
| Isaac Lab H1 cfg | `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/{rough_env_cfg.py,flat_env_cfg.py,agents/rsl_rl_ppo_cfg.py,__init__.py}` |
| K1 標準移植先 | `source/booster_train/booster_train/tasks/manager_based/locomotion/velocity/config/k1/{flat_env_cfg.py,agents/rsl_rl_ppo_cfg.py,__init__.py}` |
| K1 MDP helper | `source/booster_train/booster_train/tasks/manager_based/locomotion/velocity/mdp/{observations.py,rewards.py}` |
| K1 移植元 | `source/booster_train/booster_train/tasks/manager_based/locomotion/robots/k1/walk/{env_cfg.py,ppo_cfg.py}` |

Digit / Cassie / direct humanoid / humanoid_amp / classic humanoid は今回の主比較対象ではありません。二足歩行 task ではありますが、今回の K1 移植方針は Isaac Lab の `manager_based/locomotion/velocity` にある G1/H1 の構成へ寄せるものです。

## Isaac Lab 標準と同じ部分

| 領域 | Isaac Lab G1/H1 | K1 flat velocity | 判断 | 根拠 |
| --- | --- | --- | --- | --- |
| task family | `manager_based/locomotion/velocity` の velocity tracking task | 同じ velocity tracking task として `locomotion/velocity/config/k1` に配置 | 同じ | G1/H1 `config/{g1,h1}`、K1 `config/k1` |
| env base | `ManagerBasedRLEnvCfg` + `Scene/Commands/Actions/Observations/Event/Rewards/Terminations/Curriculum` | 同じ manager-based cfg 群を定義 | 同じ | `velocity_env_cfg.py`, `flat_env_cfg.py` |
| Gym ID 型 | `Isaac-Velocity-Flat-G1-v0`, `Isaac-Velocity-Flat-H1-v0` | `Booster-Velocity-Flat-K1-v0` | 同じ命名型 | `config/{g1,h1}/__init__.py`, `config/k1/__init__.py` |
| entry point | `isaaclab.envs:ManagerBasedRLEnv` | 同じ | 同じ | 各 `__init__.py` |
| flat terrain | G1/H1 flat は plane、height scanner 無効 | K1 は flat plane のみ、height scanner なし | 最終挙動は同じ | G1/H1 `flat_env_cfg.py`, K1 `K1FlatSceneCfg` |
| 主要 tracking reward | `track_lin_vel_xy_yaw_frame_exp`, `track_ang_vel_z_world_exp` | 同じ helper を使用 | 同じ family | G1/H1 `G1Rewards` / `H1Rewards`, K1 `RewardsCfg` |
| 二足歩行 reward | `feet_air_time_positive_biped`, `feet_slide`, `termination_penalty` | 同じ helper family を使用 | 同じ family | G1/H1 reward cfg, K1 reward cfg |
| PPO algorithm skeleton | RSL-RL PPO、`num_steps_per_env=24`, adaptive schedule, `lr=1e-3`, `gamma=0.99`, `lam=0.95` | 同じ基本 skeleton | 同じ | 各 `agents/rsl_rl_ppo_cfg.py` |

## K1 で意図的に違う部分

| 領域 | Isaac Lab G1/H1 velocity | K1 flat velocity | 判断 | 根拠 |
| --- | --- | --- | --- | --- |
| cfg 構造 | 共通 rough cfg を持ち、G1/H1 rough が継承し、flat は rough から override | K1 は flat-only を直接定義 | 残す | 今回は K1 flat だけが対象で、rough / height scanner を移植しないため |
| Gym 登録数 | G1/H1 は rough/flat の train/play を登録 | K1 は flat train/play のみ登録 | 残す | `Booster-Velocity-Flat-K1-v0`, `Booster-Velocity-Flat-K1-Play-v0` のみに絞るため |
| skrl 登録 | G1/H1 は `skrl_cfg_entry_point` も登録 | K1 は RSL-RL cfg のみ | 残す | 既存 K1 学習は RSL-RL 前提で、skrl config は未用意 |
| actor observation | term 分割: `base_lin_vel`, `base_ang_vel`, `projected_gravity`, command, joint pos/vel, last action、rough では height scan | `k1_deploy_locomotion_observation` 1本にまとめる | 残す | export/deploy 入力契約を固定するため |
| actor の base linear velocity | policy observation に `base_lin_vel` を入れる | actor には `base_lin_vel` / `root_lin_vel_b` を入れない | 残す | 実機で simulation truth 相当の base linear velocity を直接安定取得しない前提 |
| actor 履歴 | 標準 flat cfg では flatten 履歴を使わない | 1フレーム 69 次元、10フレーム flatten 後 690 次元 | 残す | base linear velocity なしで運動状態を推定しやすくするため |
| critic observation | G1/H1 の該当 cfg には separate critic group がない | `critic` group を追加し、actor obs + `base_lin_vel` + 足接触を使う | 残す | asymmetric actor-critic。actor は deploy 可能なまま、critic だけ学習中の privileged 情報を使う |
| observation corruption | G1/H1 train は `enable_corruption=True`、play で無効化 | K1 は policy/critic とも corruption 無効 | 残す | deploy 互換観測を決定的に保つ。ノイズ注入を戻す場合は挙動変更として扱う |
| action 関節指定 | `joint_names=[".*"]`, scale `0.5`。G1/H1 asset では脚・胴・腕・手指などの actuated joints が対象 | deploy 順序の 20 関節を明示、`preserve_order=True`, scale は `K1_ACTION_SCALE` 由来の関節別値 | 残す | action 順序と scale を実機 controller / export policy と一致させるため |
| 頭部 action | G1/H1 velocity asset cfg には K1 の `.*Head.*` 相当の head actuator がないため、`joint_names=[".*"]` でも頭は action 対象にならない | K1 asset には `AAHead_yaw` / `Head_pitch` 用の `.*Head.*` actuator があるが、velocity policy では deploy 互換 20DoF に絞って頭を除外 | 残す | K1 で `joint_names=[".*"]` にすると頭2DoFまで action に入るため、G1/H1 と同じ感覚で `.*` は使えない |
| 腕 action | G1/H1 も腕 action を含み得る。`joint_deviation_arms` で default 姿勢からのずれを抑える | K1 も腕を action に含め、腕8DoFを index 指定して `arm_joint_deviation_l1`, `arm_action_l2`, `arm_action_rate_l2` で抑える | 残す | K1 は腕を姿勢安定用の可動質量として過剰に使うことがあるため、deploy action 順序のまま腕の大振りを抑える |
| default joint pose | G1/H1 は `G1_MINIMAL_CFG` / `H1_MINIMAL_CFG` を task 側で `replace` するだけで、初期姿勢は asset cfg 側の `init_state.joint_pos` に置かれている | K1 は task cfg 側で `K1_DEFAULT_JOINT_POS_BY_NAME` により全22関節の初期姿勢を上書き | 残す、非reward差分として明記 | K1 locomotion/deploy 用の立ち姿勢を task 側で固定しているため。RewardA 実験でも reward 以外の前提差分として残る |
| command heading | 共通 cfg は heading command をサポートし、G1/H1 で range を調整 | `heading_command=False`, `rel_heading_envs=0.0` | 残す | K1 deploy policy は direct velocity command を受ける設計 |
| command range | G1 flat は x `(0,1)`, y `(-0.5,0.5)`, yaw `(-1,1)`。H1 は y `0` | K1 は x/y/yaw すべて `(-1,1)`, `rel_standing_envs=0.2` | 残すが要実験レビュー | 旧 K1 walk の command randomization を維持。後退・横移動まで含むため G1/H1 flat より広い |
| contact body | G1/H1 は torso / ankle link regex | K1 は base `Trunk`, 足 `left_foot_link` / `right_foot_link` | 残す | K1 URDF body 名に合わせるため |
| termination | G1/H1 は timeout + torso contact | K1 は timeout + `Trunk` contact + `bad_orientation(limit_angle=0.8)` | 残す | 旧 K1 walk の転倒判定を維持 |
| play randomization | G1/H1 flat play は corruption と一部 event を無効化 | K1 play は physics material / mass / reset / push など randomization event を無効化 | 残す | play/export で挙動確認しやすくするため |

## 要レビューな差分

以下は今回のリファクタでは既存 K1 walk から維持します。ただし、Isaac Lab 標準に完全に寄せるか、学習安定性を優先して K1 独自値を残すかは、別途実験でレビューする余地があります。

| 領域 | Isaac Lab G1/H1 velocity | K1 flat velocity | 今回の扱い | 根拠 |
| --- | --- | --- | --- | --- |
| event randomization | 共通 cfg には friction, base mass, base COM, external force, reset, push がある。G1/H1 では push/add mass/base COM を無効化し、reset velocity を 0 にする | K1 は friction range を広げ、`Trunk` mass randomization、reset velocity randomization、push `(-1,1)` を有効。base COM / external force は持たない | 維持、要レビュー | 旧 K1 walk の domain randomization を維持しているが、G1/H1 より強い |
| torque helper | 標準は generic `joint_torques_l2` | K1 は `k1_joint_torques_l2` を使う | 維持、要整理 | 実装は generic に近い。K1 scope / naming のために残すかは後で整理可能 |
| energy reward | G1/H1 標準 reward には同種の `joint_energy` term がない | K1 は `k1_joint_energy` を追加 | 維持、要レビュー | 旧 K1 walk の省エネ penalty を維持 |
| foot/contact 追加 reward | G1/H1 は `feet_air_time`, `feet_slide`, ankle limit, joint deviation が中心 | K1 は `feet_force`, `feet_too_near`, `feet_stumble`, all non-foot `undesired_contacts` を追加 | 維持、要レビュー | K1 の足幅・接触・転倒傾向に合わせた追加項目 |
| joint deviation | G1/H1 は hip/arms/torso、G1 は fingers も default deviation を抑える | K1 は腕 deviation を明示し、hip/torso deviation は追加していない | 維持、要レビュー | K1 では腕暴れ抑制を優先。hip/torso deviation を足すなら挙動変更 |
| reward weights | G1/H1 flat は robot ごとに weights を override。例: G1 flat `feet_air_time=0.75`, H1 flat `feet_air_time=1.0` | K1 は旧 K1 walk weights を維持。例: `feet_air_time=0.5`, `lin_vel_z_l2=-1.0`, `action_rate_l2=-0.01` | 維持、要実験レビュー | 既存 K1 checkpoint の挙動を変えないため |
| PPO hidden dims | G1 rough/H1 rough は `[512,256,128]`、G1 flat は `[256,128,128]`、H1 flat は `[128,128,128]` | K1 flat は `[512,256,128]` | 維持、要レビュー | 旧 K1 walk の network size を維持。G1/H1 flat より大きい |
| PPO iterations | G1 flat `1500`, H1 flat `1000`, rough `3000` | K1 `50000` | 維持、要実験レビュー | K1 の既存学習予算を維持 |
| PPO normalization | G1/H1 RSL-RL は `empirical_normalization=False` | K1 は `True` | 維持、要レビュー | 旧 K1 walk の設定を維持 |
| PPO entropy | G1 `0.008`, H1 `0.01` | K1 `0.005` | 維持、要レビュー | 旧 K1 walk の設定を維持 |
| `clip_actions` | G1/H1 RSL-RL cfg では明示なし | K1 は `clip_actions=None` を明示 | 維持、要確認 | 旧 K1 walk の挙動維持。RSL-RL wrapper 側の解釈を変えない |

## 旧 K1 walk から維持しているもの

| 項目 | 扱い |
| --- | --- |
| deploy-compatible actor obs | `root_ang_vel_b`, `projected_gravity_b`, command, joint pos rel, scaled joint vel, last action の 69 次元を維持 |
| actor history | 10フレーム flatten、690 次元を維持 |
| actor に入れない情報 | `base_lin_vel` / `root_lin_vel_b` は actor に入れない |
| critic privileged obs | `base_lin_vel` と足接触を critic のみへ追加 |
| action order | K1 deploy 順序の 20 関節と `preserve_order=True` を維持。頭2DoFは action 対象外 |
| action scale | 旧 walk の一律 `0.25` ではなく、BeyondMimic K1 と同じ `K1_ACTION_SCALE` 由来の関節別 scale を velocity 20DoF に解決して使用 |
| command randomization | 旧 K1 walk の x/y/yaw `(-1,1)` と standing env ratio を維持 |
| reward set/weights | 旧 K1 walk の main flat task の reward set/weights を維持 |
| termination | `bad_orientation(limit_angle=0.8)` を維持 |
| PPO主要値 | `max_iterations=50000`, `empirical_normalization=True`, `clip_actions=None`, `[512,256,128]` を維持 |

## 今回移植しないもの

| 項目 | 扱い | 理由 |
| --- | --- | --- |
| `Booster-K1-Locomotion-*` Gym ID | 登録しない | Isaac Lab 標準型の Gym ID へ寄せるため |
| `IdealFlatForward*` | 移植しない | debug/check variant であり、本命 flat velocity task ではないため |
| `IdealFlatCommandRandom*` | 移植しない | debug/check variant であり、本命 flat velocity task ではないため |
| rough variant | 作らない | 今回は flat velocity task のみ対象 |
| height scanner | 作らない | rough terrain policy ではなく flat/deploy 重視 |
| actor 側 `base_lin_vel` | 追加しない | deploy 入力契約を変える挙動変更になるため |
| `K1_WALK_LEG_JOINT_NAMES` | 移植しない | 未使用 constant のため |

## 確認コマンド

```bash
rg -n 'joint_names=\\[\"\\.\\*\"\\]|joint_deviation_arms|arm_action_l2|arm_action_rate_l2' \
  /mnt/ssd2/booster/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity \
  source/booster_train/booster_train/tasks/manager_based/locomotion/velocity

rg -n 'base_lin_vel|root_lin_vel_b|k1_deploy_locomotion_observation|k1_privileged_locomotion_observation' \
  source/booster_train/booster_train/tasks/manager_based/locomotion/velocity

python -m compileall source/booster_train/booster_train/tasks/manager_based/locomotion/velocity
```
