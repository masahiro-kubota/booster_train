# K1 Walk と Isaac Lab 二足 Velocity タスクの設計差分

このメモは、K1 walk タスクを Isaac Lab の G1/H1 velocity タスクに近いコードスタイルへリファクタするときに、残すべき意図的な挙動差分を整理するためのものです。

参照ファイル:

- K1 walk タスク:
  `source/booster_train/booster_train/tasks/manager_based/locomotion/robots/k1/walk/env_cfg.py`
- K1 walk 観測ヘルパー:
  `source/booster_train/booster_train/tasks/manager_based/locomotion/mdp/observations.py`
- K1 walk 報酬ヘルパー:
  `source/booster_train/booster_train/tasks/manager_based/locomotion/mdp/rewards.py`
- ローカルの Isaac Lab 参照実装:
  `../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/`

## 概要

K1 walk タスクは、Isaac Lab の G1/H1 二足 velocity タスクを単純に K1 アセットへ差し替えたものではなく、K1 固有の派生実装として扱うべきです。

共通している主な設計:

- Isaac Lab の manager-based task 構造を使っている。
- base velocity command を使っている。
- default offset 付きの joint position action を使っている。
- RSL-RL PPO runner と標準的な PPO ハイパーパラメータ構成を使っている。
- 二足 velocity 系の報酬ファミリを使っている:
  `track_lin_vel_xy_yaw_frame_exp`, `track_ang_vel_z_world_exp`,
  `feet_air_time_positive_biped`, `feet_slide`, `flat_orientation_l2`,
  `is_terminated`。

K1 固有の主な設計:

- actor 観測を実機デプロイ互換にしている。
- actor は base linear velocity を観測しない。
- actor 観測は時間履歴を flatten した入力になっている。
- critic だけがシミュレーション中の privileged 情報を受け取る。
- K1 の関節順、デフォルト姿勢、足/contact body 名を明示している。
- 現在の標準タスクは平地前提で、height scan を使わない。

## 残すべき意図的な差分

`新 velocity / RewardA での状態` は、`Booster-Velocity-Flat-K1-v0` と
`Booster-Velocity-Flat-K1-RewardA-v0` で、その差分がどう残っているかを示します。
`次タスク方針` は、command range を維持したまま、消せる差分を消す追加 task での扱いです。

| 領域 | Isaac Lab G1/H1 velocity のスタイル | K1 walk の設計 | 残す? | 新 velocity / RewardA での状態 | 次タスク方針 | 理由 |
| --- | --- | --- | --- | --- | --- | --- |
| ロボットアセット | robot-specific config で `G1_MINIMAL_CFG` や `H1_MINIMAL_CFG` を設定 | `FlatEnvCfg.__post_init__` で `BOOSTER_K1_CFG` を設定 | Yes | 残る。`K1FlatEnvCfg.__post_init__` で `BOOSTER_K1_CFG` と K1 locomotion 初期姿勢を設定する。RewardA も同じ。 | 残す。K1 固有 asset / 初期姿勢は変更しない。 | K1 固有の articulation、初期 base height、デフォルト関節姿勢が必要。 |
| action 対象関節 | `[".*"]` のような広い regex を使い、報酬側で subset を指定することが多い | `K1_WALK_POLICY_JOINT_NAMES` を明示し、`preserve_order=True` を使う | Yes | 残る。新 velocity は 20DoF を明示し、head 2DoF を除外し、`preserve_order=True` を維持する。 | 残す。20DoF deploy order と head 除外は変更しない。 | policy の action 順序を K1 deploy/export と一致させる必要がある。 |
| action scale | Isaac Lab の base velocity config は `0.5`、robot config 側で調整 | K1 は `K1_WALK_ACTION_SCALE = 0.25` | Yes | 一部更新。旧 walk の固定 `0.25` ではなく、`K1_ACTION_SCALE` 由来の per-joint dict に修正済み。G1/H1 の `0.5` との差分は残る。 | 残す。`K1_ACTION_SCALE` dict を維持する。 | K1 の actuator range と deploy policy の期待値が違う。 |
| actor 観測の形 | 複数の observation term を manager が結合する | deploy 互換の observation term 1本を履歴付きで flatten する | Yes | 残る。policy group は `deploy_locomotion_obs` 1本、shape は 690。RewardA も同じ。 | 残す。deploy 互換入力は変更しない。 | ONNX/deploy の入力契約を固定するため。 |
| actor の base linear velocity | base velocity task では policy observation に入ることが多い | actor 観測に入れない | Yes | 残る。RewardA でも actor に `base_lin_vel` / `root_lin_vel_b` は入れない。 | 残す。actor には追加しない。 | base linear velocity は実機でシミュレーション真値と同じ品質では直接取得できない。 |
| actor の angular velocity | `base_ang_vel` observation term | deploy 観測内の `root_ang_vel_b` | Yes | 残る。deploy 互換 observation の中に含める。 | 残す。パッケージングは変更しない。 | 実機 IMU 由来で扱いやすい同等信号。パッケージングだけが違う。 |
| actor の gravity | `projected_gravity` observation term | deploy 観測内の `projected_gravity_b` | Yes | 残る。deploy 互換 observation の中に含める。 | 残す。パッケージングは変更しない。 | 実機 IMU 由来で扱いやすい姿勢信号。 |
| actor の command | `generated_commands(base_velocity)` | deploy 観測内に command vector を埋め込む | Yes | 残る。command の値は使うが、term 分割ではなく deploy 互換 observation に埋め込む。 | 残す。command range も observation への入れ方も維持する。 | command の意味は同じで、観測の作り方だけが違う。 |
| actor の joint state | `joint_pos_rel`, `joint_vel_rel` terms | K1 順序の joint position error と scaled joint velocity | Yes | 残る。20DoF deploy action 順序で joint pos rel / scaled joint vel を入れる。 | 残す。deploy joint order を維持する。 | deploy 順序と deploy 用 scale を維持するため。 |
| actor の履歴 | Isaac Lab の基本 config では通常 flatten された時間履歴を使わない | `history_length = 10`、actor 入力は `69 * 10 = 690` | Yes | 残る。新 velocity / RewardA とも `history_length=10`, `flatten_history_dim=True`。 | 残す。履歴長 10 を維持する。 | `base_lin_vel` なしで速度や運動状態を推定しやすくするため。 |
| observation corruption | 標準 config では term ごとに noise/corruption を有効化することが多い | K1 deploy 観測は corruption 無効 | Yes。ただし明示的に変更するなら別判断 | 残る。policy/critic とも `enable_corruption=False`。 | 残す。custom flatten obs への noise 設計は別実験に分ける。 | deploy 互換観測は、意図的にノイズ注入を戻すまでは決定的に保つ。 |
| critic 観測 | policy と同じ、または標準的な privileged setup | privileged critic observation に `root_lin_vel_b` と足接触を追加 | Yes | 残る。critic は `privileged_locomotion_obs` を持ち、policy obs に加えて `base_lin_vel` 相当と足接触を持つ。RewardA も同じ。 | 残す。asymmetric actor-critic を維持する。 | actor は deploy 可能なまま、critic は学習中だけシミュレーション真値を使える。 |
| terrain | Isaac Lab rough config は terrain generator と height scanner を使い、flat config で無効化 | K1 walk はデフォルトで flat plane、height scanner なし | 現在の K1 walk では Yes | 残る。新 velocity は flat-only 直定義で、rough variant / height scanner は追加していない。 | 残す。flat-only のままにする。 | 現在のタスクは flat/deploy 重視。地形認識が必要なら別 rough variant を追加する。 |
| base contact body | `torso_link` など robot 固有の base body | `Trunk` | Yes | 残る。termination の `base_contact` は `Trunk`。 | 残す。K1 body 名のため変更しない。 | K1 の body 名に合わせる。 |
| foot body names | G1/H1 の ankle/foot link regex | `left_foot_link`, `right_foot_link` | Yes | 残る。contact sensor と feet reward は K1 の foot body 名を使う。 | 残す。K1 foot body 名のため変更しない。 | K1 の body 名と contact sensor の scope に合わせる。 |
| heading command | Isaac Lab の base config は heading command をサポートし、G1/H1 側で range を調整 | K1 は heading command を無効化 | Yes | 残る。`heading_command=False`, `rel_heading_envs=0.0`。 | 残す。direct velocity command のままにする。 | 現在の policy は direct velocity command を受け取る設計。 |
| command range | Isaac Lab の robot config は robot ごとに x/y/yaw range を調整 | K1 の command random variant は x/y/yaw を広く取り、fixed-forward variant は固定 command | Yes。ただし weight/range は別途レビュー対象 | 残る。新 velocity / RewardA は x/y/yaw すべて `(-1.0, 1.0)`、`rel_standing_envs=0.2`。G1/H1 flat とは違う。 | 残す。joystick/deploy 用途を優先し、x/y/yaw `(-1.0, 1.0)` と `rel_standing_envs=0.2` を維持する。 | これはコードスタイルではなくタスク設計。 |
| event randomization | G1/H1 は humanoid config 側で push / add mass / base COM を無効化し、reset velocity と joint reset の揺らぎも小さくする | K1 は friction range、`Trunk` mass randomization、reset velocity randomization、push `(-1,1)` を有効化 | 要レビュー | RewardA でも K1 base env を継承するため残る。 | 消す。physics material を固定寄りにし、`add_base_mass=None`, `push_robot=None`, reset base velocity を 0、joint reset range を `(1.0, 1.0)` に寄せる。 | command range とは別軸の外乱差分なので、標準寄せ実験では消してよい。 |
| reward helper names | Isaac Lab は使える限り generic helper を使う | torque, energy, contact force, stumble, foot spacing に `k1_*` helper を追加 | K1 固有の意味がある箇所は Yes | base velocity では残る。RewardA では K1 固有 helper の多くを外し、generic helper 中心に寄せた。 | 消す。RewardA reward を維持し、K1 固有 reward helper は入れない。 | K1 固有の報酬意味、または deploy 用 scope を表すため。 |
| 腕抑制 reward | G1/H1 も `joint_names=[".*"]` の action 指定で腕を含み、generic `joint_deviation_arms` で default 姿勢からのずれを抑える | K1 も腕を action に含めるが、K1 deploy action 順序の腕8DoFを index 指定し、`arm_joint_deviation_l1`, `arm_action_l2`, `arm_action_rate_l2` で姿勢ずれ・action・action rate を追加で抑える | Yes | base velocity では残る。RewardA では `joint_deviation_arms` のみに寄せ、`arm_action_l2` / `arm_action_rate_l2` は外した。 | 消す。RewardA と同じく `joint_deviation_arms` のみを使う。 | K1 では腕を姿勢安定用の可動質量として過剰に使うことがある。obs/action 互換を変えずに、deploy action 順序のまま腕の大振りを抑えるため。 |
| PPO architecture | G1/H1 rough は actor/critic に `[512, 256, 128]` を使う。G1/H1 flat はより小さい hidden dims にする | K1 も `[512, 256, 128]` | Yes | 残る。新 velocity / RewardA とも `[512, 256, 128]`。ただし G1/H1 flat runner は rough より小さい hidden dims に上書きする。 | 残す。K1 actor obs は 690 次元なので、network capacity は別実験に分ける。 | hidden dims 変更は標準寄せではあるが、容量差分が大きく学習成否の比較軸を変える。 |
| PPO entropy | G1 は `0.008`、H1 は `0.01` | K1 は `0.005` | 要レビュー | RewardA でも `0.005` のまま。 | 消す。RewardA が G1 flat 寄せなので `entropy_coef=0.008` に寄せる。 | PPO探索量の差分で、deploy interface や command range を壊さず標準寄せできる。 |
| PPO iterations | Isaac Lab examples は比較的小さい iteration 数 | K1 は `max_iterations = 50000` | Yes。ただし実験で調整 | 残る。新 velocity / RewardA とも `max_iterations=50000`。 | 残す。学習停止位置は run 側で管理する。 | 学習予算はプロジェクト側の選択。 |
| PPO empirical normalization | Isaac Lab G1/H1 examples は `False` | K1 walk は `True` | Yes。ただし学習結果次第 | 残る。新 velocity / RewardA とも `empirical_normalization=True`。 | 消す。`empirical_normalization=False` にする。 | G1/H1 RSL-RL cfg と合わせられる差分で、deploy input contract は変えない。 |

## スタイルだけ寄せてよい候補

以下は、挙動を変えない限り Isaac Lab 風に整理してよい項目です。

| 現在の K1 walk 項目 | Isaac Lab 風の候補 | 制約 |
| --- | --- | --- |
| `K1WalkSceneCfg` | `K1FlatSceneCfg` など env class と揃う名前 | flat terrain と contact sensor の挙動は維持する。 |
| `FlatEnvCfg` / `PlayFlatEnvCfg` | Isaac Lab の `*FlatEnvCfg` / `*FlatEnvCfg_PLAY` パターンに寄せる | Gym 登録 ID を維持するか、登録側も同時に更新する。 |
| `JointPositionAction` または `joint_pos` の term 名 | 対象ファイルで使う Isaac Lab のローカル慣習へ寄せる | action term の順序や `preserve_order=True` は変えない。 |
| `joint_torques_l2` などの reward term 名 | helper の意味が一致する場合だけ Isaac Lab 風の名前に寄せる | K1 固有 helper を、意味が違うのに generic 名へ変えない。 |
| ファイル先頭の K1 定数群 | action、observation、body names、default pose のように役割ごとに整理 | joint order は変えない。 |
| 繰り返しの `SceneEntityCfg` | 明確になるならローカル定数化してよい | 必要な scope と `preserve_order` は維持する。 |
| `IdealFlatForward*` と `IdealFlatCommandRandom*` variants | debug/check variant として名前や配置を整理 | main deploy task の挙動と混ぜない。 |

## リファクタ前にレビューしたい余計・不明瞭な項目

以下は必ずしも間違いではありませんが、Isaac Lab から K1 への設計差分として必要かは明確ではありません。

| 項目 | レビュー理由 |
| --- | --- |
| `K1_WALK_LEG_JOINT_NAMES` | 定義されているが現在未使用。不要なら削除、必要なら scoped reward/event に使う。 |
| `IdealFlatForward*` と `IdealFlatCommandRandom*` classes | 検証用として有用だが、main task file を大きくしている。移動または明示的な文書化を検討する。 |
| `clip_actions = None` in PPO config | export/deploy 互換のためかもしれない。変更前に確認する。 |
| `empirical_normalization = True` | Isaac Lab G1/H1 examples と違う。学習に必要なら残す。不要なら比較実験する。 |
| locomotion files から削除された Isaac Lab license headers | 実質的に Isaac Lab 派生コードが残っているなら、 attribution と license handling をレビューする。 |

## リファクタ規則

K1 walk のコードスタイルを Isaac Lab に寄せるときは、次の規則を守る。

1. `残す? = Yes` の行は、別途「挙動変更」として合意しない限り維持する。
2. actor input、critic input、action order、reward value、command distribution、termination condition を変えない範囲でのみ、Isaac Lab 風の命名や class layout を優先する。
3. observation shape は外部インターフェースとして扱う:
   actor input は 1フレーム 69 次元、10フレーム flatten 後 690 次元を維持する。
4. K1 joint order と `preserve_order=True` は deploy-critical として扱う。
5. actor に `base_lin_vel` を再導入する変更は、スタイルリファクタではなく挙動変更として扱う。
