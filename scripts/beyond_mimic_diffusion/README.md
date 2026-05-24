# BeyondMimic Diffusion Joystick Scripts

Run these scripts from the Isaac Lab environment used by the existing Booster
K1 and BeyondMimic workflows.

```bash
cd /mnt/ssd2/booster/booster_train
source /mnt/ssd2/booster/env_isaaclab/bin/activate
export DISPLAY=:1
```

Collect locomotion teacher rollouts from an RSL-RL checkpoint. These teacher
policies are data sources only; the exported joystick controller does not load
a runtime teacher prior by default.

```bash
python scripts/beyond_mimic_diffusion/collect_isaac_rollouts.py \
  --task Booster-K1-Locomotion-v0-Play \
  --checkpoint logs/rsl_rl/k1_locomotion/<run>/model_<iter>.pt \
  --teacher-type rsl_rl \
  --teacher-id k1_locomotion \
  --num_envs 16 \
  --steps 1000 \
  --output-dir logs/beyond_mimic_diffusion/datasets/k1_joystick_v1
```

Collect BeyondMimic motion-tracking rollouts from an exported TorchScript
teacher:

```bash
python scripts/beyond_mimic_diffusion/collect_isaac_rollouts.py \
  --task Booster-K1-Fight_001-v0-Play \
  --checkpoint /mnt/ssd2/booster/booster_deploy/tasks/beyond_mimic/models/k1_fight_001_model_11000.pt \
  --teacher-type jit \
  --teacher-id k1_fight \
  --command-name motion \
  --num_envs 16 \
  --steps 1000 \
  --output-dir logs/beyond_mimic_diffusion/datasets/k1_joystick_v1
```

Train the VAE from teacher rollouts:

```bash
python scripts/beyond_mimic_diffusion/train_vae.py \
  --shards logs/beyond_mimic_diffusion/datasets/k1_joystick_v1/*.npz \
  --output-dir logs/beyond_mimic_diffusion/k1_joystick_v1/vae
```

Then collect diffusion rollouts by running the VAE policy in
Isaac Lab with the paper's OU action noise (`theta=0.8`, `mu=0`, `sigma=0.1`,
`dt=1.0`). Each shard records 2.5 seconds and rejects envs that terminate before
5 seconds:

```bash
python scripts/beyond_mimic_diffusion/collect_vae_rollouts.py \
  --task Booster-K1-Locomotion-v0-Play \
  --vae-decoder logs/beyond_mimic_diffusion/k1_joystick_v1/vae/vae_decoder.pt \
  --normalizer logs/beyond_mimic_diffusion/k1_joystick_v1/vae/normalizer.json \
  --output-dir logs/beyond_mimic_diffusion/datasets/k1_joystick_v1_diffusion \
  --num_envs 16
```

Optionally mirror shards with K1 sagittal symmetry before diffusion training:

```bash
python scripts/beyond_mimic_diffusion/augment_sagittal_symmetry.py \
  --shards logs/beyond_mimic_diffusion/datasets/k1_joystick_v1_diffusion/*.npz \
  --manifest logs/beyond_mimic_diffusion/datasets/k1_joystick_v1_diffusion/manifest.json \
  --output-dir logs/beyond_mimic_diffusion/datasets/k1_joystick_v1_diffusion_mirrored
```

Train diffusion and export:

```bash
python scripts/beyond_mimic_diffusion/train_diffusion.py \
  --shards logs/beyond_mimic_diffusion/datasets/k1_joystick_v1_diffusion/*.npz \
  --vae-checkpoint logs/beyond_mimic_diffusion/k1_joystick_v1/vae/vae.pt \
  --normalizer logs/beyond_mimic_diffusion/k1_joystick_v1/vae/normalizer.json \
  --output-dir logs/beyond_mimic_diffusion/k1_joystick_v1/diffusion \
  --latent-loss-weight 1.0 \
  --state-loss-weight 1.0

python scripts/beyond_mimic_diffusion/export.py \
  --vae-checkpoint logs/beyond_mimic_diffusion/k1_joystick_v1/vae/vae.pt \
  --diffusion-checkpoint logs/beyond_mimic_diffusion/k1_joystick_v1/diffusion/diffusion.pt \
  --normalizer logs/beyond_mimic_diffusion/k1_joystick_v1/vae/normalizer.json \
  --output-dir /mnt/ssd2/booster/booster_deploy/tasks/beyond_mimic/models/k1_beyondmimic_joystick \
  --policy-dt 0.04 \
  --num-bodies 23 \
  --action-joint-names ALeft_Shoulder_Pitch ARight_Shoulder_Pitch Left_Hip_Pitch Right_Hip_Pitch Left_Shoulder_Roll Right_Shoulder_Roll Left_Hip_Roll Right_Hip_Roll Left_Elbow_Pitch Right_Elbow_Pitch Left_Hip_Yaw Right_Hip_Yaw Left_Elbow_Yaw Right_Elbow_Yaw Left_Knee_Pitch Right_Knee_Pitch Left_Ankle_Pitch Right_Ankle_Pitch Left_Ankle_Roll Right_Ankle_Roll
```

Export writes `vae_decoder.pt`, `diffusion.pt`, `normalizer.json`,
`state_projection.pt`, and `manifest.json`. The diffusion window follows the
paper-style layout: `history_length=4`, `current_index=4`, `horizon=16`,
`window_length=21`, and the deploy controller runs at `policy_dt=0.04` (25 Hz).

Deploy and evaluate in MuJoCo:

```bash
cd /mnt/ssd2/booster/booster_deploy
source /mnt/ssd2/booster/env_isaaclab/bin/activate
export DISPLAY=:1

python scripts/deploy.py --task k1_beyondmimic_joystick --mujoco
python scripts/evaluate_mujoco_velocity.py \
  --task k1_beyondmimic_joystick \
  --duration-s 10.0 \
  --schedules forward backward lateral yaw mixed
```

The deploy task uses joystick classifier guidance inside the diffusion
denoising loop. Runtime teacher-prior blending is kept only as an ablation knob
in code and is disabled by default; normal evaluation should leave it at `0.0`.
With the repository-scale teacher data, stability may be worse than the paper's
large human-motion corpus, so acceptance should report metrics rather than
silently re-enabling a runtime teacher.
