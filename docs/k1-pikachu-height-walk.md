# Pikachu fixed-height walking task

`Booster-K1-Pikachu-HeightWalk-Scale0842-v0` is the simplified successor to the
first `IdleWalk` experiment. It does not imitate the time-varying Idle motion.
The policy learns its own joint posture while following a velocity command.

## Task constraints

- Track a fixed Trunk-link height of `0.45 m`.
- Penalize K1-leg proximity to the hard torso shell below `10 mm`.
- End an episode at `2 mm` shell clearance or below.
- End an episode when Trunk height falls below `0.33 m`.
- Start with forward commands from `0` to `0.10 m/s`; lateral and yaw commands
  remain zero.
- Keep locomotion regularizers for foot slip, flight, undesired contacts,
  action rate, acceleration, and joint limits.

The task deliberately removes upper-body Idle imitation, Trunk roll/pitch
tracking, knee-angle constraints, the Idle phase, and Idle reference
observations. Its actor observation is 750 values: ten frames of velocity
command, angular velocity, projected gravity, joint position, joint velocity,
and previous action. Simulated linear velocity, fixed-height error, and shell
clearance are critic-only observations.

## RTX 4090 parallelism benchmark

The benchmark used Isaac Sim 5.0, Isaac Lab v2.2.0, an RTX 4090 with 24 GiB
VRAM, 32 GiB host RAM, and `--video --video_length 24`. Each successful case
ran one PPO update.

| Environments | Result | Peak VRAM | Host-memory observation | Throughput |
|---:|---|---:|---|---:|
| 64 | Passed; MP4 written | 7.7 GiB | 15.2 GiB process RSS during first shader build | not used for sizing |
| 4,096 | Passed; MP4 written | 11.6 GiB | 20.5 GiB process RSS; about 9.1 GiB available | 4,408 steps/s |
| 6,144 | Failed during PhysX startup with CUDA illegal memory access | 10.9 GiB | 28.6 GiB process RSS; 367 MiB available | did not reach PPO |
| 10,240 | Stopped before PPO to protect the host | 11.5 GiB | about 30 GiB used and all 8 GiB swap consumed | did not reach PPO |

The practical limit on this workstation is host RAM and PhysX startup, not
VRAM. The selected long-run value is therefore `4,096`, which retains useful
headroom. `20,480` is not viable on this 32 GiB host.

## Training command

```bash
cd ~/booster/IsaacLab
./isaaclab.sh -p ~/booster/booster_train/scripts/rsl_rl/train.py \
  --task Booster-K1-Pikachu-HeightWalk-Scale0842-v0 \
  --num_envs 4096 \
  --max_iterations 6000 \
  --headless \
  --device cuda:0 \
  --video \
  --video_interval 2400 \
  --video_length 300 \
  --run_name height_only_4096env_6000_video
```

The video interval is 2,400 environment steps, equal to 100 PPO updates for
the current 24-step rollout. Each recorded clip contains 300 environment
steps, or six seconds at the 50 Hz control rate.
