# K1 Walk vs Isaac Lab Biped Velocity Design Differences

This note records the intentional behavior differences that should survive a
style refactor of the K1 walk task toward the Isaac Lab G1/H1 velocity task
style.

Reference files:

- K1 walk task:
  `source/booster_train/booster_train/tasks/manager_based/locomotion/robots/k1/walk/env_cfg.py`
- K1 walk observation helpers:
  `source/booster_train/booster_train/tasks/manager_based/locomotion/mdp/observations.py`
- K1 walk reward helpers:
  `source/booster_train/booster_train/tasks/manager_based/locomotion/mdp/rewards.py`
- Isaac Lab references, in the local checkout:
  `../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/`

## Summary

The K1 walk task should be treated as a K1-specific derivative of the Isaac Lab
G1/H1 biped velocity tasks, not as a pure robot-asset swap.

The main shared design is:

- Manager-based Isaac Lab task structure.
- Uniform base velocity command.
- Joint position actions with default offsets.
- RSL-RL PPO runner and standard PPO hyperparameter shape.
- Biped velocity reward family:
  `track_lin_vel_xy_yaw_frame_exp`, `track_ang_vel_z_world_exp`,
  `feet_air_time_positive_biped`, `feet_slide`, `flat_orientation_l2`,
  and `is_terminated`.

The main K1-specific design is:

- Deployment-compatible actor observation.
- Actor does not observe base linear velocity.
- Actor observation uses flattened temporal history.
- Critic receives privileged simulation-only signals.
- K1 joint order, K1 default pose, and K1 foot/contact body names are explicit.
- Flat-ground, no-height-scan task by default.

## Intentional Differences To Keep

| Area | Isaac Lab G1/H1 velocity style | K1 walk design | Keep? | Reason |
| --- | --- | --- | --- | --- |
| Robot asset | `G1_MINIMAL_CFG` or `H1_MINIMAL_CFG` in robot-specific config | `BOOSTER_K1_CFG` injected in `FlatEnvCfg.__post_init__` | Yes | K1 needs its own articulation, initial base height, and default joint pose. |
| Action joints | Usually broad regex such as `[".*"]`, then robot-specific penalties scope subsets | Explicit `K1_WALK_POLICY_JOINT_NAMES` with `preserve_order=True` | Yes | Policy action order must match K1 deploy/export order. |
| Action scale | Isaac Lab base velocity config uses `0.5`; robot configs tune terms | K1 uses `K1_WALK_ACTION_SCALE = 0.25` | Yes | K1 actuator range and deploy policy expectations differ. |
| Actor observation shape | Standard observation terms are listed independently and concatenated by the manager | One deploy-compatible observation term, flattened over history | Yes | This fixes the ONNX/deploy input contract. |
| Actor base linear velocity | Often included in Isaac Lab policy observations, especially in base velocity tasks | Not included in the actor observation | Yes | Base linear velocity is not directly available on hardware with the same quality as simulation truth. |
| Actor angular velocity | `base_ang_vel` observation term | `root_ang_vel_b` inside the deploy observation | Yes | Equivalent hardware-friendly IMU signal, just packaged differently. |
| Actor gravity | `projected_gravity` observation term | `projected_gravity_b` inside the deploy observation | Yes | Hardware-friendly IMU-derived orientation signal. |
| Actor command | `generated_commands(base_velocity)` | Command vector embedded in the deploy observation | Yes | Same command concept, different packaging. |
| Actor joint state | `joint_pos_rel`, `joint_vel_rel` terms | Ordered K1 joint position error and scaled joint velocity | Yes | Preserves deploy order and deploy scaling. |
| Actor history | Usually no flattened temporal history in the base Isaac Lab config | `history_length = 10`, `69 * 10 = 690` actor input | Yes | History helps the actor infer velocity/state without `base_lin_vel`. |
| Observation corruption | Standard configs often enable noise/corruption per term | K1 deploy observation has corruption disabled | Yes, unless explicitly changed | Deployment-compatible observation should remain deterministic unless noise injection is deliberately reintroduced. |
| Critic observation | Often same policy group or standard privileged setup depending on task | Privileged critic observation adds `root_lin_vel_b` and foot contacts | Yes | Asymmetric actor-critic: actor stays deployable, critic can use simulation truth during training. |
| Terrain | Isaac Lab rough configs use terrain generator and height scanner; flat configs disable them | K1 walk is flat plane by default and has no height scanner | Yes for current K1 walk task | Current task is flat/deploy-focused. Add a separate rough variant if terrain perception is needed. |
| Base contact body | `torso_link` or robot-specific base body | `Trunk` | Yes | K1 body naming. |
| Foot body names | G1/H1 ankle/foot link regexes | `left_foot_link`, `right_foot_link` | Yes | K1 body naming and contact sensor scoping. |
| Command heading | Isaac Lab base config supports heading commands; G1/H1 tune ranges | K1 disables heading commands | Yes | Current policy receives direct velocity command only. |
| Command range | Isaac Lab robot configs tune x/y/yaw ranges per robot | K1 random command variant uses full x/y/yaw ranges; fixed-forward variant pins command | Yes, but review weights/ranges separately | These are task choices, not style choices. |
| Reward helper names | Isaac Lab uses generic helpers where available | K1 adds `k1_*` helpers for torque, energy, contact force, stumble, and foot spacing | Yes where K1-specific behavior differs | These encode K1-specific reward semantics or deploy-specific scoping. |
| PPO architecture | G1/H1 rough use `[512, 256, 128]` for actor/critic | K1 uses `[512, 256, 128]` | Yes | Same shape is acceptable; not a naming/style issue. |
| PPO iterations | Isaac Lab examples use smaller iteration counts | K1 uses `max_iterations = 50000` | Yes, but tune experimentally | Training budget is a project choice. |
| PPO empirical normalization | Isaac Lab G1/H1 examples use `False` | K1 walk uses `True` | Yes, unless training evidence says otherwise | This is a behavior change, not a style refactor item. |

## Style-Only Refactor Candidates

These can be changed to look more like Isaac Lab as long as behavior remains
unchanged.

| Current K1 walk item | Isaac Lab-style target | Constraint |
| --- | --- | --- |
| `K1WalkSceneCfg` | A name aligned with the environment class, such as `K1FlatSceneCfg` | Keep flat terrain and contact sensor behavior. |
| `FlatEnvCfg` / `PlayFlatEnvCfg` | Names that mirror Isaac Lab's `*FlatEnvCfg` / `*FlatEnvCfg_PLAY` pattern | Keep registered Gym IDs stable or update registrations together. |
| `JointPositionAction` or `joint_pos` naming | Prefer the Isaac Lab local convention used in the target file | Do not change action term order or `preserve_order=True`. |
| Reward term names such as `joint_torques_l2` | Use Isaac Lab-style names when the helper semantics match | Do not rename K1-specific helpers into generic names if semantics differ. |
| Grouped K1 constants near the top of the file | Keep constants, but group by role: action, observation, body names, default pose | Do not change joint order. |
| Repeated `SceneEntityCfg` blocks | Extract local constants only if it improves clarity | Keep explicit scoping and `preserve_order` where required. |
| `IdealFlatForward*` and `IdealFlatCommandRandom*` variants | Move or name them as debug/check variants | Avoid mixing debug variants with the main deploy task behavior. |

## Extra Or Suspicious Items To Review Before Refactor

These are not necessarily wrong, but they are not clearly required by the
Isaac Lab-to-K1 design difference alone.

| Item | Why review it |
| --- | --- |
| `K1_WALK_LEG_JOINT_NAMES` | Defined but currently unused. Remove it or use it in a scoped reward/event if needed. |
| `IdealFlatForward*` and `IdealFlatCommandRandom*` classes | Useful for checks, but they make the main task file noisy. Consider moving debug variants or documenting them. |
| `clip_actions = None` in PPO config | This may be intentional for export/deploy compatibility. Confirm before changing. |
| `empirical_normalization = True` | Behavior differs from Isaac Lab G1/H1 examples. Keep if training depends on it; otherwise test both. |
| Removed Isaac Lab license headers in locomotion files | If substantial code remains derived from Isaac Lab, attribution and license handling should be reviewed. |

## Refactor Rule

When aligning K1 walk code style to Isaac Lab:

1. Preserve all rows marked `Keep? = Yes` unless a separate behavior-change
   decision is made.
2. Prefer Isaac Lab naming and class layout only where it does not alter the
   actor input, critic input, action order, reward values, command
   distribution, or termination conditions.
3. Treat observation shape as an external interface:
   the actor input remains 69 values per frame and 690 values after 10-frame
   flattening.
4. Treat K1 joint order and `preserve_order=True` as deployment-critical.
5. Any change that reintroduces `base_lin_vel` to the actor is a behavior
   change, not a style refactor.
