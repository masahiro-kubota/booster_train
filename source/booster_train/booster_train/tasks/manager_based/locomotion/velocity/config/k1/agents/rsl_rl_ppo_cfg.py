# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class K1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for K1 flat velocity locomotion."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "k1_flat"
    empirical_normalization = True
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class K1FlatRewardAPPORunnerCfg(K1FlatPPORunnerCfg):
    """PPO runner configuration for K1 flat velocity Reward-A ablation."""

    experiment_name = "k1_flat_reward_a"
