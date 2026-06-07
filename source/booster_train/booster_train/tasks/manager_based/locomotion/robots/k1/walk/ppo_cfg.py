from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for K1 deploy-compatible locomotion."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "k1_locomotion"
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
class IdealFlatForwardPPORunnerCfg(PPORunnerCfg):
    """PPO runner configuration for the ideal flat-forward K1 check."""

    experiment_name = "k1_locomotion_ideal_flat_forward"


@configclass
class IdealFlatCommandRandomPPORunnerCfg(PPORunnerCfg):
    """PPO runner configuration for the ideal flat command-random K1 check."""

    experiment_name = "k1_locomotion_ideal_flat_command_random"
