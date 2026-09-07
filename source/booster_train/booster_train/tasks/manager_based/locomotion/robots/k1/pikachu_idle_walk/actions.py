"""Observe shell separation at 200 Hz while retaining standard position actions."""
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from .mdp import shell_guard


class GuardedJointPositionAction(JointPositionAction):
    def apply_actions(self):
        env=self._env
        guard=shell_guard(env)
        if (env._sim_step_counter-1)%env.cfg.decimation==0:
            guard.begin_control_step()
        robot=env.scene['robot']
        guard.observe(robot.data.body_link_pos_w,robot.data.body_link_quat_w,env._sim_step_counter-1)
        super().apply_actions()
