"""Franka Panda tabletop pickup environment using MolmoSpaces assets.

This environment trains a Franka Panda to pick up a cup from a table.
The cup USD comes from MolmoSpaces (THOR assets); the robot uses
IsaacLab's built-in Franka Panda asset.

IMPORTANT: This module must be imported AFTER isaaclab AppLauncher is created.

Usage:
    python -m molmo_spaces_isaac.envs.train --num_envs 512 --headless
"""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils import configclass

from molmo_spaces_isaac.envs.franka_table_env import (
    TABLE_HEIGHT,
    FrankaTableEnv,
    FrankaTableEnvCfg,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[4]  # molmospaces/
_DEFAULT_CUP_USD = (
    _REPO_ROOT / "assets" / "usd" / "objects" / "thor" / "Cup_1_mesh" / "Cup_1_mesh.usda"
)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
@configclass
class FrankaPickupEnvCfg(FrankaTableEnvCfg):
    """Config for the Franka tabletop cup-pickup task."""

    # -- spaces --
    observation_space: int = 25  # see _get_observations

    # -- cup (MolmoSpaces THOR asset) --
    cup_usd_path: str = _DEFAULT_CUP_USD.as_posix()
    cup_cfg: RigidObjectCfg = MISSING  # built dynamically in __post_init__

    # -- reward scales --
    reward_reaching: float = 1.0
    reward_grasp: float = 5.0
    reward_lift: float = 10.0
    reward_action_penalty: float = -0.01

    # -- task thresholds --
    lift_height: float = 0.15  # metres above table to count as "lifted"
    grasp_threshold: float = 0.02  # finger closure threshold

    def __post_init__(self) -> None:
        """Build the cup RigidObjectCfg from the USD path."""
        self.cup_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Cup",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.cup_usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.0, TABLE_HEIGHT + 0.07),
                rot=(0.7071068, 0.7071068, 0.0, 0.0),  # 90-deg around X (Y-up -> Z-up)
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class FrankaPickupEnv(FrankaTableEnv):
    """Franka Panda picks up a MolmoSpaces cup from a table."""

    cfg: FrankaPickupEnvCfg

    # ------------------------------------------------------------------
    # Task objects
    # ------------------------------------------------------------------
    def _setup_task_objects(self) -> None:
        self._cup = RigidObject(self.cfg.cup_cfg)
        self.scene.rigid_objects["cup"] = self._cup

    # ------------------------------------------------------------------
    # Observations  (dim = 25)
    #   hand_pos(3) + hand_quat(4) + joint_pos(7) + joint_vel(7) +
    #   cup_pos_rel(3) + finger_width(1) = 25
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        hand_pos_w = self._robot.data.body_pos_w[:, self._hand_idx]
        hand_quat_w = self._robot.data.body_quat_w[:, self._hand_idx]
        joint_pos = self._robot.data.joint_pos[:, :7]
        joint_vel = self._robot.data.joint_vel[:, :7]
        cup_pos_w = self._cup.data.root_pos_w
        cup_pos_rel = cup_pos_w - hand_pos_w
        finger_width = (
            self._robot.data.joint_pos[:, self._finger_joint_ids[0]]
            + self._robot.data.joint_pos[:, self._finger_joint_ids[1]]
        ).unsqueeze(-1)

        obs = torch.cat(
            [hand_pos_w, hand_quat_w, joint_pos, joint_vel, cup_pos_rel, finger_width],
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        hand_pos = self._robot.data.body_pos_w[:, self._hand_idx]
        cup_pos = self._cup.data.root_pos_w

        dist = torch.norm(hand_pos - cup_pos, dim=-1)
        reaching = -dist * self.cfg.reward_reaching

        finger_w = (
            self._robot.data.joint_pos[:, self._finger_joint_ids[0]]
            + self._robot.data.joint_pos[:, self._finger_joint_ids[1]]
        )
        is_near = dist < 0.05
        is_closed = finger_w < self.cfg.grasp_threshold
        grasp = (is_near & is_closed).float() * self.cfg.reward_grasp

        cup_height_above_table = cup_pos[:, 2] - TABLE_HEIGHT
        lifted = torch.clamp(cup_height_above_table - 0.08, min=0.0) * self.cfg.reward_lift

        action_cost = torch.sum(self._actions**2, dim=-1) * self.cfg.reward_action_penalty

        return reaching + grasp + lifted + action_cost

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cup_pos = self._cup.data.root_pos_w

        success = cup_pos[:, 2] > (TABLE_HEIGHT + self.cfg.lift_height)
        fell = cup_pos[:, 2] < 0.0

        terminated = success | fell
        timed_out = self.episode_length_buf >= self.max_episode_length

        return terminated, timed_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self._reset_robot(env_ids)
        self._place_object_on_table(self._cup, env_ids)
