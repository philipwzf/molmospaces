"""Franka tabletop pinch-point scene: grasp a mug next to a fragile object.

Ported from ``task_generation/pinch_point_pipeline.py``.  A mug (target) is
placed on the table with a fragile object (e.g. a vase) positioned right next
to it, creating a tight-clearance "pinch point" for the gripper.

IMPORTANT: This module must be imported AFTER isaaclab AppLauncher is created.
"""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils import configclass

from molmo_spaces_isaac.envs.asset_registry import AssetRegistry
from molmo_spaces_isaac.envs.franka_table_env import (
    TABLE_HEIGHT,
    FrankaTableEnv,
    FrankaTableEnvCfg,
)

TARGET_CATEGORIES = ["Mug", "Cup"]
FRAGILE_CATEGORIES = ["Vase_Flat", "Vase_Medium", "Vase_Open", "Candle"]

_THOR_QUAT = (0.7071068, 0.7071068, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@configclass
class FrankaPinchEnvCfg(FrankaTableEnvCfg):
    """Config for the Franka tabletop pinch-point task."""

    observation_space: int = 25

    # Gap between target and fragile object (metres).
    pinch_gap: float = 0.005

    max_object_dim: float = 0.25
    asset_seed: int = 42

    target_cfg: RigidObjectCfg = MISSING  # type: ignore[assignment]
    fragile_cfg: RigidObjectCfg = MISSING  # type: ignore[assignment]

    # Bbox widths used for placement offset (set in __post_init__)
    _target_half_w: float = 0.05
    _fragile_half_w: float = 0.05

    def __post_init__(self) -> None:
        import numpy as np

        registry = AssetRegistry(max_dim=self.max_object_dim)
        rng = np.random.default_rng(self.asset_seed)

        def _pick(categories: list[str]) -> tuple[str, Path]:
            for cat in rng.permutation(categories):
                try:
                    return registry.sample(cat, rng=rng)
                except RuntimeError:
                    continue
            return registry.sample(rng=rng)

        target_id, target_path = _pick(TARGET_CATEGORIES)
        fragile_id, fragile_path = _pick(FRAGILE_CATEGORIES)

        target_bbox = registry.bbox(target_id)
        fragile_bbox = registry.bbox(fragile_id)

        # Half-widths in XZ plane (which becomes XY after Y-up → Z-up rotation)
        self._target_half_w = max(target_bbox[0], target_bbox[2]) / 2.0
        self._fragile_half_w = max(fragile_bbox[0], fragile_bbox[2]) / 2.0

        self.target_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/pinch_target",
            spawn=sim_utils.UsdFileCfg(
                usd_path=target_path.as_posix(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.0, TABLE_HEIGHT + 0.07),
                rot=_THOR_QUAT,
            ),
        )

        # Fragile placed adjacent to target along Y axis
        offset = self._target_half_w + self._fragile_half_w + self.pinch_gap
        self.fragile_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/pinch_fragile",
            spawn=sim_utils.UsdFileCfg(
                usd_path=fragile_path.as_posix(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, offset, TABLE_HEIGHT + 0.07),
                rot=_THOR_QUAT,
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class FrankaPinchEnv(FrankaTableEnv):
    """Franka Panda grasps a target with a fragile object in close proximity."""

    cfg: FrankaPinchEnvCfg

    def _setup_task_objects(self) -> None:
        self._target = RigidObject(self.cfg.target_cfg)
        self.scene.rigid_objects["pinch_target"] = self._target

        self._fragile = RigidObject(self.cfg.fragile_cfg)
        self.scene.rigid_objects["pinch_fragile"] = self._fragile

    # ------------------------------------------------------------------
    # Observations (placeholder)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        hand_pos_w = self._robot.data.body_pos_w[:, self._hand_idx]
        hand_quat_w = self._robot.data.body_quat_w[:, self._hand_idx]
        joint_pos = self._robot.data.joint_pos[:, :7]
        joint_vel = self._robot.data.joint_vel[:, :7]
        target_rel = self._target.data.root_pos_w - hand_pos_w
        finger_width = (
            self._robot.data.joint_pos[:, self._finger_joint_ids[0]]
            + self._robot.data.joint_pos[:, self._finger_joint_ids[1]]
        ).unsqueeze(-1)

        obs = torch.cat(
            [hand_pos_w, hand_quat_w, joint_pos, joint_vel, target_rel, finger_width],
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards / dones (stub)
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        timed_out = self.episode_length_buf >= self.max_episode_length
        return terminated, timed_out

    # ------------------------------------------------------------------
    # Reset — place target with jitter, then fragile adjacent to it
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self._reset_robot(env_ids)

        n = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]
        quat = torch.tensor(_THOR_QUAT, device=self.device).unsqueeze(0).expand(n, -1)
        zero_vel = torch.zeros(n, 6, device=self.device)

        # Target position with jitter
        tx = 0.5 + (torch.rand(n, device=self.device) - 0.5) * 0.10
        ty = (torch.rand(n, device=self.device) - 0.5) * 0.06

        target_pos = torch.zeros(n, 3, device=self.device)
        target_pos[:, 0] = tx
        target_pos[:, 1] = ty
        target_pos[:, 2] = TABLE_HEIGHT + 0.07
        self._target.write_root_pose_to_sim(
            torch.cat([target_pos + env_origins, quat], dim=-1), env_ids=env_ids
        )
        self._target.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        # Fragile — adjacent to target along a random direction (±X or ±Y)
        offset = self.cfg._target_half_w + self.cfg._fragile_half_w + self.cfg.pinch_gap
        direction = torch.randint(0, 4, (n,), device=self.device)
        dx = torch.zeros(n, device=self.device)
        dy = torch.zeros(n, device=self.device)
        dx[direction == 0] = offset
        dx[direction == 1] = -offset
        dy[direction == 2] = offset
        dy[direction == 3] = -offset

        frag_pos = torch.zeros(n, 3, device=self.device)
        frag_pos[:, 0] = tx + dx
        frag_pos[:, 1] = ty + dy
        frag_pos[:, 2] = TABLE_HEIGHT + 0.07
        self._fragile.write_root_pose_to_sim(
            torch.cat([frag_pos + env_origins, quat], dim=-1), env_ids=env_ids
        )
        self._fragile.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
