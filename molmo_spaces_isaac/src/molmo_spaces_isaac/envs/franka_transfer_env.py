"""Franka tabletop food-transfer scene: move food between containers.

Ported from ``task_generation/transfer_scene_pipeline.py``.  A food item sits
on/in a source container; a destination container is placed nearby.  The robot
must move the food to the destination without touching it directly (tool use)
or simply pick-and-place it.

**Gate check**: after physics settling, verifies the food is still above
the source container (XY proximity + Z above table).  If the food rolled
off, the env is immediately terminated and re-reset with a new seed.

IMPORTANT: This module must be imported AFTER isaaclab AppLauncher is created.
"""

from __future__ import annotations

import random
from dataclasses import MISSING
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils import configclass

from molmo_spaces_isaac.envs.asset_registry import AssetRegistry
from molmo_spaces_isaac.envs.franka_table_env import (
    TABLE_HEIGHT,
    TABLE_SIZE,
    FrankaTableEnv,
    FrankaTableEnvCfg,
)

FOOD_CATEGORIES = ["Apple", "Tomato", "Potato", "Egg", "Bread"]
SOURCE_CATEGORIES = ["Plate", "Bowl"]
DEST_CATEGORIES = ["Bowl", "Plate"]

_THOR_QUAT = (0.7071068, 0.7071068, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@configclass
class FrankaTransferEnvCfg(FrankaTableEnvCfg):
    """Config for the Franka tabletop transfer task."""

    observation_space: int = 25
    max_object_dim: float = 0.25
    asset_seed: int = 42

    # Gate check
    gate_settle_steps: int = 10
    # Max XY distance between food and source centre to count as "on source"
    gate_food_xy_tol: float = 0.12
    # Min Z for food to be considered on the table (not fallen)
    gate_min_z: float = TABLE_HEIGHT - 0.05
    gate_max_retries: int = 5

    food_cfg: RigidObjectCfg = MISSING  # type: ignore[assignment]
    source_cfg: RigidObjectCfg = MISSING  # type: ignore[assignment]
    dest_cfg: RigidObjectCfg = MISSING  # type: ignore[assignment]

    # Source bbox height (Y in THOR coords) — used to place food above source
    _source_height: float = 0.05

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

        # Pick containers first so we can size-filter the food
        source_id, source_path = _pick(SOURCE_CATEGORIES)
        dest_id, dest_path = _pick(DEST_CATEGORIES)

        source_bbox = registry.bbox(source_id)
        self._source_height = source_bbox[1]  # Y = height in THOR coords

        # Source opening footprint: XZ in THOR coords (horizontal after rotation).
        # Food must fit inside — use the smaller of the two container openings
        # with some margin so the food doesn't sit right on the rim.
        dest_bbox = registry.bbox(dest_id)
        container_opening = min(
            min(source_bbox[0], source_bbox[2]),
            min(dest_bbox[0], dest_bbox[2]),
        )
        # Food max XZ footprint must be smaller than the container opening
        food_max_footprint = container_opening * 0.8  # 20% margin

        # Pick food that fits inside the containers
        food_id, food_path = None, None
        for cat in rng.permutation(FOOD_CATEGORIES):
            candidates = registry.assets_in_category(cat)
            rng.shuffle(candidates)
            for aid in candidates:
                bbox = registry.bbox(aid)
                footprint = max(bbox[0], bbox[2])  # XZ = horizontal after rotation
                if footprint <= food_max_footprint:
                    food_id = aid
                    food_path = registry.usd_path(aid)
                    break
            if food_id is not None:
                break

        # Fallback: pick smallest available food if nothing fits
        if food_id is None:
            best_id, best_size = None, float("inf")
            for cat in FOOD_CATEGORIES:
                for aid in registry.assets_in_category(cat):
                    bbox = registry.bbox(aid)
                    size = max(bbox[0], bbox[2])
                    if size < best_size:
                        best_id, best_size = aid, size
            if best_id is not None:
                food_id = best_id
                food_path = registry.usd_path(best_id)
            else:
                food_id, food_path = registry.sample(rng=rng)

        # Source container — left side of table
        self.source_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/source",
            spawn=sim_utils.UsdFileCfg(
                usd_path=source_path.as_posix(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.45, -0.12, TABLE_HEIGHT + 0.07),
                rot=_THOR_QUAT,
            ),
        )

        # Destination container — right side of table
        self.dest_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/dest",
            spawn=sim_utils.UsdFileCfg(
                usd_path=dest_path.as_posix(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.45, 0.12, TABLE_HEIGHT + 0.07),
                rot=_THOR_QUAT,
            ),
        )

        # Food item — on top of source container
        food_z = TABLE_HEIGHT + 0.07 + self._source_height + 0.01

        self.food_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/food",
            spawn=sim_utils.UsdFileCfg(
                usd_path=food_path.as_posix(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.45, -0.12, food_z),
                rot=_THOR_QUAT,
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class FrankaTransferEnv(FrankaTableEnv):
    """Franka Panda transfers food between containers on a table."""

    cfg: FrankaTransferEnvCfg

    def __init__(self, cfg: FrankaTransferEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._reset_counter = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._gate_retry_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # Once True, skip all further gate checks for that env (max retries exceeded)
        self._gate_accepted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_task_objects(self) -> None:
        self._source = RigidObject(self.cfg.source_cfg)
        self.scene.rigid_objects["source"] = self._source

        self._dest = RigidObject(self.cfg.dest_cfg)
        self.scene.rigid_objects["dest"] = self._dest

        self._food = RigidObject(self.cfg.food_cfg)
        self.scene.rigid_objects["food"] = self._food

    # ------------------------------------------------------------------
    # Gate check
    # ------------------------------------------------------------------
    def _check_food_on_source(self) -> torch.Tensor:
        """Return bool tensor: True for envs where food is NOT on the source."""
        food_pos = self._food.data.root_pos_w  # (num_envs, 3)
        source_pos = self._source.data.root_pos_w  # (num_envs, 3)

        # XY distance between food and source
        xy_dist = torch.norm(food_pos[:, :2] - source_pos[:, :2], dim=-1)

        # Food fell off table entirely
        food_fell = food_pos[:, 2] < self.cfg.gate_min_z

        # Food rolled off source container
        food_off_source = xy_dist > self.cfg.gate_food_xy_tol

        return food_fell | food_off_source

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        hand_pos_w = self._robot.data.body_pos_w[:, self._hand_idx]
        hand_quat_w = self._robot.data.body_quat_w[:, self._hand_idx]
        joint_pos = self._robot.data.joint_pos[:, :7]
        joint_vel = self._robot.data.joint_vel[:, :7]
        food_rel = self._food.data.root_pos_w - hand_pos_w
        finger_width = (
            self._robot.data.joint_pos[:, self._finger_joint_ids[0]]
            + self._robot.data.joint_pos[:, self._finger_joint_ids[1]]
        ).unsqueeze(-1)

        obs = torch.cat(
            [hand_pos_w, hand_quat_w, joint_pos, joint_vel, food_rel, finger_width],
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards / dones
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        timed_out = self.episode_length_buf >= self.max_episode_length

        # Gate: continuously check food is on source after settling
        check_mask = (
            (self.episode_length_buf >= self.cfg.gate_settle_steps)
            & (~self._gate_accepted)
        )
        if check_mask.any():
            food_bad = self._check_food_on_source()
            gate_failed = check_mask & food_bad

            if gate_failed.any():
                failed_ids = gate_failed.nonzero(as_tuple=False).squeeze(-1)
                self._gate_retry_count[failed_ids] += 1

                if self.cfg.gate_max_retries > 0:
                    exceeded = self._gate_retry_count[failed_ids] > self.cfg.gate_max_retries
                    if exceeded.any():
                        exceeded_ids = failed_ids[exceeded]
                        print(
                            f"[TransferEnv] Gate max retries ({self.cfg.gate_max_retries}) "
                            f"exceeded for envs {exceeded_ids.tolist()}, accepting",
                            flush=True,
                        )
                        self._gate_accepted[exceeded_ids] = True
                        gate_failed[exceeded_ids] = False

                terminated |= gate_failed

        return terminated, timed_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self._reset_robot(env_ids)

        n = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]
        quat = torch.tensor(_THOR_QUAT, device=self.device).unsqueeze(0).expand(n, -1)
        zero_vel = torch.zeros(n, 6, device=self.device)

        # Increment reset counter for unique seeds; clear gate-accepted flag
        self._reset_counter[env_ids] += 1
        self._gate_accepted[env_ids] = False

        # Per-env randomized placement
        src_pos = torch.zeros(n, 3, device=self.device)
        dst_pos = torch.zeros(n, 3, device=self.device)
        for i in range(n):
            env_id = int(env_ids[i].item())
            counter = int(self._reset_counter[env_id].item())
            rng = random.Random(self.cfg.asset_seed + env_id * 997 + counter * 7919)

            y_spread = rng.uniform(0.10, 0.18)
            flip = 1.0 if rng.random() > 0.5 else -1.0
            x_base = 0.45 + rng.uniform(-0.04, 0.04)

            src_pos[i, 0] = x_base
            src_pos[i, 1] = -y_spread * flip
            src_pos[i, 2] = TABLE_HEIGHT + 0.07

            dst_pos[i, 0] = x_base
            dst_pos[i, 1] = y_spread * flip
            dst_pos[i, 2] = TABLE_HEIGHT + 0.07

        self._source.write_root_pose_to_sim(
            torch.cat([src_pos + env_origins, quat], dim=-1), env_ids=env_ids
        )
        self._source.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        self._dest.write_root_pose_to_sim(
            torch.cat([dst_pos + env_origins, quat], dim=-1), env_ids=env_ids
        )
        self._dest.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        # Food on top of source
        food_pos = src_pos.clone()
        food_pos[:, 2] = TABLE_HEIGHT + 0.07 + self.cfg._source_height + 0.01
        self._food.write_root_pose_to_sim(
            torch.cat([food_pos + env_origins, quat], dim=-1), env_ids=env_ids
        )
        self._food.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
