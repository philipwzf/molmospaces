"""Franka tabletop stack-retrieval scene: retrieve a target from under a stack.

Ported from ``task_generation/stack_scene_pipeline.py``.  Three task variants
controlled by the ``stack_mode`` config field:

* **homogeneous** — target is same category as the rest of the stack.
  E.g. "retrieve the bottom plate from a stack of plates."
* **container_target** — target is a different container (Bowl, Plate, Pot …)
  at the bottom, with a same-category stack on top.
  E.g. "retrieve the bowl from under a stack of plates."
* **flat_target** — a flat non-container object (CreditCard, Cloth …) at the
  bottom, with a same-category stack on top.
  E.g. "retrieve the credit card from under a stack of bowls."

**Gate check**: after physics settling, verifies the stack is stable (all
objects above table, none drifted, near-zero velocity).  Collapsed stacks
trigger a re-reset with a new seed.

IMPORTANT: This module must be imported AFTER isaaclab AppLauncher is created.
"""

from __future__ import annotations

import random
from dataclasses import field
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils import configclass

from molmo_spaces_isaac.envs.asset_registry import AssetRegistry, _asset_category
from molmo_spaces_isaac.envs.franka_table_env import (
    TABLE_HEIGHT,
    FrankaTableEnv,
    FrankaTableEnvCfg,
)

# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------
STACK_MODES = ("homogeneous", "container_target", "flat_target")

_MODE_TARGET_CATEGORIES: dict[str, list[str]] = {
    "homogeneous": ["Plate", "Bowl"],
    "container_target": ["Bowl", "Plate", "Pot", "Vase_Flat", "Cup"],
    "flat_target": [
        "Cloth", "Vase_Flat", "CreditCard", "Cellphone",
        "Egg_Cracked", "Dish_Sponge", "Soap_Bar", "Keychain",
    ],
}

_MODE_STACK_CATEGORIES: dict[str, list[str]] = {
    "homogeneous": ["Plate", "Bowl"],
    "container_target": ["Plate", "Bowl"],
    "flat_target": ["Plate", "Bowl"],
}

_THOR_QUAT = (0.7071068, 0.7071068, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@configclass
class FrankaStackEnvCfg(FrankaTableEnvCfg):
    """Config for the Franka tabletop stack-retrieval task."""

    observation_space: int = 25

    # Task variant
    stack_mode: str = "flat_target"  # one of STACK_MODES

    # Number of same-category items stacked ON TOP of the target.
    stack_height: int = 2

    max_object_dim: float = 0.25
    asset_seed: int = 42

    # Gate check
    gate_settle_steps: int = 30
    gate_min_z: float = TABLE_HEIGHT - 0.05
    gate_xy_tol: float = 0.06
    gate_max_vel: float = 0.10
    gate_max_angvel: float = 0.5
    gate_max_retries: int = 5

    # Minimum XZ footprint for stack items.
    min_stack_footprint: float = 0.10

    # Populated in __post_init__
    target_cfg: RigidObjectCfg | None = None  # type: ignore[assignment]
    stack_cfgs: list[tuple[str, RigidObjectCfg]] = field(default_factory=list)  # type: ignore[assignment]
    _target_height: float = 0.05
    _stack_z_offsets: list[float] = field(default_factory=list)  # type: ignore[assignment]
    _homogeneous_cat: str = ""

    def __post_init__(self) -> None:
        if self.stack_mode not in STACK_MODES:
            raise ValueError(
                f"Unknown stack_mode={self.stack_mode!r}, expected one of {STACK_MODES}"
            )

        import numpy as np

        registry = AssetRegistry(max_dim=self.max_object_dim)
        rng = np.random.default_rng(self.asset_seed)

        target_id = self._select_target(registry, rng)
        self._build_stack_cfgs(registry, rng, target_id)

    # -- Target selection (mode-aware) ------------------------------------

    def _select_target(self, registry: AssetRegistry, rng) -> str:
        target_categories = _MODE_TARGET_CATEGORIES[self.stack_mode]

        if self.stack_mode == "homogeneous":
            return self._select_target_homogeneous(registry, rng, target_categories)
        else:
            return self._select_target_generic(registry, rng, target_categories)

    def _select_target_homogeneous(self, registry: AssetRegistry, rng, categories: list[str]) -> str:
        """Pick a category with enough variants for target + full stack."""
        for cat in rng.permutation(categories):
            assets = registry.assets_in_category(cat)
            suitable = [
                a for a in assets
                if min(registry.bbox(a)[0], registry.bbox(a)[2]) >= self.min_stack_footprint
            ]
            if len(suitable) >= self.stack_height + 1:
                self._homogeneous_cat = cat
                target_id = suitable[int(rng.integers(len(suitable)))]
                self._set_target_cfg(target_id, registry)
                return target_id

        # Fallback: pick whichever category has the most
        best_cat, best_assets = categories[0], []
        for cat in categories:
            assets = registry.assets_in_category(cat)
            suitable = [
                a for a in assets
                if min(registry.bbox(a)[0], registry.bbox(a)[2]) >= self.min_stack_footprint
            ]
            if len(suitable) > len(best_assets):
                best_cat, best_assets = cat, suitable

        self._homogeneous_cat = best_cat
        target_id = best_assets[int(rng.integers(len(best_assets)))]
        self._set_target_cfg(target_id, registry)
        return target_id

    def _select_target_generic(self, registry: AssetRegistry, rng, categories: list[str]) -> str:
        """Pick any asset from the target categories."""
        for cat in rng.permutation(categories):
            candidates = registry.assets_in_category(cat)
            if candidates:
                target_id = candidates[int(rng.integers(len(candidates)))]
                self._set_target_cfg(target_id, registry)
                return target_id

        # Fallback
        target_id, _ = registry.sample(rng=rng)
        self._set_target_cfg(target_id, registry)
        return target_id

    def _set_target_cfg(self, target_id: str, registry: AssetRegistry) -> None:
        target_path = registry.usd_path(target_id)
        target_bbox = registry.bbox(target_id)
        self._target_height = target_bbox[1]

        self.target_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/stack_target",
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

    # -- Stack items selection --------------------------------------------

    def _build_stack_cfgs(self, registry: AssetRegistry, rng, target_id: str) -> None:
        if self.stack_mode == "homogeneous":
            stack_cat = self._homogeneous_cat
        else:
            # Pick a stack category with enough suitable assets
            stack_cat = None
            stack_categories = _MODE_STACK_CATEGORIES[self.stack_mode]
            for cat in rng.permutation(stack_categories):
                assets = registry.assets_in_category(cat)
                suitable = [
                    a for a in assets
                    if min(registry.bbox(a)[0], registry.bbox(a)[2]) >= self.min_stack_footprint
                ]
                if len(suitable) >= self.stack_height:
                    stack_cat = cat
                    break
            if stack_cat is None:
                best_cat, best_count = stack_categories[0], 0
                for cat in stack_categories:
                    count = sum(
                        1 for a in registry.assets_in_category(cat)
                        if min(registry.bbox(a)[0], registry.bbox(a)[2]) >= self.min_stack_footprint
                    )
                    if count > best_count:
                        best_cat, best_count = cat, count
                stack_cat = best_cat

        suitable_assets = [
            a for a in registry.assets_in_category(stack_cat)
            if min(registry.bbox(a)[0], registry.bbox(a)[2]) >= self.min_stack_footprint
        ]

        self.stack_cfgs = []
        self._stack_z_offsets = []
        cumulative_z = self._target_height
        used: set[str] = {target_id}  # exclude target from stack pool

        for i in range(self.stack_height):
            available = [a for a in suitable_assets if a not in used]
            if available:
                s_id = available[int(rng.integers(len(available)))]
            else:
                s_id = suitable_assets[int(rng.integers(len(suitable_assets)))]
            used.add(s_id)

            s_path = registry.usd_path(s_id)
            s_bbox = registry.bbox(s_id)
            item_height = s_bbox[1]
            z_off = 0.07 + cumulative_z + item_height / 2.0

            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/stack_item_{i}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=s_path.as_posix(),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=1.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.5, 0.0, TABLE_HEIGHT + z_off),
                    rot=_THOR_QUAT,
                ),
            )
            self.stack_cfgs.append((f"stack_item_{i}", cfg))
            self._stack_z_offsets.append(z_off)
            cumulative_z += item_height


# ---------------------------------------------------------------------------
# Environment (identical for all modes)
# ---------------------------------------------------------------------------
class FrankaStackEnv(FrankaTableEnv):
    """Franka Panda retrieves a target from under a vertical stack.

    Layout (bottom to top): table → target → stack_item_0 → stack_item_1 → …
    """

    cfg: FrankaStackEnvCfg

    def __init__(self, cfg: FrankaStackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._reset_counter = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._gate_retry_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._gate_accepted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stack_centre_xy = torch.zeros(self.num_envs, 2, device=self.device)

    def _setup_task_objects(self) -> None:
        self._target = RigidObject(self.cfg.target_cfg)
        self.scene.rigid_objects["stack_target"] = self._target

        self._stack_items: list[RigidObject] = []
        for name, cfg in self.cfg.stack_cfgs:
            obj = RigidObject(cfg)
            self.scene.rigid_objects[name] = obj
            self._stack_items.append(obj)

    # ------------------------------------------------------------------
    # Gate check — stack stability
    # ------------------------------------------------------------------
    def _check_stack_unstable(self) -> torch.Tensor:
        """True for envs where the stack is unstable.

        Checks four conditions for every object in the stack:
        1. Fell below table (z too low)
        2. Drifted too far from stack centre (XY)
        3. Still moving (linear velocity above threshold)
        4. Still spinning (angular velocity above threshold)
        """
        unstable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        all_objects = [self._target] + self._stack_items
        for obj in all_objects:
            pos = obj.data.root_pos_w
            vel = obj.data.root_lin_vel_w
            ang_vel = obj.data.root_ang_vel_w

            fell = pos[:, 2] < self.cfg.gate_min_z
            obj_local_xy = pos[:, :2] - self.scene.env_origins[:, :2] - self._stack_centre_xy
            drifted = torch.norm(obj_local_xy, dim=-1) > self.cfg.gate_xy_tol
            moving = torch.norm(vel, dim=-1) > self.cfg.gate_max_vel
            spinning = torch.norm(ang_vel, dim=-1) > self.cfg.gate_max_angvel

            unstable |= fell | drifted | moving | spinning

        return unstable

    # ------------------------------------------------------------------
    # Observations
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
    # Rewards / dones
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        timed_out = self.episode_length_buf >= self.max_episode_length

        check_mask = (
            (self.episode_length_buf >= self.cfg.gate_settle_steps)
            & (~self._gate_accepted)
        )
        if check_mask.any():
            collapsed = self._check_stack_unstable()
            gate_failed = check_mask & collapsed

            if gate_failed.any():
                failed_ids = gate_failed.nonzero(as_tuple=False).squeeze(-1)
                self._gate_retry_count[failed_ids] += 1

                if self.cfg.gate_max_retries > 0:
                    exceeded = self._gate_retry_count[failed_ids] > self.cfg.gate_max_retries
                    if exceeded.any():
                        exceeded_ids = failed_ids[exceeded]
                        print(
                            f"[StackEnv] Gate max retries ({self.cfg.gate_max_retries}) "
                            f"exceeded for envs {exceeded_ids.tolist()}, accepting",
                            flush=True,
                        )
                        self._gate_accepted[exceeded_ids] = True
                        gate_failed[exceeded_ids] = False

                terminated |= gate_failed

        return terminated, timed_out

    # ------------------------------------------------------------------
    # Reset — place target on table, build stack on top
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self._reset_robot(env_ids)

        n = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]
        quat = torch.tensor(_THOR_QUAT, device=self.device).unsqueeze(0).expand(n, -1)
        zero_vel = torch.zeros(n, 6, device=self.device)

        self._reset_counter[env_ids] += 1
        self._gate_accepted[env_ids] = False

        # Per-env random stack centre
        for i in range(n):
            env_id = int(env_ids[i].item())
            counter = int(self._reset_counter[env_id].item())
            rng = random.Random(self.cfg.asset_seed + env_id * 997 + counter * 7919)
            cx = 0.5 + rng.uniform(-0.05, 0.05)
            cy = rng.uniform(-0.05, 0.05)
            self._stack_centre_xy[env_id, 0] = cx
            self._stack_centre_xy[env_id, 1] = cy

        cx_t = self._stack_centre_xy[env_ids, 0]
        cy_t = self._stack_centre_xy[env_ids, 1]

        # Place target at the bottom
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = cx_t
        pos[:, 1] = cy_t
        pos[:, 2] = TABLE_HEIGHT + 0.07
        self._target.write_root_pose_to_sim(
            torch.cat([pos + env_origins, quat], dim=-1), env_ids=env_ids
        )
        self._target.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        # Stack items on top, ascending
        for (name, cfg), item, z_off in zip(
            self.cfg.stack_cfgs, self._stack_items, self.cfg._stack_z_offsets
        ):
            pos_i = torch.zeros(n, 3, device=self.device)
            pos_i[:, 0] = cx_t
            pos_i[:, 1] = cy_t
            pos_i[:, 2] = TABLE_HEIGHT + z_off
            item.write_root_pose_to_sim(
                torch.cat([pos_i + env_origins, quat], dim=-1), env_ids=env_ids
            )
            item.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
