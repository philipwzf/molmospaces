"""Franka tabletop clutter scene: grasp a target among fragile and clutter objects.

Ported from ``task_generation/clutter_scene_pipeline.py``.  A pool of objects
is spawned in every env.  At each reset, a random subset is placed on the table
using ring-based greedy packing (target at centre, others on concentric rings);
the rest are parked underground so each parallel env gets a different scene
composition.

IMPORTANT: This module must be imported AFTER isaaclab AppLauncher is created.
"""

from __future__ import annotations

import math
import random
from dataclasses import field
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

# ---------------------------------------------------------------------------
# Object-role categories (mirroring OmniGibson pipeline pools)
# ---------------------------------------------------------------------------
TARGET_CATEGORIES = ["Cup", "Mug", "Apple", "Tomato", "Potato", "Egg"]
FRAGILE_CATEGORIES = ["Bowl", "Plate", "Vase_Flat", "Vase_Medium", "Vase_Open"]
CLUTTER_CATEGORIES = [
    "Candle", "Cellphone", "Alarm_Clock", "Soap_Bottle",
    "Pepper_Shaker", "Salt_Shaker", "Remote", "Pencil", "Pen",
]

# Y-up → Z-up rotation shared by all THOR assets
_THOR_QUAT = (0.7071068, 0.7071068, 0.0, 0.0)

# Table centre in local env coords and surface bounds for packing
_TABLE_CX = 0.5
_TABLE_CY = 0.0
_TABLE_HALF_X = TABLE_SIZE[0] / 2.0  # 0.3
_TABLE_HALF_Y = TABLE_SIZE[1] / 2.0  # 0.4
_EDGE_MARGIN = 0.04

# Z position to park inactive objects (well below ground)
_PARK_Z = -10.0


def _build_rigid_cfg(prim_suffix: str, usd_path: str) -> RigidObjectCfg:
    """Create a RigidObjectCfg for a THOR asset (parked underground initially)."""
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_suffix}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, _PARK_Z),
            rot=_THOR_QUAT,
        ),
    )


# ---------------------------------------------------------------------------
# Ring-based clutter packing (adapted from clutter_pack.py)
# ---------------------------------------------------------------------------
def _effective_radius(half_xy: tuple[float, float]) -> float:
    return math.hypot(half_xy[0], half_xy[1])


def _collides(
    cx: float,
    cy: float,
    half_xy: tuple[float, float],
    placed: list[tuple[tuple[float, float], float, float]],
    min_clearance: float,
) -> bool:
    r = max(half_xy)
    for other_half, ox, oy in placed:
        other_r = max(other_half)
        if math.hypot(cx - ox, cy - oy) < r + other_r + min_clearance:
            return True
    return False


def _ring_candidates(
    half_xy: tuple[float, float],
    placed: list[tuple[tuple[float, float], float, float]],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    min_clearance: float,
    noise_margin: float = 0.02,
    n_angles: int = 36,
) -> list[tuple[float, float]]:
    """Sweep concentric rings outward, return positions on innermost free ring(s)."""
    r_new = _effective_radius(half_xy)
    (x0, y0), (x1, y1) = bounds
    x_lo, x_hi = min(x0, x1) + r_new, max(x0, x1) - r_new
    y_lo, y_hi = min(y0, y1) + r_new, max(y0, y1) - r_new
    if x_lo > x_hi or y_lo > y_hi:
        return []

    if not placed:
        cx = min(max(0.0, x_lo), x_hi)
        cy = min(max(0.0, y_lo), y_hi)
        return [(cx, cy)]

    min_ring_r = r_new + min_clearance
    for half_p, px, py in placed:
        if math.hypot(px, py) < 0.05:
            min_ring_r = max(min_ring_r, _effective_radius(half_p) + r_new + min_clearance)

    ring_step = max(0.005, 0.5 * r_new)
    max_radius = math.hypot(max(abs(x_lo), abs(x_hi)), max(abs(y_lo), abs(y_hi)))

    best_r: float | None = None
    pool: list[tuple[float, float]] = []
    ring_r = min_ring_r

    while ring_r <= max_radius:
        ring_valid: list[tuple[float, float]] = []
        for i in range(n_angles):
            theta = 2.0 * math.pi * i / n_angles
            cx = ring_r * math.cos(theta)
            cy = ring_r * math.sin(theta)
            if cx < x_lo or cx > x_hi or cy < y_lo or cy > y_hi:
                continue
            if not _collides(cx, cy, half_xy, placed, min_clearance):
                ring_valid.append((cx, cy))

        if ring_valid:
            if best_r is None:
                best_r = ring_r
            if ring_r <= best_r + noise_margin + 1e-12:
                pool.extend(ring_valid)
            else:
                break
        elif best_r is not None and ring_r > best_r + noise_margin:
            break

        ring_r += ring_step

    return pool


def run_ring_pack(
    half_extents: list[tuple[float, float]],
    roles: list[str],
    seed: int,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    min_clearance: float = 0.025,
    jitter_xy: float = 0.015,
) -> list[tuple[float, float]]:
    """Compute ring-packed (x, y) positions relative to table centre.

    Target goes near centre; remaining objects are placed on the innermost
    collision-free ring.  Returns positions in the same order as inputs.
    """
    rng = random.Random(seed)

    role_priority = {"target": 0, "fragile": 1, "clutter": 2}
    indices = sorted(range(len(roles)), key=lambda i: (role_priority.get(roles[i], 3), i))

    placed: list[tuple[tuple[float, float], float, float]] = []
    result_xy: list[tuple[float, float] | None] = [None] * len(roles)

    for idx in indices:
        half = half_extents[idx]

        if not placed and roles[idx] == "target":
            j = min(jitter_xy, 0.02)
            cx = rng.uniform(-j, j)
            cy = rng.uniform(-j, j)
            if not _collides(cx, cy, half, placed, min_clearance):
                result_xy[idx] = (cx, cy)
                placed.append((half, cx, cy))
                continue

        pool = _ring_candidates(half, placed, bounds, min_clearance)
        if not pool:
            (x0, y0), (x1, y1) = bounds
            r = _effective_radius(half)
            cx = rng.uniform(min(x0, x1) + r, max(x0, x1) - r)
            cy = rng.uniform(min(y0, y1) + r, max(y0, y1) - r)
            result_xy[idx] = (cx, cy)
            placed.append((half, cx, cy))
            continue

        cx, cy = rng.choice(pool)
        result_xy[idx] = (cx, cy)
        placed.append((half, cx, cy))

    return result_xy  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@configclass
class FrankaClutterEnvCfg(FrankaTableEnvCfg):
    """Config for the Franka tabletop clutter task.

    A *pool* of objects is spawned in every env.  At each reset a random
    subset is activated (placed on the table) and the rest are parked
    underground, so each parallel env gets a different scene composition.

    **Gate check**: after ``gate_settle_steps`` physics steps the env
    checks whether every active object is still above the table surface.
    If any fell, the episode is immediately terminated so that
    ``_reset_idx`` is called again with a different seed.
    """

    observation_space: int = 25

    # Pool sizes — total objects spawned per env (superset).
    pool_targets: int = 3
    pool_fragile: int = 5
    pool_clutter: int = 8

    # Active counts — how many of each role are placed on the table per episode.
    active_targets: int = 1
    active_fragile: int = 3
    active_clutter: int = 4

    # Packing parameters
    max_object_dim: float = 0.20
    min_clearance: float = 0.020

    # Gate check: physics steps to wait for settling, then validate.
    gate_settle_steps: int = 10
    # Minimum Z for an active object to be considered "on the table".
    # Slightly below TABLE_HEIGHT to tolerate small settling offsets.
    gate_min_z: float = TABLE_HEIGHT - 0.05
    # Maximum number of consecutive gate failures before giving up
    # (0 = unlimited retries).
    gate_max_retries: int = 5

    # Random seed for asset selection.
    asset_seed: int = 42

    # Populated in __post_init__
    object_cfgs: list[tuple[str, RigidObjectCfg]] = field(default_factory=list)  # type: ignore[assignment]
    object_roles: list[str] = field(default_factory=list)  # type: ignore[assignment]
    object_half_extents: list[tuple[float, float]] = field(default_factory=list)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        import numpy as np

        registry = AssetRegistry(max_dim=self.max_object_dim)
        rng = np.random.default_rng(self.asset_seed)

        self.object_cfgs = []
        self.object_roles = []
        self.object_half_extents = []

        def _pick_pool(role: str, categories: list[str], count: int) -> None:
            used: set[str] = set()
            for i in range(count):
                for cat in rng.permutation(categories):
                    try:
                        asset_id, path = registry.sample(
                            cat, rng=rng, exclude=used,
                        )
                        break
                    except RuntimeError:
                        continue
                else:
                    asset_id, path = registry.sample(rng=rng, exclude=used)
                used.add(asset_id)
                bbox = registry.bbox(asset_id)
                half_xy = (bbox[0] / 2.0, bbox[2] / 2.0)
                prim = f"{role}_{i}"
                cfg = _build_rigid_cfg(prim, path.as_posix())
                self.object_cfgs.append((prim, cfg))
                self.object_roles.append(role)
                self.object_half_extents.append(half_xy)

        _pick_pool("target", TARGET_CATEGORIES, self.pool_targets)
        _pick_pool("fragile", FRAGILE_CATEGORIES, self.pool_fragile)
        _pick_pool("clutter", CLUTTER_CATEGORIES, self.pool_clutter)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class FrankaClutterEnv(FrankaTableEnv):
    """Franka Panda grasps a target on a cluttered table.

    Each parallel env gets a different random subset of the object pool
    placed on the table at every reset.  After a short physics settling
    period the gate check verifies all active objects remain on the table;
    failed envs are immediately terminated (triggering a fresh reset with
    a new seed).
    """

    cfg: FrankaClutterEnvCfg

    def __init__(self, cfg: FrankaClutterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Per-env tracking for gate check
        self._active_per_env: dict[int, list[int]] = {}  # env_id → active pool indices
        # Monotonically increasing reset counter per env — ensures every reset
        # gets a unique seed even when the gate passes (fixes same-layout bug).
        self._reset_counter = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        # Consecutive gate failures (reset when gate passes).
        self._gate_retry_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def _setup_task_objects(self) -> None:
        self._task_objects: list[RigidObject] = []
        for prim_name, obj_cfg in self.cfg.object_cfgs:
            obj = RigidObject(obj_cfg)
            self.scene.rigid_objects[prim_name] = obj
            self._task_objects.append(obj)

        # Build per-role index lists for fast sampling at reset
        self._target_indices = [
            i for i, r in enumerate(self.cfg.object_roles) if r == "target"
        ]
        self._fragile_indices = [
            i for i, r in enumerate(self.cfg.object_roles) if r == "fragile"
        ]
        self._clutter_indices = [
            i for i, r in enumerate(self.cfg.object_roles) if r == "clutter"
        ]

    # ------------------------------------------------------------------
    # Gate check
    # ------------------------------------------------------------------
    def _check_any_active_fell(self) -> torch.Tensor:
        """Return a bool tensor: True for envs where any active object is below the table."""
        fell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for env_id, active_indices in self._active_per_env.items():
            for j in active_indices:
                z = self._task_objects[j].data.root_pos_w[env_id, 2]
                if z < self.cfg.gate_min_z:
                    fell[env_id] = True
                    break

        return fell

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        hand_pos_w = self._robot.data.body_pos_w[:, self._hand_idx]
        hand_quat_w = self._robot.data.body_quat_w[:, self._hand_idx]
        joint_pos = self._robot.data.joint_pos[:, :7]
        joint_vel = self._robot.data.joint_vel[:, :7]

        # Use the first target in the pool for relative position.
        target_obj = self._task_objects[self._target_indices[0]]
        target_rel = target_obj.data.root_pos_w - hand_pos_w

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

        # Gate check: after settle window, continuously check if any active
        # object has fallen off the table.  If so, terminate immediately
        # to trigger a re-reset with a new seed.
        past_settle = self.episode_length_buf >= self.cfg.gate_settle_steps
        if past_settle.any():
            fell = self._check_any_active_fell()
            gate_failed = past_settle & fell

            if gate_failed.any():
                failed_ids = gate_failed.nonzero(as_tuple=False).squeeze(-1)
                self._gate_retry_count[failed_ids] += 1

                # Give up after max_retries to avoid infinite loops
                if self.cfg.gate_max_retries > 0:
                    exceeded = self._gate_retry_count[failed_ids] > self.cfg.gate_max_retries
                    if exceeded.any():
                        exceeded_ids = failed_ids[exceeded]
                        print(
                            f"[ClutterEnv] Gate max retries ({self.cfg.gate_max_retries}) "
                            f"exceeded for envs {exceeded_ids.tolist()}, "
                            f"accepting placement",
                            flush=True,
                        )
                        # Don't terminate these — let them run
                        gate_failed[exceeded_ids] = False

                terminated |= gate_failed

        return terminated, timed_out

    # ------------------------------------------------------------------
    # Reset — sample a subset, ring-pack on table, park the rest
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self._reset_robot(env_ids)

        n = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]
        quat_t = torch.tensor(_THOR_QUAT, device=self.device).unsqueeze(0)
        zero_vel = torch.zeros(1, 6, device=self.device)

        pack_bounds = (
            (-_TABLE_HALF_X + _EDGE_MARGIN, -_TABLE_HALF_Y + _EDGE_MARGIN),
            (_TABLE_HALF_X - _EDGE_MARGIN, _TABLE_HALF_Y - _EDGE_MARGIN),
        )

        for i in range(n):
            env_id = int(env_ids[i].item())
            self._reset_counter[env_id] += 1
            counter = int(self._reset_counter[env_id].item())
            rng = random.Random(
                self.cfg.asset_seed + env_id * 997 + counter * 7919
            )

            # Sample active indices from the pool
            active_t = rng.sample(self._target_indices, self.cfg.active_targets)
            active_f = rng.sample(self._fragile_indices, self.cfg.active_fragile)
            active_c = rng.sample(self._clutter_indices, self.cfg.active_clutter)
            active_list = active_t + active_f + active_c
            active_set = set(active_list)

            # Track active objects for gate check
            self._active_per_env[env_id] = active_list

            # Build ring-pack input for active objects only
            active_halves = [self.cfg.object_half_extents[j] for j in active_list]
            active_roles = [self.cfg.object_roles[j] for j in active_list]

            packed_xy = run_ring_pack(
                active_halves,
                active_roles,
                seed=rng.randint(0, 2**31),
                bounds=pack_bounds,
                min_clearance=self.cfg.min_clearance,
            )

            origin = env_origins[i: i + 1]
            eid = env_ids[i: i + 1]

            # Place active objects on table
            for j_active, (rx, ry) in zip(active_list, packed_xy):
                obj = self._task_objects[j_active]
                pos = torch.tensor(
                    [[_TABLE_CX + rx, _TABLE_CY + ry, TABLE_HEIGHT + 0.07]],
                    device=self.device,
                )
                state = torch.cat([pos + origin, quat_t], dim=-1)
                obj.write_root_pose_to_sim(state, env_ids=eid)
                obj.write_root_velocity_to_sim(zero_vel, env_ids=eid)

            # Park inactive objects underground
            for j_pool in range(len(self._task_objects)):
                if j_pool in active_set:
                    continue
                obj = self._task_objects[j_pool]
                pos = torch.tensor(
                    [[0.0, 0.0, _PARK_Z]], device=self.device,
                )
                state = torch.cat([pos + origin, quat_t], dim=-1)
                obj.write_root_pose_to_sim(state, env_ids=eid)
                obj.write_root_velocity_to_sim(zero_vel, env_ids=eid)
