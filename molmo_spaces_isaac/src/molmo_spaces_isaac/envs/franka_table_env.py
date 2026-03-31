"""Base tabletop environment: Franka Panda at a table with ground plane and lighting.

Subclasses add task-specific objects (via ``_setup_task_objects``), observations,
rewards, termination, and reset randomization.

IMPORTANT: This module must be imported AFTER isaaclab AppLauncher is created.
"""

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TABLE_HEIGHT = 0.4
TABLE_SIZE = (0.6, 0.8, TABLE_HEIGHT)


# ---------------------------------------------------------------------------
# Base configuration
# ---------------------------------------------------------------------------
@configclass
class FrankaTableEnvCfg(DirectRLEnvCfg):
    """Shared config for all Franka tabletop tasks."""

    # -- sim --
    decimation: int = 2
    episode_length_s: float = 8.0
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=3.0)

    # -- spaces (subclasses must override) --
    action_space: int = 9   # 7 arm + 2 finger
    observation_space: int = 0
    state_space: int = 0

    # -- robot (IsaacLab built-in Franka Panda) --
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.785,
                "panda_joint3": 0.0,
                "panda_joint4": -2.356,
                "panda_joint5": 0.0,
                "panda_joint6": 1.571,
                "panda_joint7": 0.785,
                "panda_finger_joint.*": 0.04,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # -- table --
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.45, 0.3),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, TABLE_HEIGHT / 2.0),
        ),
    )

    # -- ground plane --
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# Base environment
# ---------------------------------------------------------------------------
class FrankaTableEnv(DirectRLEnv):
    """Base Franka tabletop environment.

    Handles robot, table, terrain, lighting, and common action logic.
    Subclasses implement ``_setup_task_objects``, ``_get_observations``,
    ``_get_rewards``, ``_get_dones``, and ``_reset_idx``.
    """

    cfg: FrankaTableEnvCfg

    def __init__(self, cfg: FrankaTableEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Cache robot body / joint indices
        self._hand_idx = self._robot.find_bodies("panda_hand")[0][0]
        self._finger_l_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self._finger_r_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self._arm_joint_ids = list(range(7))
        self._finger_joint_ids = [
            self._robot.find_joints("panda_finger_joint1")[0][0],
            self._robot.find_joints("panda_finger_joint2")[0][0],
        ]

        # Action buffer
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        # Robot
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # Table
        self._table = RigidObject(self.cfg.table_cfg)
        self.scene.rigid_objects["table"] = self._table

        # Task-specific objects (subclass hook)
        self._setup_task_objects()

        # Ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone environments and filter collisions
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_task_objects(self) -> None:
        """Subclasses override this to add task-specific rigid/articulated objects."""

    # ------------------------------------------------------------------
    # Actions (shared across all tabletop tasks)
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # Arm: delta joint position control
        arm_targets = self._robot.data.joint_pos[:, :7] + self._actions[:, :7] * 0.05
        self._robot.set_joint_position_target(arm_targets, joint_ids=self._arm_joint_ids)

        # Fingers: absolute position mapped from [-1, 1] → [0, 0.04]
        finger_pos = (self._actions[:, 7:9] + 1.0) * 0.02
        self._robot.set_joint_position_target(finger_pos, joint_ids=self._finger_joint_ids)

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------
    def _reset_robot(self, env_ids: torch.Tensor) -> None:
        """Reset robot joints and root pose for the given env IDs."""
        n = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]

        default_joint_pos = self._robot.data.default_joint_pos[env_ids]
        default_joint_vel = torch.zeros_like(default_joint_pos)
        self._robot.set_joint_position_target(default_joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

        robot_pos = env_origins.clone()
        robot_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).unsqueeze(0).expand(n, -1)
        self._robot.write_root_pose_to_sim(
            torch.cat([robot_pos, robot_quat], dim=-1), env_ids=env_ids
        )
        self._robot.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids=env_ids
        )

    def _place_object_on_table(
        self,
        obj: RigidObject,
        env_ids: torch.Tensor,
        *,
        x_range: tuple[float, float] = (0.425, 0.575),
        y_range: tuple[float, float] = (-0.075, 0.075),
        z_offset: float = 0.07,
        quat: tuple[float, float, float, float] = (0.7071068, 0.7071068, 0.0, 0.0),
    ) -> None:
        """Randomise an object's position on the table surface."""
        n = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]

        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = x_range[0] + torch.rand(n, device=self.device) * (x_range[1] - x_range[0])
        pos[:, 1] = y_range[0] + torch.rand(n, device=self.device) * (y_range[1] - y_range[0])
        pos[:, 2] = TABLE_HEIGHT + z_offset

        q = torch.tensor(quat, device=self.device).unsqueeze(0).expand(n, -1)
        state = torch.cat([pos + env_origins, q], dim=-1)
        obj.write_root_pose_to_sim(state, env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)
