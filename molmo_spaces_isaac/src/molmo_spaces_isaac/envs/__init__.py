# Environment classes must be imported AFTER IsaacSim AppLauncher is created.
# Use lazy imports:
#
#   from isaaclab.app import AppLauncher
#   ...  # create AppLauncher
#
#   from molmo_spaces_isaac.envs.franka_pickup_env import FrankaPickupEnv, FrankaPickupEnvCfg
#   from molmo_spaces_isaac.envs.franka_clutter_env import FrankaClutterEnv, FrankaClutterEnvCfg
#   from molmo_spaces_isaac.envs.franka_stack_env import FrankaStackEnv, FrankaStackEnvCfg
#   from molmo_spaces_isaac.envs.franka_transfer_env import FrankaTransferEnv, FrankaTransferEnvCfg
#   from molmo_spaces_isaac.envs.franka_pinch_env import FrankaPinchEnv, FrankaPinchEnvCfg
#
# The AssetRegistry can be imported without AppLauncher:
#   from molmo_spaces_isaac.envs.asset_registry import AssetRegistry
