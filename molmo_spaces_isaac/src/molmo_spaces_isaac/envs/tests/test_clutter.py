"""Smoke-test for FrankaClutterEnv: create the env, reset, step, and record video.

Usage:
    python -m molmo_spaces_isaac.envs.tests.test_clutter --headless --num_envs 4
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test FrankaClutterEnv")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--save_dir", type=str, default="output/test_clutter")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Always enable cameras for video recording in headless mode
if args.headless:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- Now safe to import IsaacLab modules ----
import gymnasium as gym
import torch

from molmo_spaces_isaac.envs.franka_clutter_env import FrankaClutterEnv, FrankaClutterEnvCfg


def main():
    cfg = FrankaClutterEnvCfg()
    cfg.scene.num_envs = args.num_envs

    print(f"[Test] Creating FrankaClutterEnv with {args.num_envs} envs ...", flush=True)
    print(f"[Test] Object pool: {len(cfg.object_cfgs)} total "
          f"(target={cfg.pool_targets}, fragile={cfg.pool_fragile}, "
          f"clutter={cfg.pool_clutter})", flush=True)
    print(f"[Test] Active per episode: target={cfg.active_targets}, "
          f"fragile={cfg.active_fragile}, clutter={cfg.active_clutter}", flush=True)
    for idx, (prim, obj_cfg) in enumerate(cfg.object_cfgs):
        role = cfg.object_roles[idx]
        fname = obj_cfg.spawn.usd_path.split("/")[-1]
        print(f"  {prim} ({role}): {fname}", flush=True)

    try:
        env = FrankaClutterEnv(cfg, render_mode="rgb_array")
    except Exception as e:
        print(f"[Test] ERROR creating env: {e}", flush=True)
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return

    # Wrap with video recorder — timestamped subfolder to avoid overwrites
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = os.path.join(
        args.save_dir, "videos", f"{timestamp}_envs{args.num_envs}_steps{args.steps}"
    )
    os.makedirs(video_dir, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,  # record from the start
        video_length=args.steps,
        disable_logger=True,
    )

    print("[Test] Resetting ...", flush=True)
    obs, info = env.reset()
    print(f"[Test] Obs shape: {obs['policy'].shape}", flush=True)

    print(f"[Test] Stepping {args.steps} times (recording video) ...", flush=True)
    total_resets = 0
    for i in range(args.steps):
        action = torch.zeros(args.num_envs, cfg.action_space, device="cuda:0")
        obs, reward, terminated, truncated, info = env.step(action)
        n_term = terminated.sum().item() if hasattr(terminated, 'sum') else 0
        if n_term > 0:
            total_resets += int(n_term)
            print(f"  step {i}: {int(n_term)} env(s) terminated (gate fail or timeout)", flush=True)
        elif i % 50 == 0:
            print(f"  step {i}/{args.steps}", flush=True)
    print(f"[Test] Total env resets during run: {total_resets}", flush=True)

    print(f"[Test] Done — video saved to {video_dir}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
