"""Smoke-test for FrankaStackEnv: create, reset, step, and record video.

Usage:
    python -m molmo_spaces_isaac.envs.tests.test_stack --headless --num_envs 4 --mode homogeneous
    python -m molmo_spaces_isaac.envs.tests.test_stack --headless --num_envs 4 --mode container_target
    python -m molmo_spaces_isaac.envs.tests.test_stack --headless --num_envs 4 --mode flat_target
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test FrankaStackEnv")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--mode", type=str, default="flat_target",
                    choices=["homogeneous", "container_target", "flat_target"])
parser.add_argument("--save_dir", type=str, default="output/test_stack")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.headless:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from molmo_spaces_isaac.envs.franka_stack_env import FrankaStackEnv, FrankaStackEnvCfg


def main():
    cfg = FrankaStackEnvCfg(stack_mode=args.mode)
    cfg.scene.num_envs = args.num_envs

    print(f"[Test] Creating FrankaStackEnv with {args.num_envs} envs ...", flush=True)
    print(f"[Test] Mode: {args.mode}", flush=True)
    print(f"[Test] Stack height: {cfg.stack_height} items on top of target", flush=True)
    print(f"[Test] Target: {cfg.target_cfg.spawn.usd_path.split('/')[-1]}", flush=True)
    for name, item_cfg in cfg.stack_cfgs:
        print(f"  {name}: {item_cfg.spawn.usd_path.split('/')[-1]}", flush=True)

    try:
        env = FrankaStackEnv(cfg, render_mode="rgb_array")
    except Exception as e:
        print(f"[Test] ERROR creating env: {e}", flush=True)
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = os.path.join(
        args.save_dir, "videos", f"{timestamp}_{args.mode}_envs{args.num_envs}_steps{args.steps}"
    )
    os.makedirs(video_dir, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,
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
            print(f"  step {i}: {int(n_term)} env(s) gate-failed (stack collapsed)", flush=True)
        elif i % 100 == 0:
            print(f"  step {i}/{args.steps}", flush=True)

    print(f"[Test] Total gate resets: {total_resets}", flush=True)
    print(f"[Test] Done — video saved to {video_dir}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
