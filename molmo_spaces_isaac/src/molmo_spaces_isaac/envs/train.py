"""Train a Franka Panda to pick up a MolmoSpaces cup using PPO.

Prerequisites:
    1. pip install -e .[dev,sim]              (from molmo_spaces_isaac/)
    2. ms-download --type usd --install-dir assets/usd --assets thor

Usage:
    # Headless training with video recording
    python -m molmo_spaces_isaac.envs.train --num_envs 64 --max_iterations 100 \
        --headless --enable_cameras --record_video

    # Custom cup asset
    python -m molmo_spaces_isaac.envs.train --num_envs 256 --headless --enable_cameras \
        --record_video --cup_usd assets/usd/objects/thor/Bottle_1_mesh/Bottle_1_mesh.usda
"""

from __future__ import annotations

import argparse

# ---- AppLauncher MUST be created before any other Omniverse imports ----
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Franka pickup with MolmoSpaces assets")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel envs")
parser.add_argument("--max_iterations", type=int, default=1500, help="PPO iterations")
parser.add_argument("--cup_usd", type=str, default="", help="Override path to cup USD file")
parser.add_argument("--save_dir", type=str, default="output/franka_pickup", help="Log / checkpoint dir")
parser.add_argument("--horizon", type=int, default=24, help="Rollout horizon per iteration")
parser.add_argument("--record_video", action="store_true", help="Record videos during training")
parser.add_argument("--video_interval", type=int, default=50, help="Record a video every N iterations")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- Now safe to import everything else ----
import os

import gymnasium as gym
import torch
import torch.nn as nn

from molmo_spaces_isaac.envs.franka_pickup_env import FrankaPickupEnv, FrankaPickupEnvCfg

# ---------------------------------------------------------------------------
# Minimal PPO (self-contained, no external RL library needed)
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, act_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor):
        mean = self.actor_mean(obs)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(obs).squeeze(-1)
        return mean, std, value

    def act(self, obs: torch.Tensor):
        mean, std, value = self(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(-1, 1), log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, std, value = self(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy


def train_ppo(
    env,
    model: ActorCritic,
    max_iterations: int = 1500,
    horizon: int = 24,
    mini_epochs: int = 5,
    mini_batch_size: int = 512,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    lr: float = 3e-4,
    save_dir: str = "output/franka_pickup",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Get underlying env for num_envs/device (may be wrapped by RecordVideo)
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    device = base_env.device
    num_envs = base_env.num_envs

    os.makedirs(save_dir, exist_ok=True)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    for iteration in range(max_iterations):
        # ---- Rollout ----
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        for _ in range(horizon):
            with torch.no_grad():
                action, log_prob, value = model.act(obs)

            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(log_prob)
            val_buf.append(value)

            obs_dict, reward, terminated, truncated, _ = env.step(action)
            obs = obs_dict["policy"]
            rew_buf.append(reward)
            done_buf.append(terminated | truncated)

        obs_t = torch.stack(obs_buf)
        act_t = torch.stack(act_buf)
        logp_t = torch.stack(logp_buf)
        rew_t = torch.stack(rew_buf)
        val_t = torch.stack(val_buf)
        done_t = torch.stack(done_buf)

        # ---- GAE ----
        with torch.no_grad():
            _, _, last_val = model(obs)
        advantages = torch.zeros_like(rew_t)
        last_gae = torch.zeros(num_envs, device=device)
        for t in reversed(range(horizon)):
            next_val = val_t[t + 1] if t + 1 < horizon else last_val
            delta = rew_t[t] + gamma * next_val * (~done_t[t]).float() - val_t[t]
            last_gae = delta + gamma * lam * (~done_t[t]).float() * last_gae
            advantages[t] = last_gae
        returns = advantages + val_t

        B = horizon * num_envs
        obs_flat = obs_t.reshape(B, -1)
        act_flat = act_t.reshape(B, -1)
        logp_flat = logp_t.reshape(B)
        adv_flat = advantages.reshape(B)
        ret_flat = returns.reshape(B)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # ---- PPO update ----
        total_loss = 0.0
        for _ in range(mini_epochs):
            perm = torch.randperm(B, device=device)
            for start in range(0, B, mini_batch_size):
                idx = perm[start : start + mini_batch_size]
                new_logp, new_val, entropy = model.evaluate(obs_flat[idx], act_flat[idx])
                ratio = (new_logp - logp_flat[idx]).exp()

                surr1 = ratio * adv_flat[idx]
                surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_flat[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (new_val - ret_flat[idx]).pow(2).mean()
                entropy_loss = -entropy.mean() * 0.01

                loss = policy_loss + value_loss + entropy_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

        mean_rew = rew_t.sum(dim=0).mean().item() / horizon
        if iteration % 10 == 0:
            print(
                f"[Iter {iteration:4d}/{max_iterations}]  "
                f"mean_reward={mean_rew:.4f}  loss={total_loss:.4f}"
            )

        if (iteration + 1) % 200 == 0 or iteration == max_iterations - 1:
            ckpt_path = os.path.join(save_dir, f"ckpt_{iteration + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = FrankaPickupEnvCfg()
    cfg.scene.num_envs = args.num_envs

    if args.cup_usd:
        cfg.cup_usd_path = args.cup_usd
        cfg.__post_init__()

    env = FrankaPickupEnv(cfg, render_mode="rgb_array")

    # Wrap with video recorder if requested
    if args.record_video:
        video_dir = os.path.join(args.save_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step % (args.video_interval * 24) == 0,  # every N iters * horizon
            video_length=200,  # ~200 steps per clip
            disable_logger=True,
        )
        print(f"Recording videos to: {video_dir}")

    model = ActorCritic(
        obs_dim=cfg.observation_space,
        act_dim=cfg.action_space,
    ).to(env.unwrapped.device)

    print(f"Training on {cfg.scene.num_envs} envs | device={env.unwrapped.device}")
    print(f"Cup USD: {cfg.cup_usd_path}")

    train_ppo(env, model, max_iterations=args.max_iterations, horizon=args.horizon, save_dir=args.save_dir)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
