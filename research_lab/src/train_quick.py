"""
Quick PPO Baseline Training — Colab-compatible script.

This trains a single PPO agent on the full dataset with no walk-forward splitting.
Purpose: prove the TradingEnv works before running autoresearch on Colab.

Run on Colab:
1. Upload market_data.parquet and market_data.npy
2. pip install gymnasium stable-baselines3 torch
3. Run this script

Saves model to: models/ppo_baseline.pt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

# Ensure parent is on path for TradingEnv
_parent = Path(__file__).resolve().parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from trading_env import TradingEnv


class EpisodeLogger(BaseCallback):
    """Log reward and value predictions every N steps."""

    def __init__(self, verbose: int = 0, log_every: int = 10000):
        super().__init__(verbose)
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every == 0:
            rewards = [ep["r"] for ep in self.model.ep_info_buffer] if self.model.ep_info_buffer else [0]
            print(f"  Step {self.n_calls:,} | Mean reward: {np.mean(rewards):.4f}")
        return True


def make_env(data: pd.DataFrame, close_prices: np.ndarray) -> DummyVecEnv:
    """Create a vectorised environment for SB3."""

    def env_fn():
        return TradingEnv(
            data=data,
            close_prices=close_prices,
            initial_balance=10_000.0,
            commission_pct=0.001,
            slippage_pct=0.001,
            max_position_size=0.1,
        )

    return DummyVecEnv([env_fn])


def train(
    data: pd.DataFrame,
    close_prices: np.ndarray,
    total_timesteps: int,
    device: str,
) -> PPO:
    """Train a PPO agent and return the model."""
    print(f"Data: {data.shape[1]} features, {len(data)} rows, {len(close_prices)} prices")
    print(f"Device: {device}")
    print(f"Training for {total_timesteps:,} timesteps...")

    vec_env = make_env(data, close_prices)
    
    config = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "total_timesteps": total_timesteps,
    }

    run = wandb.init(
        project="quant-fund-rl",
        config=config,
        sync_tensorboard=True,
    )

    model = PPO(
        policy=config["policy"],
        env=vec_env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        verbose=1,
        device=device,
        tensorboard_log=f"./ppo_tensorboard/{run.id}" if run else "./ppo_tensorboard/",
    )

    callbacks = [EpisodeLogger(log_every=5000)]
    if run:
        callbacks.append(WandbCallback(gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2))

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
    )
    
    if run:
        run.finish()

    return model


def evaluate(
    model: PPO, data: pd.DataFrame, close_prices: np.ndarray, n_episodes: int = 10
) -> tuple[float, float]:
    """Evaluate the trained model on held-out data."""
    vec_env = make_env(data, close_prices)
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=n_episodes)
    print(f"\nEvaluation: reward = {mean_reward:.4f} +/- {std_reward:.4f} ({n_episodes} episodes)")
    return mean_reward, std_reward


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Quick PPO baseline training")
    parser.add_argument("--data", default="data/market_data.parquet", help="Path to features parquet")
    parser.add_argument("--close-prices", default="data/market_data.npy", help="Path to close prices .npy")
    parser.add_argument("--model-out", default="../models/ppo_baseline.pt", help="Output model path")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Training steps")
    parser.add_argument("--train-pct", type=float, default=0.8, help="Fraction of data for training")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load features and close prices
    data = pd.read_parquet(args.data)
    close_prices = np.load(args.close_prices)
    print(f"Loaded {len(data)} feature rows, {len(close_prices)} close prices")

    assert len(data) == len(close_prices), (
        f"Feature rows ({len(data)}) != close prices ({len(close_prices)})"
    )

    # Train/test split
    split_idx = int(len(data) * args.train_pct)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    train_prices = close_prices[:split_idx]
    test_prices = close_prices[split_idx:]

    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

    # Train
    model = train(train_data, train_prices, total_timesteps=args.timesteps, device=device)

    # Evaluate
    evaluate(model, test_data, test_prices)

    # Save
    output_path = Path(args.model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"\nModel saved to {output_path}")

    # Run one human-readable episode
    print("\n--- Sample Episode ---")
    env = TradingEnv(
        test_data,
        close_prices=test_prices,
        initial_balance=10_000.0,
        commission_pct=0.001,
        slippage_pct=0.001,
        max_position_size=0.1,
        render_mode="human",
    )
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.render()


if __name__ == "__main__":
    main()
