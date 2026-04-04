"""
Walk-Forward PPO Training — Colab-compatible script.

This splits data into K rolling windows. For each window:
1. Train on window T
2. Validate on window T+1 (out-of-sample)
3. Record out-of-sample Sharpe as the fitness score

This is more realistic than single-split training and matches
how autoresearch would evaluate mutated strategies.

Run on Colab:
1. Upload market_data.parquet
2. pip install gymnasium stable-baselines3 torch matplotlib
3. Run this script

Saves best model to: models/best_walk_forward.pt
Saves results to: models/walk_forward_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch

# Ensure parent is on path for TradingEnv
_parent = Path(__file__).resolve().parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from trading_env import TradingEnv


def make_vec_env(data: pd.DataFrame, close_prices: np.ndarray) -> DummyVecEnv:
    """Create a vectorized environment for Stable-Baselines3."""

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


def compute_episodic_sharpe(
    model: PPO, data: pd.DataFrame, close_prices: np.ndarray, n_episodes: int = 5
) -> float:
    """Run episodes and compute mean Sharpe ratio as performance metric."""
    env = TradingEnv(
        data=data,
        close_prices=close_prices,
        initial_balance=10_000.0,
        commission_pct=0.001,
        slippage_pct=0.001,
        max_position_size=0.1,
    )

    sharpe_values = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if env._episode_returns and len(env._episode_returns) >= 10:
            returns = np.array(env._episode_returns)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
            sharpe_values.append(sharpe)

    return float(np.mean(sharpe_values)) if sharpe_values else 0.0


def walk_forward_train(
    data: pd.DataFrame,
    close_prices: np.ndarray,
    window_size: int = 10000,
    forward_size: int = 5000,
    timesteps: int = 30_000,
    device: str = "cpu",
) -> tuple[PPO | None, list[dict]]:
    """
    Walk-forward training loop.

    Returns the best model and a list of per-window results.
    """
    results = []
    best_sharpe = float("-inf")
    best_model = None

    n_windows = (len(data) - window_size) // forward_size
    print(f"Data length: {len(data)}, windows: {n_windows}")
    print(f"Each: train {window_size} -> validate {forward_size}")
    print(f"Timesteps per window: {timesteps:,}")

    for i in range(n_windows):
        start = i * forward_size
        train_end = start + window_size
        val_end = min(train_end + forward_size, len(data))

        train_data = data.iloc[start:train_end].copy()
        val_data = data.iloc[train_end:val_end].copy()
        train_prices = close_prices[start:train_end]
        val_prices = close_prices[train_end:val_end]

        if len(val_data) < 100:
            continue

        print(f"\n--- Window {i + 1}/{n_windows} ---")
        print(f"  Train: {start}-{train_end} ({len(train_data)} rows)")
        print(f"  Val:   {train_end}-{val_end} ({len(val_data)} rows)")

        vec_env = make_vec_env(train_data, train_prices)
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            device=device,
        )

        print("  Training...", end=" ", flush=True)
        model.learn(total_timesteps=timesteps)
        print("Done")

        # Evaluate out-of-sample
        val_sharpe = compute_episodic_sharpe(model, val_data, val_prices)
        mean_reward, _ = evaluate_policy(model, make_vec_env(val_data, val_prices), n_eval_episodes=3)

        result = {
            "window": i + 1,
            "train_range": [start, train_end],
            "val_range": [train_end, val_end],
            "out_of_sample_sharpe": round(val_sharpe, 4),
            "mean_reward": round(mean_reward, 4),
        }
        results.append(result)

        print(f"  Out-of-sample Sharpe: {val_sharpe:.4f}")
        print(f"  Mean reward: {mean_reward:.4f}")

        if val_sharpe > best_sharpe:
            best_sharpe = val_sharpe
            best_model = model
            print(f"  ** New best model! **")

    return best_model, results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Walk-forward PPO training")
    parser.add_argument("--data", default="data/market_data.parquet", help="Path to features parquet")
    parser.add_argument("--close-prices", default="data/market_data.npy", help="Path to close prices .npy")
    parser.add_argument(
        "--model-out", default="../models/best_walk_forward.pt", help="Output model path"
    )
    parser.add_argument(
        "--results-out",
        default="../models/walk_forward_results.json",
        help="Output results JSON",
    )
    parser.add_argument("--window", type=int, default=10000, help="Training window size")
    parser.add_argument("--forward", type=int, default=5000, help="Validation window size")
    parser.add_argument("--timesteps", type=int, default=30_000, help="Steps per window")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = pd.read_parquet(args.data)
    close_prices = np.load(args.close_prices)
    print(f"Loaded: {len(data)} rows, {data.shape[1]} features, {len(close_prices)} prices")

    assert len(data) == len(close_prices), (
        f"Feature rows ({len(data)}) != close prices ({len(close_prices)})"
    )

    best_model, results = walk_forward_train(
        data,
        close_prices,
        window_size=args.window,
        forward_size=args.forward,
        timesteps=args.timesteps,
        device=device,
    )

    if best_model is None:
        print("\nNo valid model found — adjust window sizes or check data.")
        return

    # Save model
    output_path = Path(args.model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_model.save(str(output_path))
    print(f"\nBest model saved to {output_path}")

    # Save results
    results_path = Path(args.results_out)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {results_path}")

    # Summary
    print("\n=== Walk-Forward Summary ===")
    for r in results:
        print(
            f"  Window {r['window']:2d}: Sharpe={r['out_of_sample_sharpe']:.4f}, "
            f"Reward={r['mean_reward']:.4f}"
        )
    avg_sharpe = np.mean([r["out_of_sample_sharpe"] for r in results])
    print(f"  Average OOS Sharpe: {avg_sharpe:.4f}")
    print(f"  Best Sharpe: {max(r['out_of_sample_sharpe'] for r in results):.4f}")


if __name__ == "__main__":
    main()
