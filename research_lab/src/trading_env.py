"""
Gymnasium Trading Environment for RL Agent Training.

The agent observes market features and decides position sizing [-1.0, 1.0].
Negative = bearish (reduce position), positive = bullish (increase position).
Reward = Sharpe ratio - drawdown penalty - transaction costs.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class TradeRecord:
    """Records a single step in a trade episode."""
    step: int
    action: float
    price: float
    position: float
    portfolio_value: float
    reward: float
    done: bool


class TradingEnv(gym.Env):
    """
    Continuous-action trading environment compatible with Gymnasium.

    Observation space: [n_features] array of market features.
    Action space: Box(-1.0, 1.0) — controls portfolio allocation fraction.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        data: pd.DataFrame,
        close_prices: Optional[np.ndarray] = None,
        initial_balance: float = 10_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.001,
        max_position_size: float = 1.0,
        reward_shaping: Optional[Dict[str, float]] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.max_position_size = max_position_size
        self.render_mode = render_mode

        # Close prices for PnL — fallback: try 'close' column if not provided
        if close_prices is not None:
            self.close_prices = np.asarray(close_prices, dtype=np.float64)
        elif "close" in self.data.columns:
            self.close_prices = self.data["close"].values.astype(np.float64)
        else:
            raise ValueError(
                "close_prices must be provided when 'close' column is absent from data"
            )

        # Default reward configuration
        self.reward_shaping = reward_shaping or {
            "sharpe_weight": 1.0,
            "drawdown_penalty": 0.5,
            "transaction_cost_penalty": 0.01,
        }

        # Derive action/observation space shape from data
        self.n_features = int(data.shape[1])
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # State tracking
        self._current_step = 0
        self._balance = 0.0
        self._position = 0.0  # Number of coins/units held
        self._portfolio_value = 0.0
        self._entry_price = 0.0
        self._episode_returns: list[float] = []
        self._trade_log: list[TradeRecord] = []
        self._peak_value = 0.0
        self._max_drawdown = 0.0

    def _get_observation(self) -> np.ndarray:
        """Return current feature vector, clipped to avoid extreme values."""
        row = self.data.iloc[self._current_step]
        obs = row.values.astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _apply_slippage_for_buy(self, price: float) -> float:
        """Slippage on a buy — pay more than market price."""
        slip = float(np.random.uniform(0, self.slippage_pct))
        return price * (1.0 + slip)

    def _apply_slippage_for_sell(self, price: float) -> float:
        """Slippage on a sell — receive less than market price."""
        slip = float(np.random.uniform(0, self.slippage_pct))
        return price * (1.0 - slip)

    def _calculate_reward(self, step_return: float) -> float:
        """Composite reward: Sharpe - drawdown penalty - transaction cost penalty."""
        self._episode_returns.append(step_return)

        if len(self._episode_returns) < 10:
            return step_return

        returns_array = np.array(self._episode_returns)
        mean_ret = float(np.mean(returns_array))
        std_ret = float(np.std(returns_array))
        sharpe = mean_ret / std_ret if std_ret > 1e-8 else 0.0

        # Drawdown component
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = float(cumulative[-1] / running_max[-1]) - 1.0
        self._max_drawdown = min(self._max_drawdown, drawdown)

        # Transaction cost penalty
        txn_penalty = 0.0
        if len(self._trade_log) >= 2:
            action_diff = abs(self._trade_log[-1].action - self._trade_log[-2].action)
            if action_diff > 0.05:
                txn_penalty = action_diff * self.reward_shaping["transaction_cost_penalty"]

        reward = (
            self.reward_shaping["sharpe_weight"] * sharpe
            - self.reward_shaping["drawdown_penalty"] * abs(drawdown)
            - txn_penalty
        )
        return float(reward)

    def step(self, action: np.ndarray):  # noqa: ANN201
        """
        Execute one timestep: apply action, compute PnL with costs.

        Args:
            action: Box(-1, 1) — positive increases long position, negative reduces.
        """
        action_val = float(np.clip(action[0], -1.0, 1.0))

        current_price = float(self.close_prices[self._current_step])
        new_fill_price = 0.0

        # --- Position Management (incremental adjustment) ---
        prev_portfolio_value = self._portfolio_value

        # Action maps to target dollar exposure as fraction of balance
        # Positive → long, negative → short (only long supported for now)
        target_dollar_exposure = action_val * self._balance * self.max_position_size

        if target_dollar_exposure > self._balance * 0.001:
            # Calculate target units at current price
            buy_price = self._apply_slippage_for_buy(current_price)
            target_units = target_dollar_exposure / buy_price if buy_price > 0 else 0.0

            delta_units = target_units - self._position

            if abs(delta_units) > 1e-10:
                if delta_units > 0:
                    # Increase position — buy delta
                    cost = abs(delta_units) * buy_price
                    commission_cost = cost * self.commission_pct
                    self._balance -= cost + commission_cost
                    self._position = target_units
                    self._entry_price = (
                        (self._position - delta_units) * self._entry_price
                        + delta_units * buy_price
                    ) / self._position if self._position > 0 else buy_price
                else:
                    # Decrease position — sell delta
                    sell_price = self._apply_slippage_for_sell(current_price)
                    revenue = abs(delta_units) * sell_price
                    commission_cost = revenue * self.commission_pct
                    self._balance += revenue - commission_cost
                    self._position = target_units
        else:
            # Close position if one exists
            if abs(self._position) > 1e-10:
                sell_price = self._apply_slippage_for_sell(current_price)
                revenue = abs(self._position) * sell_price
                commission_cost = revenue * self.commission_pct
                self._balance += revenue - commission_cost
                self._position = 0.0
                self._entry_price = 0.0

        # Update portfolio value = balance + unrealized position value
        if self._position > 0:
            unrealized = self._position * current_price
        else:
            unrealized = 0.0
        self._portfolio_value = self._balance + unrealized

        step_return = (
            (self._portfolio_value / prev_portfolio_value - 1.0)
            if prev_portfolio_value > 0 else 0.0
        )

        # Peak tracking for drawdown
        if self._portfolio_value > self._peak_value:
            self._peak_value = self._portfolio_value

        reward = self._calculate_reward(step_return)

        self._trade_log.append(
            TradeRecord(
                step=self._current_step,
                action=action_val,
                price=new_fill_price if new_fill_price > 0 else current_price,
                position=float(self._position),
                portfolio_value=float(self._portfolio_value),
                reward=reward,
                done=False,
            )
        )

        self._current_step += 1

        # Check termination
        terminated = (
            self._portfolio_value < self.initial_balance * 0.5
            or self._current_step >= len(self.data) - 1
        )
        truncated = False
        info = {
            "balance": float(self._balance),
            "portfolio_value": float(self._portfolio_value),
            "step_return": step_return,
            "max_drawdown": float(self._max_drawdown),
            "n_trades": len(self._trade_log),
        }

        return self._get_observation(), reward, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):  # noqa: ANN201
        """Reset environment to a random position in the data for variety."""
        super().reset(seed=seed)

        min_offset = 50
        if len(self.data) > min_offset * 2:
            self._current_step = int(self.np_random.integers(min_offset, len(self.data) - min_offset))
        else:
            self._current_step = min_offset

        self._balance = self.initial_balance
        self._position = 0.0
        self._portfolio_value = self.initial_balance
        self._entry_price = 0.0
        self._episode_returns = []
        self._trade_log = []
        self._peak_value = self.initial_balance
        self._max_drawdown = 0.0

        return self._get_observation(), {}

    def render(self):  # noqa: ANN201
        """Print episode summary."""
        if self._trade_log:
            total_return = (
                (self._portfolio_value - self.initial_balance) / self.initial_balance * 100
            )
            print(f"\n--- Episode Summary ---")
            print(f"Steps: {self._current_step}")
            print(f"Trades: {len(self._trade_log)}")
            print(f"Final: ${self._portfolio_value:.2f} ({total_return:+.2f}%)")
            print(f"Max Drawdown: {self._max_drawdown*100:.2f}%")
            print(f"-----------------------")
