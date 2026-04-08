# Code Review — Autonomous AI Trading System

> Date: 2026-04-07
> Scope: All Python files in research_lab/src/, execution_engine/src/, config/

---

## Bugs Found

### BUG-1: Position Never Held — Liquidation Logic Erases Everything 【CRITICAL】
**File**: `research_lab/src/trading_env.py:161-182`

```python
# Liquidation: close previous position
if abs(prev_position) > 1e-10:
    liq_price = self._apply_slippage_for_sell(current_price)
    liquidation_value = prev_position * liq_price
    self._balance += liquidation_value
    ...

# Enter New Position
self._position = 0.0  # <-- BUG: resets before any position exists
```

**Issue**: `self._position` is set to `0.0` BEFORE `target_dollar_exposure` is applied. Even though `self._position` is only used for PnL display (not for balance), the comment says "Enter New Position" but the position is immediately zeroed. The environment actually converts everything to cash at each step — it's a "fully liquidate then re-enter" model, not a continuous position model.

**Impact**: The agent can't maintain a position over time — it must re-enter every single step. This means:
- No concept of "holding" a position
- Each step incurs full liquidation + re-entry costs
- Sharpe ratio is unfairly penalized by transaction costs

**Fix**: The agent should be able to adjust position size incrementally rather than fully liquidating each step.

---

### BUG-2: Position.update_price Uses Frozen dataclass 【CRITICAL】
**File**: `execution_engine/src/models.py:97-98`
```python
def update_price(self, price: float) -> None:
    object.__setattr__(self, "current_price", price)
```

**Issue**: `Position` is NOT frozen (line 82: `@dataclass` without `frozen=True`), so `update_price` uses the awkward `object.__setattr__` pattern normally reserved for frozen dataclasses. For a mutable dataclass, regular attribute assignment would work.

**Fix**: Either make `Position` frozen (`@dataclass(frozen=True)`) for consistency with other models, or use direct assignment.

---

### BUG-3: Balance Reset on Every Tick via Liquidation PnL 【HIGH】
**File**: `execution_engine/src/paper_executor.py:138-186`

In `_close_position()`, the balance is updated:
```python
self.balance += pos.quantity * current_price - commission
```

But in `_open_position()` (line 95):
```python
self.balance -= total_cost  # total_cost = exposure + commission
```

**Double-counting**: When a position is closed, unrealized PnL is credited to balance. But `self.equity` property (line 206-217) STILL adds the position value to balance when there IS a position open. If a position is marked-to-market each tick via `execute_signal()` → `update_price()`, the equity calculation is correct but the balance changes during open/close are also modifying the same value.

Looking at `_close_position()`: it returns cash to balance AND records a trade. The `balance` is correctly the "cash on hand" while `equity` is `balance + position_value`. This is actually correct.

---

### BUG-4: Signal Confidence Misleading 【MEDIUM】
**File**: `execution_engine/src/signal_generator.py:102`
```python
confidence = float(abs(action[0]))
```

**Issue**: The PPO model outputs actions in the range `[-1.0, 1.0]`. Using `abs(action)` as "confidence" means a strong SHORT signal (action = -0.9) has 90% "confidence" as a LONG signal would at 0.9. This conflates magnitude with direction certainty.

**Fix**: Consider using the model's action distribution (entropy/probability) instead of raw action magnitude for confidence.

---

### BUG-5: _sleep Is an async_staticMethod 【LOW】
**File**: `execution_engine/src/main.py:122-128`
```python
@staticmethod
async def _sleep(seconds: int) -> None:
    try:
        await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        pass
```

**Issue**: `@staticmethod` with `async def` works technically but the `CancelledError` from `asyncio.sleep` won't propagate to the outer `engine.run()` loop properly because `main.py:54` calls `await self._tick()` inside a `while self._running:` loop where `_sleep` is called inside `_tick` at line 58.

Wait — looking at the code, `_tick()` does NOT call `_sleep`. Line 58 calls `await self._sleep(loop_interval)` in the `while` loop after `_tick()`. This is correct and the `_sleep` static method works fine. Not a bug.

---

### BUG-6: No Stop-Loss Triggered Trade Recording Has Wrong reason Field 【MEDIUM】
**File**: `execution_engine/src/paper_executor.py:197`

When stop-loss triggers via `check_stop_loss()`:
```python
self._close_position(self.position.stop_loss)
```

The `_close_position()` method always sets `reason="signal"`. It should be `"stop_loss"` for audit trail purposes.

**Fix**: Add a `reason` parameter to `_close_position()`:
```python
def _close_position(self, current_price: float, reason: str = "signal") -> float:
```

And in `check_stop_loss`, call with `reason="stop_loss"`.

---

### BUG-7: Position Side Mismatch on Close 【MEDIUM】
**File**: `execution_engine/src/paper_executor.py:61-67`

In `execute_signal()`, when there's an open position:
```python
if self.position is not None:
    self._close_position(signal.price)
```

This closes the position at the signal price regardless of whether the new signal agrees with the existing position. If current position is LONG and signal is still LONG (just slightly different), the system liquidates and re-enters, paying commission + slippage twice unnecessarily.

**Fix**: Only close if the new signal disagrees with the current position:
```python
if self.position is not None:
    should_close = (
        (self.position.side == Side.LONG and not signal.is_long) or
        (self.position.side == Side.SHORT and not signal.is_short)
    )
    if should_close:
        self._close_position(signal.price)
    return None
```

---

## Code Quality Issues

### QUALITY-1: Circular Import Risk
**File**: `execution_engine/src/risk_manager.py:18`
```python
from .signal_generator import Signal
```

`risk_manager.py` imports from `signal_generator.py`. If `signal_generator.py` ever imports from `risk_manager.py`, this creates a circular import. Currently it doesn't, but the dependency `risk_manager → signal_generator` is unusual (risk management shouldn't know about the signal format).

---

### QUALITY-2: Hardcoded Feature Columns
**File**: `execution_engine/src/signal_generator.py:21-33`

The `FEATURE_COLUMNS` list must perfectly match the order `feature_engine.py` produces them in. These are in two different files with no automated sync. Currently they match (verified), but this is a ticking time bomb.

---

### QUALITY-3: Magic Numbers
| Location | Value | Meaning |
|----------|-------|---------|
| `main.py:40` | `10_000.0` | Initial balance — should be in `Settings` |
| `trading_env.py:115` | `10` | Minimum returns for Sharpe calc |
| `trading_env.py:209` | `0.5` | 50% blowup threshold |
| `feature_engine.py:101` | `550` | Min rows for features |
| `feature_engine.py:138` | `50` | NaN drop after rolling zscore |
| `signal_generator.py:94` | `-10.0, 10.0` | Clipping range |

---

### QUALITY-4: No `pyproject.toml` in execution_engine
The `research_lab` has `pyproject.toml` but `execution_engine` does not. Dependencies are managed ad-hoc.

---

## Summary

| Category | Count | Description |
|----------|-------|-------------|
| CRITICAL Bugs | 2 | Position liquidation logic, frozen dataclass method |
| HIGH Bugs | 1 | Balance/PnL accounting concerns |
| MEDIUM Bugs | 3 | Stop-loss reason field, unnecessary re-entry, misleading confidence |
| Quality Issues | 4 | Circular import risk, hardcoded columns, magic numbers, missing pyproject.toml |
