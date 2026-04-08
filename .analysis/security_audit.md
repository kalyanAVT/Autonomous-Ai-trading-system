# Security Audit — Autonomous AI Trading System

> Date: 2026-04-07
> Scope: Security vulnerabilities, credential management, attack surface

---

## Findings by Severity

### CRITICAL-1: No .env.example or .gitignore for Secrets
**Location**: Project root

**Issue**: No `.env` file exists, no `.env.example` template, and the `.gitignore` status is unknown. API keys are configured as plain strings in `config.py:14-15`:
```python
api_key: str = ""
api_secret: str = ""
```

**Risk**: Without `.gitignore`, a developer might accidentally commit `.env` with real API keys. The plan uses real exchange API keys for live trading (Phase 5).

**Remediation**:
1. Create `.env.example` with placeholder values for all required variables
2. Create/verify `.gitignore` includes `.env`, `*.key`, `*.pem`, `*.pt`
3. Add `.env` to `.gitignore`
4. Consider using environment variable injection or cloud secrets manager for production

---

### CRITICAL-2: Model Files (.pt) Not Protected
**Location**: `config.py:24` — `model_path: str = ""`

**Issue**: Trained .pt model files contain serialized Python objects. Loading an untrusted .pt via `PPO.load()` (`signal_generator.py:72`) is a **remote code execution** vector — PyTorch `torch.load()` unpickles objects which can execute arbitrary code.

**Risk**: If a model file is tampered with, it gains full system access. In a live trading context, this could lead to unauthorized fund transfers.

**Remediation**:
1. Use `torch.load(path, weights_only=True)` where possible (PyTorch 2.0+)
2. Validate model file hashes before loading
3. Restrict file permissions on model directory
4. Add model signature verification

---

### HIGH-1: Exchange API Keys Transmitted Insecurely
**Location**: `data_feed.py:37-39` — `config.py:14-15`

**Issue**: Binance API connections use default HTTP. While Binance requires HTTPS for API calls, the code does not explicitly enforce SSL verification:
```python
self.exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
})
```

**Risk**: MITM attack could intercept or modify API responses.

**Remediation**:
```python
self.exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
    "nonce": lambda: ccxt.Exchange.milliseconds(),
})
# ccxt enforces HTTPS by default for exchange APIs, verify no override path exists
```

---

### HIGH-2: No Rate Limiting on ccxt Calls
**Location**: `data_feed.py:37-38` — `"enableRateLimit": True`

**Issue**: `enableRateLimit` is set (good), but there's no application-level rate limit monitoring. If the exchange changes rate limits, the code will keep hammering until banned.

**Risk**: Exchange IP ban, lost trading opportunities, or in live mode — orphaned positions with no stop-loss monitoring.

**Remediation**: Add rate limit tracking and backoff logic. On repeated 429 responses, pause the execution loop.

---

### HIGH-3: No Process Isolation Between Research and Execution
**Location**: `data_feed.py:25-27` — `sys.path.insert`

**Issue**: `execution_engine` directly imports from `research_lab` at import time. A compromised or buggy research module has full access to the execution pipeline.

**Risk**: If research_lab is updated during live trading, import-time side effects could crash the execution engine.

**Remediation**: Use subprocess-based communication (socket, message queue) between the two modules, or package research_lab as an immutable dependency.

---

### MEDIUM-1: Insufficient Error Context Could Leak Info
**Location**: `main.py:55-56`
```python
except Exception as e:
    logger.error("Error in execution loop: %s", e, exc_info=True)
```

**Issue**: `exc_info=True` logs full stack traces which may contain API keys, endpoint URLs, or trade details. In production, this goes to log files accessible by system admins.

**Remediation**: Sanitize log output — exclude sensitive fields from exception chain. Use structured logging with sensitive field masking.

---

### MEDIUM-2: No Request Validation on Exchange Responses
**Location**: `data_feed.py:51-74` — `fetch_history()`

**Issue**: Raw exchange responses are consumed without validation. If Binance returns anomalous prices (flash crash, bad ticker update), the system accepts them.

**Risk**: A $0.01 BTC price causes the system to attempt massive positions or misprice all risk calculations.

**Remediation**: Add sanity checks:
```python
assert 1000 < close < 1_000_000, f"BTC price sanity fail: {close}"
```

---

### MEDIUM-3: Candle Buffer Memory Growth
**Location**: `data_feed.py:166-168`
```python
max_buffer = self.settings.feature_lookback + 200
if len(self._candle_buffer) > max_buffer:
    self._candle_buffer = self._candle_buffer[-max_buffer:]
```

**Issue**: Buffer is capped (good) but the cap is ~700 candle objects. At 6 fields each, this is manageable. However if `feature_lookback` is misconfigured to a large value, memory grows proportionally.

**Remediation**: Add hard cap independent of settings, e.g., `max_buffer = min(max_buffer, 5000)`.

---

### LOW-1: No Input Sanitization on CLI Args
**Location**: `main.py:134` — `"--verbose" in sys.argv`

**Issue**: Using `sys.argv` directly for config is basic but acceptable for a CLI tool. Not a significant risk.

---

### LOW-2: UUID-Based Order IDs
**Location**: `models.py:63` — `order_id: str = field(default_factory=lambda: uuid4().hex)`

**Issue**: 32-char hex strings are predictable enough for collision analysis. Not a real security issue for paper trading; for live trading, use exchange-provided order IDs.

---

## Summary Table

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 2 | No secret protection, model file RCE |
| HIGH | 3 | MITM risk, rate limit gaps, no process isolation |
| MEDIUM | 3 | Logging leaks, no price validation, memory growth |
| LOW | 2 | CLI args, UUID order IDs |

## Recommended Immediate Actions

1. Create `.env.example` and `.gitignore` — protect credentials
2. Add price sanity bounds to `data_feed.py`
3. Validate model file integrity before loading
4. Never commit `.env` — verify `.gitignore` before first trade
