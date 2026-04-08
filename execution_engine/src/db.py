"""SQLite persistence layer for trade logging, session tracking, and model metadata.

Provides durable storage so trade history survives restarts,
enables analytics across sessions, and supports audit trails.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class TradeDB:
    """SQLite-backed store for trades, sessions, and models."""

    def __init__(self, db_path: str = "data/trades.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                initial_balance REAL NOT NULL,
                final_equity REAL,
                total_trades INTEGER DEFAULT 0,
                mode TEXT NOT NULL,
                model_path TEXT
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                pnl REAL NOT NULL,
                commission REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                reason TEXT NOT NULL,
                model_version TEXT,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                holding_hours REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                training_date TEXT NOT NULL,
                backtest_sharpe REAL,
                backtest_return_pct REAL,
                paper_days INTEGER DEFAULT 0,
                promoted_to_live INTEGER DEFAULT 0
            );
        """)
        self.conn.commit()

    def create_session(
        self,
        session_id: str,
        initial_balance: float,
        mode: str = "paper",
        model_path: str = "",
    ) -> None:
        self.conn.execute(
            "INSERT INTO sessions (id, start_time, initial_balance, mode, model_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, datetime.now(timezone.utc).isoformat(), initial_balance, mode, model_path),
        )
        self.conn.commit()

    def close_session(self, session_id: str, final_equity: float, total_trades: int) -> None:
        self.conn.execute(
            "UPDATE sessions SET end_time=?, final_equity=?, total_trades=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), final_equity, total_trades, session_id),
        )
        self.conn.commit()

    def log_trade(
        self,
        session_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        commission: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        reason: str,
        entry_time: datetime,
        exit_time: datetime,
        holding_hours: float,
        model_version: str = "",
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades
               (session_id, symbol, side, entry_price, exit_price, quantity, pnl,
                commission, stop_loss, take_profit, reason, model_version,
                entry_time, exit_time, holding_hours)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, symbol, side, entry_price, exit_price, quantity, pnl,
                commission, stop_loss, take_profit, reason, model_version,
                entry_time.isoformat(), exit_time.isoformat(), holding_hours,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def register_model(
        self,
        path: str,
        backtest_sharpe: float = 0.0,
        backtest_return_pct: float = 0.0,
    ) -> None:
        self.conn.execute(
            "INSERT INTO models (path, training_date, backtest_sharpe, backtest_return_pct) "
            "VALUES (?, ?, ?, ?)",
            (path, datetime.now(timezone.utc).isoformat(), backtest_sharpe, backtest_return_pct),
        )
        self.conn.commit()

    def query_trades(self, session_id: Optional[str] = None) -> list[dict]:
        q = "SELECT * FROM trades"
        params: list = []
        if session_id:
            q += " WHERE session_id = ?"
            params.append(session_id)
        q += " ORDER BY exit_time DESC"
        return [dict(row) for row in self.conn.execute(q, params).fetchall()]

    def close(self) -> None:
        self.conn.close()
