"""
trade_memory.py - Zero-Cost SQLite Trade Memory & Mistake Analyzer

Persists every approved trade's pre-trade context, post-trade lifecycle metrics
(MFE, MAE, R-multiple, fees), dynamic account equity, and delayed 4h/12h
post-exit autopsies in `data/trade_memory.db` using Python's built-in sqlite3.
"""

import contextlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "trade_memory.db")


def infer_market_session(ts: Optional[datetime] = None) -> str:
    """Classify UTC hour into Asia (00-08), London (08-13), or NY (13-24)."""
    if ts is None:
        ts = datetime.now(timezone.utc)
    elif isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    hour = ts.astimezone(timezone.utc).hour
    if 0 <= hour < 8:
        return "Asia"
    if 8 <= hour < 13:
        return "London"
    return "NY"


def _parse_iso(ts_str: str) -> datetime:
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class TradeMemory:
    """SQLite-backed quantitative trade journal and equity store."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, initial_equity: Optional[float] = None):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._initial_equity = (
            float(initial_equity)
            if initial_equity is not None
            else float(os.getenv("INITIAL_CAPITAL", "10000.0"))
        )
        self._mem_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:")
            self._mem_conn.row_factory = sqlite3.Row
            self._mem_conn.execute("PRAGMA foreign_keys = ON")
        self.init_schema()

    @contextlib.contextmanager
    def _connect(self):
        if self._mem_conn is not None:
            with self._mem_conn:
                yield self._mem_conn
            return
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS equity_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    account_equity REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades_context (
                    trade_id TEXT PRIMARY KEY,
                    entry_time TEXT NOT NULL,
                    side TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    trade_horizon TEXT NOT NULL DEFAULT 'INTRADAY',
                    market_session TEXT NOT NULL,
                    confluence_score REAL NOT NULL,
                    xgb_confidence REAL NOT NULL,
                    top_shap_features TEXT NOT NULL,
                    pattern_detected TEXT NOT NULL,
                    atr_pct REAL NOT NULL,
                    adx_14 REAL NOT NULL,
                    rsi_14 REAL NOT NULL,
                    volume_ratio REAL NOT NULL,
                    funding_rate REAL NOT NULL,
                    news_sentiment_score REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    initial_sl REAL NOT NULL,
                    risk_pct REAL NOT NULL,
                    account_equity_before REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN'
                );

                CREATE TABLE IF NOT EXISTS trades_lifecycle (
                    trade_id TEXT PRIMARY KEY REFERENCES trades_context(trade_id),
                    exit_time TEXT NOT NULL,
                    exit_state TEXT NOT NULL,
                    exit_reason TEXT NOT NULL,
                    mfe_r REAL NOT NULL,
                    mae_r REAL NOT NULL,
                    candles_to_tp1 INTEGER,
                    total_candles_held INTEGER NOT NULL,
                    net_pnl_usd REAL NOT NULL,
                    net_r_multiple REAL NOT NULL,
                    fees_paid_usd REAL NOT NULL,
                    account_equity_after REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS post_exit_autopsy (
                    trade_id TEXT PRIMARY KEY REFERENCES trades_context(trade_id),
                    exit_time TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    initial_sl REAL NOT NULL,
                    check_4h_due TEXT NOT NULL,
                    check_12h_due TEXT NOT NULL,
                    price_4h_later REAL,
                    price_12h_later REAL,
                    max_favorable_after REAL,
                    was_stop_hunt INTEGER NOT NULL DEFAULT 0,
                    autopsy_complete INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            try:
                conn.execute("ALTER TABLE trades_context ADD COLUMN trade_horizon TEXT DEFAULT 'INTRADAY'")
            except Exception:
                pass
            cur = conn.execute("SELECT account_equity FROM equity_state WHERE id = 1")
            if cur.fetchone() is None:
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO equity_state (id, account_equity, updated_at) VALUES (1, ?, ?)",
                    (self._initial_equity, now),
                )

    def get_account_equity(self) -> float:
        with self._connect() as conn:
            row = conn.execute("SELECT account_equity FROM equity_state WHERE id = 1").fetchone()
            return float(row["account_equity"]) if row else self._initial_equity

    def set_account_equity(self, equity: float, updated_at: Optional[str] = None) -> None:
        ts = updated_at or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE equity_state SET account_equity = ?, updated_at = ? WHERE id = 1",
                (round(float(equity), 2), ts),
            )

    def is_drawdown_shield_active(self, consecutive_losses: int = 3) -> bool:
        """Return True if the last `consecutive_losses` closed trades were all losses."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT net_r_multiple FROM trades_lifecycle ORDER BY exit_time DESC, rowid DESC LIMIT ?",
                (consecutive_losses,),
            ).fetchall()
        if len(rows) < consecutive_losses:
            return False
        return all(float(r["net_r_multiple"]) < 0 for r in rows)

    def log_entry(self, ctx: Dict[str, Any]) -> str:
        trade_id = str(ctx.get("trade_id") or uuid.uuid4().hex[:12])
        entry_time = str(ctx.get("entry_time") or datetime.now(timezone.utc).isoformat())
        session = str(ctx.get("market_session") or infer_market_session(_parse_iso(entry_time)))
        shap_raw = ctx.get("top_shap_features", [])
        shap_json = shap_raw if isinstance(shap_raw, str) else json.dumps(shap_raw)
        equity_before = float(
            ctx.get("account_equity_before")
            if ctx.get("account_equity_before") is not None
            else self.get_account_equity()
        )
        horizon = str(ctx.get("trade_horizon") or ctx.get("horizon") or "INTRADAY")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades_context (
                    trade_id, entry_time, side, timeframe, trade_horizon, market_session,
                    confluence_score, xgb_confidence, top_shap_features, pattern_detected,
                    atr_pct, adx_14, rsi_14, volume_ratio, funding_rate,
                    news_sentiment_score, entry_price, initial_sl, risk_pct,
                    account_equity_before, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                """,
                (
                    trade_id,
                    entry_time,
                    str(ctx["side"]),
                    str(ctx.get("timeframe", "15m")),
                    horizon,
                    session,
                    float(ctx.get("confluence_score", 0.0)),
                    float(ctx.get("xgb_confidence", 0.0)),
                    shap_json,
                    str(ctx.get("pattern_detected", "NONE")),
                    float(ctx.get("atr_pct", 0.0)),
                    float(ctx.get("adx_14", 0.0)),
                    float(ctx.get("rsi_14", 50.0)),
                    float(ctx.get("volume_ratio", 0.0)),
                    float(ctx.get("funding_rate", 0.0)),
                    float(ctx.get("news_sentiment_score", 0.0)),
                    float(ctx["entry_price"]),
                    float(ctx["initial_sl"]),
                    float(ctx.get("risk_pct", 0.01)),
                    equity_before,
                ),
            )
        return trade_id

    def log_exit(self, trade_id: str, lifecycle: Dict[str, Any]) -> float:
        """Record trade exit, update dynamic equity, and schedule autopsy on INITIAL_SL losses."""
        with self._connect() as conn:
            ctx = conn.execute(
                "SELECT side, entry_price, initial_sl, account_equity_before FROM trades_context WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()
            if ctx is None:
                raise KeyError(f"Unknown trade_id: {trade_id}")

            exit_time = str(lifecycle.get("exit_time") or datetime.now(timezone.utc).isoformat())
            exit_state = str(lifecycle["exit_state"])
            exit_reason = str(lifecycle["exit_reason"])
            net_pnl = float(lifecycle["net_pnl_usd"])
            current_equity = self.get_account_equity()
            equity_after = round(
                float(lifecycle.get("account_equity_after", max(0.0, current_equity + net_pnl))),
                2,
            )

            conn.execute(
                """
                INSERT OR REPLACE INTO trades_lifecycle (
                    trade_id, exit_time, exit_state, exit_reason, mfe_r, mae_r,
                    candles_to_tp1, total_candles_held, net_pnl_usd, net_r_multiple,
                    fees_paid_usd, account_equity_after
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_id,
                    exit_time,
                    exit_state,
                    exit_reason,
                    round(float(lifecycle.get("mfe_r", 0.0)), 4),
                    round(float(lifecycle.get("mae_r", 0.0)), 4),
                    lifecycle.get("candles_to_tp1"),
                    int(lifecycle.get("total_candles_held", 0)),
                    round(net_pnl, 2),
                    round(float(lifecycle.get("net_r_multiple", 0.0)), 4),
                    round(float(lifecycle.get("fees_paid_usd", 0.0)), 2),
                    equity_after,
                ),
            )
            conn.execute("UPDATE trades_context SET status = 'CLOSED' WHERE trade_id = ?", (trade_id,))
            conn.execute(
                "UPDATE equity_state SET account_equity = ?, updated_at = ? WHERE id = 1",
                (equity_after, exit_time),
            )

            if exit_reason == "INITIAL_SL" and float(lifecycle.get("net_r_multiple", 0.0)) < 0:
                exit_dt = _parse_iso(exit_time)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO post_exit_autopsy (
                        trade_id, exit_time, side, entry_price, initial_sl,
                        check_4h_due, check_12h_due
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_id,
                        exit_time,
                        ctx["side"],
                        float(ctx["entry_price"]),
                        float(ctx["initial_sl"]),
                        (exit_dt + timedelta(hours=4)).isoformat(),
                        (exit_dt + timedelta(hours=12)).isoformat(),
                    ),
                )
        return equity_after

    def process_due_autopsies(
        self,
        current_price: float,
        current_time: Optional[str] = None,
        window_high: Optional[float] = None,
        window_low: Optional[float] = None,
    ) -> int:
        """
        Check pending 4h/12h post-exit autopsies against current market price/extremes.
        Flags `was_stop_hunt = 1` if price subsequently reached +1.5R in the original direction.
        Returns count of autopsies newly completed or updated.
        """
        now_dt = _parse_iso(current_time) if current_time else datetime.now(timezone.utc)
        hi = float(window_high if window_high is not None else current_price)
        lo = float(window_low if window_low is not None else current_price)
        updated = 0

        with self._connect() as conn:
            pending = conn.execute(
                "SELECT * FROM post_exit_autopsy WHERE autopsy_complete = 0"
            ).fetchall()
            for row in pending:
                entry = float(row["entry_price"])
                sl = float(row["initial_sl"])
                r_dist = abs(entry - sl)
                side = str(row["side"]).upper()
                is_long = side in ("LONG", "BUY")
                target_15r = entry + 1.5 * r_dist if is_long else entry - 1.5 * r_dist

                hit_15r = (hi >= target_15r) if is_long else (lo <= target_15r)
                was_hunt = 1 if (int(row["was_stop_hunt"]) == 1 or hit_15r) else 0
                favorable_move = (
                    (hi - entry) / r_dist if is_long and r_dist > 0
                    else (entry - lo) / r_dist if r_dist > 0
                    else 0.0
                )
                prev_fav = float(row["max_favorable_after"]) if row["max_favorable_after"] is not None else favorable_move
                max_fav = round(max(prev_fav, favorable_move), 4)

                price_4h = row["price_4h_later"]
                price_12h = row["price_12h_later"]
                complete = 0
                changed = False

                if price_4h is None and now_dt >= _parse_iso(row["check_4h_due"]):
                    price_4h = float(current_price)
                    changed = True
                if price_12h is None and now_dt >= _parse_iso(row["check_12h_due"]):
                    price_12h = float(current_price)
                    complete = 1
                    changed = True
                elif was_hunt != int(row["was_stop_hunt"]) or max_fav != prev_fav:
                    changed = True

                if changed:
                    conn.execute(
                        """
                        UPDATE post_exit_autopsy
                        SET price_4h_later = ?, price_12h_later = ?,
                            max_favorable_after = ?, was_stop_hunt = ?, autopsy_complete = ?
                        WHERE trade_id = ?
                        """,
                        (price_4h, price_12h, max_fav, was_hunt, complete, row["trade_id"]),
                    )
                    updated += 1
        return updated

    def get_cumulative_metrics(self) -> Dict[str, Any]:
        """Return total trades, win rate, net R, profit factor, and equity for Telegram broadcasts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT net_r_multiple, net_pnl_usd FROM trades_lifecycle ORDER BY exit_time ASC"
            ).fetchall()
        total = len(rows)
        if total == 0:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "net_r": 0.0,
                "net_pnl_usd": 0.0,
                "account_equity": self.get_account_equity(),
                "drawdown_shield": False,
            }
        rs = [float(r["net_r_multiple"]) for r in rows]
        pnls = [float(r["net_pnl_usd"]) for r in rows]
        wins = sum(1 for r in rs if r > 0)
        losses = sum(1 for r in rs if r < 0)
        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total * 100.0, 1),
            "net_r": round(sum(rs), 2),
            "net_pnl_usd": round(sum(pnls), 2),
            "account_equity": self.get_account_equity(),
            "drawdown_shield": self.is_drawdown_shield_active(),
        }
