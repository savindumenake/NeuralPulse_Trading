"""
analyze_performance.py - Quantitative Diagnostic Report from data/trade_memory.db

Queries `trades_context`, `trades_lifecycle`, and `post_exit_autopsy` to print:
- Overall Summary (Trades, Win Rate, Net R, Net PnL, Current Equity)
- Win Rate & Expectancy by Pattern
- Win Rate & Expectancy by Market Session (Asia / London / NY)
- Win Rate & Expectancy by ADX_14 Regime
- MFE / MAE Optimization Insights
- Stop-Hunt Frequency from 4h/12h Post-Exit Autopsies

Usage:
    python analyze_performance.py
    python analyze_performance.py --db data/backtest_memory.db
"""

import argparse
import os
import sqlite3
from typing import Any, Dict, List

from agents.trade_memory import DEFAULT_DB_PATH, TradeMemory


def _bucket_stats(rows: List[sqlite3.Row]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0, "net_r": 0.0}
    rs = [float(r["net_r_multiple"]) for r in rows]
    wins = sum(1 for r in rs if r > 0)
    return {
        "trades": total,
        "win_rate": round(wins / total * 100.0, 1),
        "expectancy_r": round(sum(rs) / total, 3),
        "net_r": round(sum(rs), 2),
    }


def _adx_bucket(adx: float) -> str:
    if adx < 22.0:
        return "<22 (Choppy)"
    if adx < 30.0:
        return "22-30 (Emerging)"
    if adx < 45.0:
        return "30-45 (Strong)"
    return "45+ (Extreme)"


def generate_diagnostic_report(db_path: str = DEFAULT_DB_PATH) -> str:
    if not os.path.exists(db_path):
        return f"No database found at {db_path}. Run live trading or backtest.py first."

    mem = TradeMemory(db_path=db_path)
    summary = mem.get_cumulative_metrics()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        joined = conn.execute(
            """
            SELECT c.*, l.exit_state, l.exit_reason, l.mfe_r, l.mae_r,
                   l.candles_to_tp1, l.total_candles_held, l.net_pnl_usd,
                   l.net_r_multiple, l.fees_paid_usd
            FROM trades_context c
            JOIN trades_lifecycle l ON c.trade_id = l.trade_id
            ORDER BY l.exit_time ASC
            """
        ).fetchall()
        autopsies = conn.execute("SELECT * FROM post_exit_autopsy").fetchall()
    finally:
        conn.close()

    lines = [
        "=" * 72,
        f"  NeuralPulse Quantitative Diagnostic Report ({os.path.basename(db_path)})",
        "=" * 72,
    ]
    if not joined:
        lines.append("\nNo closed trades logged yet.")
        lines.append(f"Current Account Equity: ${summary['account_equity']:,.2f}")
        return "\n".join(lines)

    lines.extend([
        f"\nClosed Trades:   {summary['total_trades']} ({summary['wins']}W / {summary['losses']}L)",
        f"Win Rate:        {summary['win_rate']:.1f}%",
        f"Net Profit (R):  {summary['net_r']:+.2f}R (${summary['net_pnl_usd']:+,.2f})",
        f"Account Equity:  ${summary['account_equity']:,.2f}",
        f"Drawdown Shield: {'ACTIVE' if summary['drawdown_shield'] else 'OFF'}",
    ])

    def _print_group(title: str, key_fn) -> None:
        groups: Dict[str, List[sqlite3.Row]] = {}
        for row in joined:
            groups.setdefault(key_fn(row), []).append(row)
        lines.append(f"\n--- {title} ---")
        for name, grp in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            st = _bucket_stats(grp)
            lines.append(
                f"  {name:28s} | n={st['trades']:3d} | Win={st['win_rate']:5.1f}% | "
                f"Exp={st['expectancy_r']:+.3f}R | Net={st['net_r']:+.2f}R"
            )

    _print_group("By Pattern Detected", lambda r: str(r["pattern_detected"]))
    _print_group("By Market Session (UTC)", lambda r: str(r["market_session"]))
    _print_group("By ADX_14 Regime", lambda r: _adx_bucket(float(r["adx_14"])))

    # MFE / MAE insights
    mfes = [float(r["mfe_r"]) for r in joined]
    maes = [float(r["mae_r"]) for r in joined]
    winners_mae = [float(r["mae_r"]) for r in joined if float(r["net_r_multiple"]) > 0]
    losers_mfe = [float(r["mfe_r"]) for r in joined if float(r["net_r_multiple"]) < 0]
    lines.append("\n--- MFE / MAE Optimization Insights ---")
    lines.append(f"  Avg MFE (All Trades):   {sum(mfes)/len(mfes):+.2f}R (Max: {max(mfes):+.2f}R)")
    lines.append(f"  Avg MAE (All Trades):   {sum(maes)/len(maes):+.2f}R (Worst: {min(maes):+.2f}R)")
    if winners_mae:
        lines.append(f"  Avg MAE (Winners):      {sum(winners_mae)/len(winners_mae):+.2f}R")
    if losers_mfe:
        lines.append(f"  Avg MFE (Losers):       {sum(losers_mfe)/len(losers_mfe):+.2f}R")

    # Stop-Hunt Autopsy
    lines.append("\n--- Post-Exit Stop-Hunt Autopsy (INITIAL_SL Losses) ---")
    if autopsies:
        total_sl = len(autopsies)
        hunts = sum(1 for a in autopsies if int(a["was_stop_hunt"]) == 1)
        lines.append(
            f"  INITIAL_SL Losses Audited: {total_sl} | "
            f"Stop-Hunts (later hit +1.5R): {hunts} ({hunts / total_sl * 100.0:.1f}%)"
        )
    else:
        lines.append("  No INITIAL_SL losses recorded yet.")

    # Individual Trades Breakdown Table
    lines.append("\n--- Individual Closed Trades Log ---")
    header = (
        f"  {'#':<3s} {'Entry Time (UTC)':<20s} {'Side':<5s} {'Horizon':<8s} "
        f"{'Pattern':<28s} {'Entry':<10s} {'SL':<10s} {'MFE':<7s} {'MAE':<7s} "
        f"{'Exit Reason':<20s} {'Net R':<8s} {'Net USD':<10s}"
    )
    lines.append(header)
    lines.append("  " + "-" * 135)
    for idx, r in enumerate(joined, 1):
        horizon = r["trade_horizon"] if "trade_horizon" in r.keys() else "INTRADAY"
        lines.append(
            f"  {idx:<3d} {str(r['entry_time'])[:19]:<20s} {str(r['side']):<5s} "
            f"{str(horizon):<8s} {str(r['pattern_detected'])[:28]:<28s} "
            f"${float(r['entry_price']):<9.2f} ${float(r['initial_sl']):<9.2f} "
            f"{float(r['mfe_r']):+5.2f}R  {float(r['mae_r']):+5.2f}R  "
            f"{str(r['exit_reason'])[:20]:<20s} {float(r['net_r_multiple']):+6.2f}R "
            f"${float(r['net_pnl_usd']):+8.2f}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze NeuralPulse trade_memory.db performance")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Path to SQLite trade_memory.db")
    args = parser.parse_args()
    print(generate_diagnostic_report(args.db))
