"""
telegram_notifier.py - VIP Trader Signal Formatter & Telegram Channel Broadcaster

Formats approved trades into a clean, human-readable Trader Signal Card
and broadcasts them to a Telegram Channel / Group / Chat via the Telegram Bot API.
Includes duplicate-signal cooldown protection.
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

from agents.state import TradingState

load_dotenv()

logger = logging.getLogger("TelegramNotifier")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SIGNAL_COOLDOWN_MINUTES = int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "60"))

# In-memory cooldown tracker: {"direction": str, "timestamp": float, "price": float}
_LAST_SENT_SIGNAL: dict = {"direction": None, "timestamp": 0.0, "price": 0.0}


def reset_signal_cooldown() -> None:
    """Reset the duplicate signal cooldown state (useful for tests)."""
    _LAST_SENT_SIGNAL["direction"] = None
    _LAST_SENT_SIGNAL["timestamp"] = 0.0
    _LAST_SENT_SIGNAL["price"] = 0.0


def is_in_cooldown(direction: str, current_ts: Optional[float] = None) -> bool:
    """Return True if the same direction signal was already sent within cooldown window."""
    if current_ts is None:
        current_ts = time.time()
    last_dir = _LAST_SENT_SIGNAL.get("direction")
    last_ts = float(_LAST_SENT_SIGNAL.get("timestamp", 0.0))
    if last_dir == direction and (current_ts - last_ts) < (SIGNAL_COOLDOWN_MINUTES * 60):
        return True
    return False


def mark_signal_sent(direction: str, price: float, current_ts: Optional[float] = None) -> None:
    """Record that a signal was just broadcast."""
    _LAST_SENT_SIGNAL["direction"] = direction
    _LAST_SENT_SIGNAL["timestamp"] = time.time() if current_ts is None else current_ts
    _LAST_SENT_SIGNAL["price"] = price


def format_trade_signal_message(state: TradingState) -> str:
    """
    Create a clean, trader-friendly VIP Signal Card for Console and Telegram.
    """
    direction = state.get("final_decision", "HOLD")
    symbol = state.get("symbol", "BTC/USDT")
    entry = float(state.get("current_price", 0.0))
    sl = float(state.get("stop_loss", 0.0))
    pos_size = float(state.get("position_size", 0.0))
    capital = float(state.get("capital_allocation", 0.0))
    confidence = float(state.get("confidence", 0.0)) * 100.0
    sentiment = float(state.get("sentiment_score", 0.0))
    sentiment_summary = state.get("sentiment_summary", "Neutral / No major news headwinds")

    indicators = state.get("indicators", {}) or {}
    regime = indicators.get("market_regime", "TRENDING")
    patterns = indicators.get("detected_patterns", []) or []
    market_type = str(indicators.get("market_type", "FUTURES")).upper()
    margin_mode = str(indicators.get("margin_mode", "ISOLATED")).upper()
    leverage = int(indicators.get("recommended_leverage") or 1)
    notional_usd = float(indicators.get("notional_value_usd") or capital)
    margin_usd = float(indicators.get("margin_required_usd") or (notional_usd / max(leverage, 1)))
    risk_usd = float(indicators.get("risk_amount_usd", 0.0))
    risk_pct = float(indicators.get("risk_pct", 2.0))
    est_liq = float(indicators.get("est_liquidation_price", 0.0))

    contract_name = (
        f"{symbol.replace('/', '')} Perpetual (Futures)"
        if market_type == "FUTURES"
        else symbol
    )

    sl_dist = abs(entry - sl) if sl > 0 else 0.0
    entry_band = sl_dist * 0.15
    entry_zone_low = round(entry - entry_band, 2)
    entry_zone_high = round(entry + entry_band, 2)

    if direction == "BUY":
        tp1 = round(float(indicators.get("take_profit_1") or (entry + sl_dist * 1.5)), 2)
        tp2 = round(float(indicators.get("take_profit_2") or indicators.get("take_profit") or (entry + sl_dist * 2.0)), 2)
        tp3 = round(float(indicators.get("take_profit_3") or (entry + sl_dist * 3.0)), 2)
        dir_badge = "🟢 BUY / LONG"
    else:
        tp1 = round(float(indicators.get("take_profit_1") or (entry - sl_dist * 1.5)), 2)
        tp2 = round(float(indicators.get("take_profit_2") or indicators.get("take_profit") or (entry - sl_dist * 2.0)), 2)
        tp3 = round(float(indicators.get("take_profit_3") or (entry - sl_dist * 3.0)), 2)
        dir_badge = "🔴 SELL / SHORT"

    sl_pct = (sl_dist / entry * 100.0) if entry > 0 else 0.0
    tp2_pct = (abs(tp2 - entry) / entry * 100.0) if entry > 0 else 0.0

    raw_reasons = state.get("technical_reason", "")
    reason_list = [r.strip() for r in raw_reasons.split(";") if r.strip()]
    reasons_formatted = "\n".join(f"  • {r}" for r in reason_list) if reason_list else "  • Multi-timeframe confluence aligned"
    patterns_str = ", ".join(patterns) if patterns else "Structure + Price Action"

    tier = str(indicators.get("trade_tier", "A_GRADE"))
    sniper_score = float(indicators.get("sniper_score", confidence))
    shield_tag = " | 🛡️ Drawdown Shield (50% Risk)" if indicators.get("drawdown_shield_active") else ""
    funding_rate = float(indicators.get("funding_rate", 0.0))
    funding_line = f"💹 Funding Rate: {funding_rate * 100:+.4f}%\n" if funding_rate != 0.0 else ""

    cum = indicators.get("cumulative_metrics") or {}
    cum_line = (
        f"📈 Track Record: {cum['total_trades']} Trades | "
        f"Win Rate: {cum['win_rate']:.1f}% | Net: {cum['net_r']:+.2f}R\n"
        if cum.get("total_trades", 0) > 0
        else ""
    )

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    horizon = str(indicators.get("trade_horizon") or state.get("trade_horizon") or "INTRADAY").upper()
    horizon_badges = {
        "SCALP": "⚡ SCALP (5m Trigger | 1H/15m MTF)",
        "INTRADAY": "🏹 INTRADAY (15m Trigger | 1D/1H MTF)",
        "SWING": "🌊 SWING (1H Trigger | 1D/4H MTF)",
    }
    horizon_str = horizon_badges.get(horizon, horizon)

    if market_type == "FUTURES":
        margin_line = f"⚡ Margin & Leverage: {margin_mode} | Up to {leverage}x (Safe)\n"
        liq_line = (
            f"  • Est. Liq Buffer: ${est_liq:,.2f} (Outside SL)\n"
            if est_liq > 0
            else ""
        )
        sizing_block = (
            f"🛡️ Futures Risk Plan ({risk_pct:.1f}% Risk{shield_tag}):\n"
            f"  • Position Size: {pos_size:.4f} BTC (${notional_usd:,.2f} Notional)\n"
            f"  • Margin Used ({leverage}x): ${margin_usd:,.2f}"
            + (f" | Max Risk: ${risk_usd:,.2f}\n" if risk_usd > 0 else "\n")
            + liq_line
        )
    else:
        margin_line = ""
        sizing_block = f"⚖️ Paper Size: {pos_size:.6f} BTC (${capital:,.2f})\n"

    regime_mode = str(indicators.get("regime_mode", "EXPANSION")).upper()
    regime_mode_str = "🔄 ROTATIONAL (Quick Harvest)" if regime_mode == "ROTATIONAL" else "🚀 EXPANSION (Trend Runner)"

    if regime_mode == "ROTATIONAL":
        tp_block = (
            f"🎯 Take Profit 1 (+1.1R): ${tp1:,.2f} [Close 50% & SL → +0.15R]\n"
            f"🎯 Take Profit 2 (+1.8R): ${tp2:,.2f} (+{tp2_pct:.2f}%) [Close remaining 50% | Full Target Exit]\n"
            f"🎯 Take Profit 3 (Runner): N/A (Rotational Mode - Full Exit at TP2)\n"
        )
    else:
        tp_block = (
            f"🎯 Take Profit 1 (+1.4R): ${tp1:,.2f} [Close 35% & SL → -0.20R]\n"
            f"🎯 Take Profit 2 (+2.4R): ${tp2:,.2f} (+{tp2_pct:.2f}%) [Close 30% & SL → +1.00R]\n"
            f"🎯 Take Profit 3 (Runner): ${tp3:,.2f} [35% Chandelier Exit]\n"
        )

    msg = (
        f"🚨 NEURALPULSE VIP TRADE SIGNAL 🚨\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 Contract: {contract_name}\n"
        f"🎯 Horizon: {horizon_str}\n"
        f"🌐 Mode: {regime_mode_str}\n"
        f"📣 Action: {dir_badge} [{tier}]\n"
        f"{margin_line}"
        f"🕒 Time: {now_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Entry Price: ${entry:,.2f} (Zone: ${entry_zone_low:,.2f} – ${entry_zone_high:,.2f})\n"
        f"🛑 Stop Loss (SL): ${sl:,.2f} (-{sl_pct:.2f}% | Wick-Proof)\n"
        f"{tp_block}"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{sizing_block}"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Sniper Score: {sniper_score:.1f}/100 ({regime})\n"
        f"📐 Patterns: {patterns_str}\n"
        f"{funding_line}"
        f"📰 News Sentiment: {sentiment:+.2f} ({sentiment_summary})\n"
        f"{cum_line}"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ Confirmations:\n"
        f"{reasons_formatted}"
    )
    return msg


def format_lifecycle_event_message(event: dict, metrics: Optional[dict] = None) -> str:
    """Format a 5-stage trade lifecycle transition or close alert for Telegram."""
    kind = event.get("kind", "TRANSITION")
    reason = event.get("reason", "")
    side = event.get("side", "LONG")
    state_name = event.get("state", "")
    price = float(event.get("price", 0.0))
    new_sl = float(event.get("new_sl", 0.0))
    rem = float(event.get("remaining_pct", 0.0))
    net_r = float(event.get("net_r_multiple", 0.0))
    net_pnl = float(event.get("net_pnl_usd", 0.0))
    trade_id = event.get("trade_id", "")

    header = "🔄 TRADE STATE UPDATE" if kind == "TRANSITION" else "🏁 TRADE CLOSED"
    lines = [
        f"{header} [{side} #{trade_id}]",
        f"━━━━━━━━━━━━━━━━━━━━━━",
        f"📍 Event: {reason} ({state_name})",
        f"💵 Trigger Price: ${price:,.2f}",
        f"🛑 Updated SL: ${new_sl:,.2f} | Open Size: {rem:.0f}%",
        f"💰 Realized Net: {net_r:+.2f}R (${net_pnl:+,.2f})",
    ]
    if metrics and metrics.get("total_trades", 0) > 0:
        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"📊 Cumulative Accuracy: {metrics['win_rate']:.1f}% Win Rate "
            f"({metrics['wins']}W / {metrics['losses']}L, {metrics['total_trades']} Total)",
            f"🏦 Net R: {metrics['net_r']:+.2f}R | Equity: ${metrics['account_equity']:,.2f}",
        ])
    return "\n".join(lines)


def send_telegram_signal(
    message: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> bool:
    """
    Send the formatted trade signal to a Telegram channel or chat.
    Supports optional TELEGRAM_PROXY_URL and TELEGRAM_API_BASE_URL from .env.
    Returns True if sent successfully, False if unconfigured or failed.
    """
    load_dotenv(override=False)
    token = (bot_token if bot_token is not None else os.getenv("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN)).strip()
    target_chat = (chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)).strip()
    api_base = os.getenv("TELEGRAM_API_BASE_URL", "https://api.telegram.org").strip().rstrip("/")
    proxy_url = os.getenv("TELEGRAM_PROXY_URL", "").strip()

    if (
        not token
        or not target_chat
        or token == "your_telegram_bot_token_here"
        or target_chat == "your_telegram_channel_or_chat_id_here"
    ):
        logger.info("Telegram credentials not configured in .env; skipping Telegram broadcast.")
        return False

    url = f"{api_base}/bot{token}/sendMessage"
    payload = {
        "chat_id": target_chat,
        "text": message,
        "disable_web_page_preview": True,
    }
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

    try:
        resp = requests.post(url, json=payload, proxies=proxies, timeout=10)
        if not resp.ok:
            print(f"❌ Telegram API Error ({resp.status_code}): {resp.text}")
            logger.error(f"Telegram API Error ({resp.status_code}): {resp.text}")
            return False
        logger.info(f"Telegram signal sent to {target_chat}!")
        return True
    except requests.exceptions.ConnectionError as e:
        print(
            "❌ Telegram Connection Error: Could not reach api.telegram.org "
            "(Connection reset/blocked by network or ISP). "
            "Try enabling a VPN / Cloudflare WARP or setting TELEGRAM_PROXY_URL in .env."
        )
        safe_err = str(e).replace(token, "<REDACTED>") if token else str(e)
        logger.error(f"Telegram connection error: {safe_err}")
        return False
    except Exception as e:
        safe_err = str(e).replace(token, "<REDACTED>") if token else str(e)
        print(f"❌ Failed to send Telegram message: {safe_err}")
        logger.error(f"Failed to send Telegram message: {safe_err}")
        return False

