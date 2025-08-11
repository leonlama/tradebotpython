#!/usr/bin/env python3
"""
live_runner.py — run DODA LiveDodaBot on Alpaca

Providers:
  - alpaca (requires API key)

Usage examples:
  # Alpaca + QQQ
  python live_runner.py --symbol QQQ --tz UTC --session 13 22 --poll 30

Notes:
  - This is a paper runner; it logs intents and marks fills like the backtester.
  - Requires bot.py in the same folder.
"""

import os, sys, time, json, math
import datetime as dt
from dataclasses import asdict
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import pytz

from notifier import TelegramNotifier, fmt_stats_line, local_now
from bot import DodaParams, LiveDodaBot, ensure_datetime_index

TZ = pytz.timezone(os.getenv("TIMEZONE", "Europe/Vienna"))

def send_telegram(text: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        requests.get(
            f"https://api.telegram.org/bot{token}/sendMessage",
            params={"chat_id": chat_id, "text": text[:3990]},
            timeout=10,
        )
    except Exception:
        pass

def _vienna_now():
    return dt.datetime.now(tz=TZ)

def now_utc():
    return dt.datetime.now(dt.timezone.utc)

def round_down_minute(x: dt.datetime) -> dt.datetime:
    return x.replace(second=0, microsecond=0)

# --- add near the other imports ---
import os
from datetime import timezone
# optional: only import alpaca when used
def fetch_alpaca(symbol: str, n_minutes: int = 600) -> pd.DataFrame:
    try:
        from alpaca_trade_api.rest import REST as AlpacaREST, TimeFrame, TimeFrameUnit
    except Exception as e:
        print("Please: pip install alpaca-trade-api", file=sys.stderr)
        raise

    base_url = os.getenv("BROKER_URL", "https://paper-api.alpaca.markets")
    key = os.getenv("API_KEY")
    secret = os.getenv("API_SECRET")
    if not (key and secret):
        raise RuntimeError("Missing API_KEY / API_SECRET for Alpaca")

    api = AlpacaREST(key_id=key, secret_key=secret, base_url=base_url)

    end = pd.Timestamp.utcnow().tz_localize("UTC").floor("min")
    start = end - pd.Timedelta(minutes=n_minutes + 5)

    bars = api.get_bars(
        symbol,
        TimeFrame(1, TimeFrameUnit.Minute),
        start.isoformat(),
        end.isoformat(),
        adjustment=None,
        limit=n_minutes + 5
    )

    df = bars.df
    if df.empty:
        raise RuntimeError("Alpaca returned empty data")
    # When multiple symbols are requested, df has 'symbol' col + multi-index;
    # for single symbol it’s simpler, but normalize anyway:
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol]

    # Ensure UTC index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Keep canonical column names
    df = df.rename(columns={
        "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume"
    })[["open", "high", "low", "close", "volume"]]
    return df

# ---------------- Plot ----------------

def plot_equity(equity_df: pd.DataFrame, out_png: str):
    plt.figure(figsize=(9,5))
    equity_df["equity"].plot()
    plt.title("Live Equity")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default="QQQ", help="Symbol for Alpaca")
    ap.add_argument("--tz", type=str, default="UTC")
    ap.add_argument("--session", type=int, nargs=2, default=[13,22], metavar=("START_H","END_H"))
    ap.add_argument("--poll", type=int, default=30, help="Seconds between data polls")
    ap.add_argument("--initial", type=float, default=10_000.0)
    ap.add_argument("--commission", type=float, default=2.0)
    ap.add_argument("--slippage", type=float, default=0.5)
    ap.add_argument("--size", type=float, default=1.0)
    ap.add_argument("--history", type=int, default=1200, help="Minutes of history to bootstrap indicators")
    args = ap.parse_args()

    # Best-1-week params
    p = DodaParams(
        sig_fast=3, sig_mid=17,
        tr_fast=21, tr_mid=55, tr_slow=144,
        atr_period=20, sl_atr=1.0935227502529754, tp_atr=0.0,
        breakeven_rr=1.0, trail_atr=1.75,
        commission_per_trade=args.commission, slippage=args.slippage,
        confirm_on_close=False, cooldown_bars=1,
        session=(args.session[0], args.session[1]),
        size=args.size, base_equity=args.initial,
        use_stop_entries=True, stop_offset=1.100747969496732
    )

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    trades_path = logs_dir / "trades_live.csv"
    equity_path = logs_dir / "equity_live.csv"
    equity_png  = logs_dir / "equity_live.png"

    # state
    bot = LiveDodaBot(p)
    cash = 0.0
    position = 0
    trades: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    # notification setup
    notifier = TelegramNotifier()
    session_start, session_end = args.session if args.session else (None, None)

    # buffers for hourly & daily summaries  
    hourly_events: list[str] = []
    last_hour_sent_utc: Optional[int] = None
    day_accumulator = {
        "trades": 0,
        "gross_pnl": 0.0,
        "wins": 0,
        "losses": 0,
    }

    # Telegram scheduling state
    last_hourly = _vienna_now().replace(minute=0, second=0, microsecond=0)
    next_hourly = last_hourly + dt.timedelta(hours=1)

    def next_22_vienna(from_dt=None):
        now = _vienna_now() if from_dt is None else from_dt.astimezone(TZ)
        target = now.replace(hour=22, minute=0, second=0, microsecond=0)
        if now >= target:
            target = target + dt.timedelta(days=1)
        return target

    next_daily = next_22_vienna()
    trades_buffer = []

    def send_boot_message():
        if not notifier.enabled: return
        msg = (
            "✅ *DODA Live Bot started*\n"
            f"Symbol: *{args.symbol}*\n"
            f"Session (UTC): *{session_start}-{session_end}*\n"
            f"Params: sig({p.sig_fast}/{p.sig_mid}) | trend({p.tr_fast}/{p.tr_mid}/{p.tr_slow})\n"
            f"Risk: ATR{p.atr_period} SLx{p.sl_atr} TPx{p.tp_atr} "
            f"BE:{p.breakeven_rr} Trail:{p.trail_atr}\n"
            f"Stops: {'stop-entries' if p.use_stop_entries else 'market-open'} off={p.stop_offset}\n"
            f"Costs: comm={p.commission_per_trade} slip={p.slippage}\n"
            f"Size: {p.size} | Base: {p.base_equity}"
        )
        notifier.send(msg)

    def maybe_send_hourly(now_utc: dt.datetime):
        nonlocal last_hour_sent_utc, hourly_events
        hour_marker = int(now_utc.replace(minute=0, second=0, microsecond=0).timestamp())
        if last_hour_sent_utc is None:
            last_hour_sent_utc = hour_marker
            return
        if hour_marker != last_hour_sent_utc:
            if notifier.enabled:
                text = "🕐 *Hourly update*\n" + ("\n".join(hourly_events) if hourly_events else "_No trades this hour_")
                notifier.send(text)
            hourly_events = []
            last_hour_sent_utc = hour_marker

    def maybe_send_daily_summary(now_utc: dt.datetime):
        vienna_now = dt.datetime.now(ZoneInfo("Europe/Vienna"))
        if vienna_now.hour == 22 and vienna_now.minute == 0:
            if notifier.enabled:
                pnl = day_accumulator["gross_pnl"]
                msg = (
                    "📊 *Daily Summary (Vienna 22:00)*\n"
                    f"Trades: *{day_accumulator['trades']}* | Wins: *{day_accumulator['wins']}* | "
                    f"Losses: *{day_accumulator['losses']}*\n"
                    f"Net PnL: *{pnl:+.2f}*"
                )
                notifier.send(msg)
            day_accumulator["trades"] = 0
            day_accumulator["gross_pnl"] = 0.0
            day_accumulator["wins"] = 0
            day_accumulator["losses"] = 0

    def get_today_trades():
        orders = api.list_orders(status="all", limit=50)
        return [f"{o.side.upper()} {o.qty} {o.symbol} @ {o.filled_avg_price}" for o in orders]

    # Which provider? (env var, not CLI)
    provider = os.getenv("PROVIDER", "yahoo").lower()
    symbol = args.symbol or os.getenv("SYMBOL", "QQQ")

    print(f"[boot] fetching history via {provider.capitalize()} symbol={symbol}")
    if provider == "alpaca":
        df_hist = fetch_alpaca(symbol, n_minutes=args.history)
    elif provider == "yahoo":
        df_hist = fetch_yahoo(symbol, n_minutes=args.history)
    else:
        # fallback to yahoo
        df_hist = fetch_yahoo(symbol, n_minutes=args.history)

    # only feed closed bars
    for ts, row in df_hist.iloc[:-1].iterrows():
        bot.step({
            "time": ts.isoformat(),
            "open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"],
            "volume": row["volume"],
        })

    last_seen_ts: Optional[pd.Timestamp] = df_hist.index[-1] if not df_hist.empty else None
    print("[live] entering loop; Ctrl+C to stop")

    send_boot_message()
    send_telegram("✅ DODA bot started on Railway\nSymbol: "
                 f"{os.getenv('SYMBOL','QQQ')} | TF: {os.getenv('TIMEFRAME','1Min')}")

    try:
        while True:
            now_utc = dt.datetime.now(tz=ZoneInfo("UTC"))

            # fetch recent window
            try:
                if provider == "alpaca":
                    df = fetch_alpaca(symbol, n_minutes=180)
                else:
                    df = fetch_yahoo(symbol, n_minutes=180)
            except Exception as e:
                print(f"[warn] fetch error: {e}")
                time.sleep(max(5, args.poll))
                continue

            cutoff = round_down_minute(now_utc)
            df = df[df.index < cutoff]
            if last_seen_ts is not None:
                df = df[df.index > last_seen_ts]

            if not df.empty:
                for ts, row in df.iterrows():
                    intents = bot.step({
                        "time": ts.isoformat(),
                        "open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"],
                        "volume": row["volume"],
                    })
                    # execute intents via Alpaca
                    for intent in intents:
                        if intent["action"] in ("BUY","SELL"):
                            # Place order via Alpaca
                            api.submit_order(
                                symbol=args.symbol,
                                qty=p.size,
                                side=intent["action"].lower(),
                                type="market",
                                time_in_force="day"
                            )
                            send_telegram(f"📈 {intent['action']} {p.size} {args.symbol} @ market price via Alpaca Paper Trading.")
                            trades_buffer.append({
                                "t": _vienna_now().strftime("%Y-%m-%d %H:%M:%S"),
                                "side": intent["action"],
                                "qty": p.size,
                                "px": round(row["close"], 4),
                                "reason": "signal"
                            })
                        elif intent["action"]=="CLOSE":
                            # realized PnL computed using bot's entry_price and current close
                            side = 1 if "LONG" in intent.get("reason","") else ( -1 if "SHORT" in intent.get("reason","") else 0 )
                            # we'll derive side from bot state; bot.entry_price exists
                            if getattr(bot, "entry_price", None) is not None:
                                exit_px = row["close"]
                                pnl = (exit_px - bot.entry_price) * (1 if bot.position==1 else -1) * p.size - p.commission_per_trade
                                trades.append({
                                    "entry_time": ts, "exit_time": ts,
                                    "side": "LONG" if bot.position==1 else "SHORT" if bot.position==-1 else "FLAT",
                                    "entry": float(bot.entry_price), "exit": float(exit_px),
                                    "pnl": float(pnl), "reason": intent.get("reason","CLOSE")
                                })
                                cash += pnl

                                trades_buffer.append({
                                    "t": _vienna_now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "side": "CLOSE",
                                    "qty": p.size,
                                    "px": round(exit_px, 4),
                                    "reason": intent.get("reason","CLOSE")
                                })

                                # Add trade notification
                                side_str = "LONG" if bot.position==1 else "SHORT"
                                tline = f"• {side_str} closed | PnL {pnl:+.2f}"
                                hourly_events.append(tline)

                                day_accumulator["trades"] += 1
                                day_accumulator["gross_pnl"] += pnl
                                if pnl > 0:
                                    day_accumulator["wins"] += 1
                                elif pnl < 0:
                                    day_accumulator["losses"] += 1

                    # equity mark-to-market
                    px = row["close"]
                    mtm = 0.0 if bot.position==0 else (px - bot.entry_price) * (1 if bot.position==1 else -1) * p.size
                    equity = p.base_equity + cash + mtm
                    equity_rows.append({"time": ts, "equity": float(equity)})

                last_seen_ts = df.index[-1]
                # persist
                if trades:
                    pd.DataFrame(trades).to_csv(trades_path, index=False)
                if equity_rows:
                    edf = pd.DataFrame(equity_rows).set_index("time")
                    edf.to_csv(equity_path)
                    plot_equity(edf, str(equity_png))
                    print(f"[live] {now_utc.isoformat()} updated -> {equity_png} last={last_seen_ts}")

            # periodic notifications
            maybe_send_hourly(now_utc)
            maybe_send_daily_summary(now_utc)

            # Telegram periodic notifications
            now_vie = _vienna_now()

            # Hourly update
            if now_vie >= next_hourly:
                send_telegram("📊 Hourly update:\n" + "\n".join(get_today_trades()))
                next_hourly += dt.timedelta(hours=1)

            # Daily summary at 22:00 Vienna
            if now_vie >= next_daily:
                send_telegram("📅 End of day summary:\n" + "\n".join(get_today_trades()))
                trades_buffer.clear()
                next_daily = next_22_vienna(now_vie)

            time.sleep(max(5, int(args.poll)))
    except KeyboardInterrupt:
        print("Stopped by user.")
        if trades:
            pd.DataFrame(trades).to_csv(trades_path, index=False)
        if equity_rows:
            edf = pd.DataFrame(equity_rows).set_index("time")
            edf.to_csv(equity_path)
            plot_equity(edf, str(equity_png))
        print("Logs saved.")

if __name__ == "__main__":
    main()

