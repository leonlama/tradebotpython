#!/usr/bin/env python3
"""
live_stream.py — DODA live runner with Alpaca streaming (tick -> 1m aggregation)

Requires:
  pip install alpaca-py pandas numpy matplotlib pytz requests

Env (Railway):
  API_KEY, API_SECRET, PAPER=1, ALPACA_FEED=iex|sip,
  SYMBOL, INITIAL, COMMISSION, SLIPPAGE, SIZE, HISTORY_MINUTES,
  SIG_* TR_* ATR_* etc.,
  TELEGRAM_ENABLE, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TIMEZONE
"""

from __future__ import annotations

import os
import sys
import math
import json
import time
import datetime as dt
from dataclasses import asdict
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import requests
import matplotlib.pyplot as plt

# Alpaca SDK (alpaca-py)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from notifier import TelegramNotifier
from bot import DodaParams, LiveDodaBot

# ---------- helpers ----------
TZ = pytz.timezone(os.getenv("TIMEZONE", "Europe/Vienna"))

def vie_now() -> dt.datetime:
    return dt.datetime.now(TZ)

def plot_equity(equity_df: pd.DataFrame, out_png: str):
    plt.figure(figsize=(9,5))
    equity_df["equity"].plot()
    plt.title("Live Equity")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

class MinuteAggregator:
    """Aggregate trade ticks into 1-minute OHLCV and intrabar extremes."""
    def __init__(self):
        self.curr_min: Optional[pd.Timestamp] = None
        self.bar: Optional[Dict[str, float]] = None

    def push(self, ts: pd.Timestamp, price: float, size: float=0.0):
        # ensure UTC floor to minute
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        m = ts.replace(second=0, microsecond=0)
        if self.curr_min is None:
            self.curr_min = m
            self.bar = {"open":price, "high":price, "low":price, "close":price, "volume":size}
            return None
        if m == self.curr_min:
            b = self.bar
            b["high"] = max(b["high"], price)
            b["low"]  = min(b["low"],  price)
            b["close"] = price
            b["volume"] += size
            return None
        # minute rolled
        finished = (self.curr_min, self.bar.copy())
        self.curr_min = m
        self.bar = {"open":price, "high":price, "low":price, "close":price, "volume":size}
        return finished

def must_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"[fatal] Missing env var: {name}", file=sys.stderr)
        sys.exit(1)
    return v

# ---------- main (synchronous) ----------
def main():
    # Alpaca creds / config
    API_KEY    = must_env("API_KEY")
    API_SECRET = must_env("API_SECRET")
    PAPER      = os.getenv("PAPER","1").lower() in ("1","true","yes")
    ALPACA_FEED = os.getenv("ALPACA_FEED","iex").lower()
    FEED_ENUM   = DataFeed.IEX if ALPACA_FEED == "iex" else DataFeed.SIP

    symbol   = os.getenv("SYMBOL", "QQQ")
    initial  = float(os.getenv("INITIAL","10000"))
    commission = float(os.getenv("COMMISSION","2"))
    slippage   = float(os.getenv("SLIPPAGE","0.5"))
    size       = float(os.getenv("SIZE","1"))
    history_m  = int(os.getenv("HISTORY_MINUTES","1200"))

    # DODA params (from env with sane defaults)
    p = DodaParams(
        sig_fast=int(os.getenv("SIG_FAST","3")),
        sig_mid=int(os.getenv("SIG_MID","21")),
        tr_fast=int(os.getenv("TR_FAST","13")),
        tr_mid=int(os.getenv("TR_MID","55")),
        tr_slow=int(os.getenv("TR_SLOW","144")),
        atr_period=int(os.getenv("ATR_PERIOD","15")),
        sl_atr=float(os.getenv("SL_ATR","1.7006674335")),
        tp_atr=float(os.getenv("TP_ATR","2.5")),
        breakeven_rr=float(os.getenv("BREAKEVEN_RR","1.0")),
        trail_atr=float(os.getenv("TRAIL_ATR","1.75")),
        commission_per_trade=commission,
        slippage=slippage,
        confirm_on_close=(os.getenv("CONFIRM_ON_CLOSE","0")=="1"),
        cooldown_bars=int(os.getenv("COOLDOWN_BARS","1")),
        session=(int(os.getenv("SESSION_START","13")), int(os.getenv("SESSION_END","22"))),
        size=size,
        base_equity=initial,
        use_stop_entries=(os.getenv("USE_STOP_ENTRIES","1")=="1"),
        stop_offset=float(os.getenv("STOP_OFFSET","0.5")),
        name="DODA_M1_M30"
    )

    # Telegram notifier
    notifier = TelegramNotifier()
    if notifier.enabled:
        notifier.send(
            "✅ *DODA Live Bot (streaming)*\n"
            f"Symbol: *{symbol}* | Feed: *{ALPACA_FEED.upper()}* | Paper: *{PAPER}*\n"
            f"Session UTC: {p.session}\n"
            f"sig({p.sig_fast}/{p.sig_mid}) trend({p.tr_fast}/{p.tr_mid}/{p.tr_slow})\n"
            f"ATR{p.atr_period} SLx{p.sl_atr} TPx{p.tp_atr} BE:{p.breakeven_rr} Trail:{p.trail_atr}\n"
            f"Stops: {'stop-entries' if p.use_stop_entries else 'market-open'} off={p.stop_offset}\n"
            f"Costs: comm={p.commission_per_trade} slip={p.slippage}"
        )

    # Clients
    trading = TradingClient(API_KEY, API_SECRET, paper=PAPER)
    hist    = StockHistoricalDataClient(API_KEY, API_SECRET)

    # Warm-up history (minute bars)
    end   = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(minutes=history_m)
    req   = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=FEED_ENUM,
        adjustment=None
    )
    bars = hist.get_stock_bars(req)
    df_hist = bars.df
    if isinstance(df_hist.index, pd.MultiIndex) or "symbol" in df_hist.columns:
        # normalize for single symbol
        try:
            df_hist = df_hist.xs(symbol)
        except Exception:
            df_hist = df_hist[df_hist.get("symbol", symbol) == symbol]
    # Bot state
    bot = LiveDodaBot(p)
    equity_rows: List[Dict[str, Any]] = []
    logs_dir = Path("logs"); logs_dir.mkdir(parents=True, exist_ok=True)
    trades_path = logs_dir/"trades_live.csv"
    equity_path = logs_dir/"equity_live.csv"
    equity_png  = logs_dir/"equity_live.png"

    # Seed bot with closed bars
    if not df_hist.empty:
        seed = df_hist.iloc[:-1]
        for ts, row in seed.iterrows():
            bot.step({
                "time": ts.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low":  float(row["low"]),
                "close":float(row["close"]),
                "volume": float(row.get("volume", 0.0))
            })

    # Aggregator & reporting timers
    agg = MinuteAggregator()
    hourly_anchor = vie_now().replace(minute=0, second=0, microsecond=0)
    next_hourly   = hourly_anchor + dt.timedelta(hours=1)
    next_daily    = vie_now().replace(hour=22, minute=0, second=0, microsecond=0)
    if vie_now() >= next_daily:
        next_daily += dt.timedelta(days=1)

    trades_buffer: List[str] = []
    realized_cash: float = 0.0  # realized PnL minus commissions

    # Prime aggregator with last unfinished minute (if exists)
    if not df_hist.empty:
        last = df_hist.iloc[-1]
        last_ts = last.name
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")
        agg.curr_min = last_ts.replace(second=0, microsecond=0)
        agg.bar = {
            "open":  float(last["open"]),
            "high":  float(last["high"]),
            "low":   float(last["low"]),
            "close": float(last["close"]),
            "volume": float(last.get("volume", 0.0))
        }

    # Trade handler (async per SDK), but we keep whole program synchronous via stream.run()
    async def on_trade(t):
        nonlocal next_hourly, next_daily, realized_cash

        # ts can be ns; pandas handles
        ts = pd.to_datetime(getattr(t, "timestamp", None), utc=True)
        price = float(getattr(t, "price", np.nan))
        size  = float(getattr(t, "size", 0.0) or 0.0)
        if not np.isfinite(price):
            return

        # Intrabar SL/TP: act immediately
        if bot.position != 0:
            if bot.position == 1:
                if bot.sl is not None and price <= bot.sl:
                    # CLOSE LONG
                    order = MarketOrderRequest(symbol=symbol, qty=p.size, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                    try: trading.submit_order(order_data=order)
                    except Exception as e: print(f"[order] close long failed: {e}")
                    # Compute PnL
                    if bot.entry_price is not None:
                        pnl = (price - bot.entry_price) * p.size - p.commission_per_trade
                        realized_cash += pnl
                    trades_buffer.append(f"{vie_now():%H:%M:%S} CLOSE LONG @ ~{price:.2f} (SL)")
                    # reset internal position immediately
                    bot.position = 0; bot.entry_price=None; bot.sl=None; bot.tp=None
                elif bot.tp is not None and price >= bot.tp:
                    order = MarketOrderRequest(symbol=symbol, qty=p.size, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                    try: trading.submit_order(order_data=order)
                    except Exception as e: print(f"[order] close long TP failed: {e}")
                    if bot.entry_price is not None:
                        pnl = (price - bot.entry_price) * p.size - p.commission_per_trade
                        realized_cash += pnl
                    trades_buffer.append(f"{vie_now():%H:%M:%S} CLOSE LONG @ ~{price:.2f} (TP)")
                    bot.position = 0; bot.entry_price=None; bot.sl=None; bot.tp=None
            else:
                if bot.sl is not None and price >= bot.sl:
                    order = MarketOrderRequest(symbol=symbol, qty=p.size, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                    try: trading.submit_order(order_data=order)
                    except Exception as e: print(f"[order] close short SL failed: {e}")
                    if bot.entry_price is not None:
                        pnl = (bot.entry_price - price) * p.size - p.commission_per_trade
                        realized_cash += pnl
                    trades_buffer.append(f"{vie_now():%H:%M:%S} CLOSE SHORT @ ~{price:.2f} (SL)")
                    bot.position = 0; bot.entry_price=None; bot.sl=None; bot.tp=None
                elif bot.tp is not None and price <= bot.tp:
                    order = MarketOrderRequest(symbol=symbol, qty=p.size, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                    try: trading.submit_order(order_data=order)
                    except Exception as e: print(f"[order] close short TP failed: {e}")
                    if bot.entry_price is not None:
                        pnl = (bot.entry_price - price) * p.size - p.commission_per_trade
                        realized_cash += pnl
                    trades_buffer.append(f"{vie_now():%H:%M:%S} CLOSE SHORT @ ~{price:.2f} (TP)")
                    bot.position = 0; bot.entry_price=None; bot.sl=None; bot.tp=None

        # Tick -> minute bar
        finished = agg.push(ts, price, size)
        if finished is not None:
            bar_ts, bar = finished
            intents = bot.step({
                "time": bar_ts.isoformat(),
                "open": bar["open"], "high": bar["high"], "low": bar["low"], "close": bar["close"],
                "volume": bar["volume"]
            })
            for intent in intents:
                if intent["action"] in ("BUY","SELL"):
                    order = MarketOrderRequest(
                        symbol=symbol,
                        qty=p.size,
                        side=OrderSide.BUY if intent["action"]=="BUY" else OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    try:
                        trading.submit_order(order_data=order)
                        realized_cash -= p.commission_per_trade
                    except Exception as e:
                        print(f"[order] entry failed: {e}")
                    trades_buffer.append(f"{vie_now():%H:%M:%S} {intent['action']} {p.size} {symbol} @ ~{bar['close']:.2f}")
                elif intent["action"] == "CLOSE":
                    # Safety net close on minute roll (usually handled intrabar already)
                    side = OrderSide.SELL if bot.position==1 else OrderSide.BUY
                    order = MarketOrderRequest(symbol=symbol, qty=p.size, side=side, time_in_force=TimeInForce.DAY)
                    try:
                        trading.submit_order(order_data=order)
                        realized_cash -= p.commission_per_trade
                    except Exception as e:
                        print(f"[order] safety close failed: {e}")
                    trades_buffer.append(f"{vie_now():%H:%M:%S} CLOSE via minute bar")

            # Mark-to-market equity on bar close
            if bot.position == 0 or bot.entry_price is None:
                mtm = 0.0
            else:
                mtm = (bar["close"] - bot.entry_price) * (1 if bot.position==1 else -1) * p.size
            equity = p.base_equity + realized_cash + mtm
            equity_rows.append({"time": bar_ts, "equity": float(equity)})
            if len(equity_rows) % 20 == 0:
                edf = pd.DataFrame(equity_rows).set_index("time")
                edf.to_csv(equity_path)
                plot_equity(edf, str(equity_png))
                print(f"[live] equity updated -> {equity_png} last={bar_ts}")

        # Hourly telegram
        now_v = vie_now()
        if now_v >= next_hourly:
            if notifier.enabled:
                msg = "🕐 Hourly update\n" + ("\n".join(trades_buffer[-20:]) if trades_buffer else "No trades this hour.")
                notifier.send(msg)
            next_hourly += dt.timedelta(hours=1)

        # Daily telegram at 22:00 Vienna
        if now_v >= next_daily:
            if notifier.enabled:
                msg = "📅 End of day summary\n" + ("\n".join(trades_buffer[-100:]) if trades_buffer else "No trades.")
                notifier.send(msg)
            trades_buffer.clear()
            next_daily = next_daily + dt.timedelta(days=1)

    # Start stream (SDK manages its own asyncio loop internally)
    stream = StockDataStream(API_KEY, API_SECRET, feed=ALPACA_FEED)  # 'iex' or 'sip'
    stream.subscribe_trades(on_trade, symbol)
    print(f"[stream] starting for {symbol} | feed={ALPACA_FEED} | paper={PAPER}")
    try:
        stream.run()  # blocking; no asyncio.run() around this
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        # Persist equity on shutdown
        if equity_rows:
            edf = pd.DataFrame(equity_rows).set_index("time")
            edf.to_csv("logs/equity_live.csv")
            plot_equity(edf, "logs/equity_live.png")
        print("Logs saved.")

if __name__ == "__main__":
    main()

