#!/usr/bin/env python3
"""
live_stream.py — DODA live runner with Alpaca streaming (tick -> 1m aggregation)

Requires:
  pip install alpaca-py pandas numpy matplotlib pytz requests

Env (Railway):
  API_KEY, API_SECRET, ALPACA_FEED=iex|sip, PAPER=1,
  SYMBOL, TELEGRAM_ENABLE, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TIMEZONE, etc.
"""
from __future__ import annotations

import os, asyncio, math, json
import datetime as dt
from dataclasses import asdict
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import requests
import pytz
import matplotlib.pyplot as plt

# Alpaca SDK (new)
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
        ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
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

# ---------- CLI ----------
def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default=os.getenv("SYMBOL","QQQ"))
    ap.add_argument("--tz", type=str, default="UTC")
    ap.add_argument("--session", type=int, nargs=2, default=[13,22], metavar=("START_H","END_H"))
    ap.add_argument("--initial", type=float, default=float(os.getenv("INITIAL","10000")))
    ap.add_argument("--commission", type=float, default=float(os.getenv("COMMISSION","2")))
    ap.add_argument("--slippage", type=float, default=float(os.getenv("SLIPPAGE","0.5")))
    ap.add_argument("--size", type=float, default=float(os.getenv("SIZE","1")))
    ap.add_argument("--history", type=int, default=int(os.getenv("HISTORY_MINUTES","1200")))
    return ap.parse_args()

# ---------- main ----------
async def main():
    args = cli()

    # Alpaca creds / config
    API_KEY = os.getenv("API_KEY")
    API_SECRET = os.getenv("API_SECRET")
    if not (API_KEY and API_SECRET):
        raise SystemExit("Missing API_KEY / API_SECRET")

    ALPACA_FEED = os.getenv("ALPACA_FEED", "iex").lower()
    FEED_ENUM = DataFeed.IEX if ALPACA_FEED == "iex" else DataFeed.SIP
    PAPER = os.getenv("PAPER","1") in ("1","true","True")

    symbol = args.symbol

    # DODA params (your best week can be hard-coded/env)
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
        commission_per_trade=args.commission,
        slippage=args.slippage,
        confirm_on_close=(os.getenv("CONFIRM_ON_CLOSE","0")=="1"),
        cooldown_bars=int(os.getenv("COOLDOWN_BARS","1")),
        session=(args.session[0], args.session[1]),
        size=args.size,
        base_equity=args.initial,
        use_stop_entries=(os.getenv("USE_STOP_ENTRIES","1")=="1"),
        stop_offset=float(os.getenv("STOP_OFFSET","0.5")),
        name="DODA_M1_M30"
    )

    # Notifier
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

    # Trading + historical data clients
    trading = TradingClient(API_KEY, API_SECRET, paper=PAPER)
    hist = StockHistoricalDataClient(API_KEY, API_SECRET)

    # Warm-up history (minute bars)
    end = pd.Timestamp.utcnow().tz_convert("UTC").floor("min")
    start = end - pd.Timedelta(minutes=args.history)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
                           start=start, end=end, feed=FEED_ENUM, adjustment=None)
    bars = hist.get_stock_bars(req)
    df_hist = bars.df
    if "symbol" in df_hist.columns:  # normalize single symbol
        df_hist = df_hist.xs(symbol)

    # Bot state
    bot = LiveDodaBot(p)
    cash = 0.0
    equity_rows: List[Dict[str, Any]] = []
    logs_dir = Path("logs"); logs_dir.mkdir(parents=True, exist_ok=True)
    trades_path = logs_dir/"trades_live.csv"
    equity_path = logs_dir/"equity_live.csv"
    equity_png  = logs_dir/"equity_live.png"

    # Seed bot with closed bars (all but the current minute)
    if not df_hist.empty:
        seed = df_hist.iloc[:-1]
        for ts, row in seed.iterrows():
            bot.step({"time": ts.isoformat(),
                      "open": float(row["open"]), "high": float(row["high"]),
                      "low": float(row["low"]), "close": float(row["close"]),
                      "volume": float(row.get("volume",0))})

    # Minute aggregator over trade ticks
    agg = MinuteAggregator()
    hourly_anchor = vie_now().replace(minute=0, second=0, microsecond=0)
    next_hourly = hourly_anchor + dt.timedelta(hours=1)

    # In-memory trade log for hourly/daily summaries
    trades_buffer: List[str] = []

    async def on_trade(t):
        """
        t: alpaca.data.models.Trade
        Attributes of interest: t.symbol, t.price, t.size, t.timestamp (ns)
        """
        # Convert ns timestamp -> pandas Timestamp UTC
        ts = pd.to_datetime(t.timestamp, utc=True)
        price = float(t.price)
        size = float(getattr(t, "size", 0) or 0)

        # Intrabar SL/TP: act immediately on tick
        if bot.position != 0:
            if bot.position == 1:
                if bot.sl is not None and price <= bot.sl:
                    # close long
                    order = MarketOrderRequest(symbol=symbol, qty=p.size, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                    trading.submit_order(order_data=order)
                    trades_buffer.append(f"{vie_now():%H:%M:%S} CLOSE LONG @ ~{price:.2f} (hit SL)")
                    # reset bot state (cooldown handled on next bar)
                    bot.position = 0; bot.entry_price=None; bot.sl=None; bot.tp=None
                elif bot.tp is not None and price >= bot.tp:
                    order = MarketOrderRequest(symbol=symbol, qty=p.size, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                    trading.submit_order(order_data=order)
                    trades_buffer.append(f"{vie_now():%H:%M:%S} CLOSE LONG @ ~{price:.2f} (hit TP)")
                    bot.position = 0; bot.entry_price=None; bot.sl=None; bot.tp=None
            else:
                if bot.sl is not None and price >= bot.sl:
                    order = MarketOrderRequest(symbol=symbol, qty=p.size, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                    trading.submit_order(order_data=order)
                    trades_buffer.append(f"{vie_now():%H:%M:%S} CLOSE SHORT @ ~{price:.2f} (hit SL)")
                    bot.position = 0; bot.entry_price=None; bot.sl=None; bot.tp=None
                elif bot.tp is not None and price <= bot.tp:
                    order = MarketOrderRequest(symbol=symbol, qty=p.size, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                    trading.submit_order(order_data=order)
                    trades_buffer.append(f"{vie_now():%H:%M:%S} CLOSE SHORT @ ~{price:.2f} (hit TP)")
                    bot.position = 0; bot.entry_price=None; bot.sl=None; bot.tp=None

        # Aggregate tick -> 1m
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
                        symbol=symbol, qty=p.size,
                        side=OrderSide.BUY if intent["action"]=="BUY" else OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    trading.submit_order(order_data=order)
                    trades_buffer.append(f"{vie_now():%H:%M:%S} {intent['action']} {p.size} {symbol} @ ~{bar['close']:.2f}")
                elif intent["action"] == "CLOSE":
                    # This path will rarely trigger now (we close on ticks),
                    # but keep as safety on minute roll.
                    side = OrderSide.SELL if bot.position==1 else OrderSide.BUY
                    order = MarketOrderRequest(symbol=symbol, qty=p.size, side=side, time_in_force=TimeInForce.DAY)
                    trading.submit_order(order_data=order)
                    trades_buffer.append(f"{vie_now():%H:%M:%S} CLOSE via minute bar")

            # mark-to-market equity (simple: last close)
            mtm = 0.0 if bot.position==0 else (bar["close"] - bot.entry_price) * (1 if bot.position==1 else -1) * p.size
            equity = p.base_equity + (-len([x for x in trades_buffer if "BUY" in x or "SELL" in x]) * p.commission_per_trade) + mtm
            equity_rows.append({"time": bar_ts, "equity": float(equity)})
            if len(equity_rows) % 20 == 0:
                edf = pd.DataFrame(equity_rows).set_index("time")
                edf.to_csv("logs/equity_live.csv")
                plot_equity(edf, "logs/equity_live.png")

        # Hourly & daily Telegram pings
        now_vie = vie_now()
        if now_vie >= next_hourly[0]:
            if trades_buffer:
                msg = "🕐 Hourly update\n" + "\n".join(trades_buffer[-20:])
            else:
                msg = "🕐 Hourly update\nNo trades this hour."
            try:
                TelegramNotifier().send(msg)
            except: pass
            next_hourly[0] = next_hourly[0] + dt.timedelta(hours=1)

        if now_vie.hour == 22 and now_vie.minute == 0 and now_vie.second < 5:
            day = [ln for ln in trades_buffer if ln.split(" ")[0] == f"{now_vie:%H:%M:%S}"]  # minimal guard
            try:
                TelegramNotifier().send("📅 End of day summary\n" + ( "\n".join(trades_buffer[-100:]) if trades_buffer else "No trades." ))
            except: pass
            trades_buffer.clear()

    # mutable container for next_hourly (so closure can update it)
    next_hourly = [hourly_anchor + dt.timedelta(hours=1)]

    # Prime aggregator with the last unfinished minute bar
    if not df_hist.empty:
        last = df_hist.iloc[-1]
        last_ts = last.name
        agg.curr_min = last_ts.tz_convert("UTC") if last_ts.tzinfo else last_ts.tz_localize("UTC")
        agg.bar = {"open": float(last["open"]), "high": float(last["high"]), "low": float(last["low"]),
                   "close": float(last["close"]), "volume": float(last.get("volume",0))}

    # Start data stream
    stream = StockDataStream(API_KEY, API_SECRET, feed=FEED_ENUM)  # IEX for free; SIP if enabled
    stream.subscribe_trades(on_trade, symbol)

    print(f"[stream] starting for {symbol} | feed={ALPACA_FEED} | paper={PAPER}")
    await stream.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")

