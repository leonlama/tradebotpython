#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SODA Trader — Alpaca paper-trading bot (multi-symbol)

Strategy (per symbol):
  - SIGNAL (M1): triple-EMA crossover -> BUY/SELL/NEUTRAL
  - TREND  (30m): triple-EMA on 30-min resample -> BUY/SELL/NEUTRAL
  - ENTRY:  LONG if both BUY; SHORT if both SELL
  - EXIT:   close if Signal & Trend diverge OR if ATR-based SL/TP is hit

Risk:
  - ATR_PERIOD (minutes), SL_ATR, TP_ATR
  - Optional breakeven (BREAKEVEN_RR) + ATR trailing (TRAIL_ATR)

Ops:
  - Batches one bars API call for all symbols per poll (well under 200 req/min)
  - Hourly Telegram update, daily 22:00 Europe/Vienna recap
  - Session filter via SESSION_START..SESSION_END (UTC hours). Leave blank to trade all day.

Env (examples):
  APCA_API_KEY_ID=...
  APCA_API_SECRET_KEY=...
  PAPER=1
  ALPACA_FEED=iex        # or sip if entitled
  SYMBOLS=SPY,QQQ,DIA,DAX,GLD,USO
  POLL_SECONDS=30
  HISTORY_MINUTES=1200
  SIG_FAST=3 SIG_MID=21 SIG_SLOW=34
  TR_FAST=13 TR_MID=55 TR_SLOW=144
  ATR_PERIOD=15 SL_ATR=1.7 TP_ATR=2.5
  BREAKEVEN_RR=1.0 TRAIL_ATR=1.75
  SESSION_START=13 SESSION_END=22
  SIZE=1 COMMISSION=2 SLIPPAGE=0.5
  TELEGRAM_ENABLE=1 TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
  TIMEZONE=Europe/Vienna
"""

from __future__ import annotations

import os
import time
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import pytz
import numpy as np
import pandas as pd
import requests

# Alpaca SDK
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# TA
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# ---------------- logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("SODA")

# ---------------- env & config ----------------
API_KEY = os.getenv("APCA_API_KEY_ID") or os.getenv("API_KEY")
API_SECRET = os.getenv("APCA_API_SECRET_KEY") or os.getenv("API_SECRET")
if not (API_KEY and API_SECRET):
    raise SystemExit("Missing APCA_API_KEY_ID/APCA_API_SECRET_KEY (or API_KEY/API_SECRET)")

PAPER = os.getenv("PAPER", "1").lower() in ("1", "true", "yes")
ALPACA_FEED = os.getenv("ALPACA_FEED", "iex").lower()
FEED = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}.get(ALPACA_FEED, DataFeed.IEX)

SYMBOLS = [s.strip().upper() for s in (os.getenv("SYMBOLS") or "SPY,QQQ,DIA,DAX,GLD,USO").split(",") if s.strip()]

POLL_SECONDS = int(os.getenv("POLL_SECONDS", 30))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", 15))
HISTORY_MINUTES = max(int(os.getenv("HISTORY_MINUTES", 1200)), ATR_PERIOD + 50)  # ensure warmup

# signal (M1) triple-EMA
SIG_FAST = int(os.getenv("SIG_FAST", 3))
SIG_MID  = int(os.getenv("SIG_MID", 21))
SIG_SLOW = int(os.getenv("SIG_SLOW", 34))

# trend (30m) triple-EMA
TR_FAST = int(os.getenv("TR_FAST", 13))
TR_MID  = int(os.getenv("TR_MID", 55))
TR_SLOW = int(os.getenv("TR_SLOW", 144))

SL_ATR    = float(os.getenv("SL_ATR", "1.7"))
TP_ATR    = float(os.getenv("TP_ATR", "2.5"))
BE_RR     = float(os.getenv("BREAKEVEN_RR", "1.0"))
TRAIL_ATR = float(os.getenv("TRAIL_ATR", "1.75"))

SIZE       = float(os.getenv("SIZE", "1"))
COMMISSION = float(os.getenv("COMMISSION", "2"))
SLIPPAGE   = float(os.getenv("SLIPPAGE", "0.5"))

# Session window (UTC)
SESSION_START = os.getenv("SESSION_START")
SESSION_END   = os.getenv("SESSION_END")
SESSION_START_H = int(SESSION_START) if SESSION_START and SESSION_START.strip().isdigit() else None
SESSION_END_H   = int(SESSION_END) if SESSION_END and SESSION_END.strip().isdigit() else None

# Telegram
TELEGRAM_ENABLE = os.getenv("TELEGRAM_ENABLE", "0") == "1"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# Timezone for daily recap
TZ_VIE = pytz.timezone(os.getenv("TIMEZONE", "Europe/Vienna"))

# ---------------- utils ----------------
def vie_now() -> datetime:
    return datetime.now(TZ_VIE)

def utc_floor_min() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("min")

def in_session(ts: pd.Timestamp) -> bool:
    if SESSION_START_H is None or SESSION_END_H is None:
        return True
    hr = ts.tz_convert("UTC").hour
    if SESSION_START_H <= SESSION_END_H:
        return SESSION_START_H <= hr < SESSION_END_H
    return hr >= SESSION_START_H or hr < SESSION_END_H

def send_telegram(msg: str) -> None:
    if not TELEGRAM_ENABLE or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=8,
        )
    except Exception as e:
        log.warning(f"[telegram] send failed: {e}")

def ema(series: pd.Series, period: int) -> pd.Series:
    return EMAIndicator(close=series.astype(float), window=int(period)).ema_indicator()

def triple_state(close: pd.Series, f: int, m: int, s: int) -> pd.Series:
    ef, em, es = ema(close, f), ema(close, m), ema(close, s)
    buy  = (ef > em) & (em > es)
    sell = (ef < em) & (em < es)
    return pd.Series(np.where(buy, "BUY", np.where(sell, "SELL", "NEUTRAL")), index=close.index)

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    if len(df) < period + 2:
        return pd.Series(index=df.index, dtype=float)
    atr = AverageTrueRange(
        high=df["high"].astype(float),
        low=df["low"].astype(float),
        close=df["close"].astype(float),
        window=period
    ).average_true_range()
    return atr

def resample_30m(df_1m: pd.DataFrame) -> pd.DataFrame:
    # Right-closed, right-labeled 30m
    r = df_1m.resample("30min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return r.dropna(how="any")

def fetch_m1_multi(data_client: StockHistoricalDataClient,
                   symbols: List[str],
                   minutes: int,
                   feed: DataFeed) -> Dict[str, pd.DataFrame]:
    end = utc_floor_min()
    start = end - pd.Timedelta(minutes=minutes)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=feed,
        adjustment=None
    )
    bars = data_client.get_stock_bars(req)
    out: Dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in symbols}
    if bars is None or bars.df is None or bars.df.empty:
        return out
    df = bars.df
    if isinstance(df.index, pd.MultiIndex) or "symbol" in df.columns:
        for s in symbols:
            try:
                d = df.xs(s)
            except Exception:
                d = df[df.get("symbol", s) == s]
            if not d.empty:
                idx = d.index
                if idx.tz is None:
                    d.index = idx.tz_localize("UTC")
                else:
                    d.index = idx.tz_convert("UTC")
                out[s] = d[["open", "high", "low", "close", "volume"]].astype(float)
    else:
        # single symbol fallback
        s = symbols[0]
        d = df
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        else:
            d.index = d.index.tz_convert("UTC")
        out[s] = d[["open", "high", "low", "close", "volume"]].astype(float)
    return out

def submit_market(trading: TradingClient, symbol: str, side: OrderSide, qty: int) -> None:
    order = MarketOrderRequest(symbol=symbol, qty=int(qty), side=side, time_in_force=TimeInForce.DAY)
    trading.submit_order(order)

# ---------------- main ----------------
def main():
    trading = TradingClient(API_KEY, API_SECRET, paper=PAPER)
    data     = StockHistoricalDataClient(API_KEY, API_SECRET)

    send_telegram(
        "✅ SODA Trader started\n"
        f"Symbols: {', '.join(SYMBOLS)} | Feed: {ALPACA_FEED.upper()} | Paper: {PAPER}\n"
        f"Signal EMA: {SIG_FAST}/{SIG_MID}/{SIG_SLOW} | Trend(30m): {TR_FAST}/{TR_MID}/{TR_SLOW}\n"
        f"ATR{ATR_PERIOD} SLx{SL_ATR} TPx{TP_ATR} BE:{BE_RR} Trail:{TRAIL_ATR}\n"
        f"Session UTC: {SESSION_START or '-'}-{SESSION_END or '-'} | Size: {SIZE}"
    )

    # Per-symbol state
    state: Dict[str, Dict[str, Any]] = {
        s: {"pos": None, "entry": None, "stop": None, "take": None, "log": []} for s in SYMBOLS
    }

    # Try to sync existing open positions (paper account)
    try:
        open_positions = trading.get_all_positions()
        for p in open_positions:
            sym = p.symbol.upper()
            if sym in state:
                qty = float(p.qty)
                state[sym]["pos"] = "LONG" if qty > 0 else "SHORT"
                state[sym]["entry"] = float(p.avg_entry_price)
                # SL/TP will be (re)built on first loop when ATR is known
                log.info(f"[sync] {sym}: found {state[sym]['pos']} @ {state[sym]['entry']}")
    except Exception as e:
        log.warning(f"[sync] could not fetch open positions: {e}")

    # Scheduling for Telegram summaries
    hourly_anchor = vie_now().replace(minute=0, second=0, microsecond=0)
    next_hourly = hourly_anchor + timedelta(hours=1)
    next_daily = vie_now().replace(hour=22, minute=0, second=0, microsecond=0)
    if vie_now() >= next_daily:
        next_daily += timedelta(days=1)

    log.info(f"[loop] starting — symbols={','.join(SYMBOLS)} poll={POLL_SECONDS}s feed={ALPACA_FEED}")
    while True:
        loop_t0 = time.time()
        try:
            log.info(f"[loop] polling data — symbols={','.join(SYMBOLS)} feed={ALPACA_FEED}")
            dfs = fetch_m1_multi(data, SYMBOLS, HISTORY_MINUTES, FEED)

            for sym in SYMBOLS:
                df = dfs.get(sym, pd.DataFrame())
                if df.empty:
                    log.warning(f"[{sym}] no bars")
                    continue

                # Only closed bars
                df = df[df.index < utc_floor_min()]
                if df.empty:
                    continue

                # Session filter (if configured)
                if SESSION_START_H is not None and SESSION_END_H is not None:
                    df = df[[in_session(ts) for ts in df.index]]
                    if df.empty:
                        continue

                # Indicators
                sig_series = triple_state(df["close"], SIG_FAST, SIG_MID, SIG_SLOW)

                df30 = resample_30m(df)
                if df30.empty or len(df30) < max(TR_FAST, TR_MID, TR_SLOW):
                    log.info(f"[{sym}] trend warmup (30m) not ready")
                    continue
                tr_series_30 = triple_state(df30["close"], TR_FAST, TR_MID, TR_SLOW)
                tr_series = tr_series_30.reindex(df.index, method="ffill")

                atr = compute_atr(df, ATR_PERIOD)
                if atr.empty or not np.isfinite(atr.iloc[-1]):
                    log.info(f"[{sym}] ATR not ready")
                    continue

                last_ts = df.index[-1]
                last_close = float(df["close"].iloc[-1])
                last_atr = float(atr.iloc[-1])

                sig = str(sig_series.iloc[-1])
                trd = str(tr_series.loc[last_ts])

                log.info(f"[{sym}] signal={sig} | trend={trd} | close={last_close:.4f} | ATR={last_atr:.4f}")

                want_long  = (sig == "BUY"  and trd == "BUY")
                want_short = (sig == "SELL" and trd == "SELL")
                diverged   = not (want_long or want_short)

                st = state[sym]
                pos, entry, stop, take = st["pos"], st["entry"], st["stop"], st["take"]

                # Manage trailing/breakeven when in trade
                if pos and entry is not None and last_atr > 0:
                    if pos == "LONG":
                        # trailing
                        if TRAIL_ATR > 0:
                            new_sl = last_close - TRAIL_ATR * last_atr
                            if stop is None or new_sl > stop:
                                stop = new_sl
                        # breakeven
                        if BE_RR > 0 and (last_close - entry) >= BE_RR * last_atr:
                            stop = max(stop or -1e9, entry)
                    else:  # SHORT
                        if TRAIL_ATR > 0:
                            new_sl = last_close + TRAIL_ATR * last_atr
                            if stop is None or new_sl < stop:
                                stop = new_sl
                        if BE_RR > 0 and (entry - last_close) >= BE_RR * last_atr:
                            stop = min(stop or 1e9, entry)

                # Initialize SL/TP if missing (once we know ATR)
                if pos and entry is not None and last_atr > 0:
                    if stop is None:
                        stop = entry - SL_ATR * last_atr if pos == "LONG" else entry + SL_ATR * last_atr
                    if TP_ATR > 0 and take is None:
                        take = entry + TP_ATR * last_atr if pos == "LONG" else entry - TP_ATR * last_atr

                # SL/TP checks
                hit_sl = hit_tp = False
                if pos and entry is not None:
                    if pos == "LONG":
                        if stop is not None and last_close <= stop: hit_sl = True
                        if take is not None and last_close >= take: hit_tp = True
                    else:
                        if stop is not None and last_close >= stop: hit_sl = True
                        if take is not None and last_close <= take: hit_tp = True

                # Exits: divergence or SL/TP
                should_exit = False
                exit_reason = None
                if pos and diverged:
                    should_exit = True; exit_reason = "DIVERGENCE"
                elif pos and (hit_sl or hit_tp):
                    should_exit = True; exit_reason = "SL" if hit_sl and not hit_tp else ("TP" if hit_tp and not hit_sl else "SL/TP")

                if should_exit and pos:
                    try:
                        side = OrderSide.SELL if pos == "LONG" else OrderSide.BUY
                        submit_market(trading, sym, side, SIZE)
                        msg = f"{sym} 🔻 Exit {pos} ({exit_reason}) @ ~{last_close:.2f}"
                        st["log"].append(f"{last_ts:%H:%M} {msg}")
                        send_telegram(msg)
                        pos = None; entry = None; stop = None; take = None
                    except Exception as e:
                        log.error(f"[{sym}] exit failed: {e}")

                # Entries
                if pos is None and last_atr > 0 and not diverged:
                    if want_long:
                        try:
                            submit_market(trading, sym, OrderSide.BUY, SIZE)
                            pos = "LONG"
                            entry = last_close + SLIPPAGE
                            stop  = entry - SL_ATR * last_atr
                            take  = (entry + TP_ATR * last_atr) if TP_ATR > 0 else None
                            msg = f"{sym} 🟢 Enter LONG @ ~{entry:.2f} (sig+trend)"
                            st["log"].append(f"{last_ts:%H:%M} {msg}")
                            send_telegram(msg)
                        except Exception as e:
                            log.error(f"[{sym}] long failed: {e}")
                    elif want_short:
                        try:
                            submit_market(trading, sym, OrderSide.SELL, SIZE)
                            pos = "SHORT"
                            entry = last_close - SLIPPAGE
                            stop  = entry + SL_ATR * last_atr
                            take  = (entry - TP_ATR * last_atr) if TP_ATR > 0 else None
                            msg = f"{sym} 🔴 Enter SHORT @ ~{entry:.2f} (sig+trend)"
                            st["log"].append(f"{last_ts:%H:%M} {msg}")
                            send_telegram(msg)
                        except Exception as e:
                            log.error(f"[{sym}] short failed: {e}")

                # Save state back
                st["pos"], st["entry"], st["stop"], st["take"] = pos, entry, stop, take

            # Telegram summaries
            now_v = vie_now()
            if now_v >= next_hourly:
                chunks: List[str] = []
                for s in SYMBOLS:
                    if state[s]["log"]:
                        chunks.append(f"*{s}*\n" + "\n".join(state[s]["log"][-6:]))
                send_telegram("🕐 Hourly update\n" + ("\n\n".join(chunks) if chunks else "No trades in the last hour."))
                next_hourly += timedelta(hours=1)

            if now_v >= next_daily:
                chunks: List[str] = []
                for s in SYMBOLS:
                    if state[s]["log"]:
                        chunks.append(f"*{s}*\n" + "\n".join(state[s]["log"][-40:]))
                send_telegram("📅 Daily recap (22:00 Vienna)\n" + ("\n\n".join(chunks) if chunks else "No trades today."))
                for s in SYMBOLS: state[s]["log"].clear()
                next_daily += timedelta(days=1)

        except Exception as e:
            log.error(f"[loop] error: {e}")

        # Respect rate limits
        elapsed = time.time() - loop_t0
        time.sleep(max(1.0, POLL_SECONDS - elapsed))

if __name__ == "__main__":
    main()

