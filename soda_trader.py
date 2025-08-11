#!/usr/bin/env python3
"""
soda_trader.py — Alpaca paper-trading bot with dual 3-EMA confirmation

Logic (per your spec):
- SIGNAL DODA (M1): triple EMA crossover on close (SIG_FAST, SIG_MID, SIG_SLOW)
- TREND SODA (M30): triple EMA crossover on 30m close (TR_FAST, TR_MID, TR_SLOW)
- Entry: go LONG if both == BUY; go SHORT if both == SELL
- Exit: close position immediately if Signal and Trend diverge (no longer both BUY/SELL)
- Risk mgmt: ATR-based SL/TP (ATR_PERIOD, SL_ATR, TP_ATR), optional breakeven (BREAKEVEN_RR) + trailing (TRAIL_ATR)

Data:
- Pulls M1 bars via alpaca-py (IEX or SIP feed, controlled by ALPACA_FEED=iex|sip)
- 30m trend is resampled from the same M1 stream

Ops:
- Poll loop every POLL_SECONDS (rate-friendly; under 200 req/min)
- Session filter (UTC hours) SESSION_START..SESSION_END
- Hourly Telegram update, daily 22:00 Europe/Vienna recap

Env (examples):
  APCA_API_KEY_ID=...
  APCA_API_SECRET_KEY=...
  APCA_API_BASE_URL=https://paper-api.alpaca.markets/v2
  ALPACA_FEED=iex
  SYMBOL=QQQ
  POLL_SECONDS=20
  HISTORY_MINUTES=1200
  SIG_FAST=3 SIG_MID=21 SIG_SLOW=34
  TR_FAST=13 TR_MID=55 TR_SLOW=144
  ATR_PERIOD=15 SL_ATR=1.7006674335 TP_ATR=2.5
  BREAKEVEN_RR=1.0 TRAIL_ATR=1.75
  USE_STOP_ENTRIES=0 STOP_OFFSET=0.5  (stop-entries not used here)
  CONFIRM_ON_CLOSE=1 (we always trade on closed bars)
  SESSION_START=13 SESSION_END=22
  SIZE=1 INITIAL=10000 COMMISSION=2 SLIPPAGE=0.5
  TELEGRAM_ENABLE=1 TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
  TIMEZONE=Europe/Vienna
"""

from __future__ import annotations

import os
import time
import math
import json
import logging
import datetime as dt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import pytz
import requests

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# =======================
# Config & utilities
# =======================

LOG_FMT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
log = logging.getLogger("SODA")

TZ_VIE = pytz.timezone(os.getenv("TIMEZONE", "Europe/Vienna"))

def vie_now() -> dt.datetime:
    return dt.datetime.now(TZ_VIE)

def utc_now_floor_min() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("min")

def session_mask(index: pd.DatetimeIndex, start_h: int, end_h: int) -> np.ndarray:
    hrs = index.tz_convert("UTC").hour
    if start_h <= end_h:
        return (hrs >= start_h) & (hrs < end_h)
    else:
        return (hrs >= start_h) | (hrs < end_h)

def send_telegram(text: str) -> None:
    if os.getenv("TELEGRAM_ENABLE", "0") != "1":
        return
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:3900], "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        log.warning(f"Telegram send error: {e}")

# =======================
# Indicator helpers
# =======================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=int(period), adjust=False).mean()

def compute_signal_states(df_1m: pd.DataFrame,
                          sig_fast: int, sig_mid: int, sig_slow: int,
                          tr_fast: int, tr_mid: int, tr_slow: int) -> Tuple[pd.Series, pd.Series]:
    """Return last-bar states: signal_state (M1 triple EMA), trend_state (M30 triple EMA)."""
    close_1m = df_1m["close"]

    # --- SIGNAL (M1) ---
    s_f = ema(close_1m, sig_fast)
    s_m = ema(close_1m, sig_mid)
    s_s = ema(close_1m, sig_slow)
    sig_buy  = (s_f > s_m) & (s_m > s_s)
    sig_sell = (s_f < s_m) & (s_m < s_s)
    signal_state = pd.Series(np.where(sig_buy, "BUY", np.where(sig_sell, "SELL", "NEUTRAL")),
                             index=df_1m.index)

    # --- TREND (M30 from M1) ---
    df_30 = df_1m.resample("30min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    if df_30.empty:
        trend_state = pd.Series(index=df_1m.index, dtype="object")
        return signal_state, trend_state

    t_f = ema(df_30["close"], tr_fast)
    t_m = ema(df_30["close"], tr_mid)
    t_s = ema(df_30["close"], tr_slow)
    tr_buy  = (t_f > t_m) & (t_m > t_s)
    tr_sell = (t_f < t_m) & (t_m < t_s)
    trend_30 = pd.Series(np.where(tr_buy, "BUY", np.where(tr_sell, "SELL", "NEUTRAL")),
                         index=df_30.index)
    # align to 1m index (ffill last known 30m trend)
    trend_state = trend_30.reindex(df_1m.index, method="ffill")
    return signal_state, trend_state

def compute_atr(df_1m: pd.DataFrame, period: int) -> pd.Series:
    h = df_1m["high"].astype(float)
    l = df_1m["low"].astype(float)
    c = df_1m["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=int(period), adjust=False, min_periods=1).mean()
    return atr

# =======================
# Alpaca clients
# =======================

def build_clients():
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("API_SECRET")
    if not (key and secret):
        raise SystemExit("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY (or API_KEY/API_SECRET).")
    paper = os.getenv("PAPER", "1").lower() in ("1","true","yes")
    trading = TradingClient(key, secret, paper=paper)
    data = StockHistoricalDataClient(key, secret)
    return trading, data

def fetch_m1(data_client: StockHistoricalDataClient, symbol: str, minutes: int, feed: DataFeed) -> pd.DataFrame:
    end = utc_now_floor_min()
    start = end - pd.Timedelta(minutes=minutes)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
                           start=start, end=end, feed=feed, adjustment=None)
    bars = data_client.get_stock_bars(req)
    df = bars.df
    if "symbol" in df.columns or isinstance(df.index, pd.MultiIndex):
        try:
            df = df.xs(symbol)
        except Exception:
            df = df[df.get("symbol", symbol) == symbol]
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df[["open","high","low","close","volume"]].astype(float)
    return df

# =======================
# Main trading loop
# =======================

def main():
    # -------- env & params --------
    symbol = os.getenv("SYMBOL", "QQQ")
    poll_s = int(os.getenv("POLL_SECONDS", "20"))
    hist_m = int(os.getenv("HISTORY_MINUTES", "1200"))

    sig_fast = int(os.getenv("SIG_FAST", "3"))
    sig_mid  = int(os.getenv("SIG_MID", "21"))
    sig_slow = int(os.getenv("SIG_SLOW", "34"))  # NEW: default if not set
    tr_fast  = int(os.getenv("TR_FAST", "13"))
    tr_mid   = int(os.getenv("TR_MID", "55"))
    tr_slow  = int(os.getenv("TR_SLOW", "144"))

    atr_period = int(os.getenv("ATR_PERIOD", "15"))
    sl_atr     = float(os.getenv("SL_ATR", "1.7"))
    tp_atr     = float(os.getenv("TP_ATR", "2.5"))
    be_rr      = float(os.getenv("BREAKEVEN_RR", "1.0"))
    trail_atr  = float(os.getenv("TRAIL_ATR", "1.75"))

    session_start = int(os.getenv("SESSION_START", "13"))
    session_end   = int(os.getenv("SESSION_END", "22"))

    size      = float(os.getenv("SIZE", "1"))
    commission= float(os.getenv("COMMISSION", "2"))
    slippage  = float(os.getenv("SLIPPAGE", "0.5"))

    feed_str = os.getenv("ALPACA_FEED", "iex").lower()
    feed = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}.get(feed_str, DataFeed.IEX)

    # -------- clients --------
    trading, data = build_clients()

    # -------- start message --------
    send_telegram(
        "✅ *SODA Trader started*\n"
        f"Symbol: *{symbol}* | Feed: *{feed_str.upper()}*\n"
        f"Signal EMA: {sig_fast}/{sig_mid}/{sig_slow} | Trend EMA(30m): {tr_fast}/{tr_mid}/{tr_slow}\n"
        f"ATR{atr_period} SLx{sl_atr} TPx{tp_atr} BE:{be_rr} Trail:{trail_atr}\n"
        f"Session UTC: {session_start}-{session_end} | Size: {size}"
    )

    # -------- state --------
    position_side: Optional[str] = None   # "LONG" | "SHORT" | None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_price: Optional[float] = None
    trades_log: List[str] = []
    hourly_anchor = vie_now().replace(minute=0, second=0, microsecond=0)
    next_hourly = hourly_anchor + dt.timedelta(hours=1)
    next_daily  = vie_now().replace(hour=22, minute=0, second=0, microsecond=0)
    if vie_now() >= next_daily:
        next_daily += dt.timedelta(days=1)

    # Try to sync any open position (optional)
    try:
        pos = trading.get_open_position(symbol)
        if pos and float(pos.qty) != 0:
            position_side = "LONG" if float(pos.qty) > 0 else "SHORT"
            entry_price = float(pos.avg_entry_price)
            log.info(f"[sync] Found open position {position_side} @ {entry_price}")
    except Exception:
        pass

    # -------- loop --------
    log.info(f"[loop] starting — symbol={symbol} poll={poll_s}s feed={feed_str}")
    while True:
        loop_started = time.time()
        try:
            # Fetch history
            log.info(f"[loop] polling data — symbol={symbol} feed={feed_str}")
            df = fetch_m1(data, symbol, hist_m, feed)
            if df.empty:
                log.warning("No data fetched (empty). Sleeping...")
                time.sleep(poll_s)
                continue

            # Session filter (on closed bars)
            df = df[df.index < utc_now_floor_min()]
            if df.empty:
                time.sleep(poll_s)
                continue
            mask = session_mask(df.index, session_start, session_end)
            df = df[mask]
            if df.empty:
                # outside session: close any open position (optional). We'll just idle.
                time.sleep(poll_s)
                continue

            # Indicators
            signal_state, trend_state = compute_signal_states(df, sig_fast, sig_mid, sig_slow,
                                                              tr_fast, tr_mid, tr_slow)
            atr = compute_atr(df, atr_period)
            last_ts = df.index[-1]
            last_close = float(df["close"].iloc[-1])
            last_atr = float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) else None
            sig = str(signal_state.iloc[-1]) if signal_state.size else "NEUTRAL"
            trd = str(trend_state.loc[last_ts]) if trend_state.size and last_ts in trend_state.index else "NEUTRAL"

            log.info(f"[signal] DODA Signal = {sig}")
            log.info(f"[trend]  DODA Trend  = {trd}")

            want_long  = (sig == "BUY"  and trd == "BUY")
            want_short = (sig == "SELL" and trd == "SELL")

            # Management: breakeven & trailing while in a trade
            if position_side and last_atr:
                if position_side == "LONG" and entry_price is not None:
                    # trailing
                    if trail_atr > 0:
                        new_sl = last_close - trail_atr * last_atr
                        if stop_price is None or new_sl > stop_price:
                            stop_price = new_sl
                    # breakeven at X*ATR (approx rr via atr multiple to be conservative)
                    if be_rr > 0 and (last_close - entry_price) >= be_rr * last_atr:
                        stop_price = max(stop_price or -1e9, entry_price)
                elif position_side == "SHORT" and entry_price is not None:
                    if trail_atr > 0:
                        new_sl = last_close + trail_atr * last_atr
                        if stop_price is None or new_sl < stop_price:
                            stop_price = new_sl
                    if be_rr > 0 and (entry_price - last_close) >= be_rr * last_atr:
                        stop_price = min(stop_price or 1e9, entry_price)

            # Exits:
            # 1) Divergence exit: if we have a position and not (both BUY or both SELL)
            diverged = not (want_long or want_short)
            # 2) ATR stops or TP
            hit_stop = False
            hit_tp   = False
            if position_side and last_atr and entry_price is not None:
                # set initial SL/TP if missing
                if stop_price is None:
                    stop_price = (entry_price - sl_atr * last_atr) if position_side=="LONG" else (entry_price + sl_atr * last_atr)
                if tp_atr and tp_atr > 0 and take_price is None:
                    take_price = (entry_price + tp_atr * last_atr) if position_side=="LONG" else (entry_price - tp_atr * last_atr)

                if position_side == "LONG":
                    if stop_price is not None and last_close <= stop_price: hit_stop = True
                    if take_price is not None and last_close >= take_price: hit_tp = True
                else:
                    if stop_price is not None and last_close >= stop_price: hit_stop = True
                    if take_price is not None and last_close <= take_price: hit_tp = True

            should_exit = False
            exit_reason = None
            if position_side and diverged:
                should_exit = True
                exit_reason = "DIVERGENCE"
            elif position_side and (hit_stop or hit_tp):
                should_exit = True
                exit_reason = "SL" if hit_stop and not hit_tp else ("TP" if hit_tp and not hit_stop else "SL/TP")

            if should_exit and position_side:
                side = OrderSide.SELL if position_side == "LONG" else OrderSide.BUY
                try:
                    trading.submit_order(MarketOrderRequest(
                        symbol=symbol, qty=size, side=side, time_in_force=TimeInForce.DAY
                    ))
                    msg = f"🔻 Exit {position_side} ({exit_reason}) @ ~{last_close:.2f}"
                    trades_log.append(f"{last_ts.strftime('%H:%M')} {msg}")
                    send_telegram(msg)
                except Exception as e:
                    log.error(f"[order] exit failed: {e}")
                # reset local state
                position_side = None
                entry_price = None
                stop_price = None
                take_price = None

            # Entries
            if position_side is None and last_atr:
                if want_long:
                    try:
                        trading.submit_order(MarketOrderRequest(
                            symbol=symbol, qty=size, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
                        ))
                        position_side = "LONG"
                        # apply slippage on our "entry" accounting only
                        entry_price = last_close + slippage
                        stop_price = entry_price - sl_atr * last_atr
                        take_price = (entry_price + tp_atr * last_atr) if tp_atr and tp_atr > 0 else None
                        msg = f"🟢 Enter LONG @ ~{entry_price:.2f} (sig+trend)"
                        trades_log.append(f"{last_ts.strftime('%H:%M')} {msg}")
                        send_telegram(msg)
                    except Exception as e:
                        log.error(f"[order] long failed: {e}")

                elif want_short:
                    try:
                        trading.submit_order(MarketOrderRequest(
                            symbol=symbol, qty=size, side=OrderSide.SELL, time_in_force=TimeInForce.DAY
                        ))
                        position_side = "SHORT"
                        entry_price = last_close - slippage
                        stop_price = entry_price + sl_atr * last_atr
                        take_price = (entry_price - tp_atr * last_atr) if tp_atr and tp_atr > 0 else None
                        msg = f"🔴 Enter SHORT @ ~{entry_price:.2f} (sig+trend)"
                        trades_log.append(f"{last_ts.strftime('%H:%M')} {msg}")
                        send_telegram(msg)
                    except Exception as e:
                        log.error(f"[order] short failed: {e}")

            # Hourly update
            now_v = vie_now()
            if now_v >= next_hourly:
                if trades_log:
                    chunk = "\n".join(trades_log[-20:])
                else:
                    chunk = "No trades in the last hour."
                send_telegram("🕐 Hourly update\n" + chunk)
                next_hourly += dt.timedelta(hours=1)

            # Daily recap at 22:00 Vienna
            if now_v >= next_daily:
                if trades_log:
                    again = "\n".join(trades_log[-100:])
                else:
                    again = "No trades today."
                send_telegram("📅 Daily recap (22:00 Vienna)\n" + again)
                trades_log.clear()
                next_daily += dt.timedelta(days=1)

        except Exception as e:
            log.error(f"[loop] error: {e}")

        # pacing to respect rate limits
        elapsed = time.time() - loop_started
        sleep_s = max(1.0, poll_s - elapsed)
        time.sleep(sleep_s)

if __name__ == "__main__":
    main()

