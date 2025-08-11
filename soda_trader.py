#!/usr/bin/env python3.11
"""
SODA Trader — multi-symbol Alpaca paper bot with dual 3-EMA confirmation

Per symbol:
- SIGNAL (M1): triple EMA crossover on close -> BUY/SELL/NEUTRAL
- TREND (30m): triple EMA on 30-min resample -> BUY/SELL/NEUTRAL
- ENTRY: go LONG if both BUY; go SHORT if both SELL
- EXIT: close immediately if Signal/Trend diverge (not both BUY/SELL) OR SL/TP hit
- Risk: ATR-based SL/TP, optional breakeven RR and ATR trailing

Rate usage:
- One bars request per poll for *all* symbols (batched), very safe under 200 req/min.

Env (examples)
  APCA_API_KEY_ID=...
  APCA_API_SECRET_KEY=...
  APCA_API_BASE_URL=https://paper-api.alpaca.markets
  PAPER=1
  ALPACA_FEED=iex              # or sip (requires entitlement)
  SYMBOLS=SPY,QQQ,DIA,DAX,GLD,USO

  POLL_SECONDS=30
  HISTORY_MINUTES=1200

  SIG_FAST=3 SIG_MID=21 SIG_SLOW=34
  TR_FAST=13 TR_MID=55 TR_SLOW=144

  ATR_PERIOD=15 SL_ATR=1.7006674335 TP_ATR=2.5
  BREAKEVEN_RR=1.0 TRAIL_ATR=1.75

  SESSION_START=13 SESSION_END=22
  SIZE=1 COMMISSION=2 SLIPPAGE=0.5

  TELEGRAM_ENABLE=1 TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
  TIMEZONE=Europe/Vienna
"""

from __future__ import annotations

import os, time, logging, datetime as dt
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import pytz
import requests

# Alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# ---------- logging ----------
LOG_FMT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
log = logging.getLogger("SODA")

# ---------- time helpers ----------
TZ_VIE = pytz.timezone(os.getenv("TIMEZONE", "Europe/Vienna"))

def vie_now() -> dt.datetime:
    return dt.datetime.now(TZ_VIE)

def utc_floor_min() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("min")

# ---------- telegram ----------
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
            timeout=10,
        )
    except Exception as e:
        log.warning(f"Telegram send error: {e}")

# ---------- indicators ----------
def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=int(period), adjust=False).mean()

def triple_state(close: pd.Series, f: int, m: int, s: int) -> pd.Series:
    ef, em, es = ema(close, f), ema(close, m), ema(close, s)
    buy  = (ef > em) & (em > es)
    sell = (ef < em) & (em < es)
    return pd.Series(
        np.where(buy, "BUY", np.where(sell, "SELL", "NEUTRAL")),
        index=close.index,
        dtype="object"
    )

def resample_30m(df_1m: pd.DataFrame) -> pd.DataFrame:
    return df_1m.resample("30min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h = df["high"].astype(float); l = df["low"].astype(float); c = df["close"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=int(period), adjust=False, min_periods=1).mean()

# ---------- env/clients ----------
def must_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing env var: {name}")
    return v

def build_clients():
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("API_KEY")
    sec = os.getenv("APCA_API_SECRET_KEY") or os.getenv("API_SECRET")
    if not (key and sec):
        raise SystemExit("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY (or API_KEY/API_SECRET).")
    paper = os.getenv("PAPER","1").lower() in ("1","true","yes")
    trading = TradingClient(key, sec, paper=paper)
    data = StockHistoricalDataClient(key, sec)
    return trading, data

def parse_symbols() -> List[str]:
    raw = os.getenv("SYMBOLS")
    if raw and raw.strip():
        syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
    else:
        # sensible defaults
        syms = ["SPY","QQQ","DIA","DAX","GLD","USO"]
    return sorted(list(dict.fromkeys(syms)))  # de-dupe, keep order

def session_mask(index: pd.DatetimeIndex, start_h: int, end_h: int) -> np.ndarray:
    hrs = index.tz_convert("UTC").hour
    return (hrs >= start_h) & (hrs < end_h) if start_h <= end_h else ((hrs >= start_h) | (hrs < end_h))

# ---------- data fetch (batched) ----------
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
    df = bars.df  # MultiIndex (symbol, timestamp) or single-index if one symbol
    out: Dict[str, pd.DataFrame] = {}
    if df.empty:
        for s in symbols:
            out[s] = df.copy()
        return out
    if isinstance(df.index, pd.MultiIndex) or "symbol" in df.columns:
        # split by symbol
        for s in symbols:
            try:
                d = df.xs(s)
            except Exception:
                d = df[df.get("symbol", s) == s]
            if not d.empty:
                if d.index.tz is None:
                    d.index = d.index.tz_localize("UTC")
                else:
                    d.index = d.index.tz_convert("UTC")
                out[s] = d[["open","high","low","close","volume"]].astype(float)
            else:
                out[s] = pd.DataFrame(columns=["open","high","low","close","volume"])
    else:
        # single symbol case
        s = symbols[0]
        d = df
        if d.index.tz is None: d.index = d.index.tz_localize("UTC")
        else: d.index = d.index.tz_convert("UTC")
        out[s] = d[["open","high","low","close","volume"]].astype(float)
        for rest in symbols[1:]:
            out[rest] = pd.DataFrame(columns=["open","high","low","close","volume"])
    return out

# ---------- main ----------
def main():
    # params
    symbols = parse_symbols()
    poll_s = int(os.getenv("POLL_SECONDS","30"))
    hist_m = int(os.getenv("HISTORY_MINUTES","1200"))

    sig_fast = int(os.getenv("SIG_FAST","3"))
    sig_mid  = int(os.getenv("SIG_MID","21"))
    sig_slow = int(os.getenv("SIG_SLOW","34"))

    tr_fast  = int(os.getenv("TR_FAST","13"))
    tr_mid   = int(os.getenv("TR_MID","55"))
    tr_slow  = int(os.getenv("TR_SLOW","144"))

    atr_period = int(os.getenv("ATR_PERIOD","15"))
    sl_atr     = float(os.getenv("SL_ATR","1.7006674335"))
    tp_atr     = float(os.getenv("TP_ATR","2.5"))
    be_rr      = float(os.getenv("BREAKEVEN_RR","1.0"))
    trail_atr  = float(os.getenv("TRAIL_ATR","1.75"))

    session_start = int(os.getenv("SESSION_START","13"))
    session_end   = int(os.getenv("SESSION_END","22"))

    size      = float(os.getenv("SIZE","1"))
    commission= float(os.getenv("COMMISSION","2"))
    slippage  = float(os.getenv("SLIPPAGE","0.5"))

    feed_str = os.getenv("ALPACA_FEED","iex").lower()
    feed = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}.get(feed_str, DataFeed.IEX)

    trading, data = build_clients()

    send_telegram(
        "✅ *SODA Trader (multi-symbol) started*\n"
        f"Symbols: *{', '.join(symbols)}* | Feed: *{feed_str.upper()}*\n"
        f"Signal EMA: {sig_fast}/{sig_mid}/{sig_slow} | Trend(30m): {tr_fast}/{tr_mid}/{tr_slow}\n"
        f"ATR{atr_period} SLx{sl_atr} TPx{tp_atr} BE:{be_rr} Trail:{trail_atr}\n"
        f"Session UTC: {session_start}-{session_end} | Size: {size}"
    )

    # per-symbol state
    state: Dict[str, Dict[str, Any]] = {
        s: {
            "position": None,       # "LONG" | "SHORT" | None
            "entry": None,          # float
            "stop": None,           # float
            "take": None,           # float
            "log": []               # list of trade lines (for telegram summaries)
        }
        for s in symbols
    }

    hourly_anchor = vie_now().replace(minute=0, second=0, microsecond=0)
    next_hourly = hourly_anchor + dt.timedelta(hours=1)
    next_daily  = vie_now().replace(hour=22, minute=0, second=0, microsecond=0)
    if vie_now() >= next_daily:
        next_daily += dt.timedelta(days=1)

    log.info(f"[loop] starting — symbols={','.join(symbols)} poll={poll_s}s feed={feed_str}")

    while True:
        loop_t0 = time.time()
        try:
            log.info(f"[loop] polling data — symbols={','.join(symbols)} feed={feed_str}")
            dfs = fetch_m1_multi(data, symbols, hist_m, feed)

            for sym in symbols:
                df = dfs.get(sym) or pd.DataFrame()
                if df is None or df.empty:
                    log.warning(f"[{sym}] empty bars")
                    continue

                # closed bars only
                df = df[df.index < utc_floor_min()]
                if df.empty:
                    continue

                # session window
                df = df[session_mask(df.index, session_start, session_end)]
                if df.empty:
                    continue

                # compute states
                sig_state = triple_state(df["close"], sig_fast, sig_mid, sig_slow)

                df30 = resample_30m(df)
                if df30.empty:
                    continue
                tr_state_30 = triple_state(df30["close"], tr_fast, tr_mid, tr_slow)
                tr_state = tr_state_30.reindex(df.index, method="ffill")

                atr = compute_atr(df, atr_period)

                last_ts = df.index[-1]
                last_close = float(df["close"].iloc[-1])
                last_atr = float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) else None

                sig = str(sig_state.iloc[-1])
                trd = str(tr_state.loc[last_ts])

                log.info(f"[{sym}] signal={sig} | trend={trd}")

                want_long  = (sig == "BUY"  and trd == "BUY")
                want_short = (sig == "SELL" and trd == "SELL")
                diverged   = not (want_long or want_short)

                st = state[sym]
                pos, entry, stop, take = st["position"], st["entry"], st["stop"], st["take"]

                # manage trailing / breakeven if in trade
                if pos and last_atr and entry is not None:
                    if pos == "LONG":
                        if trail_atr > 0:
                            new_sl = last_close - trail_atr * last_atr
                            if stop is None or new_sl > stop:
                                stop = new_sl
                        if be_rr > 0 and (last_close - entry) >= be_rr * last_atr:
                            stop = max(stop or -1e9, entry)
                    else:  # SHORT
                        if trail_atr > 0:
                            new_sl = last_close + trail_atr * last_atr
                            if stop is None or new_sl < stop:
                                stop = new_sl
                        if be_rr > 0 and (entry - last_close) >= be_rr * last_atr:
                            stop = min(stop or 1e9, entry)

                # SL/TP checks
                hit_sl = hit_tp = False
                if pos and last_atr and entry is not None:
                    if stop is None:
                        stop = entry - sl_atr * last_atr if pos=="LONG" else entry + sl_atr * last_atr
                    if (take is None) and tp_atr and tp_atr > 0:
                        take = entry + tp_atr * last_atr if pos=="LONG" else entry - tp_atr * last_atr

                    if pos == "LONG":
                        if stop is not None and last_close <= stop: hit_sl = True
                        if take is not None and last_close >= take: hit_tp = True
                    else:
                        if stop is not None and last_close >= stop: hit_sl = True
                        if take is not None and last_close <= take: hit_tp = True

                # exits: divergence or SL/TP
                should_exit = False
                exit_reason = None
                if pos and diverged:
                    should_exit = True; exit_reason = "DIVERGENCE"
                elif pos and (hit_sl or hit_tp):
                    should_exit = True; exit_reason = "SL" if hit_sl and not hit_tp else ("TP" if hit_tp and not hit_sl else "SL/TP")

                if should_exit and pos:
                    side = OrderSide.SELL if pos == "LONG" else OrderSide.BUY
                    try:
                        trading.submit_order(MarketOrderRequest(
                            symbol=sym, qty=int(size), side=side, time_in_force=TimeInForce.DAY
                        ))
                        msg = f"{sym} 🔻 Exit {pos} ({exit_reason}) @ ~{last_close:.2f}"
                        st["log"].append(f"{last_ts:%H:%M} {msg}")
                        send_telegram(msg)
                        pos = None; entry = None; stop = None; take = None
                    except Exception as e:
                        log.error(f"[{sym}] exit failed: {e}")

                # entries
                if pos is None and last_atr:
                    if want_long:
                        try:
                            trading.submit_order(MarketOrderRequest(
                                symbol=sym, qty=int(size), side=OrderSide.BUY, time_in_force=TimeInForce.DAY
                            ))
                            pos = "LONG"
                            entry = last_close + float(slippage)
                            stop  = entry - sl_atr * last_atr
                            take  = (entry + tp_atr * last_atr) if tp_atr and tp_atr > 0 else None
                            msg = f"{sym} 🟢 Enter LONG @ ~{entry:.2f} (sig+trend)"
                            st["log"].append(f"{last_ts:%H:%M} {msg}")
                            send_telegram(msg)
                        except Exception as e:
                            log.error(f"[{sym}] long failed: {e}")
                    elif want_short:
                        try:
                            trading.submit_order(MarketOrderRequest(
                                symbol=sym, qty=int(size), side=OrderSide.SELL, time_in_force=TimeInForce.DAY
                            ))
                            pos = "SHORT"
                            entry = last_close - float(slippage)
                            stop  = entry + sl_atr * last_atr
                            take  = (entry - tp_atr * last_atr) if tp_atr and tp_atr > 0 else None
                            msg = f"{sym} 🔴 Enter SHORT @ ~{entry:.2f} (sig+trend)"
                            st["log"].append(f"{last_ts:%H:%M} {msg}")
                            send_telegram(msg)
                        except Exception as e:
                            log.error(f"[{sym}] short failed: {e}")

                # persist updated per-symbol state
                st["position"], st["entry"], st["stop"], st["take"] = pos, entry, stop, take

            # hourly summary
            now_v = vie_now()
            if now_v >= next_hourly:
                lines: List[str] = []
                for s in symbols:
                    if state[s]["log"]:
                        lines.append(f"*{s}*\n" + "\n".join(state[s]["log"][-6:]))
                send_telegram("🕐 Hourly update\n" + ("\n\n".join(lines) if lines else "No trades in the last hour."))
                next_hourly += dt.timedelta(hours=1)

            # daily 22:00 Vienna
            if now_v >= next_daily:
                lines: List[str] = []
                for s in symbols:
                    if state[s]["log"]:
                        lines.append(f"*{s}*\n" + "\n".join(state[s]["log"][-40:]))
                send_telegram("📅 Daily recap (22:00 Vienna)\n" + ("\n\n".join(lines) if lines else "No trades today."))
                for s in symbols: state[s]["log"].clear()
                next_daily += dt.timedelta(days=1)

        except Exception as e:
            log.error(f"[loop] error: {e}")

        # pacing
        elapsed = time.time() - loop_t0
        sleep_s = max(1.0, poll_s - elapsed)
        time.sleep(sleep_s)

if __name__ == "__main__":
    main()

