
#!/usr/bin/env python3
"""
live_runner.py — run DODA LiveDodaBot on near-real-time 1-minute data

Providers:
  - twelvedata (requires API key; may be paywalled for NDX)
  - yahoo (free; 1m, near-real-time but sometimes delayed; last ~1 day)

Usage examples:
  # Yahoo + QQQ (free)
  python live_runner.py --provider yahoo --symbol QQQ --tz UTC --session 13 22 --poll 30

  # Twelve Data + QQQ (free tier typically OK)
  export TWELVEDATA_API_KEY=YOUR_KEY
  python live_runner.py --provider twelvedata --symbol QQQ --tz UTC --session 13 22 --poll 30

Notes:
  - This is a paper runner; it logs intents and marks fills like the backtester.
  - Requires bot.py in the same folder.
"""

import os, time, sys, json, argparse, math
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# --- import bot primitives ---
from bot import DodaParams, LiveDodaBot, ensure_datetime_index

def now_utc():
    return datetime.now(timezone.utc)

def round_down_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)

# ---------------- Data providers ----------------

def fetch_yahoo(symbol: str, n_minutes: int = 600) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("Please: pip install yfinance", file=sys.stderr)
        sys.exit(1)
    # Yahoo supports 1m for last ~7d via period param; we'll fetch '1d' for live
    period = "1d" if n_minutes <= 1440 else "5d"
    df = yf.download(symbol, interval="1m", period=period, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("Yahoo returned empty data")
    # Ensure DatetimeIndex UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df = df[["open","high","low","close","volume"]]
    return df

def fetch_twelvedata(apikey: str, symbol: str, n_minutes: int = 600) -> pd.DataFrame:
    import requests
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1min",
        "format": "JSON",
        "order": "ASC",
        "timezone": "UTC",
        "outputsize": min(n_minutes, 5000),
        "apikey": apikey,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if "status" in payload and payload["status"] == "error":
        raise RuntimeError(f"Twelve Data error: {payload.get('message','unknown')}")
    vals = payload.get("values") or payload.get("data") or []
    rows = []
    for d in vals:
        rows.append({
            "time": pd.to_datetime(d["datetime"], utc=True),
            "open": float(d["open"]),
            "high": float(d["high"]),
            "low": float(d["low"]),
            "close": float(d["close"]),
            "volume": float(d.get("volume", 0.0)),
        })
    df = pd.DataFrame(rows).sort_values("time").set_index("time")
    if df.empty:
        raise RuntimeError("Twelve Data returned empty data")
    # Already UTC
    return df[["open","high","low","close","volume"]]

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
    ap.add_argument("--provider", choices=["twelvedata","yahoo"], default="yahoo")
    ap.add_argument("--symbol", type=str, default=None, help="Symbol (^NDX/QQQ for Yahoo; NDX/QQQ for TwelveData)")
    ap.add_argument("--tz", type=str, default="UTC")
    ap.add_argument("--session", type=int, nargs=2, default=[13,22], metavar=("START_H","END_H"))
    ap.add_argument("--poll", type=int, default=30, help="Seconds between data polls")
    ap.add_argument("--initial", type=float, default=10_000.0)
    ap.add_argument("--commission", type=float, default=2.0)
    ap.add_argument("--slippage", type=float, default=0.5)
    ap.add_argument("--size", type=float, default=1.0)
    ap.add_argument("--history", type=int, default=1200, help="Minutes of history to bootstrap indicators")
    args = ap.parse_args()

    # Defaults
    symbol = args.symbol or ("QQQ" if args.provider=="yahoo" else "QQQ")
    apikey = os.getenv("TWELVEDATA_API_KEY") if args.provider=="twelvedata" else None
    if args.provider == "twelvedata" and not apikey:
        print("Set TWELVEDATA_API_KEY or use --provider yahoo", file=sys.stderr)
        sys.exit(1)

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

    # bootstrap history
    print(f"[boot] fetching history via {args.provider} symbol={symbol}")
    if args.provider == "twelvedata":
        df_hist = fetch_twelvedata(apikey, symbol, n_minutes=args.history)
    else:
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

    try:
        while True:
            # fetch recent window
            try:
                if args.provider == "twelvedata":
                    df = fetch_twelvedata(apikey, symbol, n_minutes=180)
                else:
                    df = fetch_yahoo(symbol, n_minutes=180)
            except Exception as e:
                print(f"[warn] fetch error: {e}")
                time.sleep(max(5, args.poll))
                continue

            cutoff = round_down_minute(now_utc())
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
                    # execute intents (paper)
                    for intent in intents:
                        if intent["action"] in ("BUY","SELL"):
                            # commission on entry
                            cash -= p.commission_per_trade
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
                    print(f"[live] {now_utc().isoformat()} updated -> {equity_png} last={last_seen_ts}")

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

