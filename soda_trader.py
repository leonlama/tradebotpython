import os
import time
import logging
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange
import requests

# --- CONFIG ---
SYMBOLS = ["SPY", "QQQ", "DIA", "DAX", "GLD", "USO"]
POLL_SECONDS = int(os.getenv("POLL_SECONDS", 30))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", 15))
HISTORY_MINUTES = max(int(os.getenv("HISTORY_MINUTES", 1200)), ATR_PERIOD + 5)
SIG_FAST = int(os.getenv("SIG_FAST", 3))
SIG_MID = int(os.getenv("SIG_MID", 21))
TR_FAST = int(os.getenv("TR_FAST", 13))
TR_MID = int(os.getenv("TR_MID", 55))
TR_SLOW = int(os.getenv("TR_SLOW", 144))
TIMEZONE = os.getenv("TIMEZONE", "Europe/Vienna")
API_KEY = os.getenv("APCA_API_KEY_ID") or os.getenv("API_KEY")
API_SECRET = os.getenv("APCA_API_SECRET_KEY") or os.getenv("API_SECRET")
if not (API_KEY and API_SECRET):
    raise SystemExit("Missing APCA_API_KEY_ID/APCA_API_SECRET_KEY (or API_KEY/API_SECRET).")

# Telegram config
TELEGRAM_ENABLE = os.getenv("TELEGRAM_ENABLE", "0") == "1"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- LOGGING ---
logging.basicConfig(format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.INFO)

# --- Alpaca Client ---
client = StockHistoricalDataClient(API_KEY, API_SECRET)

def send_telegram(msg):
    if TELEGRAM_ENABLE:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logging.error(f"[telegram] failed: {e}")

def fetch_data(symbols):
    end_dt = datetime.now(pytz.UTC).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(minutes=HISTORY_MINUTES)
    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start_dt,
            end=end_dt,
            feed=DataFeed.IEX,
            adjustment=None
        )
        bars = client.get_stock_bars(req)
        if bars is None or bars.df is None or bars.df.empty:
            return None
        df = bars.df.reset_index()
        df.rename(columns={"timestamp": "time"}, inplace=True)
        return df
    except Exception as e:
        logging.error(f"[data] fetch error: {e}")
        return None

def doda_signal(df):
    sma_fast = SMAIndicator(df['close'], SIG_FAST).sma_indicator()
    sma_mid = SMAIndicator(df['close'], SIG_MID).sma_indicator()
    if sma_fast.iloc[-1] > sma_mid.iloc[-1]:
        return "BUY"
    elif sma_fast.iloc[-1] < sma_mid.iloc[-1]:
        return "SELL"
    else:
        return "NEUTRAL"

def doda_trend(df):
    sma_fast = SMAIndicator(df['close'], TR_FAST).sma_indicator()
    sma_mid = SMAIndicator(df['close'], TR_MID).sma_indicator()
    sma_slow = SMAIndicator(df['close'], TR_SLOW).sma_indicator()
    if sma_fast.iloc[-1] > sma_mid.iloc[-1] > sma_slow.iloc[-1]:
        return "BUY"
    elif sma_fast.iloc[-1] < sma_mid.iloc[-1] < sma_slow.iloc[-1]:
        return "SELL"
    else:
        return "NEUTRAL"

def atr_stop_levels(df):
    if len(df) < ATR_PERIOD:
        logging.warning(f"[atr] Not enough data for ATR (have {len(df)}, need {ATR_PERIOD})")
        return None
    atr = AverageTrueRange(df['high'], df['low'], df['close'], ATR_PERIOD).average_true_range()
    return round(atr.iloc[-1], 4)

# --- Main Loop ---
logging.info(f"[loop] starting — symbols={','.join(SYMBOLS)} poll={POLL_SECONDS}s feed=iex")

# Initialize a dictionary to track positions for each symbol
positions = {sym: None for sym in SYMBOLS}

while True:
    logging.info(f"[loop] polling data — symbols={','.join(SYMBOLS)} feed=iex")
    data = fetch_data(SYMBOLS)
    if data is None or data.empty:
        logging.warning("[loop] no data received")
        time.sleep(POLL_SECONDS)
        continue

    for sym in SYMBOLS:
        df_sym = data[data['symbol'] == sym].copy()
        if df_sym.empty:
            logging.warning(f"[data] no bars for {sym}")
            continue
        df_sym.sort_values("time", inplace=True)

        sig = doda_signal(df_sym)
        trd = doda_trend(df_sym)
        atr_val = atr_stop_levels(df_sym)

        logging.info(f"[signal] {sym} DODA Signal = {sig}")
        logging.info(f"[trend]  {sym} DODA Trend  = {trd}")
        logging.info(f"[atr]    {sym} ATR = {atr_val:.4f}" if atr_val is not None else f"[atr]    {sym} ATR = N/A")

        # Trade decision logic
        action = "NO TRADE"
        if atr_val is not None and atr_val > 0:
            if sig.upper() == "BUY" and trd.upper() == "BUY":
                if positions[sym] != "LONG":
                    action = "BUY"
                    positions[sym] = "LONG"
                    logging.info(f"[trade] {sym} Opening LONG")
            elif sig.upper() == "SELL" and trd.upper() == "SELL":
                if positions[sym] != "SHORT":
                    action = "SELL"
                    positions[sym] = "SHORT"
                    logging.info(f"[trade] {sym} Opening SHORT")
            elif positions[sym] is not None:
                action = "CLOSE"
                logging.info(f"[trade] {sym} Closing position")
                positions[sym] = None

        logging.info(f"[decision] {sym} Signal={sig} Trend={trd} => action={action}")

    time.sleep(POLL_SECONDS)

