import os
import pytz
import logging
import asyncio
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import ta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from telegram import Bot

# -------------------------
# CONFIG
# -------------------------
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL")
SYMBOL = os.getenv("SYMBOL", "QQQ")

# MA Periods
SIG_FAST = int(os.getenv("SIG_FAST", 3))
SIG_MID = int(os.getenv("SIG_MID", 21))
TR_FAST = int(os.getenv("TR_FAST", 13))
TR_MID = int(os.getenv("TR_MID", 55))
TR_SLOW = int(os.getenv("TR_SLOW", 144))

# Telegram
TELEGRAM_ENABLE = os.getenv("TELEGRAM_ENABLE", "0") == "1"
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Timezone
TZ = pytz.timezone(os.getenv("TIMEZONE", "Europe/Vienna"))

# -------------------------
# INIT
# -------------------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("SodaTrader")

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
bot = Bot(token=BOT_TOKEN) if TELEGRAM_ENABLE else None

# -------------------------
# HELPERS
# -------------------------
def send_telegram(msg):
    if TELEGRAM_ENABLE:
        asyncio.create_task(bot.send_message(chat_id=CHAT_ID, text=msg))

def fetch_bars(symbol, minutes):
    end_dt = datetime.now(TZ)
    start_dt = end_dt - timedelta(minutes=minutes)
    bars = data_client.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_dt,
            end=end_dt,
            adjustment="raw"
        )
    )
    df = pd.DataFrame([b.__dict__ for b in bars[symbol]])
    df.set_index("timestamp", inplace=True)
    return df

def compute_doda(df):
    # Signal Doda
    df["sig_fast"] = df["close"].rolling(SIG_FAST).mean()
    df["sig_mid"] = df["close"].rolling(SIG_MID).mean()
    # Trend Doda
    df["tr_fast"] = df["close"].rolling(TR_FAST).mean()
    df["tr_mid"] = df["close"].rolling(TR_MID).mean()
    df["tr_slow"] = df["close"].rolling(TR_SLOW).mean()
    return df

def get_signal(df):
    last = df.iloc[-1]
    trend_up = last.tr_fast > last.tr_mid > last.tr_slow
    trend_down = last.tr_fast < last.tr_mid < last.tr_slow
    buy_sig = last.sig_fast > last.sig_mid
    sell_sig = last.sig_fast < last.sig_mid

    if buy_sig and trend_up:
        return "BUY"
    elif sell_sig and trend_down:
        return "SELL"
    else:
        return None

def place_trade(side, qty=1):
    try:
        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL,
                qty=qty,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
        )
        logger.info(f"Placed {side} order for {SYMBOL}")
        send_telegram(f"📈 Trade executed: {side} {SYMBOL} at market")
    except Exception as e:
        logger.error(f"Trade failed: {e}")

# -------------------------
# TASKS
# -------------------------
async def trading_loop():
    df = fetch_bars(SYMBOL, 500)
    df = compute_doda(df)
    signal = get_signal(df)
    if signal:
        place_trade(signal)

async def hourly_status():
    df = fetch_bars(SYMBOL, 120)
    last = df.iloc[-1]
    msg = f"Hourly Update {SYMBOL} | Last Price: {last.close:.2f} | Time: {last.name}"
    send_telegram(msg)

async def daily_recap():
    df = fetch_bars(SYMBOL, 1440)
    change = (df.close.iloc[-1] - df.close.iloc[0]) / df.close.iloc[0] * 100
    msg = f"📊 Daily Recap {SYMBOL}: {df.close.iloc[-1]:.2f} ({change:.2f}%)"
    send_telegram(msg)

# -------------------------
# MAIN
# -------------------------
async def main():
    scheduler = AsyncIOScheduler(timezone=TZ)
    scheduler.add_job(trading_loop, "interval", minutes=1)   # Run strategy every minute
    scheduler.add_job(hourly_status, "interval", hours=1)    # Hourly updates
    scheduler.add_job(daily_recap, "cron", hour=22, minute=0) # Daily recap 22:00 Vienna
    scheduler.start()

    logger.info("Soda Trader started (Paper Mode)")
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())

