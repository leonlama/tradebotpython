import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import pytz

from notifier import TelegramNotifier

# ========= CONFIG =========
API_KEY = "YOUR_ALPACA_KEY"
API_SECRET = "YOUR_ALPACA_SECRET"
SYMBOL = "QQQ"
VIENNA_TZ = ZoneInfo("Europe/Vienna")

BAR_TIMEFRAME = TimeFrame.Minute  # 1-min bars
HISTORY_LOOKBACK = 200  # minutes for initial calc

TRADE_SIZE = 1
COMMISSION = 0.0
SLIPPAGE = 0.0

SHORT_MA = 5
MID_MA = 10
LONG_MA = 20

# Alpaca Clients
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Telegram notifier
notifier = TelegramNotifier()

# Trade log
trade_log = []

# ========= STRATEGY =========
def fetch_latest_data():
    end_dt = datetime.now(pytz.UTC)
    start_dt = end_dt - timedelta(minutes=HISTORY_LOOKBACK)
    request_params = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=BAR_TIMEFRAME,
        start=start_dt,
        end=end_dt
    )
    bars = data_client.get_stock_bars(request_params).df
    if bars.empty:
        return pd.DataFrame()
    df = bars.xs(SYMBOL, level=0)
    df.index = df.index.tz_convert(VIENNA_TZ)
    return df

def calculate_doda(df):
    df["ma_short"] = df["close"].rolling(SHORT_MA).mean()
    df["ma_mid"] = df["close"].rolling(MID_MA).mean()
    df["ma_long"] = df["close"].rolling(LONG_MA).mean()

    df["signal_doda"] = np.where(df["ma_short"] > df["ma_mid"], 1, -1)
    df["trend_doda"] = np.where(df["ma_mid"] > df["ma_long"], 1, -1)
    return df

def decide_trade(df, position):
    latest = df.iloc[-1]
    if latest["signal_doda"] == 1 and latest["trend_doda"] == 1:
        if position <= 0:
            return "buy"
    elif latest["signal_doda"] == -1 and latest["trend_doda"] == -1:
        if position >= 0:
            return "sell"
    return None

def execute_trade(side):
    order = MarketOrderRequest(
        symbol=SYMBOL,
        qty=TRADE_SIZE,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    trading_client.submit_order(order)
    price = fetch_latest_data().iloc[-1]["close"]
    log_line = f"{datetime.now(VIENNA_TZ):%Y-%m-%d %H:%M} {side.upper()} {SYMBOL} @ {price:.2f}"
    trade_log.append(log_line)
    print(log_line)
    if notifier.enabled:
        notifier.send(f"📈 {log_line}")

def get_position():
    positions = trading_client.get_all_positions()
    for pos in positions:
        if pos.symbol == SYMBOL:
            return int(pos.qty)
    return 0

# ========= SCHEDULERS =========
async def hourly_report_loop():
    while True:
        now = datetime.now(VIENNA_TZ)
        next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
        await asyncio.sleep((next_hour - now).total_seconds())
        if notifier.enabled:
            msg = "🕐 Hourly update\n" + ("\n".join(trade_log[-10:]) if trade_log else "No trades this hour.")
            notifier.send(msg)

async def daily_report_loop():
    while True:
        now = datetime.now(VIENNA_TZ)
        target = now.replace(hour=22, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())
        if notifier.enabled:
            msg = "📅 Daily summary\n" + ("\n".join(trade_log) if trade_log else "No trades today.")
            notifier.send(msg)

# ========= MAIN LOOP =========
async def trading_loop():
    while True:
        df = fetch_latest_data()
        if not df.empty:
            df = calculate_doda(df)
            pos = get_position()
            action = decide_trade(df, pos)
            if action:
                execute_trade(action)
        await asyncio.sleep(60)  # check every 1 minute

async def main():
    asyncio.create_task(hourly_report_loop())
    asyncio.create_task(daily_report_loop())
    await trading_loop()

if __name__ == "__main__":
    asyncio.run(main())

