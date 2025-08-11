import asyncio
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import ta
from tqdm import tqdm
import matplotlib.pyplot as plt

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ================= CONFIG =================
ALPACA_API_KEY = "YOUR_API_KEY"
ALPACA_SECRET_KEY = "YOUR_SECRET_KEY"

SYMBOL = "QQQ"
BAR_TIMEFRAME = TimeFrame.Minute
HISTORY_LOOKBACK = 60  # minutes
VIENNA_TZ = pytz.timezone("Europe/Vienna")

ORDER_QTY = 1
EMA_SHORT = 9
EMA_LONG = 21
RSI_PERIOD = 14

# Alpaca clients
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# ================= STRATEGY =================
def fetch_latest_data():
    """Fetch recent price data using free IEX feed."""
    end_dt = datetime.now(pytz.UTC)
    start_dt = end_dt - timedelta(minutes=HISTORY_LOOKBACK)

    request_params = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=BAR_TIMEFRAME,
        start=start_dt,
        end=end_dt,
        feed="iex"  # 👈 Free plan compatible
    )

    bars = data_client.get_stock_bars(request_params).df
    if bars.empty:
        print("⚠ No data returned from Alpaca IEX feed.")
        return pd.DataFrame()

    df = bars.xs(SYMBOL, level=0)
    df.index = df.index.tz_convert(VIENNA_TZ)
    return df

def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA and RSI indicators."""
    if df.empty:
        return df
    df["EMA_short"] = ta.trend.ema_indicator(df["close"], window=EMA_SHORT)
    df["EMA_long"] = ta.trend.ema_indicator(df["close"], window=EMA_LONG)
    df["RSI"] = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    return df

def generate_signal(df: pd.DataFrame) -> str:
    """Generate BUY/SELL/HOLD signals."""
    if len(df) < max(EMA_SHORT, EMA_LONG, RSI_PERIOD):
        return "HOLD"

    latest = df.iloc[-1]
    if latest["EMA_short"] > latest["EMA_long"] and latest["RSI"] < 70:
        return "BUY"
    elif latest["EMA_short"] < latest["EMA_long"] and latest["RSI"] > 30:
        return "SELL"
    return "HOLD"

def execute_trade(signal: str):
    """Place a market order if BUY or SELL."""
    if signal not in ["BUY", "SELL"]:
        return

    order_data = MarketOrderRequest(
        symbol=SYMBOL,
        qty=ORDER_QTY,
        side=OrderSide.BUY if signal == "BUY" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )

    try:
        order = trading_client.submit_order(order_data)
        print(f"✅ Placed {signal} order: {order.id}")
    except Exception as e:
        print(f"❌ Order failed: {e}")

# ================= MAIN LOOP =================
async def trading_loop():
    while True:
        df = fetch_latest_data()
        if df.empty:
            await asyncio.sleep(60)
            continue

        df = apply_indicators(df)
        signal = generate_signal(df)
        print(f"[{datetime.now(VIENNA_TZ).strftime('%H:%M:%S')}] Signal: {signal}")

        execute_trade(signal)
        await asyncio.sleep(60)  # wait 1 minute

async def main():
    print("🚀 Soda Trader started with Alpaca IEX feed")
    await trading_loop()

if __name__ == "__main__":
    asyncio.run(main())

