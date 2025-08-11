from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed  # ✅ import the enum
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import os

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
SYMBOL = os.getenv("SYMBOL", "QQQ")

# Use the enum instead of string
feed_str = os.getenv("ALPACA_FEED", "iex").lower()
feed_enum = DataFeed.IEX if feed_str == "iex" else DataFeed.SIP

# Trading client for paper trading
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Streaming client
stream = StockDataStream(API_KEY, API_SECRET, feed=feed_enum)

# Bar handler
async def on_bar(bar):
    print(f"BAR: {bar}")
    # Dummy condition to fire a trade quickly
    if float(bar.close) % 2 < 0.5:  # Just a fake trigger
        print(f"🚀 Triggered BUY for {SYMBOL} at {bar.close}")
        try:
            market_order_data = MarketOrderRequest(
                symbol=SYMBOL,
                qty=1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            order = trading_client.submit_order(order_data=market_order_data)
            print(f"✅ Order submitted: {order}")
        except Exception as e:
            print(f"❌ Order failed: {e}")

# Subscribe to bars
stream.subscribe_bars(on_bar, SYMBOL)

print(f"[stream] starting for {SYMBOL} | feed={feed_str} | paper=True")
stream.run()  # blocking; do NOT wrap with asyncio.run()

