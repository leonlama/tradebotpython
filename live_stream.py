from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import os, asyncio

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
ALPACA_FEED = "iex"  # or "sip" if you have permission
SYMBOL = os.getenv("SYMBOL", "QQQ")

# Trading client for paper trading
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Streaming client
stream = StockDataStream(API_KEY, API_SECRET, feed=ALPACA_FEED)

@stream.on_bar(SYMBOL)
async def handle_bar(bar):
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

async def main():
    print(f"[stream] starting for {SYMBOL} | feed={ALPACA_FEED} | paper=True")
    await stream.subscribe_bars(SYMBOL)
    await stream.run()

if __name__ == "__main__":
    asyncio.run(main())

