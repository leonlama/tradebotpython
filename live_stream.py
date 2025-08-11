import os
import asyncio
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

API_KEY    = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
SYMBOL     = os.getenv("SYMBOL", "QQQ")
feed_str   = os.getenv("ALPACA_FEED", "iex").lower()
FEED_ENUM  = DataFeed.IEX if feed_str == "iex" else DataFeed.SIP

stream = StockDataStream(API_KEY, API_SECRET, feed=FEED_ENUM)

async def on_bar(bar):
    print(f"[BAR] {bar.symbol} {bar.timestamp} O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close} V:{bar.volume}")

async def on_trade(t):
    print(f"[TRADE] {t.symbol} {t.timestamp} Price:{t.price} Size:{t.size}")

stream.subscribe_bars(on_bar, SYMBOL)
stream.subscribe_trades(on_trade, SYMBOL)

print(f"[stream] starting for {SYMBOL} | feed={feed_str} | paper={os.getenv('PAPER','1')}")
stream.run()

