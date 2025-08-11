import os
import asyncio
import logging
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed  # <-- correct home for DataFeed

# ---------------------------
# CONFIGURATION
# ---------------------------
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
ALPACA_FEED = os.getenv("ALPACA_FEED", "iex").lower()
SYMBOLS = ["QQQ"]

# Map string from env to correct DataFeed enum
feed_map = {
    "iex": DataFeed.IEX,
    "sip": DataFeed.SIP,
    "otc": DataFeed.OTC
}
if ALPACA_FEED not in feed_map:
    raise ValueError(f"Invalid ALPACA_FEED: {ALPACA_FEED}. Use iex|sip|otc.")

# ---------------------------
# LOGGING SETUP
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LiveStream")

# ---------------------------
# STREAM HANDLERS
# ---------------------------
async def on_trade(trade):
    logger.info(f"[TRADE] {trade.symbol} price={trade.price} size={trade.size}")

async def on_bar(bar):
    logger.info(
        f"[BAR] {bar.symbol} | Time={bar.timestamp} | Open={bar.open} "
        f"High={bar.high} Low={bar.low} Close={bar.close} Volume={bar.volume}"
    )

# ---------------------------
# STREAM LOOP WITH RECONNECT
# ---------------------------
async def main():
    retry_delay = 5
    while True:
        # create a fresh stream object every reconnect
        stream = StockDataStream(API_KEY, API_SECRET, feed=feed_map[ALPACA_FEED])

        # subscribe
        for symbol in SYMBOLS:
            stream.subscribe_trades(on_trade, symbol)
            stream.subscribe_bars(on_bar, symbol)

        try:
            logger.info(f"Starting stream for {', '.join(SYMBOLS)} | feed={ALPACA_FEED}")
            await stream.run()
        except Exception as e:
            logger.error(f"Stream error: {e}")
            logger.info(f"Reconnecting in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)

# ---------------------------
# MAIN ENTRY POINT
# ---------------------------
if __name__ == "__main__":
    asyncio.run(main())

