import os
import sys
import time
import asyncio
import signal
import logging
from alpaca.data.live import StockDataStream, DataFeed

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
    raise ValueError(f"Invalid ALPACA_FEED value: {ALPACA_FEED}")

# Rate limit: 200 requests/minute
MAX_CALLS_PER_MIN = 200
REQUEST_INTERVAL = 60 / MAX_CALLS_PER_MIN  # seconds between calls

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
# GLOBALS
# ---------------------------
last_request_time = 0
stream = None
stop_event = asyncio.Event()

# ---------------------------
# RATE LIMIT HELPER
# ---------------------------
async def rate_limit():
    global last_request_time
    elapsed = time.time() - last_request_time
    if elapsed < REQUEST_INTERVAL:
        await asyncio.sleep(REQUEST_INTERVAL - elapsed)
    last_request_time = time.time()

# ---------------------------
# BAR HANDLER
# ---------------------------
async def on_bar(bar):
    await rate_limit()
    logger.info(
        f"BAR | {bar.symbol} | Time={bar.timestamp} | Open={bar.open} "
        f"High={bar.high} Low={bar.low} Close={bar.close} Volume={bar.volume}"
    )

# ---------------------------
# GRACEFUL SHUTDOWN
# ---------------------------
def shutdown_handler(sig, frame):
    logger.warning("Shutdown signal received. Stopping stream...")
    stop_event.set()

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# ---------------------------
# STREAM LOOP WITH RECONNECT
# ---------------------------
async def start_stream():
    global stream
    retry_delay = 1  # seconds, grows exponentially on failure

    while not stop_event.is_set():
        try:
            logger.info(f"Starting stream for {', '.join(SYMBOLS)} | feed={ALPACA_FEED}")
            stream = StockDataStream(API_KEY, API_SECRET, feed=feed_map[ALPACA_FEED])

            for symbol in SYMBOLS:
                stream.subscribe_bars(on_bar, symbol)

            await stream.run()
        except Exception as e:
            logger.error(f"Stream error: {e}")
            logger.info(f"Reconnecting in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)  # max 1 minute
        else:
            retry_delay = 1  # reset after successful run

    logger.info("Stream stopped.")

# ---------------------------
# MAIN ENTRY POINT
# ---------------------------
if __name__ == "__main__":
    try:
        asyncio.run(start_stream())
    except KeyboardInterrupt:
        logger.info("Manual interrupt received. Exiting.")
    finally:
        if stream:
            try:
                stream.stop()
            except Exception:
                pass

