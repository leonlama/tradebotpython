from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import os
import time
import pandas as pd
from pathlib import Path

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
SYMBOL = os.getenv("SYMBOL", "QQQ")

feed_str = os.getenv("ALPACA_FEED", "iex").lower()
feed_enum = DataFeed.IEX if feed_str == "iex" else DataFeed.SIP

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

RECONNECT_BACKOFF_S = int(os.getenv("RECONNECT_BACKOFF_S", "60"))
REST_POLL_S = int(os.getenv("REST_POLL_S", "15"))

async def on_bar(bar):
    print(f"BAR: {bar}")
    if float(bar.close) % 2 < 0.5:
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

def rest_poll_loop(symbol, hist_client, feed_enum, bot, p, notifier):
    logs_dir = Path("logs"); logs_dir.mkdir(parents=True, exist_ok=True)
    equity_rows = []
    equity_path = logs_dir / "equity_live.csv"
    equity_png  = logs_dir / "equity_live.png"

    last_ts = None
    while True:
        try:
            now = pd.Timestamp.now(tz="UTC").floor("min")
            start = (now - pd.Timedelta(minutes=5))
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=now,
                feed=feed_enum,
                adjustment=None
            )
            bars = hist_client.get_stock_bars(req).df
            if isinstance(bars.index, pd.MultiIndex) or "symbol" in getattr(bars, "columns", []):
                try:
                    bars = bars.xs(symbol)
                except Exception:
                    bars = bars[bars.get("symbol", symbol) == symbol]

            if not bars.empty:
                bars = bars.sort_index()
                if last_ts is not None:
                    bars = bars[bars.index > last_ts]
                cutoff = pd.Timestamp.now(tz="UTC").floor("min")
                bars = bars[bars.index < cutoff]

                for ts, row in bars.iterrows():
                    intents = bot.step({
                        "time": ts.isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low":  float(row["low"]),
                        "close":float(row["close"]),
                        "volume": float(row.get("volume", 0.0))
                    })
                    if float(row["close"]) % 2 < 0.5:
                        print(f"🚀 Triggered BUY for {SYMBOL} at {row['close']}")
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

                if not bars.empty:
                    last_ts = bars.index[-1]

            time.sleep(REST_POLL_S)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[rest] poll error: {e}")
            time.sleep(REST_POLL_S)

def make_stream():
    return StockDataStream(API_KEY, API_SECRET, feed=feed_enum)

async def on_trade(t):
    await on_bar(t)

def start_stream_or_fallback(stream_ctor, symbol, on_trade_cb, hist_client, feed_enum, bot, p, notifier):
    while True:
        try:
            stream = stream_ctor()
            stream.subscribe_bars(on_trade_cb, symbol)
            print(f"[stream] starting for {symbol} | websocket")
            stream.run()
        except ValueError as e:
            msg = str(e).lower()
            if "connection limit exceeded" in msg or "auth failed" in msg:
                print("[stream] connection limit exceeded — falling back to REST polling.")
                if notifier.enabled:
                    notifier.send("⚠️ Alpaca stream connection limit. Falling back to REST polling; will retry stream periodically.")
                t0 = time.time()
                while time.time() - t0 < RECONNECT_BACKOFF_S:
                    rest_poll_loop(symbol, hist_client, feed_enum, bot, p, notifier)
                print("[stream] retrying websocket...")
                continue
            else:
                print(f"[stream] error: {e}; retry in {RECONNECT_BACKOFF_S}s")
                time.sleep(RECONNECT_BACKOFF_S)
                continue
        except Exception as e:
            print(f"[stream] unexpected error: {e}; retry in {RECONNECT_BACKOFF_S}s")
            time.sleep(RECONNECT_BACKOFF_S)
            continue

start_stream_or_fallback(
    stream_ctor=make_stream,
    symbol=SYMBOL,
    on_trade_cb=on_trade,
    hist_client=None,  # Replace with actual historical client
    feed_enum=feed_enum,
    bot=None,  # Replace with actual bot instance
    p=None,  # Replace with actual parameter if needed
    notifier=None  # Replace with actual notifier instance
)

