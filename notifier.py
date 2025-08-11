# notifier.py
from __future__ import annotations
import os, time, json, datetime as dt
from typing import List, Dict, Any, Optional
import requests

class TelegramNotifier:
    def __init__(self):
        self.enabled = os.getenv("TELEGRAM_ENABLE", "0") == "1"
        self.token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.timeout = 10
        if self.enabled and (not self.token or not self.chat_id):
            # If misconfigured, silently disable to avoid crashes
            self.enabled = False

    def send(self, text: str, disable_web_page_preview: bool=True):
        if not self.enabled: return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            r = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": disable_web_page_preview
            }, timeout=self.timeout)
            return r.ok
        except Exception:
            return False


def fmt_stats_line(stats: Dict[str, Any]) -> str:
    pf   = stats.get("profit_factor")
    wr   = stats.get("winrate")
    pnl  = stats.get("net_pnl")
    tr   = stats.get("trades")
    dd   = stats.get("max_dd_pct")
    pfs  = f"{pf:.2f}" if isinstance(pf,(int,float)) and pf is not None else "–"
    wrs  = f"{(wr*100):.1f}%" if isinstance(wr,(int,float)) and wr is not None else "–"
    pnls = f"{pnl:+.2f}" if isinstance(pnl,(int,float)) and pnl is not None else "–"
    dds  = f"{dd:.2f}%" if isinstance(dd,(int,float)) and dd is not None else "–"
    return f"PnL: *{pnls}* | PF: *{pfs}* | Winrate: *{wrs}* | Trades: *{tr}* | MaxDD: *{dds}*"


def local_now(tzname: str="Europe/Vienna") -> dt.datetime:
    # zoneinfo is in stdlib (Python 3.9+)
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(tz=ZoneInfo(tzname))
    except Exception:
        # Fallback: naive UTC, then pretend Vienna (not ideal but prevents crash)
        return dt.datetime.utcnow()

