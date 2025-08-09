#!/usr/bin/env python3
"""
DODA Dual-Gate Trade Bot (MT4-style) — Python
Now with a neural-network surrogate optimizer (`optimize` subcommand),
built-in visualizations, and tqdm progress bars.

- Signal gate on M1: EMA(fast) crosses EMA(mid)
- Trend gate on M30: EMA(fast) > EMA(mid) > EMA(slow) (bull) or reversed (bear)
- Entries: market-at-open OR momentum stop entries
- Exits: ATR SL/TP; optional break-even at X*R; optional ATR trailing
- Costs: commission, slippage
- Filters: session, cooldown, confirm-on-close
- Proper 30min resample; UTC-safe timestamps; base equity for DD%
- Visuals: equity curve PNG for backtests; leaderboard chart for optimizer
- Robust JSON: safe serialization of numpy/pandas types
- Progress: tqdm bars for init sampling and optimization loops

Install:
  pip install numpy pandas scikit-learn matplotlib tqdm

Usage (backtest):
  python bot.py backtest --csv ./US100_1m.csv --tz UTC --start 2024-01-01 --end 2025-08-08 \
    --sig-fast 4 --sig-mid 10 --tr-fast 21 --tr-mid 55 --tr-slow 89 \
    --atr-period 14 --sl-atr 1.0 --tp-atr 2.0 --initial 10000

Usage (optimize with NN surrogate):
  python bot.py optimize --csv ./US100_1m.csv --tz UTC --start 2024-01-01 --end 2025-08-08 \
    --initial 10000 --commission 2 --slippage 0.5 --session 13 22 \
    --iterations 60 --init-samples 25 --seed 7 --min-trades 200
"""
from __future__ import annotations

import argparse, json, warnings
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Literal, Dict, Any, List
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm  # progress bars

# ---------------- JSON helper ----------------
def _json_default(o):
    import numpy as _np
    import pandas as _pd
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.ndarray,)):
        return o.tolist()
    if isinstance(o, (_pd.Timestamp,)):
        return o.isoformat()
    return str(o)

# ---------------- Indicators ----------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    # Robust, vectorized TR (avoids ambiguous reductions on live/dirty data)
    h = pd.to_numeric(df['high'], errors='coerce')
    l = pd.to_numeric(df['low'], errors='coerce')
    c = pd.to_numeric(df['close'], errors='coerce')
    prev = c.shift(1)
    a = (h - l).to_numpy()
    b = (h - prev).abs().to_numpy()
    d = (l - prev).abs().to_numpy()
    tr = np.maximum.reduce([a, b, d])
    return pd.Series(tr, index=df.index, name="tr")

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(span=period, adjust=False, min_periods=1).mean()

# ---------------- Parameters ----------------
@dataclass
class DodaParams:
    # Signal EMA periods (M1)
    sig_fast: int = 5
    sig_mid: int = 13
    # Trend EMA periods (M30)
    tr_fast: int = 21
    tr_mid: int = 55
    tr_slow: int = 89
    # Risk/Exit
    atr_period: int = 14
    sl_atr: float = 1.5
    tp_atr: float = 3.0            # 0 => no fixed TP
    stop_priority: Literal["SL","TP"] = "SL"
    # Optional dynamic management
    breakeven_rr: float = 0.0      # e.g. 1.0 => move SL to entry at 1R
    trail_atr: float = 0.0         # e.g. 1.0 => trail by 1*ATR from close
    # Costs
    commission_per_trade: float = 0.0  # absolute currency units per trade side
    slippage: float = 0.0              # price slippage per entry/exit
    # Rules
    confirm_on_close: bool = True      # confirm M1 signal on closed bar
    cooldown_bars: int = 2             # bars after exit before a new entry
    session: Optional[Tuple[int,int]] = None  # (hour_start, hour_end) end exclusive
    # Entries
    use_stop_entries: bool = False     # momentum entries via stop triggers
    stop_offset: float = 2.0           # price units added above/below prior high/low
    # Sizing & Equity
    size: float = 1.0                  # notional multiplier
    base_equity: float = 10_000.0      # starting equity for DD%
    name: str = "DODA_M1_M30"

# ---------------- Helpers ----------------
def ensure_datetime_index(df: pd.DataFrame, tz: Optional[str]=None) -> pd.DataFrame:
    df = df.copy()
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        if tz:
            if df['time'].dt.tz is None:
                df['time'] = df['time'].dt.tz_localize(tz)
            else:
                df['time'] = df['time'].dt.tz_convert(tz)
        df = df.set_index('time')
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex or a 'time' column.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()

def resample_30m(df_1m: pd.DataFrame) -> pd.DataFrame:
    df_30 = df_1m.resample("30min", label="right", closed="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    return df_30

def build_trend_on_m1(df_1m: pd.DataFrame, p: DodaParams) -> pd.DataFrame:
    df_30 = resample_30m(df_1m)
    df_30['ema_f'] = ema(df_30['close'], p.tr_fast)
    df_30['ema_m'] = ema(df_30['close'], p.tr_mid)
    df_30['ema_s'] = ema(df_30['close'], p.tr_slow)
    tr_sig = df_30[['ema_f','ema_m','ema_s']].shift(1)
    tr_1m = tr_sig.reindex(df_1m.index, method="ffill")
    out = pd.DataFrame(index=df_1m.index)
    out['bull'] = (tr_1m['ema_f'] > tr_1m['ema_m']) & (tr_1m['ema_m'] > tr_1m['ema_s'])
    out['bear'] = (tr_1m['ema_f'] < tr_1m['ema_m']) & (tr_1m['ema_m'] < tr_1m['ema_s'])
    return out

def build_signal_m1(df_1m: pd.DataFrame, p: DodaParams) -> pd.DataFrame:
    s = pd.DataFrame(index=df_1m.index)
    s['ema_f'] = ema(df_1m['close'], p.sig_fast)
    s['ema_m'] = ema(df_1m['close'], p.sig_mid)
    sh1 = 1 if p.confirm_on_close else 0
    sh2 = sh1 + 1
    s['cross_up']   = (s['ema_f'].shift(sh2) <= s['ema_m'].shift(sh2)) & (s['ema_f'].shift(sh1) > s['ema_m'].shift(sh1))
    s['cross_down'] = (s['ema_f'].shift(sh2) >= s['ema_m'].shift(sh2)) & (s['ema_f'].shift(sh1) < s['ema_m'].shift(sh1))
    return s[['cross_up','cross_down']]

# ---------------- Backtester ----------------
@dataclass
class BacktestStats:
    net_pnl: float
    profit_factor: Optional[float]
    winrate: Optional[float]
    max_dd: Optional[float]
    max_dd_pct: Optional[float]
    trades: int
    start: str
    end: str
    params: Dict[str, Any]

class DodaBacktester:
    def __init__(self, params: DodaParams):
        self.p = params

    def _session_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.p.session is None:
            return np.ones(len(index), dtype=bool)
        s, e = self.p.session
        hrs = index.hour
        if s <= e:
            return (hrs >= s) & (hrs < e)
        else:
            return (hrs >= s) | (hrs < e)

    def run(self, df_1m: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, BacktestStats]:
        p = self.p
        df = df_1m.copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df[["open","high","low","close","volume"]].dropna()
        df = df[self._session_mask(df.index)]
        if df.empty:
            raise ValueError("Filtered dataframe is empty after session mask.")
        sig = build_signal_m1(df, p)
        trd = build_trend_on_m1(df, p)
        df['atr'] = atr(df, p.atr_period)

        long_sig  = (sig['cross_up']   & trd['bull']).astype(bool)
        short_sig = (sig['cross_down'] & trd['bear']).astype(bool)

        position = 0
        entry_price = None
        entry_ts = None
        entry_risk = None
        sl = None
        tp = None
        cooldown = 0
        cash = 0.0
        equity_points: List[tuple[pd.Timestamp, float]] = []
        trades: List[Dict[str,Any]] = []

        for ts, row in df.iterrows():
            o,h,l,c = row['open'], row['high'], row['low'], row['close']
            this_atr = row['atr']
            mtm = 0.0 if position == 0 else (c - entry_price) * position * p.size
            equity_points.append((ts, p.base_equity + cash + mtm))

            if cooldown > 0:
                cooldown -= 1

            if position != 0:
                # Break-even move
                if entry_risk and p.breakeven_rr > 0:
                    if position == 1:
                        reward = (c - entry_price)
                        if reward >= p.breakeven_rr * entry_risk and (sl is None or sl < entry_price):
                            sl = entry_price
                    else:
                        reward = (entry_price - c)
                        if reward >= p.breakeven_rr * entry_risk and (sl is None or sl > entry_price):
                            sl = entry_price
                # ATR trailing
                if p.trail_atr and not np.isnan(this_atr):
                    if position == 1:
                        new_sl = c - p.trail_atr * this_atr
                        if sl is None or new_sl > sl:
                            sl = new_sl
                    else:
                        new_sl = c + p.trail_atr * this_atr
                        if sl is None or new_sl < sl:
                            sl = new_sl

                # Exit checks
                hit_sl = hit_tp = False
                if position == 1:
                    hit_sl = (sl is not None) and (l <= sl)
                    hit_tp = (tp is not None) and (h >= tp)
                else:
                    hit_sl = (sl is not None) and (h >= sl)
                    hit_tp = (tp is not None) and (l <= tp)

                if hit_sl and hit_tp:
                    exit_price = sl if p.stop_priority == "SL" else tp
                    reason = p.stop_priority
                elif hit_sl:
                    exit_price = sl; reason = "SL"
                elif hit_tp:
                    exit_price = tp; reason = "TP"
                else:
                    exit_price = None; reason = None

                if exit_price is not None:
                    exit_price = exit_price - p.slippage if position==1 else exit_price + p.slippage
                    pnl = (exit_price - entry_price) * position * p.size - (2 * p.commission_per_trade)
                    cash += pnl
                    trades.append(dict(
                        entry_time=entry_ts, exit_time=ts, side="LONG" if position==1 else "SHORT",
                        entry=entry_price, exit=exit_price, pnl=pnl, reason=reason
                    ))
                    position = 0; entry_price=None; entry_ts=None; sl=None; tp=None; entry_risk=None
                    cooldown = p.cooldown_bars
                    continue

            # entries
            if position == 0 and cooldown == 0:
                go_long  = bool(long_sig.get(ts, False))
                go_short = bool(short_sig.get(ts, False))
                if go_long:
                    if p.use_stop_entries:
                        trigger = (df['high'].shift(1).get(ts, np.nan) + p.stop_offset)
                        entry_price_calc = trigger + p.slippage if not np.isnan(trigger) and h >= trigger else None
                    else:
                        entry_price_calc = o + p.slippage
                    if entry_price_calc is not None:
                        entry_price = entry_price_calc
                        entry_ts = ts
                        position = 1
                        sl = entry_price - p.sl_atr * this_atr if not np.isnan(this_atr) else None
                        tp = entry_price + p.tp_atr * this_atr if (p.tp_atr>0 and not np.isnan(this_atr)) else None
                        entry_risk = abs(entry_price - sl) if sl is not None else None
                        cash -= p.commission_per_trade
                elif go_short:
                    if p.use_stop_entries:
                        trigger = (df['low'].shift(1).get(ts, np.nan) - p.stop_offset)
                        entry_price_calc = trigger - p.slippage if not np.isnan(trigger) and l <= trigger else None
                    else:
                        entry_price_calc = o - p.slippage
                    if entry_price_calc is not None:
                        entry_price = entry_price_calc
                        entry_ts = ts
                        position = -1
                        sl = entry_price + p.sl_atr * this_atr if not np.isnan(this_atr) else None
                        tp = entry_price - p.tp_atr * this_atr if (p.tp_atr>0 and not np.isnan(this_atr)) else None
                        entry_risk = abs(entry_price - sl) if sl is not None else None
                        cash -= p.commission_per_trade

        if position != 0:
            c = df['close'].iloc[-1]
            exit_price = c - p.slippage if position==1 else c + p.slippage
            pnl = (exit_price - entry_price) * position * p.size - p.commission_per_trade
            cash += pnl
            trades.append(dict(
                entry_time=entry_ts, exit_time=df.index[-1], side="LONG" if position==1 else "SHORT",
                entry=entry_price, exit=exit_price, pnl=pnl, reason="EOD"
            ))
            position = 0

        equity = pd.Series([e[1] for e in equity_points], index=[e[0] for e in equity_points], name="equity")

        wins = sum(t['pnl'] for t in trades if t['pnl']>0)
        losses = -sum(t['pnl'] for t in trades if t['pnl']<0)
        pf = (wins / losses) if losses>0 else None
        wr = (sum(1 for t in trades if t['pnl']>0) / len(trades)) if trades else None
        dd_series = equity.cummax() - equity
        max_dd = float(dd_series.max()) if not dd_series.empty else None
        max_dd_pct = float((dd_series / equity.cummax()).max() * 100.0) if not dd_series.empty else None
        stats = BacktestStats(
            net_pnl=float(equity.iloc[-1] - p.base_equity) if not equity.empty else 0.0,
            profit_factor=(float(pf) if pf is not None else None),
            winrate=(float(wr) if wr is not None else None),
            max_dd=max_dd,
            max_dd_pct=max_dd_pct,
            trades=len(trades),
            start=str(df.index[0]),
            end=str(df.index[-1]),
            params=asdict(p)
        )

        return pd.DataFrame(trades), equity, stats

# ---------------- Surrogate Optimizer (NN) ----------------
def param_space():
    return {
        "sig_fast": (3, 8, "int"),
        "sig_mid": (8, 21, "int"),
        "tr_fast": ([13, 21, 34], "choice"),
        "tr_mid": ([34, 55, 89], "choice"),
        "tr_slow": ([55, 89, 144], "choice"),
        "atr_period": (10, 20, "int"),
        "sl_atr": (0.75, 1.75, "float"),
        "tp_atr": ([0.0, 1.5, 2.0, 2.5, 3.0], "choice"),
        "breakeven_rr": ([0.0, 1.0], "choice"),
        "trail_atr": ([0.0, 1.25, 1.75], "choice"),
        "confirm_on_close": ([0, 1], "choice"),  # encode bool as 0/1
        "cooldown_bars": (0, 2, "int"),
        "use_stop_entries": ([0, 1], "choice"),
        "stop_offset": (1.0, 2.5, "float"),
    }

def sample_params(space, rng):
    cfg = {}
    for k, spec in space.items():
        # choice space
        if isinstance(spec[0], list):
            cfg[k] = rng.choice(spec[0])
        else:
            a, b, typ = spec
            if typ == "int":
                cfg[k] = int(rng.integers(a, b+1))
            elif typ == "float":
                cfg[k] = float(rng.uniform(a, b))
            else:
                raise ValueError(f"Unknown spec for {k}: {spec}")
    return cfg

def cfg_to_params(base: DodaParams, cfg: Dict[str, Any]) -> DodaParams:
    p = DodaParams(**asdict(base))
    p.sig_fast = int(cfg["sig_fast"])
    p.sig_mid = int(cfg["sig_mid"])
    p.tr_fast = int(cfg["tr_fast"])
    p.tr_mid  = int(cfg["tr_mid"])
    p.tr_slow = int(cfg["tr_slow"])
    p.atr_period = int(cfg["atr_period"])
    p.sl_atr = float(cfg["sl_atr"])
    p.tp_atr = float(cfg["tp_atr"])
    p.breakeven_rr = float(cfg["breakeven_rr"])
    p.trail_atr = float(cfg["trail_atr"])
    p.confirm_on_close = bool(int(cfg["confirm_on_close"]))
    p.cooldown_bars = int(cfg["cooldown_bars"])
    p.use_stop_entries = bool(int(cfg["use_stop_entries"]))
    p.stop_offset = float(cfg["stop_offset"])
    return p

def score_stats(stats: BacktestStats, min_trades: int, dd_weight: float, pf_weight: float, pnl_weight: float) -> float:
    if stats.trades < min_trades:
        return -1e6 + stats.trades  # harsh penalty
    pf = stats.profit_factor if stats.profit_factor is not None else 0.0
    pnl = stats.net_pnl
    ddp = stats.max_dd_pct if stats.max_dd_pct is not None else 0.0
    score = pf_weight * pf + pnl_weight * (pnl / max(1.0, abs(pnl)))  # sign of pnl
    score -= dd_weight * (ddp / 100.0)
    return float(score)

def nn_surrogate_optimize(df: pd.DataFrame, base: DodaParams, iterations: int, init_samples: int,
                          seed: int, min_trades: int, dd_weight: float, pf_weight: float, pnl_weight: float):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

    rng = np.random.default_rng(seed)
    space = param_space()

    X, y, tried = [], [], []

    def encode_cfg(cfg: Dict[str, Any]) -> List[float]:
        order = ["sig_fast","sig_mid","tr_fast","tr_mid","tr_slow","atr_period",
                 "sl_atr","tp_atr","breakeven_rr","trail_atr","confirm_on_close",
                 "cooldown_bars","use_stop_entries","stop_offset"]
        return [float(cfg[k]) for k in order]

    def eval_cfg(cfg):
        p = cfg_to_params(base, cfg)
        bt = DodaBacktester(p)
        _, _, stats = bt.run(df)
        sc = score_stats(stats, min_trades, dd_weight, pf_weight, pnl_weight)
        return sc, stats

    results = []

    # Initial random evaluations with progress bar
    for _ in tqdm(range(init_samples), desc="Initial sampling", ncols=100):
        cfg = sample_params(space, rng)
        sc, st = eval_cfg(cfg)
        X.append(encode_cfg(cfg)); y.append(sc); tried.append(cfg); results.append((sc, st, cfg))

    # Surrogate loop with progress bar
    for _ in tqdm(range(iterations), desc="Optimizing", ncols=100):
        if len(X) < 12:
            candidates = [sample_params(space, rng) for _ in range(32)]
            to_try = candidates[:5]
        else:
            surrogate = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=(64, 64),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=1e-3,
                    max_iter=3000,
                    early_stopping=True,
                    n_iter_no_change=50,
                    tol=1e-4,
                    random_state=seed
                ))
            ])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                surrogate.fit(np.array(X), np.array(y))

            K = 96
            candidates = [sample_params(space, rng) for _ in range(K)]
            Xc = np.array([encode_cfg(c) for c in candidates])
            yp = surrogate.predict(Xc)
            order = np.argsort(yp)[::-1]
            M = 5
            to_try = [candidates[i] for i in order[:M]] + [sample_params(space, rng) for _ in range(2)]

        for cfg in to_try:
            sc, st = eval_cfg(cfg)
            X.append(encode_cfg(cfg)); y.append(sc); tried.append(cfg); results.append((sc, st, cfg))

    best = max(results, key=lambda t: t[0])
    best_score, best_stats, best_cfg = best
    return best_score, best_stats, best_cfg, results

# ---------------- Live Bot Skeleton ----------------
class LiveDodaBot:
    def __init__(self, params: DodaParams):
        self.p = params
        self.df = pd.DataFrame(columns=['open','high','low','close','volume'])
        self.position = 0
        self.entry_price = None
        self.sl = None
        self.tp = None
        self.cooldown = 0

    def _compute_signals(self) -> Tuple[bool,bool]:
        p = self.p
        df = self.df
        if len(df) < max(p.sig_mid, p.tr_slow)*2:
            return False, False
        sig = build_signal_m1(df, p)
        trd = build_trend_on_m1(df, p)
        ts = df.index[-1]
        long_sig = bool(sig['cross_up'].iloc[-1] and trd['bull'].loc[ts])
        short_sig = bool(sig['cross_down'].iloc[-1] and trd['bear'].loc[ts])
        return long_sig, short_sig

    def _apply_session(self, ts: pd.Timestamp) -> bool:
        if self.p.session is None: return True
        s,e = self.p.session
        hr = ts.hour
        return (s <= hr < e) if s <= e else (hr >= s or hr < e)

    def step(self, bar: Dict[str, Any]) -> List[Dict[str, Any]]:
        intents: List[Dict[str, Any]] = []
        ts = pd.to_datetime(bar['time'], utc=True)
        row = pd.DataFrame([[bar['open'],bar['high'],bar['low'],bar['close'],bar.get('volume',0)]],
                           columns=['open','high','low','close','volume'],
                           index=[ts])
        if row.index.tz is None:
            row.index = row.index.tz_localize("UTC")
        else:
            row.index = row.index.tz_convert("UTC")

        self.df = pd.concat([self.df, row], axis=0, copy=False).sort_index()
        if not self._apply_session(ts):
            return intents

        # exit management on the incoming (closed) bar
        if self.position != 0:
            h = row['high'].iloc[0]; l = row['low'].iloc[0]
            if self.position == 1:
                hit_sl = self.sl is not None and l <= self.sl
                hit_tp = self.tp is not None and h >= self.tp
            else:
                hit_sl = self.sl is not None and h >= self.sl
                hit_tp = self.tp is not None and l <= self.tp
            if hit_sl or hit_tp:
                reason = self.p.stop_priority if (hit_sl and hit_tp) else ("SL" if hit_sl else "TP")
                intents.append({'action':'CLOSE', 'reason':reason})
                self.position = 0; self.entry_price=None; self.sl=None; self.tp=None
                self.cooldown = self.p.cooldown_bars
                return intents

        if self.cooldown > 0:
            self.cooldown -= 1
            return intents

        # entries (need ATR warm-up)
        last_atr = atr(self.df, self.p.atr_period).iloc[-1]
        if not np.isfinite(last_atr) or np.isnan(last_atr):
            return intents  # wait for enough history
        a = self.p.sl_atr * last_atr

        long_sig, short_sig = self._compute_signals()
        if self.position == 0:
            if long_sig:
                entry = row['open'].iloc[0] + self.p.slippage
                self.sl = entry - a; self.tp = entry + self.p.tp_atr * a if self.p.tp_atr>0 else None
                intents.append({'action':'BUY','ref_price':entry,'sl':self.sl,'tp':self.tp})
                self.position = 1; self.entry_price = entry
            elif short_sig:
                entry = row['open'].iloc[0] - self.p.slippage
                self.sl = entry + a; self.tp = entry - self.p.tp_atr * a if self.p.tp_atr>0 else None
                intents.append({'action':'SELL','ref_price':entry,'sl':self.sl,'tp':self.tp})
                self.position = -1; self.entry_price = entry
        return intents

# ---------------- CLI ----------------
def load_csv(path: Path, tz: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = ensure_datetime_index(df, tz=tz)
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna()

def cli_args(argv: Optional[list[str]]=None):
    ap = argparse.ArgumentParser(description="DODA M1/M30 MT4-style trade bot (Python) with NN optimizer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="Run backtest on 1-minute CSV")
    bt.add_argument("--csv", type=Path, required=True)
    bt.add_argument("--start", type=str, default=None)
    bt.add_argument("--end", type=str, default=None)
    bt.add_argument("--tz", type=str, default="UTC")
    bt.add_argument("--sig-fast", type=int, default=5)
    bt.add_argument("--sig-mid", type=int, default=13)
    bt.add_argument("--tr-fast", type=int, default=21)
    bt.add_argument("--tr-mid", type=int, default=55)
    bt.add_argument("--tr-slow", type=int, default=89)
    bt.add_argument("--atr-period", type=int, default=14)
    bt.add_argument("--sl-atr", type=float, default=1.5)
    bt.add_argument("--tp-atr", type=float, default=3.0)
    bt.add_argument("--breakeven-rr", type=float, default=0.0)
    bt.add_argument("--trail-atr", type=float, default=0.0)
    bt.add_argument("--confirm-on-close", action="store_true", default=True)
    bt.add_argument("--no-confirm-on-close", action="store_false", dest="confirm_on_close")
    bt.add_argument("--cooldown", type=int, default=2)
    bt.add_argument("--commission", type=float, default=0.0)
    bt.add_argument("--slippage", type=float, default=0.0)
    bt.add_argument("--size", type=float, default=1.0)
    bt.add_argument("--session", type=int, nargs=2, default=None, metavar=('START_H','END_H'))
    bt.add_argument("--initial", type=float, default=10_000.0)
    bt.add_argument("--use-stop-entries", action="store_true", default=False)
    bt.add_argument("--stop-offset", type=float, default=2.0)
    bt.add_argument("--outdir", type=Path, default=Path("./doda_results"))

    opt = sub.add_parser("optimize", help="Neural surrogate optimization over parameters")
    opt.add_argument("--csv", type=Path, required=True)
    opt.add_argument("--start", type=str, default=None)
    opt.add_argument("--end", type=str, default=None)
    opt.add_argument("--tz", type=str, default="UTC")
    opt.add_argument("--initial", type=float, default=10_000.0)
    opt.add_argument("--commission", type=float, default=0.0)
    opt.add_argument("--slippage", type=float, default=0.0)
    opt.add_argument("--session", type=int, nargs=2, default=None, metavar=('START_H','END_H'))
    opt.add_argument("--size", type=float, default=1.0)
    opt.add_argument("--iterations", type=int, default=60)
    opt.add_argument("--init-samples", type=int, default=25)
    opt.add_argument("--seed", type=int, default=7)
    opt.add_argument("--min-trades", type=int, default=300)
    opt.add_argument("--dd-weight", type=float, default=1.0)
    opt.add_argument("--pf-weight", type=float, default=1.0)
    opt.add_argument("--pnl-weight", type=float, default=0.25)
    opt.add_argument("--outdir", type=Path, default=Path("./doda_opt"))

    return ap.parse_args(argv)

def main(argv: Optional[list[str]]=None):
    import matplotlib.pyplot as plt
    args = cli_args(argv)

    if args.cmd == "backtest":
        df = load_csv(args.csv, tz=args.tz)
        if args.start:
            df = df[df.index >= pd.to_datetime(args.start, utc=True)]
        if args.end:
            df = df[df.index <= pd.to_datetime(args.end, utc=True)]
        params = DodaParams(
            sig_fast=args.sig_fast, sig_mid=args.sig_mid,
            tr_fast=args.tr_fast, tr_mid=args.tr_mid, tr_slow=args.tr_slow,
            atr_period=args.atr_period, sl_atr=args.sl_atr, tp_atr=args.tp_atr,
            breakeven_rr=args.breakeven_rr, trail_atr=args.trail_atr,
            commission_per_trade=args.commission, slippage=args.slippage,
            confirm_on_close=args.confirm_on_close, cooldown_bars=args.cooldown,
            session=tuple(args.session) if args.session else None, size=args.size,
            base_equity=args.initial, use_stop_entries=args.use_stop_entries, stop_offset=args.stop_offset
        )
        bt = DodaBacktester(params)
        trades, equity, stats = bt.run(df)

        outdir = args.outdir
        outdir.mkdir(parents=True, exist_ok=True)
        trades.to_csv(outdir / "trades.csv", index=False)
        equity.to_frame().to_csv(outdir / "equity.csv")
        with open(outdir / "report.json", "w") as f:
            json.dump({"stats": asdict(stats), "params": asdict(params)}, f, indent=2, default=_json_default)

        # Visualization: equity curve
        plt.figure()
        equity.plot()
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.tight_layout()
        eq_png = outdir / "equity_curve.png"
        plt.savefig(eq_png, dpi=120)
        plt.close()

        print("=== Backtest Complete ===")
        print(json.dumps(asdict(stats), indent=2, default=_json_default))
        print(f"Saved trades -> {outdir/'trades.csv'}")
        print(f"Saved equity -> {outdir/'equity.csv'}")
        print(f"Saved report -> {outdir/'report.json'}")
        print(f"Saved equity plot -> {eq_png}")

    elif args.cmd == "optimize":
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        df = load_csv(args.csv, tz=args.tz)
        if args.start:
            df = df[df.index >= pd.to_datetime(args.start, utc=True)]
        if args.end:
            df = df[df.index <= pd.to_datetime(args.end, utc=True)]

        base = DodaParams(
            commission_per_trade=args.commission,
            slippage=args.slippage,
            session=tuple(args.session) if args.session else None,
            base_equity=args.initial,
            size=args.size
        )

        best_score, best_stats, best_cfg, all_results = nn_surrogate_optimize(
            df, base,
            iterations=args.iterations,
            init_samples=args.init_samples,
            seed=args.seed,
            min_trades=args.min_trades,
            dd_weight=args.dd_weight,
            pf_weight=args.pf_weight,
            pnl_weight=args.pnl_weight
        )

        outdir = args.outdir
        outdir.mkdir(parents=True, exist_ok=True)
        # Leaderboard
        rows = []
        for sc, st, cfg in sorted(all_results, key=lambda t: t[0], reverse=True):
            row = {"score": sc, **asdict(st)}
            row.update(cfg)
            rows.append(row)
        df_leader = pd.DataFrame(rows)
        df_leader.to_csv(outdir / "leaderboard.csv", index=False)
        with open(outdir / "best.json", "w") as f:
            json.dump({
                "score": best_score,
                "stats": asdict(best_stats),
                "config": best_cfg
            }, f, indent=2, default=_json_default)

        # Visualization: top-20 scores
        import matplotlib.pyplot as plt
        topn = df_leader.head(20)
        plt.figure()
        plt.plot(range(len(topn)), topn["score"].values, marker="o")
        plt.title("Top-20 Candidate Scores")
        plt.xlabel("Rank")
        plt.ylabel("Score")
        plt.tight_layout()
        sc_png = outdir / "optimizer_scores.png"
        plt.savefig(sc_png, dpi=120)
        plt.close()

        print("=== Optimization Complete ===")
        print(json.dumps({
            "best_score": best_score,
            "best_stats": asdict(best_stats),
            "best_config": best_cfg
        }, indent=2, default=_json_default))
        print(f"Saved leaderboard -> {outdir/'leaderboard.csv'}")
        print(f"Saved best -> {outdir/'best.json'}")
        print(f"Saved scores plot -> {sc_png}")

if __name__ == "__main__":
    main()

