"""Data access and cost constants for the X01 study.

The research corpus is not in this repository: the six M5 series together are
several hundred MB. Point `M5_DIR` at a directory holding one CSV per pair and
the rest of the code runs unchanged.

CSV format, no header semantics beyond column order:

    open_utc_ms,open,high,low,close

Prices are absolute quotes. `eurusd`, `gbpusd` and `audusd` quote as 1.xxxxx;
the USDJPY/GBPJPY/EURJPY files used in the study store 1.xxxxx too, i.e. the
quote divided by 100. `pip_of()` derives the pip size from the price scale, so
both conventions work.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

PIP = 1e-4

PAIRS = ("eurusd", "gbpusd", "usdjpy", "gbpjpy", "audusd", "eurjpy")
JPY = {"usdjpy", "gbpjpy", "eurjpy", "audjpy"}

# Live costs: spread in pips, commission in account currency per round turn,
# slippage in pips. Adjust to your broker before trusting any net number.
SPREAD = {"eurusd": 0.5, "gbpusd": 0.9, "usdjpy": 0.6, "gbpjpy": 2.5,
          "audusd": 0.7, "eurjpy": 1.2}
PIPVAL = {"eurusd": 10.0, "gbpusd": 10.0, "usdjpy": 6.9, "gbpjpy": 6.9,
          "audusd": 10.0, "eurjpy": 6.9}
SLIP = 0.2
COMM = 5.0

M5_DIR = os.environ.get("M5_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "market_data"))


def m5_path(pair):
    suffix = "_5m_20y_d100.csv" if pair in JPY else "_5m_20y.csv"
    return os.path.join(M5_DIR, pair + suffix)


def load_m5(path):
    """CSV rows -> arrays, with the same validation the live engine applies."""
    df = pd.read_csv(path, header=0, usecols=[0, 1, 2, 3, 4],
                     names=["ms", "o", "h", "l", "c"],
                     dtype={"ms": np.int64, "o": np.float64, "h": np.float64,
                            "l": np.float64, "c": np.float64})
    ms = df["ms"].to_numpy(np.int64)
    o = df["o"].to_numpy(np.float64)
    h = df["h"].to_numpy(np.float64)
    l = df["l"].to_numpy(np.float64)
    c = df["c"].to_numpy(np.float64)
    if not (np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)).all():
        raise ValueError("non-finite prices")
    if not ((o > 0) & (h > 0) & (l > 0) & (c > 0)).all():
        raise ValueError("non-positive price")
    if not ((h >= np.maximum(o, c)) & (l <= np.minimum(o, c))).all():
        raise ValueError("invalid OHLC relation")
    if ms.shape[0] > 1 and not (ms[1:] > ms[:-1]).all():
        raise ValueError("open_utc_ms ordering")
    return ms, o, h, l, c


def tradeable(ms, o, h, l, c, min_run=12):
    """False for bars inside long runs of identical OHLC.

    Weekend and holiday gaps arrive forward-filled: dozens of identical flat
    bars. Trading those produces fills that never existed.
    """
    flat = (h == l) & (o == c)
    ok = np.ones(len(ms), bool)
    run = 0
    for i, f in enumerate(flat):
        run = run + 1 if f else 0
        if run >= min_run:
            ok[i] = False
            ok[max(0, i - run + 1):i + 1] = False
    return ok
