"""X01: Michael Stahl's Asian-liquidation -> MSS + FVG, coded from the thread.

Source: https://en.rattibha.com/thread/1729181209597456510 (2023-11-27)

Published rules, verbatim in substance:
  1. Wait for the Asian high or low to be liquidated. It must happen before
     03:00 New York.
  2. Once liquidated, wait for a clear MSS with an FVG.
  3. Use the first candle of the FVG as the orderblock.
  4. Enter with a limit at that orderblock, targeting the other Asian extreme.
  5. Stop slightly beyond the orderblock.
  6. If the target is hit before the limit fills, cancel.
  7. If not filled, or not at target, by 07:00 New York, close it.
  8. GBPUSD, M5. All times New York.

Choices made where the thread is silent, all fixed before the run and never
tuned:
  - Asian session      20:00 (previous NY day) to 02:00.
  - Sweep window       02:00 to 03:00, first bar that trades beyond the range.
  - "Clear MSS"        first close beyond the highest high (long) or lowest low
                       (short) of the ten bars before the sweep bar.
  - FVG                first three-bar imbalance in the leg between the sweep
                       and the MSS close: low[i] > high[i-2] for a long.
  - Orderblock         that gap's first candle, i-2.
  - Limit price        the orderblock's near edge (its high for a long), so the
                       first touch fills.
  - Stop               2 pips beyond the orderblock's far edge.
  - Target             the opposite Asian extreme.
  - One attempt per pair per NY day.

Usage: python x01_asian_mss.py [--years=2]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import (PIP, JPY, SPREAD, PIPVAL, SLIP, COMM,             # noqa: E402
                  load_m5, m5_path)
from prop_screen import screen, show                                        # noqa: E402

PAIRS = ("eurusd", "gbpusd", "usdjpy", "gbpjpy", "audusd", "eurjpy")
BUF = 2.0          # pips beyond the orderblock
LOOK = 10          # bars of structure before the sweep
DAY_MS = 86_400_000


def pip_of(pair, c):
    """Pip size in the units of the loaded file.

    Not every JPY file quotes 150.00. The local `_5m_20y_d100.csv` series store
    USDJPY as 1.15-1.58, i.e. the quote divided by 100. In that scale a pip is
    0.0001, exactly like EURUSD. Reading the scale off the median close keeps
    both conventions correct instead of hardcoding one.
    """
    if pair in JPY and float(np.median(c)) > 20.0:
        return 0.01
    return PIP


def context(ms):
    idx = pd.to_datetime(ms, unit="ms", utc=True).tz_convert("America/New_York")
    hour = idx.hour.to_numpy()
    # session day: bars from 20:00 belong to the next calendar day
    day = np.array([d.toordinal() + (h >= 20) for d, h in zip(idx.date, hour)])
    return day, hour


def asian_ranges(day, hour, h, l, h0=20, h1=2):
    m = (hour >= h0) | (hour < h1)
    d = day[m]
    if not len(d):
        return {}
    order = np.argsort(d, kind="stable")
    d, hi, lo = d[order], h[m][order], l[m][order]
    uniq, start = np.unique(d, return_index=True)
    end = np.append(start[1:], len(d))
    return {int(k): (float(hi[a:b].max()), float(lo[a:b].min()))
            for k, a, b in zip(uniq, start, end)}


def one_pair(pair, years):
    ms, o, h, l, c = load_m5(m5_path(pair))
    if years and years > 0:
        cutoff = int(ms[-1] - years * 365.25 * DAY_MS)
        keep = ms >= cutoff
        ms, o, h, l, c = ms[keep], o[keep], h[keep], l[keep], c[keep]
    day, hour = context(ms)
    ranges = asian_ranges(day, hour, h, l)

    pip = pip_of(pair, c)
    rows = []
    for d, (ahi, alo) in ranges.items():
        # sweep window 02:00-03:00
        sw = np.nonzero((day == d) & (hour >= 2) & (hour < 3))[0]
        if not len(sw):
            continue
        s = None
        bull = True
        for k in sw:
            if l[k] < alo:
                s, bull = k, True
                break
            if h[k] > ahi:
                s, bull = k, False
                break
        if s is None:
            continue

        # structure before the sweep
        lo_i = max(0, s - LOOK)
        if s - lo_i < 3:
            continue
        if bull:
            struct = float(h[lo_i:s].max())
            target = ahi
        else:
            struct = float(l[lo_i:s].min())
            target = alo

        # post-sweep window until 07:00. The published rule says the MSS follows
        # the sweep, not that it waits for 03:00, so a sweep at 02:00 may be
        # confirmed at 02:05.
        post = np.nonzero((day == d) & (hour < 7))[0]
        if not len(post):
            continue
        post = post[post > s]
        if not len(post):
            continue
        end = int(post[-1])

        # MSS: first close through structure
        mss = None
        for k in post:
            if (c[k] > struct) if bull else (c[k] < struct):
                mss = k
                break
        if mss is None:
            continue

        # FVG in the leg: first three-bar imbalance
        ob = None
        for i in range(s + 2, mss + 1):
            if bull and l[i] > h[i - 2]:
                ob = i - 2
                break
            if not bull and h[i] < l[i - 2]:
                ob = i - 2
                break
        if ob is None:
            continue

        if bull:
            entry = float(h[ob])
            stop = float(l[ob]) - BUF * pip
        else:
            entry = float(l[ob])
            stop = float(h[ob]) + BUF * pip
        R = abs(entry - stop)
        if R <= 0:
            continue
        if (target - entry) * (1 if bull else -1) <= 0:
            continue

        # from the bar after the gap, limit then management
        start = ob + 3
        if start > end:
            continue
        filled = None
        for k in range(start, end + 1):
            if bull:
                if l[k] <= entry:
                    filled = k
                    break
                if h[k] >= target:
                    break          # target hit before entry: cancel
            else:
                if h[k] >= entry:
                    filled = k
                    break
                if l[k] <= target:
                    break
        if filled is None:
            continue

        # management, stop before target inside a bar. Track the worst floating
        # excursion while the position is open: the daily-loss rule is measured
        # on equity, not on closed trades.
        exit_px, why = float(c[end]), "time"
        adv = mfe = 0.0
        for k in range(filled, end + 1):
            if bull:
                adv = max(adv, (entry - l[k]) / R)
                mfe = max(mfe, (h[k] - entry) / R)
            else:
                adv = max(adv, (h[k] - entry) / R)
                mfe = max(mfe, (entry - l[k]) / R)
            hit_sl = l[k] <= stop if bull else h[k] >= stop
            hit_tp = h[k] >= target if bull else l[k] <= target
            if hit_sl:
                exit_px, why = stop, "stop"
                break
            if hit_tp:
                exit_px, why = target, "target"
                break
        r = (exit_px - entry) / R if bull else (entry - exit_px) / R
        settle = end - filled + 1
        sp = R / pip
        cost = (SPREAD[pair] + SLIP) / sp + COMM / (sp * PIPVAL[pair])
        rows.append(dict(ts=int(ms[filled]), pair=pair, r=r - cost, gross=r,
                         adverse=float(min(max(adv, 0.0), 3.0)), why=why,
                         stop=sp, bars=settle, mfe=float(max(mfe, 0.0))))
    return rows


def one_pair_cached(pair, years):
    """One pair, memoized to disk, so a restart never recomputes a finished pair."""
    import json
    span = f"{years:.0f}y" if years else "full"
    path = os.path.join(HERE, f"x01_{pair}_{span}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
        print(f"{pair:8} {len(rows):5d} trades (cached)")
        return rows
    rows = one_pair(pair, years)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([{k: float(v) if isinstance(v, (float, int)) else v
                    for k, v in x.items()} for x in rows], fh)
    print(f"{pair:8} {len(rows):5d} trades")
    return rows


def main():
    years = 0.0
    for a in sys.argv[1:]:
        if a.startswith("--years="):
            years = float(a.split("=", 1)[1])
    label = f"{years:.0f}y" if years else "full history"

    rows = []
    for p in PAIRS:
        rows += one_pair_cached(p, years)
    rows.sort(key=lambda x: x["ts"])
    if not rows:
        print("no trades")
        return

    import collections
    why = collections.Counter(x["why"] for x in rows)
    print("\nexits:", dict(why))
    r = np.array([x["gross"] for x in rows])
    print(f"gross avg {r.mean():+.3f}R  median {np.median(r):+.2f}R  "
          f"max {r.max():+.1f}R  WR {100*np.mean(r > 0):.1f}%")

    res = screen(rows)
    show(f"X01 {label} 6 pairs", res)
    for f in res["fails"]:
        print("  -", f)


if __name__ == "__main__":
    main()
