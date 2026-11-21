"""Robustness checks for X01 before any verdict.

Four questions, because +0.76R gross on 205 trades is either an edge or a
handful of lucky 7R prints:

  1. Does it survive removing the best trades?
  2. Does each pair carry it, or is it one pair?
  3. Does each half-year carry it, or is it one regime?
  4. Does it survive doubled spread and slippage?

Also reports the stop-width distribution, because a tiny orderblock candle is
what turns a 40-pip move into an 8R print.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import SPREAD, PIPVAL, SLIP                       # noqa: E402
from x01_asian_mss import one_pair_cached, PAIRS            # noqa: E402


def stats(rows, label, mult=1.0, extra=0.0):
    if not rows:
        return float("nan")
    r = np.array([x["gross"] for x in rows])
    cost = np.array([x["stop"] for x in rows])
    cost = (mult * (np.array([SPREAD[x["pair"]] for x in rows]) + SLIP + extra)
            / cost + 5.0 / (cost * np.array([PIPVAL[x["pair"]] for x in rows])))
    net = r - cost
    w, lo = net[net > 0], net[net < 0]
    pf = w.sum() / abs(lo.sum()) if len(lo) and lo.sum() else float("inf")
    print(f"{label:26} n={len(net):5d} gross {r.mean():+.3f}R  "
          f"net {net.mean():+.3f}R  PF {pf:5.2f}  WR {100*np.mean(net > 0):5.1f}%")
    return net.mean()


def main():
    years = 0.0
    for a in sys.argv[1:]:
        if a.startswith("--years="):
            years = float(a.split("=", 1)[1])
    span = f"{years:.0f}y" if years else "full"
    print(f"X01 robustness, {span} history, {len(PAIRS)} pairs\n")

    rows = []
    for p in PAIRS:
        rows += one_pair_cached(p, years)
    rows.sort(key=lambda x: x["ts"])
    print()
    stats(rows, "all")
    print()

    order = np.argsort([-x["gross"] for x in rows])
    for pct in (1, 3, 5, 10):
        k = max(1, int(len(rows) * pct / 100))
        kept = [rows[i] for i in order[k:]]
        stats(kept, f"drop top {pct}% winners")
    print()

    for p in PAIRS:
        sub = [x for x in rows if x["pair"] == p]
        stats(sub, p)
    print()

    ts = np.array([x["ts"] for x in rows], dtype="int64")
    year = ((ts // 86_400_000) // 365).astype(int) + 1970
    for y in np.unique(year):
        sub = [x for x, yy in zip(rows, year) if yy == y]
        if len(sub) >= 20:
            stats(sub, f"year {y}")
    print()

    stats(rows, "spread x1.0", 1.0)
    stats(rows, "spread x2.0", 2.0)
    stats(rows, "spread x2.0 +1p slip", 2.0, 1.0)
    print()

    st = np.array([x["stop"] for x in rows])
    print("stop width in pips: "
          f"p10 {np.percentile(st,10):.1f}  p25 {np.percentile(st,25):.1f}  "
          f"median {np.median(st):.1f}  p75 {np.percentile(st,75):.1f}  "
          f"p90 {np.percentile(st,90):.1f}")
    r = np.array([x["gross"] for x in rows])
    print(f"R multiple: p10 {np.percentile(r,10):+.1f}  median {np.median(r):+.1f}  "
          f"p90 {np.percentile(r,90):+.1f}  max {r.max():+.1f}")
    for why in ("target", "stop", "time"):
        sub = [x for x in rows if x["why"] == why]
        if sub:
            rr = np.array([x["gross"] for x in sub])
            print(f"  {why:6} {len(sub):4d}  avg {rr.mean():+.2f}R")


if __name__ == "__main__":
    main()
