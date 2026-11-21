"""Funded-phase economics for X01 on AquaFunded 2-Step Pro, $5,000.

Rules used, from help.aquafunded.com/en/articles/15281229-2-step-pro read
2026-11-21, plus the fee and split supplied by the account owner:

  Challenge fee       $15 per attempt, no activation fee
  Profit split        80% (page states 90% standard; 80% used as instructed)
  Payout cycle        14 days
  Consistency         highest profit day must be < 50% of total profit in the
                      payout period. Not a breach: payout is delayed.
  Funded minimum      3 trading days with at least 0.5% profit each
  Daily loss          5% of the higher of balance/equity at 00:00 UTC
  Max total loss      10% of starting balance, trailing below the peak
  Minimum payout      not stated on the page; any positive profit is assumed
                      withdrawable at a cycle end

Model
  - The real trade ledger is grouped into calendar days, empty days included,
    so the 0.74 trades/day rate and same-day clustering both survive.
  - Paths are built from contiguous 5-day blocks sampled with wraparound.
  - Each funded account starts at $5,000 after passing evaluation.
  - Risk is a fixed fraction of current balance, sized per trade.
  - Payout: withdraw all profit above the cycle-start balance, credit 80%,
    reset balance and trailing peak. Delayed while the 50% rule or the
    3-qualifying-day rule is unmet.
  - Horizon 2 years. A breach ends the path with no further payouts.

Usage: python x01_funded.py [--paths=20000] [--risk=0.0025] [--shift=0]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ACCOUNT = 5_000.0
DAILY = 0.05
TOTAL = 0.10
SPLIT = 0.80
FEE = 15.0
CYCLE = 14
QUAL_DAYS = 3
QUAL_PROFIT = 0.005 * ACCOUNT
HORIZON = 730
BLOCK = 5


def load_days():
    rows = []
    for f in glob.glob(os.path.join(HERE, "x01_*_full.json")):
        with open(f, encoding="utf-8") as fh:
            rows += json.load(fh)
    rows.sort(key=lambda x: x["ts"])
    day_id = np.array([x["ts"] for x in rows], np.int64) // 86400000
    lo, hi = int(day_id.min()), int(day_id.max())
    days = [[] for _ in range(hi - lo + 1)]
    for x, d in zip(rows, day_id):
        days[int(d) - lo].append((float(x["r"]), float(x.get("adverse", 0.0))))
    return days


def run(days, risk, paths, shift, label):
    rng = np.random.default_rng(11)
    n = len(days)
    nblk = HORIZON // BLOCK + 1
    starts = rng.integers(0, n, size=(paths, nblk))
    off = np.arange(BLOCK)[None, :]
    day_seq = ((starts[:, :, None] + off[None, None, :]) % n).reshape(paths, -1)

    got = np.zeros(paths, bool)
    days_to_first = np.full(paths, -1.0)
    payouts = np.zeros(paths, np.int64)
    cash = np.zeros(paths, float)
    dead = np.zeros(paths, bool)
    dead_before = np.zeros(paths, bool)
    blocked = np.zeros(paths, np.int64)

    for p in range(paths):
        bal = peak = cycle_start = ACCOUNT
        max_day = 0.0
        qual = 0
        cday = 0
        for step, d in enumerate(day_seq[p][:HORIZON]):
            floor = bal * (1 - DAILY)
            day_start = bal
            for rr, aa in days[d]:
                if bal - aa * risk * bal < floor:
                    dead[p] = True
                    break
                bal += (rr + shift) * risk * bal
                peak = max(peak, bal)
                if bal <= peak - TOTAL * ACCOUNT:
                    dead[p] = True
                    break
            if dead[p]:
                break
            gain = bal - day_start
            max_day = max(max_day, gain)
            if gain >= QUAL_PROFIT:
                qual += 1
            cday += 1
            if cday >= CYCLE:
                profit = bal - cycle_start
                if profit > 0 and max_day < 0.5 * profit and qual >= QUAL_DAYS:
                    cash[p] += SPLIT * profit
                    payouts[p] += 1
                    if not got[p]:
                        got[p] = True
                        days_to_first[p] = step + 1
                    bal -= profit
                    peak = bal
                    cycle_start = bal
                    max_day = 0.0
                    qual = 0
                elif profit > 0:
                    blocked[p] += 1
                cday = 0

    first = got
    dead_before = dead & ~got
    surv = float(first.mean())
    print(f"\n=== {label}: risk {risk*100:.2f}%, shift {shift:+.2f}R, "
          f"{paths} paths, 2 years ===")
    print(f"  reached first payout      {surv*100:6.1f}%")
    print(f"  failed before 1st payout  {dead_before.mean()*100:6.1f}%")
    print(f"  median days to 1st payout "
          f"{np.median(days_to_first[first]):6.0f}" if first.any() else "  no payouts")
    print(f"  payouts per account       {payouts.mean():6.2f}")
    print(f"  E[gross payout]           ${cash.mean():8.2f}")
    print(f"  E[net cash] after ${FEE:.0f} fee   ${cash.mean()-FEE:8.2f}")
    print(f"  cycles blocked by 50%/3d  {blocked.mean():6.2f}")
    if cash.size:
        print(f"  payout p10/p50/p90        "
              f"${np.percentile(cash[first],10):.0f} / "
              f"${np.percentile(cash[first],50):.0f} / "
              f"${np.percentile(cash[first],90):.0f}")
    return dict(surv=surv, cash=cash.mean(), payouts=payouts.mean(),
                days=float(np.median(days_to_first[first])) if first.any() else None)


def main():
    paths = 20000
    risk = 0.0025
    for a in sys.argv[1:]:
        if a.startswith("--paths="):
            paths = int(a.split("=", 1)[1])
        if a.startswith("--risk="):
            risk = float(a.split("=", 1)[1])

    days = load_days()
    print(f"{len(days)} calendar days, "
          f"{sum(len(d) for d in days)} trades, "
          f"{sum(len(d) for d in days)/len(days):.2f} trades/day")

    run(days, risk, paths, 0.0, "realistic")
    run(days, risk, paths, -0.15, "conservative (cost +0.15R)")


if __name__ == "__main__":
    main()
