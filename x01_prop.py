"""MONEYCHECK for X01 on an AquaFunded 2-Step Pro $5,000 account.

Rules come from strategies/sunday_reopen_fade/prop/rules.py, read 2026-11-18
from help.aquafunded.com/en/articles/15281229-2-step-pro:

  Step 1 target   +10% of $5,000 = +$500
  Step 2 target   +5%  of $5,000 = +$250, phase 2 restarts at $5,000
  Daily loss      5% of the higher of balance/equity at 00:00 UTC
  Max total loss  10% of starting balance, trailing below the peak
  No time limit, no minimum trading days

Not asserted anywhere, therefore not modelled and not invented:
challenge fee, reset/activation fee, profit split, payout cycle, consistency
rule. They are reported as missing, not assumed.

Method, per the skill:
  1. Historical hard-limit replay first (§7.3): 400 fresh $5,000 windows over
     the real trade order. Any breach disqualifies that risk level outright.
  2. Block bootstrap, 25-trade blocks, 50,000 paths, through phase 1 and
     phase 2, with the daily floor marked against floating equity using each
     trade's adverse excursion.
  3. Survival = P1 and P2 complete without any breach, from the same run that
     produces the pass figures.

Usage: python x01_prop.py [--paths=50000]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from account_sim import ACCOUNT, hard_limit, run_mc     # noqa: E402

RISKS = (0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02)
DAYS_PER_TRADE = 1.0 / 0.74


def load_rows():
    rows = []
    for f in glob.glob(os.path.join(HERE, "x01_*_full.json")):
        with open(f, encoding="utf-8") as fh:
            rows += json.load(fh)
    rows.sort(key=lambda x: x["ts"])
    return rows


def main():
    paths = 50000
    for a in sys.argv[1:]:
        if a.startswith("--paths="):
            paths = int(a.split("=", 1)[1])

    rows = load_rows()
    r = np.array([x["r"] for x in rows])
    adv = np.array([x.get("adverse", 0.0) for x in rows])
    day = np.array([x["ts"] for x in rows], np.int64) // 86400000
    print(f"ledger: {len(rows)} trades, net {r.mean():+.3f}R, "
          f"maxLL {_streak(r)}, adverse p50 {np.median(adv):.2f}R "
          f"p90 {np.percentile(adv,90):.2f}R\n")

    hdr = ("risk    P1pass  P2pass  overall  survive  hist_breach  "
           "trail_x  daily_x  med_trades  med_days  attempts")
    print(hdr)
    print("-" * len(hdr))

    for risk in RISKS:
        h = hard_limit(rows, risk)
        mc = run_mc(r, adv, day, risk, paths)
        surv = float(mc["survived"].mean())
        p1 = float((mc["survived"] | mc["p2_fail"]).mean())
        p2 = surv / p1 if p1 > 0 else 0.0
        med = float(np.median(mc["steps"]))
        attempts = 1.0 / surv if surv > 0 else float("inf")
        print(f"{risk*100:5.2f}%  {p1*100:6.1f}%  {p2*100:6.1f}%  "
              f"{surv*100:6.1f}%  {'PASS' if surv >= 0.70 and h['breach'] == 0 else 'FAIL':6}  "
              f"{h['breach']*100:10.1f}%  {h['trail_mult']:6.2f}x  "
              f"{h['daily_mult']:6.2f}x  {med:10.0f}  {med*DAYS_PER_TRADE:9.0f}  "
              f"{attempts:7.2f}")

    print("\nHistorical hard-limit detail (worst window of 400)")
    for risk in RISKS:
        h = hard_limit(rows, risk)
        print(f"  risk {risk*100:5.2f}%  worst trailing {h['trail']*100:5.2f}% "
              f"of account (limit 10%, {h['trail_mult']:.2f}x)   "
              f"worst day {h['daily']*100:5.2f}% (limit 5%, {h['daily_mult']:.2f}x)")


def _streak(r):
    st = ll = 0
    for v in r:
        if v < -1e-9:
            st += 1
            ll = max(ll, st)
        else:
            st = 0
    return ll


if __name__ == "__main__":
    main()
