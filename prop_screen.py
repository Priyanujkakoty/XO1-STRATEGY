"""MONEYCHECK acceptance screen: does ANY candidate strategy survive AquaFunded?

The gates are arithmetic, not taste. They come from the $5,000 2-Step Pro
challenge: +15% total profit, 10% trailing drawdown, 5% daily.

Given a ledger of net R multiples, the only two numbers that decide the outcome
are net expectancy and the worst drawdown the strategy actually produces.

  Choose risk r so the drawdown uses at most half the trailing budget:
        dd_R * r  <=  0.05          ->  r  =  0.05 / dd_R
  Profit needed at that r:
        profit_R  =  0.15 / r  =  3 * dd_R
  Trades needed:
        trades  =  profit_R / expectancy

So the screen is: does `3 * dd_R / expectancy` fit inside a practical number of
trades, at a risk the daily limit can survive? Everything else is a restatement
of that.

Rows must be dicts with at least:
    ts   entry time, milliseconds UTC
    r    NET R multiple after spread, slippage and commission
Optional:
    gross  R multiple before costs (enables the cost gate)
    adverse  worst floating excursion against entry, in R (0..1)

Usage:
    python prop_screen.py --selftest
    python prop_screen.py --csv ledger.csv
"""
from __future__ import annotations

import csv
import sys

import numpy as np

ACCOUNT = 5_000.0
TARGET = 0.15            # step 1 +10%, step 2 +5%
BUDGET = 0.10            # trailing total loss
MARGIN = 0.50            # spend at most half the trailing budget on drawdown
DAILY = 0.05

GATES = dict(
    min_trades=300,
    min_years=5.0,
    min_per_day=2.0,       # the account must actually earn often enough
    min_exp=0.08,          # net R per trade
    dd_cap_R=25.0,         # implied risk must stay usable
    min_risk=0.0015,       # below this the challenge takes forever
    max_risk=0.0100,       # above this one trade can eat the daily limit
    max_months=24.0,
    max_cost_share=0.25,
    max_ll_risk=DAILY,     # max consecutive losses * risk must fit the day
)


def _streak(r):
    st = ll = 0
    for v in r:
        if v < -1e-9:
            st += 1
            ll = max(ll, st)
        else:
            st = 0
    return ll


def _rolling_dd(r, window=None, starts=300):
    """Peak-to-trough drawdown in R, replayed from many fresh start points."""
    n = len(r)
    if n < 20:
        return 0.0, 0.0
    w = int(window or min(500, max(100, n // 3)))
    if w >= n:
        w = n // 2
    last = n - w
    idx = np.unique(np.linspace(0, last, min(starts, last + 1)).astype(int))
    dd = []
    for s in idx:
        seg = r[s:s + w]
        eq = np.cumsum(seg)
        peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
        dd.append(float(np.max(peak - eq)))
    return float(np.percentile(dd, 90)), float(np.median(dd))


def screen(rows, gates=None):
    g = dict(GATES)
    if gates:
        g.update(gates)
    r = np.array([x["r"] for x in rows], float)
    ts = np.array([x["ts"] for x in rows], np.int64)
    n = len(r)
    if n == 0:
        return dict(verdict="INSUFFICIENT", fails=["no trades"], n=0)

    span_y = float(ts.max() - ts.min()) / (365.25 * 86400e3)
    tpd = n / (span_y * 365.25) if span_y > 0 else float(n)

    exp = float(r.mean())
    w, lo = r[r > 0], r[r < 0]
    pf = float(w.sum() / abs(lo.sum())) if len(lo) and lo.sum() else float("inf")
    maxll = _streak(r)
    dd90, dd50 = _rolling_dd(r)

    risk = (BUDGET * (1 - MARGIN) / dd90) if dd90 > 0 else float("inf")
    need_R = TARGET / risk if np.isfinite(risk) and risk > 0 else float("inf")
    need_trades = need_R / exp if exp > 0 else float("inf")
    months = need_trades / tpd / 21.0 if np.isfinite(need_trades) and tpd > 0 else float("inf")

    gross = np.array([x["gross"] for x in rows], float) if rows and "gross" in rows[0] else None
    cost_share = (1.0 - exp / gross.mean()) if gross is not None and gross.mean() > 1e-9 else None

    fails = []
    if n < g["min_trades"]:
        fails.append(f"trades {n} < {g['min_trades']}")
    if span_y < g["min_years"]:
        fails.append(f"span {span_y:.1f}y < {g['min_years']}")
    if tpd < g["min_per_day"]:
        fails.append(f"rate {tpd:.2f}/day < {g['min_per_day']:.2f}/day")
    if exp < g["min_exp"]:
        fails.append(f"expectancy {exp:+.3f}R < {g['min_exp']:+.2f}R")
    if dd90 > g["dd_cap_R"]:
        fails.append(f"dd90 {dd90:.1f}R > {g['dd_cap_R']}R")
    if risk < g["min_risk"]:
        fails.append(f"implied risk {risk*100:.2f}% < {g['min_risk']*100:.2f}%")
    if risk > g["max_risk"]:
        fails.append(f"implied risk {risk*100:.2f}% > {g['max_risk']*100:.2f}%")
    if months > g["max_months"]:
        fails.append(f"time to pass {months:.0f}mo > {g['max_months']:.0f}mo")
    if cost_share is not None and cost_share > g["max_cost_share"]:
        fails.append(f"cost {cost_share*100:.0f}% of gross > {g['max_cost_share']*100:.0f}%")
    if maxll * risk > g["max_ll_risk"]:
        fails.append(f"maxLL {maxll} x {risk*100:.2f}% > daily 5%")

    return dict(verdict="PASS" if not fails else "FAIL", fails=fails, n=n,
                years=span_y, per_day=tpd, exp=exp, pf=pf, maxll=maxll,
                dd90=dd90, dd50=dd50, risk=risk, need_R=need_R,
                need_trades=need_trades, months=months, cost_share=cost_share)


def show(name, res):
    if res["verdict"] == "INSUFFICIENT":
        print(f"{name:28} INSUFFICIENT  {res['fails'][0]}")
        return
    cs = f"{res['cost_share']*100:5.1f}%" if res["cost_share"] is not None else "   n/a"
    print(f"{name:28} {res['verdict']:5} n={res['n']:5d} {res['years']:4.1f}y "
          f"{res['per_day']:.2f}/d exp={res['exp']:+.3f}R PF={res['pf']:.2f} "
          f"dd90={res['dd90']:5.1f}R risk={res['risk']*100:5.2f}% "
          f"need={res['need_trades']:7.0f}tr {res['months']:5.0f}mo cost={cs}")
    for f in res["fails"]:
        print(f"{'':28}   - {f}")


def _selftest():
    """Score a synthetic ledger so the arithmetic is verifiable without data."""
    rng = np.random.default_rng(7)
    # 900 trades, 45% win rate, winners average 2R: positive but modest edge
    wins = rng.random(900) < 0.45
    r = np.where(wins, rng.gamma(2.0, 1.0, 900), -1.0)
    ledger = [dict(ts=1_262_304_000_000 + i * 3_600_000, r=float(v),
                   gross=float(v) + 0.15)
              for i, v in enumerate(r)]
    show("synthetic edge", screen(ledger))


def _csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            row = dict(ts=int(float(d["ts"])), r=float(d["r"]))
            for k in ("gross", "adverse"):
                if k in d and d[k] not in ("", None):
                    row[k] = float(d[k])
            rows.append(row)
    rows.sort(key=lambda x: x["ts"])
    show(path, screen(rows))


if __name__ == "__main__":
    if "--csv" in sys.argv:
        _csv(sys.argv[sys.argv.index("--csv") + 1])
    else:
        _selftest()
