"""Account-level replay for the AquaFunded 2-Step Pro $5,000 rules.

Two instruments, both fed the same trade ledger:

  hard_limit()  §7.3 evidence. Replays fresh $5,000 accounts from many
                historical start points in real trade order. A breach here is
                a real breach, not a model output, so it disqualifies the
                configuration before any Monte Carlo is believed.

  run_mc()      Block bootstrap through phase 1 and phase 2. Blocks of 25
                trades preserve clustering. Daily floor is marked against
                floating equity using each trade's adverse excursion, not just
                at exit.

Rules, read from help.aquafunded.com/en/articles/15281229-2-step-pro on
2026-11-21 (see strategies/sunday_reopen_fade/prop/rules.py in the research
tree):

  Phase 1 target    +10% of $5,000
  Phase 2 target    +5% of $5,000, phase 2 restarts at $5,000
  Daily loss        5% of the higher of balance/equity at 00:00 UTC
  Max total loss    10% of starting balance, trailing below the peak
  No time limit, no minimum trading days during evaluation
"""
from __future__ import annotations

import numpy as np

ACCOUNT = 5_000.0
STEP1, STEP2 = 0.10, 0.05
DAILY, TOTAL = 0.05, 0.10
BLOCK = 25
MAX_STEPS = 1000


def hard_limit(rows, risk, starts=400, max_trades=600):
    """Replay real challenge-sized windows from many historical start points.

    Each start point gets a fresh $5,000 and a reset trailing floor. The replay
    ends at pass, breach, or max_trades. Returns the breach rate and the worst
    drawdowns seen in those windows.
    """
    r = np.array([x["r"] for x in rows])
    adv = np.array([x.get("adverse", 0.0) for x in rows])
    day = np.array([x["ts"] for x in rows], np.int64) // 86400000
    n = len(r)
    if n < 50:
        return dict(breach=0.0, trail=0.0, daily=0.0, passed=0.0,
                    trail_mult=0.0, daily_mult=0.0)
    idx = np.linspace(0, n - 2, starts).astype(int)
    breach = passed = 0
    worst_t = worst_d = 0.0
    for s in idx:
        bal = ACCOUNT
        peak = ACCOUNT
        phase = 1
        cur = -1
        start_bal = bal
        floor = bal * (1 - DAILY)
        dead = False
        for i in range(s, min(s + max_trades, n)):
            if day[i] != cur:
                cur = day[i]
                start_bal = bal
                floor = bal * (1 - DAILY)
            low = bal - adv[i] * risk * bal
            worst_d = max(worst_d, (start_bal - low) / start_bal)
            if low < floor:
                dead = True
                break
            bal += r[i] * risk * bal
            peak = max(peak, bal)
            worst_t = max(worst_t, (peak - bal) / ACCOUNT)
            if bal <= peak - TOTAL * ACCOUNT:
                dead = True
                break
            if bal >= ACCOUNT * (1 + (STEP1 if phase == 1 else STEP2)):
                if phase == 2:
                    passed += 1
                    break
                phase = 2
                bal = ACCOUNT
                peak = ACCOUNT
                cur = -1
        breach += dead
    m = len(idx)
    return dict(breach=breach / m, passed=passed / m, trail=worst_t,
                daily=worst_d, trail_mult=worst_t / TOTAL,
                daily_mult=worst_d / DAILY)


def run_mc(r, adv, day, risk, paths, seed=7):
    """Block-bootstrap phase 1 -> phase 2. Returns per-path outcomes."""
    n = len(r)
    rng = np.random.default_rng(seed)
    nblk = MAX_STEPS // BLOCK
    starts = rng.integers(0, n, size=(paths, nblk))
    off = np.arange(BLOCK)[None, None, :]
    seq = ((starts[:, :, None] + off) % n).reshape(paths, nblk * BLOCK)

    bal = np.full(paths, ACCOUNT)
    peak = np.full(paths, ACCOUNT)
    phase = np.ones(paths, np.int8)
    floor = np.full(paths, ACCOUNT * (1 - DAILY))
    cur = np.full(paths, -1, np.int64)
    steps = np.zeros(paths, np.int64)
    live = np.ones(paths, bool)
    banked = np.zeros(paths, np.int64)

    for s in range(seq.shape[1]):
        if not live.any():
            break
        idx = seq[:, s]
        rr, aa, dd = r[idx], adv[idx], day[idx]
        newday = live & (dd != cur)
        floor = np.where(newday, bal * (1 - DAILY), floor)
        cur = np.where(newday, dd, cur)
        low = bal - aa * risk * bal
        dead = live & (low < floor)
        bal = np.where(live & ~dead, bal + rr * risk * bal, bal)
        peak = np.maximum(peak, bal)
        dead |= live & (bal <= peak - TOTAL * ACCOUNT)
        steps += live
        live &= ~dead
        banked[dead & (phase == 2)] = 2
        hit = live & (bal >= ACCOUNT * (1 + np.where(phase == 1, STEP1, STEP2)))
        p1 = hit & (phase == 1)
        bal = np.where(p1, ACCOUNT, bal)
        peak = np.where(p1, ACCOUNT, peak)
        floor = np.where(p1, ACCOUNT * (1 - DAILY), floor)
        cur = np.where(p1, -1, cur)
        phase = np.where(p1, 2, phase).astype(np.int8)
        banked[p1] = 1
        p2 = hit & (phase == 2) & ~p1
        banked[p2] = 3
        live &= ~p2

    return dict(survived=banked == 3, p1=(banked >= 1) & (banked < 3),
                p2_fail=banked == 2, steps=steps)
