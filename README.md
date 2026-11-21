# X01 — Asian liquidation, MSS and FVG

A mechanical implementation of the "Asian high/low liquidation" model published
by Michael Stahl on X, tested on 20 years of M5 FX data across six pairs.

Source thread: <https://en.rattibha.com/thread/1729181209597456510>

This repository is research code. It is not a trading system, not financial
advice, and it has never been run on a live account.

## Published rules

1. Wait for the Asian high or low to be liquidated. It must be liquidated
   before 03:00 New York.
2. Once liquidated, wait for a clear market structure shift (MSS) with a fair
   value gap (FVG).
3. Use the first candle of the FVG as the orderblock.
4. Enter with a limit order at that orderblock, targeting the opposite Asian
   extreme.
5. Place the stop slightly beyond the orderblock.
6. If the target is hit before the limit fills, cancel the order.
7. If the order is not filled, or the target is not reached, by 07:00 New York,
   close the position.
8. GBPUSD, M5, all times New York.

## Decisions taken where the source is silent

Every choice below was fixed before the first run and has not been tuned since.

| element | choice |
|---|---|
| Asian session | 20:00 (previous NY day) to 02:00 |
| Sweep window | 02:00 to 03:00, first bar trading beyond the range |
| MSS | first close beyond the highest high (long) / lowest low (short) of the ten bars before the sweep bar |
| FVG | first three-bar imbalance after the sweep: `low[i] > high[i-2]` for a long |
| Orderblock | that gap's first candle, `i-2` |
| Limit price | the orderblock's near edge, so a first touch fills |
| Stop | 2 pips beyond the orderblock's far edge |
| Target | the opposite Asian extreme |
| Attempts | one per pair per NY day |

## Results: six pairs, 20 years

Net of spread, 0.2 pip slippage and $5 commission per round turn.

| metric | value |
|---|---|
| trades | 5,434 |
| gross | +0.961R |
| net | **+0.703R** |
| profit factor | 2.00 |
| win rate | 43.2% |
| drawdown (90th pct, 500-trade windows) | 26.4R |
| trades per day | 0.74 |
| median stop | 7.2 pips |
| cost as share of gross | 27% |

Per pair, net R per trade:

| pair | trades | net R | PF |
|---|---|---|---|
| EURUSD | 1,067 | +0.768 | 2.24 |
| GBPUSD | 1,185 | +0.694 | 1.96 |
| USDJPY | 669 | +0.676 | 2.02 |
| GBPJPY | 910 | +0.658 | 1.80 |
| AUDUSD | 744 | +0.754 | 2.15 |
| EURJPY | 859 | +0.659 | 1.89 |

All 20 calendar years from 2006 to 2026 are positive net.

Tail dependence: removing the best 5% of trades leaves +0.365R (PF 1.49);
removing the best 10% leaves +0.110R (PF 1.14).

Cost stress: doubling the spread and adding 1 pip of slippage leaves +0.238R
(PF 1.24).

## Verdict against a funded-account screen

The screen (`prop_screen.py`) models an account that must make +15% while
surviving a 10% trailing drawdown and a 5% daily loss limit. X01 passes
expectancy, sample size, span, implied risk and time-to-target. It fails three
gates:

```
trade rate    0.74/day  <  2.00/day
drawdown      26.4R     >  25.0R
cost share    27%       >  25%
```

The rate gate is structural: at this per-pair rate, two trades per day needs
about sixteen pairs. The local corpus has six.

## Data

The M5 series are not included. Set `M5_DIR` to a directory containing:

```
eurusd_5m_20y.csv      gbpusd_5m_20y.csv      audusd_5m_20y.csv
usdjpy_5m_20y_d100.csv gbpjpy_5m_20y_d100.csv eurjpy_5m_20y_d100.csv
```

Each file is `open_utc_ms,open,high,low,close`, one row per five-minute bar,
strictly increasing timestamps.

Note on the JPY files: they store the quote divided by 100, so USDJPY appears
as `1.16736` rather than `116.736`. `pip_of()` reads the pip size from the
price scale instead of hardcoding it. Hardcoding a 0.01 pip for JPY pairs on
these files produces 200-pip stops, which turns the strategy into a
never-stopped target-hitter with a 78% gross win rate and a negative net. That
bug was found and fixed during this study; the fix moved the six-pair result
from -0.069R to +0.703R.

## Running

```
pip install -r requirements.txt
export M5_DIR=/path/to/csv            # Windows: set M5_DIR=...
python x01_asian_mss.py               # full history, six pairs
python x01_asian_mss.py --years=2     # last two years only
python x01_robust.py                  # tail, per-pair, per-year, cost stress
python prop_screen.py                 # screen arithmetic on a synthetic ledger
python prop_screen.py --csv ledger.csv
```

Per-pair trade rows are cached as `x01_<pair>_<span>.json`, so a rerun resumes
instead of recomputing finished pairs.

## Layout

| file | purpose |
|---|---|
| `x01_asian_mss.py` | session windows, sweep, MSS, FVG, orderblock, limit fill, management |
| `x01_robust.py` | tail removal, per pair, per year, spread and slippage stress |
| `prop_screen.py` | account-level acceptance arithmetic and gates |
| `data.py` | validated CSV loader, pip size, costs |
| `results/` | console output of the full-history run and the robustness run |

## Limitations

- One published setup per pair per day is rare: 5.2 trades per week across six
  pairs. Frequency, not edge, is the binding constraint.
- The median stop is 7.2 pips, so the result is sensitive to execution quality.
  At spread ×2 plus 1 pip of extra slippage the edge is a quarter of its base
  size, though still positive.
- Limit fills assume a touch at the orderblock edge is enough. Partial fills,
  queue position and broker stop levels are not modelled.
- Intrabar ordering is conservative: when a bar touches both stop and target,
  the stop is taken first.
- Trade rows from different pairs are evaluated as one serial ledger for the
  drawdown statistic, which overstates drawdown versus parallel positions but
  ignores the correlation between same-day trades.
