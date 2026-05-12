# Research log: does this bot have an edge?

This document records the systematic search for a tradeable edge in the strategies
this codebase implements (and several alternatives), conducted across 9 sessions
over 2026-Q2. **Headline result: the bot's stock strategy has no post-cost edge.
One alternative — vol-targeted FX carry on D1 — survives rigorous validation at
modest Sharpe (~0.25-0.37 net). Everything else tested either failed in
out-of-sample, was a drift artifact, or was a methodological mirage.**

The point of writing this down is not to claim a discovery. The point is to
record *what was actually tested, how, and what the results were*, so that
future-me (or a reader of this repo) does not waste another month rediscovering
the same negative answers.

---

## TL;DR

| Asset class | Timeframe | Best validated Sharpe (NET, OOS) | Verdict |
| --- | --- | --- | --- |
| FX majors, indicator strategies | M5 | — | None survive proper validation |
| FX majors, indicator strategies | H1 | — | None survive proper validation |
| FX majors + cross-asset signals (DXY, TNX, VIX, SPX) | H1 | — | All apparent edges fail Welch verification |
| **FX carry (top-2/bottom-2, vol-targeted)** | **D1** | **0.37** | **Real but modest** |
| FX carry + 12m momentum (Asness/Moskowitz) | D1 | — | Combo is worse than carry alone in 2005-2026 |

The bot's bundled `volatility_breakout` strategy on M5/H1 majors is, in raw
form, a **negative-expectancy generator** at realistic cost levels. No filter
combination tested rescues it. The reason is structural: spread-to-noise ratio
on M5 majors against institutional flow leaves no recoverable signal in
public price-derived indicators.

---

## Why this exercise was done

The original strategy in `src/fx_trading/strategies/volatility_breakout.py` was
producing live losses despite passing in-sample backtests on `data/sample/`.
The configs in `configs/backtest_optimized.yaml` (with comments like
`# GBPUSD removed - losing money` and `allowed_sides: short  # SHORT only
(LONG losing money)`) revealed textbook overfitting: per-symbol, per-side
parameter tuning to historical sample drift.

Rather than tune more, I went back to first principles: **does the underlying
signal have any predictive power before it gets dressed up with filters?**
This document is the answer.

---

## Methodology

The standard retail backtesting failure modes — the ones that produced the
original misleading "winning" config — were addressed explicitly:

### 1. Non-overlapping trades

Almost every false "edge" found in the early M5 work was due to trade overlap
inflating t-stats. If a signal fires at every 5-minute bar within a 3-hour
window and each "trade" holds for 24 hours, the trades share most of their PnL
and the t-statistic on the per-trade PnL is meaningless. Every test in this
document uses the rule: **once a trade fires at bar `t`, the next eligible
entry is bar `t + horizon + 1`.** Per-trade PnLs are then independent samples
suitable for a t-test.

### 2. In-sample / out-of-sample split

Each dataset is split at its temporal midpoint. A signal must be profitable
*and* statistically significant in **both** halves to be considered surviving.
Validation criteria used:
- For raw signal sweeps: net pips > 0 in both halves, |t_is| > 1.5,
  |t_oos| > 1.0.
- For the carry basket: Sharpe sign agreement IS/OOS, OOS ≥ IS preferred.

### 3. Realistic round-trip costs

Applied per trade as a flat cost in pips, calibrated to Exness/IC Markets Pro
account spreads on liquid hours plus 0.2 pip slippage:

| Symbol | Round-trip cost (pips) |
| --- | --- |
| EURUSD | 0.8 |
| GBPUSD, USDJPY, AUDUSD | 1.0 |
| USDCAD | 1.2 |
| USDCHF, NZDUSD | 1.5 |
| XAUUSD | 25-30 |

For the daily carry basket, costs are charged on **weight turnover** at
~1 bp per absolute weight change per leg — equivalent to the per-leg pip
cost on a 1-USD-notional trade for majors.

### 4. Drift detrending

Several apparent "edges" turned out to be exposure to underlying directional
drift in the sample (XAUUSD's 2025 bull market most notably). For
hour-of-day analysis, returns are demeaned per-symbol before grouping.
For long-only signals on trending instruments, a Welch t-test against the
"any other entry point" baseline is used (see survivor #1 below).

### 5. Welch verification of survivors

A signal that passes IS+OOS gets one more test: is it statistically
distinguishable from the population of returns at non-signal entry points?
This caught the gold-Monday signal (Welch t=0.29-0.78 — Monday entries are
indistinguishable from random entries on this asset).

### 6. Quarterly stability check

Real edges should be diffuse, not concentrated. For each survivor, PnL is
broken down by quarter / year. A signal where 70% of profit comes from one
quarter is not a robust edge — it's a regime artifact.

---

## Datasets

| Source | Symbols | Frequency | Span | Rows/symbol |
| --- | --- | --- | --- | --- |
| `data/sampl/`, `data/sample/`, `data/processed/` (mixed origin) | EURUSD, GBPUSD, USDJPY, XAUUSD | M5 | 2025-01-01 → 2026-03-02 | 6k-18k |
| `data/yf/` (yfinance H1) | 7 FX majors + DXY, TNX, VIX, SPX, NDX, WTI, XAUUSD | H1 | 2023-07-19 → 2026-05-06 | ~17k FX, varies for cross-asset |
| `data/yf_d1/` (yfinance D1) | 7 FX majors | D1 | 2005-01-03 → 2026-05-05 | ~5.5k |
| `data/rates/short_rates.parquet` (BIS bulk download) | USD/EUR/GBP/JPY/AUD/NZD/CAD/CHF central bank policy rates | Monthly → ffilled to D1 | 1999 → 2026-03 | ~29k after resample |

Reproducibility: see `scripts/download_yfinance.py`, `scripts/download_yfinance_d1.py`,
and `scripts/parse_bis_rates.py`. The BIS bulk CSV is downloaded from
`https://data.bis.org/static/bulk/WS_CBPOL_csv_flat.zip` after first hitting
`https://data.bis.org/bulkdownload` to obtain a session cookie.

(FRED's CSV endpoint refused responses from this network; an API-key route
works if you sign up for one. We bypassed this by using the BIS bulk
publication, which is more comprehensive anyway.)

---

## Signal universe tested

### Sweep 1: M5 raw signals (`scripts/edge_research.py`)

19 signal families × 4 symbols × 3 horizons (1h / 4h / 1d) = **228 strategy
variants** evaluated with non-overlapping trades on 1 year of M5 data.

Families:
- Momentum: 5/20/50 lookback, sign of N-bar return.
- Mean reversion: z-score of close vs N-bar mean, threshold 2.0/2.5.
- RSI extremes: 5/14 period, oversold/overbought thresholds.
- Donchian breakout (the bot's strategy, raw): N-bar high/low.
- Donchian fade: inverted breakout.
- Hour-of-day session bias: London 7-9 UTC, NY 13-15 UTC, Asia 1-5 UTC,
  long/short variants.

**Result: zero signals positive in both halves with |t| > 2.**

The "anti-edges" initially flagged (signals with consistently *negative*
net returns and large negative t-stats) turned out to be a **methodological
artifact** — the signals had ~zero gross expectancy and were losing only the
~1 pip cost margin every trade. Inverting the rule does not produce a +1 pip
edge; it produces another -1 pip loss in the opposite direction. (Tested
explicitly in `scripts/edge_verify.py`: AUDUSD inverted mom_24 4h shows
mean_net = -0.4 pips, t = -0.97 — the inversion is also money-losing once
you re-pay the spread.)

### Sweep 2: M5 hour-of-day, detrended (`scripts/edge_hour_of_day.py`)

24-hour-of-day average return per symbol, demeaned to remove drift. IS/OOS
consistency requirement: same sign, |t| > 2 in both halves.

**Result: 0/24 hours pass for any of EURUSD/GBPUSD/USDJPY/XAUUSD.** Several
hours are significant in IS but flip sign or go to noise OOS — pure
overfitting to sample drift.

### Sweep 3: H1 + cross-asset signals (`scripts/edge_xasset.py`)

11 price signals + 9 cross-asset signals × 8 symbols × 4 horizons
(4h / 1d / 3d / 1w) = **~542 evaluated** on 2.8 years of H1 data.

Cross-asset signals tested:
- VIX z-score spike → risk-off basket (long JPY/CHF, short AUD/NZD).
- DXY momentum → USD-base/quote pair direction.
- TNX (10Y yield) momentum → USD strength.
- SPX momentum → risk-on/off basket.

Initial IS+OOS filter found 5 "robust survivors":
- `dow_mon_long XAUUSD h=72,168` — long gold at Monday open.
- `dow_thu_long XAUUSD h=168` — long gold at Thursday open.
- `tnx_mom_24 USDJPY h=72` — buy USDJPY when 10Y yield rises.
- `tnx_mom_120 AUDUSD h=168` — sell AUDUSD when 10Y yield rises (5-day).

### Sweep 4: Verification of H1 survivors (`scripts/edge_verify.py`)

All 5 H1 survivors were destroyed by deeper analysis:

**Gold-Monday signals: drift artifacts.** Welch t-test of
"Monday-entry XAUUSD long" vs "any-other-bar XAUUSD long" returns:
- horizon 24h: Welch t = **0.41**
- horizon 72h: Welch t = **0.78**
- horizon 168h: Welch t = **0.29**

Monday entries are statistically *indistinguishable* from random entries.
Gold simply rallied throughout the sample (mean per-day pip move > 200);
any long-bias rule on XAUUSD inherits that drift. Not edge.

**TNX → USDJPY: regime breakdown.** Quarterly PnL:
| Quarter | PnL (pips) |
| --- | --- |
| Q1 | +1330 ✓ |
| Q2 | +1252 ✓ |
| Q3 | **-205** |
| Q4 | **-744** |

The yield-USD relationship that drove the "edge" for the first two quarters
of OOS broke and inverted in the most recent two. This is the single most
important kind of failure to catch: an academically-plausible signal that
already stopped working before live deployment.

**TNX → AUDUSD: insufficient sample.** 87 trades total over 2.8 years,
56% of cumulative PnL came from a single quarter. Edge concentrated in one
regime; not robust enough to deploy.

**Combined yield-strength basket (long USDJPY + short AUDUSD on TNX rise),
vol-normalized:** t-stat on per-trade returns = 0.28-0.41. No combined edge.

### Sweep 5: D1 FX carry basket (`scripts/carry_basket.py`)

Long top-K highest-yielding currencies vs short bottom-K lowest-yielding,
vol-targeted to 10% annualized portfolio vol, monthly rebalanced from BIS
central bank policy rates, 21 years of D1 history.

| K | NET Sharpe (full) | IS Sharpe | OOS Sharpe | Max DD | Final eq (21y) |
| --- | --- | --- | --- | --- | --- |
| 2 | **0.26** | 0.17 | **0.37** | -27.18% | 1.49× |
| 3 | 0.12 | 0.16 | 0.08 | -31.70% | 1.16× |

K=2 is the validated winner. **Out-of-sample Sharpe (0.37) exceeds in-sample
(0.17)** — the strongest possible sign that this is not curve-fit to sample.

This is the only signal in the entire research effort that survives all five
methodological filters: non-overlapping rebalances, IS+OOS, realistic costs,
drift-irrelevant (carry is a true cross-sectional bet, not a directional
exposure to one asset's drift), and quarterly diffuse PnL.

### Sweep 6: D1 carry + 12m momentum combo (`scripts/carry_momentum.py`)

Three variants compared, K=2 each:

| Strategy | Sharpe full | Sharpe IS | Sharpe OOS |
| --- | --- | --- | --- |
| Carry-only | 0.26 | 0.17 | **0.37** |
| 12m-momentum-only | 0.12 | 0.23 | **-0.01** |
| Carry + Momentum (rank-avg) | -0.03 | -0.17 | 0.16 |

The famous Asness/Moskowitz "Value & Momentum Everywhere" (2013) result —
combo Sharpe ~0.7 historically — **does not replicate on 2005-2026 FX
data**. Two reasons we can speculate about:
1. The post-GFC ZIRP era (2010-2021) compressed all rates into a narrow
   band, removing the cross-sectional momentum that drove the 1974-2013
   sample.
2. The carry-momentum factor correlation in our sample is +0.13 (not
   negative), so they don't diversify each other — the combo just dilutes
   carry without adding alpha.

12m FX momentum is, in modern data, an in-sample-only phenomenon. Don't
build on it.

---

## The single validated edge: vol-targeted carry, in detail

### What it is
Each month, on the first trading day:
1. Rank the 7 non-USD G10 currencies by their central bank policy rate.
2. Long the top K (here K=2) highest-yielding currencies vs USD.
3. Short the bottom K lowest-yielding currencies vs USD.
4. Size each leg by `1 / realized_60d_vol` (vol-parity within basket),
   then rescale gross |w| = 1, then rescale to target 10% annualized
   basket vol.
5. Hold positions flat until next month-end rebalance.

For pairs where USD is the quote (EURUSD, GBPUSD, AUDUSD, NZDUSD), "long
the currency" = long the pair. For pairs where USD is the base (USDJPY,
USDCHF, USDCAD), "long the currency" = short the pair.

### Why it works (the economic story)
This is the FX carry trade, the most studied anomaly in the FX literature.
The premise: forward exchange rates set by covered interest parity should
predict future spot moves, but empirically high-yielding currencies do
*not* depreciate enough over time to offset the rate differential. The
trader collecting the rate differential earns a positive expected return
in exchange for periodic large losses (the "carry crash" — 2008, 2015,
2020). Sharpe in the academic literature: 0.3-0.6 unlevered.

Our 0.26 NET / 0.37 OOS Sharpe is consistent with this range, on the
lower end because:
- We use central bank policy rates, not deposit/forward rates, slightly
  underestimating the carry actually capturable.
- The 2005-2026 sample is unusually carry-unfriendly: 2008 GFC + ZIRP era
  compression + rapid 2022-2024 rate normalization.

### Why it can be deployed in this codebase
Carry is a slow signal: 12 rebalances per year, 4 active legs per
rebalance = ~48 trades/year. The bot's existing `RiskEngine`,
`PortfolioManager`, and `PaperBroker` all handle this trivially. The
strategy needs only:
- A new `CarryBasket` strategy class implementing `Strategy.generate_signals`
  by reading `data/rates/short_rates.parquet` and computing
  cross-sectional ranks at month-start.
- D1 data instead of M5 — adjust the existing data loaders.
- The vol-targeting layer needs adding to `position_sizing.py` (currently
  the position sizer uses fixed risk-per-trade, not portfolio-vol
  targeting).

### Why retail-account math is still hard
At 9% annualized vol and Sharpe 0.25-0.37:
- Expected annual return ~2.2-3.3% unlevered.
- ~1-in-6 years you lose >9%.
- ~1-in-40 years you lose >27% (and one of these is in our sample —
  the 2008-2009 carry blowup hit -27%).

To turn this into "interesting" returns at retail scale (5-10% / year)
you'd lever 2-3×, which scales drawdowns to 50-80%. That's exactly what
killed the retail carry-trade community in 2008 and 2015. There is no
levered version that doesn't periodically blow up.

This is the core retail dilemma: the only edges that actually exist in
liquid public markets are too small to compound to interesting retail
returns without leverage that destroys you on the bad quarters.

---

## What didn't work and why

### The bundled volatility-breakout strategy

Tested raw (no SL/TP/filters) at M5 / H1. On EURUSD M5 with non-overlapping
trades:
- Donchian-20 4h horizon, in-sample: net = -1.5 pips/trade, t = -2.7
- Donchian-20 4h horizon, out-of-sample: net = -1.2 pips/trade, t = -3.0

The strategy is a *consistent loser* on majors at short horizons. The
filters in the optimized config (RSI, ADX, EMA, time-of-day, volatility
regime) cannot rescue a negative-expectancy generator. Each filter just
reduces the trade count, not the per-trade expectancy.

### The "anti-edge" inversion idea

Initial reading of the M5 sweep showed several signals with consistently
negative net returns (Donchian, momentum on majors at h=4). I hypothesized
that inverting these would produce edge. **It does not.** Verified
explicitly: those signals had zero gross expectancy and were losing only
the cost margin. Inverting the position just makes you pay the same cost
in the opposite direction.

**Lesson:** A negative-expectancy signal isn't a hidden positive-expectancy
signal in disguise. To find a real anti-edge you'd need t-stat << 0
*before* costs.

### Hour-of-day calendar effects

24 hours × 4 symbols × IS/OOS = 192 statistical tests. Zero pass with
same-sign |t| > 2 in both halves. Several individual hours show |t| > 2
in IS but flip in OOS — exactly what you'd expect from running 192 tests
under the null hypothesis.

### Cross-asset signals (DXY, TNX, VIX, SPX)

Conceptually sound (yield differentials should drive FX, risk-off should
move JPY/AUD), but in practice all apparent edges either failed Welch
verification or showed regime breakdown in the most recent quarter. The
TNX → USDJPY signal is the cautionary tale: 1.5 years of working, then
two quarters of breaking, just before we'd have deployed it.

### 12-month FX momentum

Famous Asness/Moskowitz factor. IS Sharpe 0.23, OOS Sharpe -0.01.
**Modern data has killed this signal.** Possibly resurrects in a future
high-rate-vol regime, but cannot be deployed today on this evidence.

### Combined carry + momentum

Sharpe -0.03 full sample, -0.17 IS. The famous "Value & Momentum
Everywhere" result is a 1974-2013 phenomenon. Adding momentum to carry
in 2005-2026 actively *hurts* — the factors are weakly positively
correlated (+0.13) so they don't diversify, and momentum is an
in-sample-only effect on this data.

---

## Implications for this codebase

### What the engine does well

- **Risk engine** (`src/fx_trading/risk/engine.py`): kill switches, exposure
  caps, anti-martingale, anti-revenge cooldown. Generic across strategies.
- **Portfolio accounting** (`src/fx_trading/portfolio/accounting.py`):
  fills, mark-to-market, position aggregation. Generic.
- **Cost model** (`src/fx_trading/costs/models.py`): spread + slippage +
  commission applied at fill time. Generic.
- **Backtest engine** (`src/fx_trading/backtesting/engine.py`): bar-by-bar
  loop with the same broker/risk/portfolio components used in live mode.
  Generic.
- **Walk-forward** (`src/fx_trading/backtesting/walkforward.py`): rolling
  window OOS framework. Already in place — should be the *primary*
  evaluator for any new strategy.

### What the engine does poorly

- **Strategy class is M5-biased.** `Strategy.generate_signals(data,
  current_index)` assumes one bar = one decision opportunity. For carry
  (monthly rebalance, cross-sectional ranking) this is wrong. The
  interface needs a `Portfolio` analog where the strategy returns a target
  weight vector across symbols, not per-symbol signals.
- **Position sizing is per-trade, not portfolio-vol.**
  `portfolio/position_sizing.py` sizes each trade independently from
  `max_risk_per_trade_pct`. Carry needs portfolio-vol targeting across
  multiple correlated legs simultaneously.
- **Multi-symbol data plumbing leaks.** The hack in
  `volatility_breakout.py:131-141` to filter by symbol when columns are
  interleaved is a smell. Real multi-asset strategies need a proper
  panel/wide-format data abstraction.

### What to keep

The whole repo, as a portfolio piece. The infrastructure quality is
genuinely above retail average. Specifically advertise:
- Pydantic v2 config models with strict validation.
- Clean separation of strategy / risk / execution / accounting.
- Realistic cost modeling (most retail bots ignore slippage entirely).
- Walk-forward harness — even though we didn't deploy strategies through
  it for this research, it's there.
- This research log. Documented disprovals are more credible than
  unverified claims of profitability.

### What to stop doing

- Tuning `volatility_breakout` parameters. Stop. The base signal is
  negative-expectancy and no amount of filter tuning fixes that.
- Backtesting on `data/sample/` synthetic data and treating the result as
  meaningful. Synthetic Brownian motion has no microstructure, no fat
  tails, no regimes — it tells you nothing about live performance.
- Removing symbols from configs because they're "losing money" without
  asking *why*. That's the textbook overfit move that produced the
  original config.
- Going live with anything that hasn't been walk-forward tested on real
  data with realistic costs across at least 3 years and 3 market regimes.

---

## Reproducibility

All scripts are self-contained and use the venv at `.venv/`.

```bash
# 1. Pull data
.venv/bin/python scripts/download_yfinance.py        # 2.8y H1 + cross-asset
.venv/bin/python scripts/download_yfinance_d1.py     # 21y D1
.venv/bin/python scripts/parse_bis_rates.py          # rates from BIS bulk

# 2. Run the sweeps (any order)
.venv/bin/python scripts/edge_research.py            # M5 sweep
.venv/bin/python scripts/edge_hour_of_day.py         # M5 hour-of-day
.venv/bin/python scripts/edge_research_h1.py         # H1 sweep
.venv/bin/python scripts/edge_xasset.py              # H1 + cross-asset
.venv/bin/python scripts/edge_verify.py              # Welch verify of survivors

# 3. The one validated edge
.venv/bin/python scripts/carry_basket.py             # carry K=2,3
.venv/bin/python scripts/carry_momentum.py           # carry vs mom vs combo
```

The only network dependencies are yfinance (FX + cross-asset proxies) and
the BIS bulk download endpoint for central bank rates. No paid data
required.

---

## What I'd do next, ranked by realism

1. **Stop running the bot live.** Expected value is negative after
   transaction costs and time cost.
2. **Polish the repo as a portfolio piece.** This document, a clean
   commit history, a top-level README that links to it. Makes the work
   credible to a quant/fintech employer.
3. **If you want carry exposure anyway:** allocate a few % of savings to
   a managed-futures ETF (DBMF, KMLM) or a multi-strategy CTA. Same
   factor exposure, professional execution, no maintenance burden, no
   leverage-blowup risk you control.
4. **If you want to keep building:** redirect the energy to where retail
   still has edge — options-selling on individual stocks (variance risk
   premium, real Sharpe 0.4-0.6), small-cap value screens, or the
   software side of fintech.

The negative result is the result. It's not a failure of the bot or the
researcher; it's the actual answer to the question "is there post-cost
edge in retail FX bot trading?" The answer, after this much testing, is
"no, except for slow modest carry that retail-scale math doesn't justify
trading directly."
