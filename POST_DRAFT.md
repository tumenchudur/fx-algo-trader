# Draft post: "I built an FX trading bot, then disproved it"

This is a draft you can edit and publish to LinkedIn / X / dev.to / Medium /
your own blog. Two versions: short (LinkedIn) and long (blog post). Both
link back to the repo and `RESEARCH.md`.

---

## Short version (LinkedIn / X — ~300 words)

**I spent a few months building a forex trading bot. Then I spent a few weeks
proving it doesn't work. Here's what I learned.**

The bot was a clean Python system: Pydantic-validated configs, backtest engine,
risk engine with kill switches, walk-forward harness, paper + MT5 execution,
proper cost modeling. It implemented a volatility-breakout strategy with
filters for trend, ADX, RSI, time-of-day, and volatility regime — the kind of
thing every retail trading YouTube video tells you to build.

It wasn't profitable in live trading. So instead of tuning more filters, I
went back to first principles and asked the question retail traders almost
never ask: **does the underlying signal have any predictive power before I
dress it up?**

I tested it properly:
→ Non-overlapping trades (so t-stats aren't inflated by overlap)
→ In-sample / out-of-sample temporal split
→ Realistic costs (0.8 pip EURUSD, 1.5 pip USDCHF, etc. — calibrated to real
  Pro account spreads)
→ Welch verification of any survivors against the null
→ Quarterly stability check (real edges are diffuse, not concentrated in one
  regime)

I tested **250+ strategy variants** across 21 years of D1, 2.8 years of H1,
and ~14 months of M5 data. Including 12-month FX momentum, the famous
Asness/Moskowitz "Value & Momentum Everywhere" combo, hour-of-day, day-of-week,
and cross-asset signals (DXY, TNX, VIX, SPX).

**Result: only one strategy survived.** A vol-targeted FX carry basket on D1.
Sharpe 0.26 net of costs, IS 0.17, OOS 0.37 (out-of-sample better than
in-sample — the strongest possible validation). The famous momentum factor
showed IS Sharpe 0.23 but **OOS Sharpe -0.01** — pure overfitting. The
carry+momentum combo? Sharpe -0.03. The Asness/Moskowitz result is a
1974-2013 phenomenon that does not replicate in modern data.

The full research log, methodology, code, and reusable scripts are open source:
[github.com/USERNAME/quant-trading](https://github.com/USERNAME/quant-trading)

The negative result is the result. If you're a retail trader thinking about
building a bot, save yourself months: read RESEARCH.md first.

---

## Long version (blog post — ~1200 words)

### I built a forex trading bot. Then I disproved it.

Like a lot of engineers, I assumed I could outsmart retail forex with
disciplined Python and a proper risk engine. I built a production-quality
trading system: Pydantic-validated configs, a bar-by-bar backtest engine, a
real risk layer with kill switches and exposure caps, walk-forward harness,
paper trading, and MT5 execution for live deployment.

The strategy was a volatility breakout with filters: trend (50-EMA), strength
(ADX > 25), session (London open + NY morning), RSI extremes, and a volatility
regime overlay. The kind of strategy every YouTube tutorial recommends.

It wasn't profitable.

What follows is the research log of how I figured out *why* — and the one
strategy that turned out to actually work, at a Sharpe so modest you wouldn't
have stayed interested if you saw it on a brokerage marketing page.

### Step 1: stop tuning. Start asking the real question.

The first instinct was to tune. Add another filter. Optimize the ATR
threshold. Remove a losing pair from the basket. The configs in the repo
became a museum of overfitting:

```yaml
symbols:
  - EURUSD
  - USDJPY
  # GBPUSD removed - losing money
allowed_sides: short  # SHORT only (LONG losing money)
```

Each comment is a textbook fitting decision dressed as a discovery. The data
told me both directions on EURUSD lost money; instead of considering that the
signal had no edge, I picked the half that had been less unlucky in-sample.

So I stopped tuning and went back to a simpler question: **does the underlying
signal — Donchian breakout — have any positive expectancy on real data, before
filters, before SL/TP, before any tuning?**

### Step 2: test honestly

I built a research framework with five rules:

1. **Non-overlapping trades.** If a signal fires every 5 minutes for 3 hours
   and each trade holds 24 hours, the trades share most of their PnL — your
   t-statistic is meaningless. Once a trade fires at bar t, the next eligible
   entry is bar t + horizon + 1. This single rule killed dozens of false
   positives.

2. **In-sample / out-of-sample split.** Cut each dataset at the temporal
   midpoint. A signal must work in both halves.

3. **Realistic costs.** Round-trip 0.8 pip EURUSD, 1.0 pip GBPUSD/USDJPY/AUDUSD,
   1.5 pip USDCHF/NZDUSD. These match real Exness/IC Markets Pro account
   spreads on liquid hours.

4. **Welch verification.** Even if a signal passes IS/OOS, run a Welch t-test
   of its returns against the population at non-signal entries. This caught
   the "long XAUUSD on Monday" signal — Welch t = 0.41, statistically
   indistinguishable from random Monday entries on a heavily-trending asset.

5. **Quarterly stability.** Real edges are diffuse. If 70% of profit comes
   from one quarter, it's a regime artifact.

### Step 3: test everything

Across 21 years of daily data, 2.8 years of hourly, and ~14 months of M5, I
tested:

- **Price-only:** momentum (5/20/50/120/240 lookback), mean-reversion z-score,
  Donchian breakout and fade, RSI extremes, hour-of-day, day-of-week.
- **Cross-asset:** DXY momentum, US 10Y yield (TNX) momentum, VIX risk-off
  basket, SPX risk-on basket, gold-as-USD-inverse.
- **Factor:** vol-targeted FX carry basket from BIS central bank policy
  rates, 12-month momentum, and the Asness/Moskowitz carry+momentum combo.

About **250 strategy variants** in total.

### Step 4: results

**Of the 250+ variants tested, exactly one survived rigorous validation.**

| Strategy class | Best Sharpe (NET, OOS) | Verdict |
| --- | --- | --- |
| FX majors, indicator strategies (M5/H1) | — | Zero pass IS+OOS+Welch |
| Cross-asset signals (DXY, TNX, VIX, SPX) | — | All apparent edges fail Welch or break in last quarter |
| Hour-of-day calendar effects | — | 0/192 hours pass |
| 12-month FX momentum | -0.01 | Pure in-sample overfit |
| Carry + momentum combo (Asness/Moskowitz) | 0.16 | Worse than carry alone in modern data |
| **Vol-targeted FX carry on D1** | **0.37** | **Real but modest** |

The bot's volatility-breakout strategy, tested raw at 4-hour horizon on EURUSD
M5: net -1.5 pips per trade, t = -2.7 in-sample, t = -3.0 out-of-sample. A
verified negative-expectancy generator. No filter combination tested rescues it.

The "Value & Momentum Everywhere" result — combo Sharpe ~0.7 in the 2013 paper —
**does not replicate** in 2005-2026 FX data. Combo Sharpe -0.03 full sample,
worse than carry alone.

Carry alone: 21 years, monthly rebalanced, K=2 long top yielders / short bottom
yielders, vol-targeted to 10%. Sharpe 0.26 NET, IS 0.17, **OOS 0.37**. The
out-of-sample being *better* than in-sample is the strongest signal you can
get that something is not curve-fit. This is a real edge.

### Step 5: face the math

At Sharpe 0.25 with 9% annualized vol, a $10,000 retail account earns about
$220/year in expectation, with 1-in-6 odds of losing >9% in a given year and
1-in-40 odds of losing >27% (which actually happened in 2008). To turn that
into "interesting" retail returns you'd lever 3-5×. That scales the drawdowns
to 80-100% — which is exactly what killed the retail carry-trade community in
2008 and 2015.

The only validated edge I found in retail FX is too small to lever safely and
too small to compound interestingly without leverage. *That is the answer.*

### What this is worth

The bot isn't profitable. The codebase is. The research log is. The
methodology is.

If you're hiring an engineer: this repo demonstrates the ability to build
serious infrastructure, test it rigorously, and kill bad ideas. Three things
that are individually rare in retail and together quite uncommon.

If you're a retail trader thinking about building a bot: please read
RESEARCH.md before you start. It will save you months. The path that ends
in "indicator-based bot generates retail income" is closed. The earlier you
believe it, the more time and money you save.

The repo: [github.com/USERNAME/quant-trading](https://github.com/USERNAME/quant-trading)

---

### Notes for posting

- Replace `USERNAME` with your GitHub handle.
- Optional: add a screenshot of the carry equity curve. To generate one:
  `.venv/bin/python -c "import matplotlib.pyplot as plt; ..."` — easy
  follow-up if you want it.
- The headline that has historically performed best on this kind of post is
  the *negative* framing: "I built X, then disproved it." Counter-intuitive,
  rare, signals confidence and rigor.
- Tag accounts that posted the original studies you contradict
  (Asness/Moskowitz on the value-momentum result) — drives engagement
  through controversy without being aggressive.
- Cross-post to: LinkedIn, /r/algotrading, Hacker News (Show HN), dev.to.
