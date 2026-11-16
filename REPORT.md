# Model test report

_Generated 2026-11-16 04:40 UTC. Data through 2026-11-13._

## Verdict

**No reliable edge. The model is not distinguishable from a coin flip / simple baselines once you account for noise.**

- Out-of-sample accuracy 54.0% vs 56.0% for simply guessing 'up' every time (difference -2.0%, 90% range -2.9% to -1.0%).
- ROC AUC 0.509 (0.5 = no skill; 90% range 0.494-0.524); best simple baseline AUC 0.505.
- The 10-90% range caught the actual move 79% of the time after calibration (target 80%).
- Backtest after costs: 47.0%/yr (Sharpe 1.37) vs 42.0%/yr (Sharpe 1.31) for just holding all the stocks. The rule beat that on risk-adjusted terms.

## Direction (out-of-sample)

| model | accuracy | auc | brier |
|---|---|---|---|
| Baseline: always up | 56.0% | 0.500 | 0.2467 |
| Baseline: last week repeats | 50.0% | 0.493 | 0.4996 |
| Baseline: historical average | 56.0% | 0.505 | 0.2467 |
| Logistic regression | 54.0% | 0.509 | 0.2495 |
| LightGBM | 54.3% | 0.499 | 0.2505 |

## Ranges (target coverage 80%)

| model | coverage | median_width | pinball |
|---|---|---|---|
| Baseline (volatility only) | 79.6% | 13.0% | 0.01540 |
| LightGBM (raw) | 76.2% | 11.5% | 0.01521 |
| LightGBM (calibrated) | 79.1% | 12.4% |  |

## Tomorrow's range (target coverage 80%)

| model | coverage | median_width |
|---|---|---|
| Baseline (volatility only) | 79.7% | 5.5% |
| LightGBM (raw) | 78.2% | 5.1% |
| LightGBM (calibrated) | 79.2% | 5.2% |

## Experiments: easier questions

Developed on data before 2025-01-01; graded once on the final exam (2025-01-01 onward).

- **Up or down, 1 week (the original question)**: No edge. Dev AUC 0.508 (0.489-0.522), ranking IC +0.018 (t=2.1), top-5 edge -3.2%/yr. Final exam: AUC 0.510, IC -0.001, top-5 edge -13.3%/yr.
- **Beats its peers, 1 week**: Small edge: passed development AND the final exam. Dev AUC 0.520 (0.513-0.527), ranking IC +0.048 (t=5.7), top-5 edge 15.0%/yr. Final exam: AUC 0.512, IC +0.031, top-5 edge 3.2%/yr.
- **Up or down, 1 month**: No edge. Dev AUC 0.505 (0.470-0.539), ranking IC +0.013 (t=0.9), top-5 edge -6.2%/yr. Final exam: AUC 0.524, IC -0.016, top-5 edge 5.1%/yr.
- **Beats its peers, 1 month**: Looked good in development, FAILED the final exam. Dev AUC 0.524 (0.512-0.537), ranking IC +0.063 (t=3.9), top-5 edge 0.1%/yr. Final exam: AUC 0.493, IC -0.006, top-5 edge 44.6%/yr.

## Backtest

![backtest](docs/backtest.png)

|  | CAGR | Volatility | Sharpe | Max drawdown | Total return | % weeks invested |
|---|---|---|---|---|---|---|
| Model rule (after costs) | 47.0% | 31.9% | 1.37 | -33.1% | 5608.5% | 85.5% |
| Hold all stocks (equal weight) | 42.0% | 30.3% | 1.31 | -40.4% | 3869.2% |  |
| Buy & hold SPY | 15.4% | 16.5% | 0.95 | -31.7% | 349.5% |  |
| Buy & hold SMH | 33.7% | 30.2% | 1.12 | -42.8% | 2011.1% |  |

Full report with charts: see `docs/report.html` / your GitHub Pages link.
