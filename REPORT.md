# Model test report

_Generated 2026-11-22 21:07 UTC. Data through 2026-11-20._

## Verdict

**No reliable edge. The model is not distinguishable from a coin flip / simple baselines once you account for noise.**

- Out-of-sample accuracy 54.0% vs 55.9% for simply guessing 'up' every time (difference -2.0%, 90% range -3.0% to -0.9%).
- ROC AUC 0.508 (0.5 = no skill; 90% range 0.492-0.524); best simple baseline AUC 0.505.
- The 10-90% range caught the actual move 79% of the time after calibration (target 80%).
- Backtest after costs: 49.0%/yr (Sharpe 1.41) vs 40.4%/yr (Sharpe 1.27) for just holding all the stocks. The rule beat that on risk-adjusted terms.

## Direction (out-of-sample)

| model | accuracy | auc | brier |
|---|---|---|---|
| Baseline: always up | 55.9% | 0.500 | 0.2469 |
| Baseline: last week repeats | 50.0% | 0.493 | 0.4999 |
| Baseline: historical average | 55.9% | 0.505 | 0.2468 |
| Logistic regression | 54.0% | 0.508 | 0.2495 |
| LightGBM | 54.4% | 0.497 | 0.2509 |

## Ranges (target coverage 80%)

| model | coverage | median_width | pinball |
|---|---|---|---|
| Baseline (volatility only) | 79.6% | 13.1% | 0.01543 |
| LightGBM (raw) | 76.1% | 11.5% | 0.01525 |
| LightGBM (calibrated) | 79.1% | 12.4% |  |

## Tomorrow's range (target coverage 80%)

| model | coverage | median_width |
|---|---|---|
| Baseline (volatility only) | 79.7% | 5.6% |
| LightGBM (raw) | 78.1% | 5.1% |
| LightGBM (calibrated) | 79.2% | 5.3% |

## Experiments: easier questions

Developed on data before 2025-01-01; graded once on the final exam (2025-01-01 onward).

- **Up or down, 1 week (the original question)**: No edge. Dev AUC 0.507 (0.489-0.520), ranking IC +0.017 (t=1.9), top-5 edge -4.2%/yr. Final exam: AUC 0.509, IC -0.003, top-5 edge -18.5%/yr.
- **Beats its peers, 1 week**: Small edge: passed development AND the final exam. Dev AUC 0.519 (0.512-0.526), ranking IC +0.047 (t=5.5), top-5 edge 16.1%/yr. Final exam: AUC 0.508, IC +0.020, top-5 edge 7.0%/yr.
- **Up or down, 1 month**: No edge. Dev AUC 0.504 (0.467-0.535), ranking IC +0.010 (t=0.7), top-5 edge -4.3%/yr. Final exam: AUC 0.517, IC -0.022, top-5 edge 4.5%/yr.
- **Beats its peers, 1 month**: Looked good in development, FAILED the final exam. Dev AUC 0.523 (0.512-0.534), ranking IC +0.062 (t=3.9), top-5 edge 1.3%/yr. Final exam: AUC 0.488, IC -0.017, top-5 edge 64.4%/yr.

## Backtest

![backtest](docs/backtest.png)

|  | CAGR | Volatility | Sharpe | Max drawdown | Total return | % weeks invested |
|---|---|---|---|---|---|---|
| Model rule (after costs) | 49.0% | 31.9% | 1.41 | -33.0% | 6535.6% | 86.7% |
| Hold all stocks (equal weight) | 40.4% | 30.4% | 1.27 | -40.4% | 3462.8% |  |
| Buy & hold SPY | 14.9% | 16.5% | 0.93 | -31.7% | 331.9% |  |
| Buy & hold SMH | 32.4% | 30.3% | 1.08 | -42.8% | 1813.7% |  |

Full report with charts: see `docs/report.html` / your GitHub Pages link.
