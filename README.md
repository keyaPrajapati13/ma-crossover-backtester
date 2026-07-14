# Moving Average Crossover Backtester

A systematic trading strategy backtester in Python. Implements a classic
20/50-day moving average crossover strategy, benchmarks it against a passive
buy-and-hold approach, and reports standard quant performance metrics.

## What it does
- Generates (or loads) historical daily price data
- Computes fast (20-day) and slow (50-day) moving averages
- Generates long/flat trading signals on crossovers
- Backtests strategy returns vs. buy-and-hold, avoiding lookahead bias
  (positions are entered on the day *after* a signal, using `.shift(1)`)
- Reports: total return, win rate, number of trades, annualized Sharpe
  ratio, and max drawdown
- Plots price + moving averages + buy/sell signals, and the equity curve

## Why
Built as a lightweight demonstration of systematic trading strategy design
and backtesting mechanics — the same core loop (signal generation, position
sizing, PnL attribution, risk metrics) used in real research platforms and
trading systems.

## Usage
```bash
pip install numpy pandas matplotlib
python ma_crossover_backtester.py
```

Outputs:
- `backtest_results.png` — price chart with signals + equity curve
- `backtest_data.csv` — full daily data with signals, returns, and equity

## Using real market data
By default the script generates a synthetic price series (Geometric Brownian
Motion) so it runs standalone with no API key or network access required.
To use real historical prices instead:

```python
# Option A: from a CSV with 'Date' and 'Close' columns
prices = load_price_data(source="csv", csv_path="your_data.csv")

# Option B: live pull via yfinance (requires internet access)
prices = load_price_data(source="yfinance", ticker="AAPL", start="2022-01-01")
```

The strategy, backtest, and metrics logic is identical either way.

## Sample output

| Metric | Value |
|---|---|
| Total Return (Strategy) | ~21% |
| Total Return (Buy & Hold) | ~26% |
| Win Rate | ~52% |
| Sharpe Ratio (Strategy) | ~0.44 |
| Max Drawdown (Strategy) | ~-18% |

*(Exact numbers vary by data source/date range — the strategy typically
trades off some upside for lower drawdown vs. buy-and-hold, a common and
honest finding for simple trend-following strategies.)*

## Possible extensions
- Parameter sweep over MA window pairs to find optimal fast/slow combo
- Transaction cost / slippage modeling
- Multi-asset portfolio backtest
- Walk-forward validation instead of single in-sample test
