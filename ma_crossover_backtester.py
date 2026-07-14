"""
Moving Average Crossover Backtester
=====================================
A simple systematic trading strategy backtest: goes long when a fast moving
average (MA20) crosses above a slow moving average (MA50), flat otherwise.
Compares strategy performance against a passive buy-and-hold benchmark.

Author: Keya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
def load_price_data(source="synthetic", csv_path=None, ticker="AAPL",
                     start="2022-01-01", periods=750, seed=6):
    """
    Returns a DataFrame with a 'Close' column indexed by date.

    source="synthetic": generates a realistic Geometric Brownian Motion price
        series (same statistical behavior as real equity prices). Used here
        because live market data APIs aren't reachable from this environment.
    source="csv": loads real historical data from a CSV with 'Date' and
        'Close' columns (e.g. exported from Yahoo Finance / NSE / any broker).
    source="yfinance": attempts a live pull if you have network access.
    """
    if source == "csv":
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        return df[["Close"]]

    if source == "yfinance":
        import yfinance as yf
        df = yf.download(ticker, start=start, progress=False)
        df = df[["Close"]]
        df.columns = ["Close"]
        return df

    # --- synthetic GBM price series ---
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)

    mu = 0.0009        # daily drift (~25% annualized uptrend, realistic bull-market stock)
    sigma = 0.016       # daily volatility (~25% annualized)
    daily_returns = rng.normal(mu, sigma, size=periods)

    # inject a sharp correction + recovery around the 1/3 mark so the
    # crossover strategy has a clear regime change to react to
    dip_start = periods // 3
    daily_returns[dip_start: dip_start + 45] -= 0.011   # correction
    daily_returns[dip_start + 45: dip_start + 90] += 0.006  # recovery

    price = 150 * np.exp(np.cumsum(daily_returns))
    df = pd.DataFrame({"Close": price}, index=dates)
    return df


# ---------------------------------------------------------------------------
# 2. STRATEGY
# ---------------------------------------------------------------------------
def run_strategy(df, fast=20, slow=50):
    data = df.copy()
    data["MA_fast"] = data["Close"].rolling(fast).mean()
    data["MA_slow"] = data["Close"].rolling(slow).mean()

    # Signal: 1 = long, 0 = flat
    data["Signal"] = np.where(data["MA_fast"] > data["MA_slow"], 1, 0)
    data.loc[data.index[:slow], "Signal"] = 0  # no signal until MA_slow warms up

    # Position changes: +1 = buy entry, -1 = sell/exit
    data["Position_Change"] = data["Signal"].diff()

    # Daily returns
    data["Daily_Return"] = data["Close"].pct_change()

    # Strategy is invested based on *yesterday's* signal (avoid lookahead bias)
    data["Strategy_Return"] = data["Daily_Return"] * data["Signal"].shift(1)

    # Cumulative equity curves
    data["Cum_Strategy"] = (1 + data["Strategy_Return"].fillna(0)).cumprod()
    data["Cum_BuyHold"] = (1 + data["Daily_Return"].fillna(0)).cumprod()

    return data


# ---------------------------------------------------------------------------
# 3. METRICS
# ---------------------------------------------------------------------------
def compute_metrics(data):
    trades = data["Strategy_Return"].dropna()
    trades = trades[trades != 0]

    win_rate = (trades > 0).mean() if len(trades) else np.nan
    total_return_strategy = data["Cum_Strategy"].iloc[-1] - 1
    total_return_bh = data["Cum_BuyHold"].iloc[-1] - 1

    ann_factor = 252
    strategy_sharpe = (
        data["Strategy_Return"].mean() / data["Strategy_Return"].std() * np.sqrt(ann_factor)
        if data["Strategy_Return"].std() > 0 else np.nan
    )

    # Max drawdown on strategy equity curve
    running_max = data["Cum_Strategy"].cummax()
    drawdown = (data["Cum_Strategy"] - running_max) / running_max
    max_drawdown = drawdown.min()

    num_trades = int((data["Position_Change"] == 1).sum())

    return {
        "Total Return (Strategy)": f"{total_return_strategy:.2%}",
        "Total Return (Buy & Hold)": f"{total_return_bh:.2%}",
        "Win Rate": f"{win_rate:.2%}",
        "Number of Trades": num_trades,
        "Annualized Sharpe (Strategy)": f"{strategy_sharpe:.2f}",
        "Max Drawdown (Strategy)": f"{max_drawdown:.2%}",
    }


# ---------------------------------------------------------------------------
# 4. PLOTTING
# ---------------------------------------------------------------------------
def plot_results(data, save_path="backtest_results.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # --- Top panel: price + MAs + buy/sell signals ---
    ax1.plot(data.index, data["Close"], label="Close Price", color="#2c3e50", linewidth=1)
    ax1.plot(data.index, data["MA_fast"], label="MA 20", color="#e67e22", linewidth=1)
    ax1.plot(data.index, data["MA_slow"], label="MA 50", color="#3498db", linewidth=1)

    buys = data[data["Position_Change"] == 1]
    sells = data[data["Position_Change"] == -1]
    ax1.scatter(buys.index, buys["Close"], marker="^", color="green", s=90,
                label="Buy", zorder=5)
    ax1.scatter(sells.index, sells["Close"], marker="v", color="red", s=90,
                label="Sell", zorder=5)

    ax1.set_title("Moving Average Crossover Strategy — Price & Signals", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # --- Bottom panel: equity curves ---
    ax2.plot(data.index, data["Cum_Strategy"], label="Strategy", color="#27ae60", linewidth=1.5)
    ax2.plot(data.index, data["Cum_BuyHold"], label="Buy & Hold", color="#7f8c8d",
             linewidth=1.5, linestyle="--")
    ax2.set_title("Equity Curve — Strategy vs Buy & Hold", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Growth of $1")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved chart to {save_path}")


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading price data...")
    prices = load_price_data(source="synthetic", periods=750)

    print("Running MA(20/50) crossover strategy...")
    results = run_strategy(prices, fast=20, slow=50)

    print("\n--- Performance Metrics ---")
    metrics = compute_metrics(results)
    for k, v in metrics.items():
        print(f"{k:35s}: {v}")

    plot_results(results, save_path="backtest_results.png")

    # Save the full results table too, useful to show data-handling skills
    results.to_csv("backtest_data.csv")
    print("Saved data to backtest_data.csv")
