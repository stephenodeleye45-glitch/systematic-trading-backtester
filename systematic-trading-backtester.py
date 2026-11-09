import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

take =pd.read_csv(r"C:\Users\steph\Desktop\cft_ds\Virtual Environment\personal scripts\SP_500.csv")
take["Date"] = pd.to_datetime(take["Date"])
take = take.sort_values("Date")

def run_strategy(take, fast_w, slow_w, cost=0.001, slippage_rate=0.001):
    take = take.copy()

    # 1. moving averages
    take["ma_fast"] = take["Value"].rolling(window=fast_w).mean()
    take["ma_slow"] = take["Value"].rolling(window=slow_w).mean()

    # 2. signal
    take["signal"] = (take["ma_fast"] > take["ma_slow"]).astype(int)

    # 3. position (no lookahead)
    take["position"] = take["signal"].shift(1)

    # 4. returns
    take["returns"] = take["Value"].pct_change()
    take["strategy_returns"] = take["position"] * take["returns"]

    # 5. costs
    take["position_change"] = take["position"].diff().abs()
    take["strategy_returns_net"] = (
        take["strategy_returns"]
        - take["position_change"] * cost
        - take["position_change"] * slippage_rate
    )

    # 6. valid net returns
    net_returns = take["strategy_returns_net"].dropna()

    # 7. equity curve
    equity = (1 + take["strategy_returns_net"].fillna(0)).cumprod()

    # 8. drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    # 9. Sharpe
    sharpe = net_returns.mean() / net_returns.std()

    # 10. Sortino
    downside_returns = np.minimum(net_returns, 0)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(12)
    annual_return = net_returns.mean() * 12

    if downside_deviation > 0:
        sortino_ratio = annual_return / downside_deviation
    else:
        sortino_ratio = np.nan

    # 11. maximum drawdown and turnover
    max_dd = drawdown.min()
    turnover = take["position_change"].sum()

    # 12. Calmar
    years = len(net_returns) / 12

    if max_dd < 0 and years > 0:
        cagr = equity.iloc[-1] ** (1 / years) - 1
        calmar_ratio = cagr / abs(max_dd)
    else:
        calmar_ratio = np.nan


    return sharpe, max_dd, turnover, sortino_ratio, calmar_ratio
   

take["returns"] = take["Value"].pct_change()

take["volatility"] = (
    take["returns"]
    .rolling(12)
    .std()
)
median_vol = take["volatility"].median()

high_vol = take[
    take["volatility"] > median_vol
]

low_vol = take[
    take["volatility"] <= median_vol
]

high_sharpe, high_dd, high_turnover, high_sortino, high_calmar = run_strategy(
    high_vol,
    9,
    12
)

low_sharpe, low_dd, low_turnover, low_sortino, low_calmar = run_strategy(
    low_vol,
    9,
    12
)



def subperiod_analysis(take, fast_w, slow_w):

    periods = [
        ("1920-01-01", "1949-12-31"),
        ("1950-01-01", "1979-12-31"),
        ("1980-01-01", "1999-12-31"),
        ("2000-01-01", "2025-12-31")
    ]

    for start, end in periods:

        subset = take[(take["Date"] >= start) & (take["Date"] <= end)]

        sharpe, dd, turnover, sortino, calmar = run_strategy(
    subset,
    fast_w,
    slow_w
)

        print(
            f"{start[:4]}-{end[:4]}",
            "Sharpe:", round(sharpe, 2),
            "MaxDD:", round(dd, 2),
            "Turnover:", round(turnover, 2),
            "Sortino:", round(sortino, 2),
            "Calmar:", round(calmar, 2)
        )

slippage_rates = [0, 0.0005, 0.001, 0.002, 0.003]

slippage_results = []

for slippage in slippage_rates:

    sharpe, dd, turnover, sortino, calmar = run_strategy(
        take, 5, 15,
        cost=0.001,
        slippage_rate=slippage
    )

    slippage_results.append(
        (slippage, sharpe, dd, turnover, sortino, calmar)
    )
slippage_df = pd.DataFrame(
    slippage_results,
    columns=[
        "Slippage Rate",
        "Sharpe",
        "Max Drawdown",
        "Turnover",
        "Sortino",
        "Calmar"
    ]
)

print(slippage_df)


def walk_forward_analysis(take):

    walk_results = []

    # 10-year step windows (monthly data → stable sample sizes)
    for start_year in range(1930, 2010, 10):

        train_start = f"{start_year}-01-01"
        train_end = f"{start_year + 39}-12-31"

        test_start = f"{start_year + 40}-01-01"
        test_end = f"{start_year + 49}-12-31"

        train = take[(take["Date"] >= train_start) & (take["Date"] <= train_end)]
        test = take[(take["Date"] >= test_start) & (take["Date"] <= test_end)]

        best_sharpe = -999
        best_fast = None
        best_slow = None

        # find best parameters on TRAIN only
        for fast_w in fast_windows:
            for slow_w in slow_windows:

                if fast_w < slow_w:

                    sharpe, _, _, _, _ = run_strategy(train, fast_w, slow_w)

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_fast = fast_w
                        best_slow = slow_w

        # evaluate on TEST
        test_sharpe, test_dd, test_turnover, test_sortino, test_calmar = run_strategy(
            test,
            best_fast,
            best_slow
        )

        walk_results.append([
            start_year,
            best_fast,
            best_slow,
            test_sharpe,
            test_dd,
            test_turnover,
            test_sortino,
            test_calmar
        ])

    return pd.DataFrame(
        walk_results,
        columns=[
            "train_start",
            "best_fast",
            "best_slow",
            "test_sharpe",
            "test_dd",
            "test_turnover",
            "test_sortino",
            "test_calmar"
        ]
    )
windows = [6, 9, 12, 15, 18, 24]

results = []

fast_windows = [3, 5, 6, 9]
slow_windows = [12, 15, 18, 24]

results = []

for fast_w in fast_windows:
    for slow_w in slow_windows:
        if fast_w < slow_w:   # important constraint
            sharpe, dd, turnover, sortino, calmar = run_strategy(take, fast_w, slow_w)
            results.append((fast_w, slow_w, sharpe, dd, turnover, sortino, calmar))

results_take = pd.DataFrame(
    results,
    columns=["fast_ma", "slow_ma", "sharpe", "max_dd", "turnover", "sortino", "calmar"]
)
print(results_take)


print("Best parameters:")
print(results_take.loc[results_take["sharpe"].idxmax()])

print("\nWorst parameters:")
print(results_take.loc[results_take["sharpe"].idxmin()])

print("Sharpe mean:", results_take["sharpe"].mean())
print("Sharpe std:", results_take["sharpe"].std())
print("Sharpe min:", results_take["sharpe"].min())
print("Sharpe max:", results_take["sharpe"].max())

positive_rate = (results_take["sharpe"] > 0).mean()

print("Fraction of parameter sets with positive Sharpe:", positive_rate)



pivot = results_take.pivot_table(
    index="fast_ma",
    columns="slow_ma",
    values="sharpe"
)

print(pivot)

subperiod_analysis(take, 5, 15)

wf_results = walk_forward_analysis(take)

print(wf_results)

print("\nHigh Volatility Regime")
print("Sharpe:", round(high_sharpe, 3))
print("Sortino:", round(high_sortino, 3))
print("Calmar:", round(high_calmar, 3))
print("Max DD:", round(high_dd, 3))
print("Turnover:", high_turnover)

print("\nLow Volatility Regime")
print("Sharpe:", round(low_sharpe, 3))
print("Sortino:", round(low_sortino, 3))
print("Calmar:", round(low_calmar, 3))
print("Max DD:", round(low_dd, 3))
print("Turnover:", low_turnover)

best_result = results_take.loc[results_take["sharpe"].idxmax()]

best_fast = int(best_result["fast_ma"])
best_slow = int(best_result["slow_ma"])

print("Best fast window:", best_fast)
print("Best slow window:", best_slow)

graph_data = take.copy()

graph_data["ma_fast"] = graph_data["Value"].rolling(window=best_fast).mean()
graph_data["ma_slow"] = graph_data["Value"].rolling(window=best_slow).mean()

graph_data["signal"] = (
    graph_data["ma_fast"] > graph_data["ma_slow"]
).astype(int)

graph_data["position"] = graph_data["signal"].shift(1)

# Calculate strategy returns
graph_data["returns"] = graph_data["Value"].pct_change()

graph_data["strategy_returns"] = (
    graph_data["position"] * graph_data["returns"]
)

graph_data["position_change"] = (
    graph_data["position"].diff().abs()
)

graph_data["strategy_returns_net"] = (
    graph_data["strategy_returns"]
    - graph_data["position_change"] * 0.001
    - graph_data["position_change"] * 0.001
)

# Calculate buy and hold returns
graph_data["buy_hold_returns"] = graph_data["returns"]

# Calculate growth of £1 invested
graph_data["strategy_equity"] = (
    1 + graph_data["strategy_returns_net"].fillna(0)
).cumprod()

graph_data["buy_hold_equity"] = (
    1 + graph_data["buy_hold_returns"].fillna(0)
).cumprod()

# Plot comparison
plt.figure(figsize=(12, 6))

plt.plot(
    graph_data["Date"],
    graph_data["strategy_equity"],
    label="Moving Average Strategy"
)

plt.plot(
    graph_data["Date"],
    graph_data["buy_hold_equity"],
    label="Buy and Hold",
)

plt.title("Moving Average Strategy vs Buy and Hold")
plt.xlabel("Date")
plt.ylabel("Growth of £1 Invested")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()