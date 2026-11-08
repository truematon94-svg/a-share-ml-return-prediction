# -*- coding: utf-8 -*-
"""G9 锁箱结果汇总、配对检验，并导出聚宽 Top30/5日调仓持仓表。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
G9 = ROOT / "data" / "g9_lock_box"
PROCESSED = G9 / "processed"
RESULTS = G9 / "results"
YEARS = (2023, 2024, 2025)
MODELS = ("random_forest", "master")
TOP_K = 30
HOLD_DAYS = 5


def prediction_path(model, year):
    return RESULTS / ("pred_%s_fold%d_s0.parquet" % (model, year))


def load_predictions(model):
    parts = []
    for year in YEARS:
        path = prediction_path(model, year)
        if not path.exists():
            raise FileNotFoundError("G9预测缺失: %s" % path)
        part = pd.read_parquet(path)
        part["date"] = pd.to_datetime(part["date"])
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(["date", "code"])


def daily_metrics(frame):
    rows = []
    for date, group in frame.groupby("date", sort=True):
        group = group.dropna(subset=["prediction", "y_true"])
        if len(group) < 5:
            continue
        pred = group["prediction"].to_numpy()
        true = group["y_true"].to_numpy()
        rows.append({
            "date": date,
            "ic": np.corrcoef(pred, true)[0, 1],
            "rankic": spearmanr(pred, true).correlation,
        })
    return pd.DataFrame(rows).set_index("date")


def newey_west(values, lag=5):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    mean = x.mean()
    e = x - mean
    lrv = np.dot(e, e) / n
    for step in range(1, min(lag, n - 1) + 1):
        gamma = np.dot(e[step:], e[:-step]) / n
        lrv += 2 * (1 - step / (lag + 1)) * gamma
    se = np.sqrt(max(lrv, 0.0) / n)
    return n, mean, se, mean / se if se > 0 else np.nan


def block_ci(values, block=20, draws=5000, seed=20261108):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    block = min(block, n)
    starts = np.arange(n - block + 1)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        sample = []
        while len(sample) < n:
            start = int(rng.choice(starts))
            sample.extend(range(start, start + block))
        means[i] = x[np.asarray(sample[:n])].mean()
    return np.quantile(means, [0.025, 0.975])


def export_holdings(model, frame):
    market = pd.read_parquet(PROCESSED / "market_status.parquet")
    market_dates = pd.DatetimeIndex(pd.to_datetime(market.index)).sort_values().unique()
    signal_dates = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())
    signal_dates = signal_dates[::HOLD_DAYS]
    holdings = {}
    rows = []
    for signal_date in signal_dates:
        candidates = frame.loc[frame["date"] == signal_date].nlargest(TOP_K, "prediction")
        position = market_dates.searchsorted(signal_date, side="right")
        if position >= len(market_dates):
            continue
        entry_date = market_dates[position].strftime("%Y-%m-%d")
        codes = candidates["code"].astype(str).tolist()
        holdings[entry_date] = codes
        rows.extend({"signal_date": signal_date.date(), "entry_date": entry_date,
                     "model": model, "rank": rank, "code": code}
                    for rank, code in enumerate(codes, 1))
    with (RESULTS / ("holdings_%s.json" % model)).open("w", encoding="utf-8") as handle:
        json.dump(holdings, handle, ensure_ascii=False, indent=2)
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    daily = {}
    summary_rows = []
    holding_parts = []
    for model in MODELS:
        pred = load_predictions(model)
        metric = daily_metrics(pred)
        daily[model] = metric
        for year in YEARS:
            part = metric.loc[metric.index.year == year]
            summary_rows.append({
                "model": model, "test_year": year, "n_days": len(part),
                "mean_ic": part["ic"].mean(), "icir": part["ic"].mean() / part["ic"].std(),
                "mean_rankic": part["rankic"].mean(),
                "rankicir": part["rankic"].mean() / part["rankic"].std(),
            })
        holding_parts.append(export_holdings(model, pred))

    pd.DataFrame(summary_rows).to_csv(RESULTS / "lockbox_summary.csv", index=False)
    pd.concat(holding_parts, ignore_index=True).to_csv(
        RESULTS / "lockbox_holdings.csv", index=False)

    paired = daily["master"].join(
        daily["random_forest"], lsuffix="_master", rsuffix="_rf", how="inner")
    paired["rankic_diff"] = paired["rankic_master"] - paired["rankic_rf"]
    paired["ic_diff"] = paired["ic_master"] - paired["ic_rf"]
    paired.reset_index().to_csv(RESULTS / "lockbox_daily_metrics.csv", index=False)
    n, mean, se, t_value = newey_west(paired["rankic_diff"])
    low, high = block_ci(paired["rankic_diff"])
    test = pd.DataFrame([{
        "comparison": "master_minus_random_forest", "metric": "daily_rankic",
        "n_days": n, "mean_diff": mean, "newey_west_se": se,
        "newey_west_t": t_value, "block_ci_low": low, "block_ci_high": high,
        "master_win_rate": (paired["rankic_diff"] > 0).mean(),
    }])
    test.to_csv(RESULTS / "lockbox_pairwise_test.csv", index=False)

    report = """# G9 Lockbox Result\n\n- Period: 2023-2025\n- Models: Random Forest and MASTER, seed=0\n- Portfolio export: Top 30, rebalance every 5 signal days, next-trading-day entry\n- Frozen rules: see configs/g9_frozen.yaml and docs/g9_freeze.md\n\nThe lockbox was opened once. Model selection and rules must not be changed using these results.\n"""
    (RESULTS / "lockbox_report.md").write_text(report, encoding="utf-8")
    print(RESULTS / "lockbox_summary.csv")
    print(RESULTS / "lockbox_pairwise_test.csv")
    print(RESULTS / "holdings_master.json")


if __name__ == "__main__":
    main()
