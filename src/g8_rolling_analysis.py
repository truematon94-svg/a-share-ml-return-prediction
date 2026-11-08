# -*- coding: utf-8 -*-
"""复算G8滚动训练汇总、同日期固定/滚动对比与配对稳健统计。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
ROLLING_DIR = ROOT / "cloud" / "results_g8_rolling"
FIXED_DIR = ROOT / "cloud" / "results_seed0"
OUT_DIR = ROOT / "results"
YEARS = (2020, 2021, 2022)


def daily_metrics(path):
    frame = pd.read_parquet(path, columns=["date", "y_true", "prediction"])
    frame["date"] = pd.to_datetime(frame["date"])
    rows = []
    for date, group in frame.groupby("date", sort=True):
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


def rolling_daily(model):
    parts = []
    for year in YEARS:
        path = ROLLING_DIR / ("pred_%s_fold%d_s0.parquet" % (model, year))
        part = daily_metrics(path)
        part["test_year"] = year
        parts.append(part)
    return pd.concat(parts).sort_index()


def newey_west_mean_test(values, lag=5):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = values.mean()
    residual = values - mean
    long_run_var = np.dot(residual, residual) / n
    for step in range(1, lag + 1):
        gamma = np.dot(residual[step:], residual[:-step]) / n
        long_run_var += 2 * (1 - step / (lag + 1)) * gamma
    se = np.sqrt(max(long_run_var, 0.0) / n)
    return {"n": n, "mean": mean, "se": se, "t": mean / se}


def moving_block_ci(values, block=20, draws=10000, seed=20261108):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    starts = np.arange(n - block + 1)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for draw in range(draws):
        sampled = []
        while len(sampled) < n:
            start = int(rng.choice(starts))
            sampled.extend(range(start, start + block))
        means[draw] = values[np.asarray(sampled[:n])].mean()
    return np.quantile(means, [0.025, 0.975])


def build_fold_summary():
    metrics = pd.read_csv(ROLLING_DIR / "metrics.csv")
    counts = {
        (model, year): len(daily_metrics(
            ROLLING_DIR / ("pred_%s_fold%d_s0.parquet" % (model, year))))
        for model in ("master", "random_forest") for year in YEARS
    }
    rows = []
    for row in metrics.itertuples(index=False):
        rows.append({
            "model": row.model,
            "test_year": str(row.test_year),
            "n_days": counts[(row.model, row.test_year)],
            "test_ic": row.test_ic,
            "test_rankic": row.test_rankic,
            "best_epoch": row.best_epoch,
            "train_seconds": row.train_seconds,
        })
    for model, group in metrics.groupby("model", sort=False):
        for stat, func in (("fold_mean", "mean"), ("fold_std", "std")):
            values = group[["test_ic", "test_rankic", "train_seconds"]].agg(func)
            rows.append({
                "model": model,
                "test_year": stat,
                "n_days": sum(counts[(model, year)] for year in YEARS)
                if stat == "fold_mean" else np.nan,
                "test_ic": values["test_ic"],
                "test_rankic": values["test_rankic"],
                "best_epoch": np.nan,
                "train_seconds": values["train_seconds"],
            })
    return pd.DataFrame(rows)


def build_fixed_comparison():
    rows = []
    for model in ("master", "random_forest"):
        fixed = daily_metrics(FIXED_DIR / ("pred_%s_s0.parquet" % model)).add_prefix("fixed_")
        rolling = rolling_daily(model).drop(columns="test_year").add_prefix("rolling_")
        paired = fixed.join(rolling, how="inner")
        rankic_diff = paired["rolling_rankic"] - paired["fixed_rankic"]
        test = newey_west_mean_test(rankic_diff)
        rows.append({
            "model": model,
            "common_start": paired.index.min().date(),
            "common_end": paired.index.max().date(),
            "n_days": len(paired),
            "fixed_ic": paired["fixed_ic"].mean(),
            "rolling_ic": paired["rolling_ic"].mean(),
            "delta_ic": (paired["rolling_ic"] - paired["fixed_ic"]).mean(),
            "fixed_rankic": paired["fixed_rankic"].mean(),
            "rolling_rankic": paired["rolling_rankic"].mean(),
            "delta_rankic": rankic_diff.mean(),
            "nw_lag": 5,
            "nw_t_rankic": test["t"],
            "daily_rankic_win_rate": (rankic_diff > 0).mean(),
        })
    return pd.DataFrame(rows)


def pairwise_master_rf():
    master = rolling_daily("master").add_prefix("master_")
    forest = rolling_daily("random_forest").add_prefix("rf_")
    paired = master.join(forest, how="inner")
    diff = paired["master_rankic"] - paired["rf_rankic"]
    test = newey_west_mean_test(diff)
    ci = moving_block_ci(diff)
    return {
        **test,
        "win_rate": (diff > 0).mean(),
        "block20_ci_low": ci[0],
        "block20_ci_high": ci[1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="写入results下两张汇总表")
    args = parser.parse_args()

    fold = build_fold_summary()
    comparison = build_fixed_comparison()
    pairwise = pairwise_master_rf()
    print("=== fold summary ===")
    print(fold.round(6).to_string(index=False))
    print("\n=== fixed vs rolling, common dates ===")
    print(comparison.round(6).to_string(index=False))
    print("\n=== rolling MASTER - RF ===")
    print(pd.Series(pairwise).round(6).to_string())

    if args.write:
        OUT_DIR.mkdir(exist_ok=True)
        fold.round(6).to_csv(OUT_DIR / "g8_rolling_summary.csv", index=False)
        comparison.round(6).to_csv(
            OUT_DIR / "g8_rolling_fixed_comparison.csv", index=False)


if __name__ == "__main__":
    main()
