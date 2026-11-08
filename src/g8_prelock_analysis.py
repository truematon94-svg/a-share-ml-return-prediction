# -*- coding: utf-8 -*-
"""G8锁箱前分析：市场状态、组合稳健性、RankICIR与日度配对检验。

只读取已经生成的预测和行情，不重新训练，也不改动冻结训练代码。
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT / "cloud"
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

YEARS = (2020, 2021, 2022)
MAIN_MODELS = (
    "ols", "ridge", "random_forest", "lightgbm",
    "xgboost", "mlp", "transformer", "master",
)
BASE_COST = 0.00025 * 2 + 0.001 + 0.001 * 2
NW_LAG = 5


def newey_west_mean_test(values, lag=NW_LAG):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return {"n_days": n, "mean_diff": np.nan, "nw_se": np.nan,
                "nw_t": np.nan, "nw_p": np.nan}
    mean = values.mean()
    residual = values - mean
    long_run_var = np.dot(residual, residual) / n
    effective_lag = min(lag, n - 1)
    for step in range(1, effective_lag + 1):
        gamma = np.dot(residual[step:], residual[:-step]) / n
        long_run_var += 2 * (1 - step / (effective_lag + 1)) * gamma
    se = np.sqrt(max(long_run_var, 0.0) / n)
    t_value = mean / se if se > 0 else np.nan
    p_value = 2 * norm.sf(abs(t_value)) if np.isfinite(t_value) else np.nan
    return {"n_days": n, "mean_diff": mean, "nw_se": se,
            "nw_t": t_value, "nw_p": p_value}


_DAILY_CACHE = {}


def daily_metrics(path):
    path = Path(path)
    if path in _DAILY_CACHE:
        return _DAILY_CACHE[path].copy()
    frame = pd.read_parquet(path, columns=["date", "y_true", "prediction"])
    frame["date"] = pd.to_datetime(frame["date"])
    rows = []
    for date, group in frame.groupby("date", sort=True):
        true = group["y_true"].to_numpy(dtype=float)
        pred = group["prediction"].to_numpy(dtype=float)
        valid = np.isfinite(true) & np.isfinite(pred)
        if valid.sum() < 5:
            continue
        rows.append({
            "date": date,
            "ic": np.corrcoef(pred[valid], true[valid])[0, 1],
            "rankic": spearmanr(pred[valid], true[valid]).correlation,
        })
    out = pd.DataFrame(rows).set_index("date").sort_index()
    _DAILY_CACHE[path] = out
    return out.copy()


def main_prediction_paths():
    paths = {}
    for seed in (0, 1, 2):
        folder = CLOUD / ("results_seed%d" % seed)
        for model in MAIN_MODELS:
            path = folder / ("pred_%s_s%d.parquet" % (model, seed))
            if path.exists():
                paths[(model, seed)] = path
    return paths


def average_daily_rankic(paths):
    series = []
    for path in paths:
        seed = pd.read_parquet(path, columns=["seed"])["seed"].iloc[0]
        series.append(daily_metrics(path)["rankic"].rename("seed_%s" % seed))
    return pd.concat(series, axis=1, join="inner").mean(axis=1)


def rankic_statistics():
    paths = main_prediction_paths()
    rows = []
    by_model = {}
    for (model, seed), path in paths.items():
        daily = daily_metrics(path)
        rankic = daily["rankic"].dropna()
        ic = daily["ic"].dropna()
        rows.append({
            "model": model,
            "seed": seed,
            "n_days": len(rankic),
            "ic_mean": ic.mean(),
            "ic_std": ic.std(ddof=1),
            "icir_ann": ic.mean() / ic.std(ddof=1) * np.sqrt(252),
            "rankic_mean": rankic.mean(),
            "rankic_std": rankic.std(ddof=1),
            "rankicir_ann": rankic.mean() / rankic.std(ddof=1) * np.sqrt(252),
            "rankic_positive_rate": (rankic > 0).mean(),
        })
        by_model.setdefault(model, []).append(path)

    seed_table = pd.DataFrame(rows).sort_values(["model", "seed"])
    aggregate = (seed_table.groupby("model", as_index=False)
                 .agg(n_seeds=("seed", "nunique"),
                      n_days=("n_days", "min"),
                      rankic_mean=("rankic_mean", "mean"),
                      rankic_seed_std=("rankic_mean", "std"),
                      rankicir_ann=("rankicir_ann", "mean"),
                      positive_rate=("rankic_positive_rate", "mean"))
                 .sort_values("rankic_mean", ascending=False))

    averaged = {model: average_daily_rankic(model_paths)
                for model, model_paths in by_model.items()}
    pair_rows = []
    for model_a, model_b in combinations(MAIN_MODELS, 2):
        if model_a not in averaged or model_b not in averaged:
            continue
        paired = pd.concat([averaged[model_a].rename("a"),
                            averaged[model_b].rename("b")], axis=1).dropna()
        diff = paired["a"] - paired["b"]
        test = newey_west_mean_test(diff)
        pair_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            **test,
            "a_win_rate": (diff > 0).mean(),
        })
    pairwise = pd.DataFrame(pair_rows)
    return seed_table, aggregate, pairwise, averaged


def ablation_pairwise(full_master):
    comparisons = {
        "master_minus_no_gate": [CLOUD / "results_g8" /
                                  ("pred_master_no_gate_s%d.parquet" % s)
                                  for s in (0, 1, 2)],
        "master_minus_no_stock": [CLOUD / "results_g8" /
                                   ("pred_master_no_stock_s%d.parquet" % s)
                                   for s in (0, 1, 2)],
        "master_minus_f0_only": [CLOUD / "results_g8_features" /
                                  ("pred_master_f0_s%d.parquet" % s)
                                  for s in (0, 1, 2)],
        "master_minus_f1_only": [CLOUD / "results_g8_features" /
                                  ("pred_master_f1_s%d.parquet" % s)
                                  for s in (0, 1, 2)],
    }
    rows = []
    for name, paths in comparisons.items():
        alternative = average_daily_rankic(paths)
        paired = pd.concat([full_master.rename("full"),
                            alternative.rename("alternative")], axis=1).dropna()
        diff = paired["full"] - paired["alternative"]
        rows.append({
            "comparison": name,
            "full_rankic": paired["full"].mean(),
            "alternative_rankic": paired["alternative"].mean(),
            **newey_west_mean_test(diff),
            "full_win_rate": (diff > 0).mean(),
        })
    return pd.DataFrame(rows)


def rolling_daily(model):
    parts = []
    for year in YEARS:
        path = CLOUD / "results_g8_rolling" / (
            "pred_%s_fold%d_s0.parquet" % (model, year))
        part = daily_metrics(path)
        part["test_year"] = year
        parts.append(part)
    return pd.concat(parts).sort_index()


def market_regime_analysis():
    market = pd.read_parquet(PROCESSED / "market_status.parquet",
                             columns=["csi300_price"])
    market.index = pd.to_datetime(market.index)
    price = market["csi300_price"].astype(float)
    state = pd.DataFrame(index=market.index)
    state["trend20"] = price / price.shift(20) - 1
    state["vol20"] = price.pct_change().rolling(20, min_periods=20).std()

    fold_metrics = pd.read_csv(CLOUD / "results_g8_rolling" / "metrics.csv")
    master_folds = fold_metrics[fold_metrics["model"] == "master"].set_index("test_year")
    states = []
    for year in YEARS:
        train_end = pd.Timestamp(master_folds.loc[year, "train_end"])
        threshold = state.loc[:train_end, "vol20"].median()
        test = state[state.index.year == year].copy()
        test["vol_threshold_train"] = threshold
        test["trend_state"] = np.where(test["trend20"] >= 0, "up", "down")
        test["vol_state"] = np.where(test["vol20"] > threshold, "high_vol", "low_vol")
        test["market_state"] = test["trend_state"] + "_" + test["vol_state"]
        test["test_year"] = year
        states.append(test)
    states = pd.concat(states).dropna(subset=["trend20", "vol20"])

    model_rows = []
    joined = {}
    for model in ("random_forest", "master"):
        daily = rolling_daily(model).join(states[["market_state", "trend20", "vol20"]],
                                          how="inner")
        joined[model] = daily
        for regime, group in daily.groupby("market_state", sort=True):
            rankic = group["rankic"].dropna()
            model_rows.append({
                "model": model,
                "market_state": regime,
                "n_days": len(rankic),
                "ic_mean": group["ic"].mean(),
                "rankic_mean": rankic.mean(),
                "rankic_std": rankic.std(ddof=1),
                "rankicir_ann": rankic.mean() / rankic.std(ddof=1) * np.sqrt(252),
                "rankic_positive_rate": (rankic > 0).mean(),
            })

    paired = (joined["master"][["rankic", "market_state"]]
              .rename(columns={"rankic": "master_rankic"})
              .join(joined["random_forest"][["rankic"]]
                    .rename(columns={"rankic": "rf_rankic"}), how="inner"))
    pair_rows = []
    for regime, group in paired.groupby("market_state", sort=True):
        diff = group["master_rankic"] - group["rf_rankic"]
        pair_rows.append({
            "market_state": regime,
            "master_rankic": group["master_rankic"].mean(),
            "rf_rankic": group["rf_rankic"].mean(),
            **newey_west_mean_test(diff),
            "master_win_rate": (diff > 0).mean(),
        })
    return pd.DataFrame(model_rows), pd.DataFrame(pair_rows)


def load_price_fields():
    fields = {"open": [], "close": [], "paused": []}
    for year in YEARS:
        panel = pd.read_parquet(PROCESSED / ("base_%d.parquet" % year))
        for field in fields:
            fields[field].append(panel[field])
    return {field: pd.concat(parts).sort_index()
            for field, parts in fields.items()}


def load_benchmark():
    raw = pd.read_csv(RAW / "index_daily.csv")
    hs300 = raw[raw["index_code"] == "000300.XSHG"].copy()
    hs300["date"] = pd.to_datetime(hs300["date"])
    return hs300.sort_values("date").set_index("date")["close"].astype(float)


def portfolio_periods(pred, price_fields, benchmark, topk, holding_days,
                      cost_multiplier):
    pred = pred[["date", "code", "prediction"]].copy()
    pred["date"] = pd.to_datetime(pred["date"])
    ranked = {date: list(group.nlargest(topk, "prediction")["code"])
              for date, group in pred.groupby("date", sort=True)}
    dates = sorted(ranked)[::holding_days]
    open_ = price_fields["open"]
    close = price_fields["close"]
    paused = price_fields["paused"]
    bench_ret = benchmark.shift(-holding_days) / benchmark.shift(-1) - 1
    rows = []
    previous = set()
    for date in dates:
        codes = [code for code in ranked[date] if code in open_.columns]
        if len(codes) != topk or date not in open_.index:
            continue
        entry = open_.shift(-1).loc[date, codes].to_numpy(dtype=float)
        exit_ = close.shift(-holding_days).loc[date, codes].to_numpy(dtype=float)
        entry_paused = paused.shift(-1).loc[date, codes].to_numpy(dtype=float)
        returns = np.where(entry_paused == 1, 0.0, exit_ / entry - 1)
        if not np.all(np.isfinite(returns)) or date not in bench_ret.index:
            continue
        gross = returns.mean()
        current = set(codes)
        turnover = 1 - len(current & previous) / topk
        cost = turnover * BASE_COST * cost_multiplier
        rows.append({"date": date, "gross": gross, "net": gross - cost,
                     "cost": cost, "turnover": turnover,
                     "bench": bench_ret.loc[date]})
        previous = current
    return pd.DataFrame(rows).dropna()


def summarize_portfolio(periods, holding_days):
    n = len(periods)
    periods_per_year = 252 / holding_days
    net = periods["net"].to_numpy()
    gross = periods["gross"].to_numpy()
    bench = periods["bench"].to_numpy()
    ann_net = (np.prod(1 + net) ** (periods_per_year / n) - 1) if n else np.nan
    ann_gross = (np.prod(1 + gross) ** (periods_per_year / n) - 1) if n else np.nan
    ann_bench = (np.prod(1 + bench) ** (periods_per_year / n) - 1) if n else np.nan
    ann_vol = np.std(net, ddof=1) * np.sqrt(periods_per_year)
    excess_period = net - bench
    ir = (excess_period.mean() / excess_period.std(ddof=1) * np.sqrt(periods_per_year)
          if excess_period.std(ddof=1) > 0 else np.nan)
    wealth = np.cumprod(1 + net)
    max_drawdown = np.min(wealth / np.maximum.accumulate(wealth) - 1)
    return {
        "n_periods": n,
        "ann_gross_return": ann_gross,
        "ann_net_return": ann_net,
        "ann_benchmark_return": ann_bench,
        "ann_excess_return": ann_net - ann_bench,
        "ann_volatility": ann_vol,
        "sharpe_zero_rf": ann_net / ann_vol if ann_vol > 0 else np.nan,
        "information_ratio": ir,
        "max_drawdown": max_drawdown,
        "avg_turnover": periods["turnover"].mean(),
        "total_cost": periods["cost"].sum(),
    }


def portfolio_robustness():
    price_fields = load_price_fields()
    benchmark = load_benchmark()
    scenarios = [
        ("topk_10", 10, 5, 1.0),
        ("main_topk_30", 30, 5, 1.0),
        ("topk_50", 50, 5, 1.0),
        ("cost_0x", 30, 5, 0.0),
        ("cost_2x", 30, 5, 2.0),
        ("holding_10d", 30, 10, 1.0),
    ]
    rows = []
    for model in MAIN_MODELS:
        path = CLOUD / "results_seed0" / ("pred_%s_s0.parquet" % model)
        if not path.exists():
            continue
        pred = pd.read_parquet(path, columns=["date", "code", "prediction"])
        for scenario, topk, holding, cost_multiple in scenarios:
            periods = portfolio_periods(pred, price_fields, benchmark, topk,
                                        holding, cost_multiple)
            rows.append({
                "model": model,
                "scenario": scenario,
                "topk": topk,
                "holding_rebalance_days": holding,
                "cost_multiple": cost_multiple,
                "roundtrip_cost_at_full_turnover": BASE_COST * cost_multiple,
                **summarize_portfolio(periods, holding),
            })
    return pd.DataFrame(rows)


def write_summary(aggregate, pairwise, ablation, regime_pair, portfolio):
    def pair(a, b):
        row = pairwise[(pairwise["model_a"] == a) &
                       (pairwise["model_b"] == b)]
        if row.empty:
            row = pairwise[(pairwise["model_a"] == b) &
                           (pairwise["model_b"] == a)].copy()
            if not row.empty:
                row.loc[:, "mean_diff"] *= -1
                row.loc[:, "nw_t"] *= -1
        return row.iloc[0]

    hypotheses = [("ridge", "ols"), ("random_forest", "ridge"),
                  ("transformer", "mlp"), ("master", "transformer"),
                  ("master", "random_forest")]
    lines = [
        "# G8锁箱前分析摘要",
        "",
        "本文件由 `python src/g8_prelock_analysis.py` 从既有预测复算；没有重新训练。",
        "RankICIR均为 `日均RankIC / 日度标准差 × sqrt(252)`。配对检验为日度RankIC差的Newey–West检验（lag=5）。",
        "",
        "## 主模型",
        "",
        "| 模型 | seeds | RankIC | 年化RankICIR |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append("| %s | %d | %.4f | %.3f |" %
                     (row.model, row.n_seeds, row.rankic_mean, row.rankicir_ann))
    lines.extend(["", "## 预先约定的主比较", "",
                  "| 比较（A-B） | 日均差 | NW t | p值 |",
                  "|---|---:|---:|---:|"])
    for a, b in hypotheses:
        row = pair(a, b)
        lines.append("| %s - %s | %.4f | %.2f | %.4f |" %
                     (a, b, row.mean_diff, row.nw_t, row.nw_p))
    lines.extend([
        "",
        "> OLS注意：当前 `pred_ols_s0.parquet` 由病态法方程直接 `solve` 生成，",
        "> 复算RankIC为0.0272；稳定 `lstsq` 单次运行记录为0.0399。",
        "> 因此Ridge−OLS配对暂作数值诊断，不用于G9模型选择。",
    ])
    lines.extend(["", "## MASTER消融", "",
                  "| 比较 | 日均差 | NW t | p值 |",
                  "|---|---:|---:|---:|"])
    for row in ablation.itertuples(index=False):
        lines.append("| %s | %.4f | %.2f | %.4f |" %
                     (row.comparison, row.mean_diff, row.nw_t, row.nw_p))
    lines.extend(["", "## 市场状态（滚动MASTER-RF）", "",
                  "| 状态 | 天数 | RankIC差 | NW t |",
                  "|---|---:|---:|---:|"])
    for row in regime_pair.itertuples(index=False):
        lines.append("| %s | %d | %.4f | %.2f |" %
                     (row.market_state, row.n_days, row.mean_diff, row.nw_t))
    lines.extend([
        "",
        "## 组合稳健性口径",
        "",
        "- 主规则仍为Top-30、5日调仓/持有、基础往返成本0.35%；稳健性结果不用于重新挑选主规则。",
        "- Top-K仅检查10/30/50；成本仅检查0/1/2倍；备选持有期固定为10日。",
        "- 10日结果是同一5日预测信号的执行持有期敏感性，不是重新训练的10日标签模型。",
        "- 本地回测为现有可执行代理；最终锁箱另用聚宽核验逐笔手续费、涨跌停和延迟成交。",
        "",
        "完整数字见 `g8_portfolio_robustness.csv`。",
    ])
    (RESULTS / "g8_prelock_summary.md").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")


def main():
    RESULTS.mkdir(exist_ok=True)
    seed_stats, aggregate, pairwise, averaged = rankic_statistics()
    ablation = ablation_pairwise(averaged["master"])
    regime, regime_pair = market_regime_analysis()
    portfolio = portfolio_robustness()

    outputs = {
        "g8_main_rankic_by_seed.csv": seed_stats,
        "g8_main_rankic_summary.csv": aggregate,
        "g8_main_pairwise_nw.csv": pairwise,
        "g8_ablation_pairwise_nw.csv": ablation,
        "g8_market_regime.csv": regime,
        "g8_market_regime_pairwise.csv": regime_pair,
        "g8_portfolio_robustness.csv": portfolio,
    }
    for filename, frame in outputs.items():
        frame.to_csv(RESULTS / filename, index=False, float_format="%.8f")
    write_summary(aggregate, pairwise, ablation, regime_pair, portfolio)

    print("=== 主模型RankIC汇总 ===")
    print(aggregate.round(4).to_string(index=False))
    print("\n=== MASTER消融配对检验 ===")
    print(ablation.round(4).to_string(index=False))
    print("\n=== 市场状态MASTER-RF ===")
    print(regime_pair.round(4).to_string(index=False))
    print("\n已写入 results/g8_* 共8个文件。")


if __name__ == "__main__":
    main()
