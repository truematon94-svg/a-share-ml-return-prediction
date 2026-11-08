# -*- coding: utf-8 -*-
"""
组合回测（论文对齐口径）：每 5 天调仓，选 Top-30，持有 5 天，扣成本，对比沪深300。

口径：
  - 信号日 t：按预测分数选 Top-30
  - 持有收益 = close[t+5]/close[t+1]-1（即 label_raw）
  - 成本：佣金 0.025% 双边 + 印花税 0.1% 卖出 + 滑点 0.1% 双边（按满仓换手 0.35% 估）
  - 基准：沪深300 同口径 5 日收益

产出：每个模型 年化收益/波动/Sharpe/最大回撤/超额/信息比率/换手率/成本占比。
"""

import numpy as np
import pandas as pd

TOP_K = 30
REBALANCE = 5          # 每 5 个交易日调仓一次
COST_ROUNDTRIP = 0.00025 * 2 + 0.001 + 0.001 * 2   # 佣金双边+印花税卖出+滑点双边 = 0.35%


def load_predictions(model, seed=0):
    df = pd.read_parquet("cloud/results_seed%d/pred_%s_s%d.parquet" % (seed, model, seed))
    return df


def load_label_raw():
    return pd.read_parquet("data/processed/label_raw.parquet")


def load_benchmark():
    idx = pd.read_csv("data/raw/index_daily.csv", dtype=str)
    hs300 = idx[idx["index_code"] == "000300.XSHG"].copy()
    hs300["date"] = pd.to_datetime(hs300["date"])
    hs300 = hs300.sort_values("date").set_index("date")
    hs300["close"] = hs300["close"].astype(float)
    return hs300["close"]


def load_base_field(field, years=(2020, 2021, 2022)):
    """从 base 面板读一个字段（open/close/paused/high_limit/low_limit），返回 date×code。"""
    frames = [pd.read_parquet("data/processed/base_%d.parquet" % y)[field] for y in years]
    return pd.concat(frames, axis=0)


def run_backtest_executable(model, seed=0):
    """可执行口径：t+1 开盘买入（停牌则跳过），t+5 收盘卖出。"""
    pred = load_predictions(model, seed)
    pred["date"] = pd.to_datetime(pred["date"])
    open_ = load_base_field("open")
    close = load_base_field("close")
    paused = load_base_field("paused")
    bench = load_benchmark()

    codes = sorted(pred["code"].unique())
    open_c = open_[codes]
    close_c = close[codes]
    paused_c = paused[codes]

    dates = sorted(pred["date"].unique())
    signal_dates = dates[::REBALANCE]
    bench_ret = bench.shift(-5) / bench.shift(-1) - 1

    rows = []
    prev_codes = set()
    for t in signal_dates:
        sub = pred[pred["date"] == t].sort_values("prediction", ascending=False)
        top = sub.head(TOP_K)
        top_codes = list(top["code"])
        try:
            entry = open_c.shift(-1).loc[t, top_codes]      # t+1 开盘
            exit_ = close_c.shift(-5).loc[t, top_codes]      # t+5 收盘
            susp = paused_c.shift(-1).loc[t, top_codes]      # t+1 是否停牌
        except KeyError:
            continue
        # 停牌（paused=1）无法买入 -> 该股收益按 0（持有现金）
        rets = np.where(susp == 1, 0.0, exit_ / entry - 1)
        gross = rets.mean()
        turnover = 1 - len(set(top_codes) & prev_codes) / TOP_K
        cost = turnover * COST_ROUNDTRIP
        net = gross - cost
        try:
            b = bench_ret.loc[t]
        except KeyError:
            b = np.nan
        rows.append({"date": t, "gross": gross, "cost": cost, "net": net, "bench": b,
                     "turnover": turnover})
        prev_codes = set(top_codes)
    res = pd.DataFrame(rows).dropna(subset=["net", "bench"])
    return res


def run_backtest(model, seed=0):
    pred = load_predictions(model, seed)
    label = load_label_raw()
    bench = load_benchmark()

    # 预测表：date, code, prediction
    pred["date"] = pd.to_datetime(pred["date"])
    dates = sorted(pred["date"].unique())

    # 信号日：每 5 天取一个
    signal_dates = dates[::REBALANCE]

    # 预计算每个信号日的持有收益（label_raw[date, code]）
    # label_raw 是 date × code 宽表，取预测的 code 列
    codes = sorted(pred["code"].unique())
    label_codes = label[codes]

    # 基准 5 日收益（close[t+5]/close[t+1]-1）
    bench_ret = bench.shift(-5) / bench.shift(-1) - 1

    rows = []
    prev_codes = set()
    for t in signal_dates:
        sub = pred[pred["date"] == t].sort_values("prediction", ascending=False)
        top = sub.head(TOP_K)
        top_codes = list(top["code"])
        # 持有收益
        try:
            rets = label_codes.loc[t, top_codes]
        except KeyError:
            continue
        gross = rets.mean()
        # 换手率（与上一期相比，换了多少股票）
        turnover = 1 - len(set(top_codes) & prev_codes) / TOP_K
        cost = turnover * COST_ROUNDTRIP
        net = gross - cost
        try:
            b = bench_ret.loc[t]
        except KeyError:
            b = np.nan
        rows.append({"date": t, "gross": gross, "cost": cost, "net": net, "bench": b,
                     "turnover": turnover})
        prev_codes = set(top_codes)

    res = pd.DataFrame(rows).dropna(subset=["net", "bench"])
    return res


def summarize(res):
    net = res["net"].values
    bench = res["bench"].values
    n = len(res)
    # 年化（用 5 日收益，一年约 50 个调仓期）
    periods_per_year = 252 / REBALANCE
    ann_ret = (1 + net).prod() ** (periods_per_year / n) - 1
    ann_vol = net.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    bench_ann = (1 + bench).prod() ** (periods_per_year / n) - 1
    excess = ann_ret - bench_ann
    # 超额波动 + IR
    ex = net - bench
    ir = ex.mean() / ex.std() * np.sqrt(periods_per_year) if ex.std() > 0 else np.nan
    # 最大回撤（累计净值）
    cum = np.cumprod(1 + net)
    dd = (cum / np.maximum.accumulate(cum) - 1).min()
    # 月度胜率
    res_month = res.copy()
    res_month["ym"] = res_month["date"].dt.strftime("%Y-%m")
    win = (res_month.groupby("ym")["net"].sum() > res_month.groupby("ym")["bench"].sum()).mean()
    cost_ratio = res["cost"].sum() / res["gross"].sum() if res["gross"].sum() != 0 else np.nan
    avg_turnover = res["turnover"].mean()

    return {
        "年化收益": ann_ret, "年化波动": ann_vol, "Sharpe": sharpe,
        "基准年化": bench_ann, "年化超额": excess, "信息比率": ir,
        "最大回撤": dd, "月度胜率": win, "平均换手率": avg_turnover,
        "成本/毛收益": cost_ratio, "调仓次数": n,
    }


def main():
    models = ["random_forest", "lightgbm", "xgboost", "master", "transformer", "mlp"]
    all_rows = []
    for m in models:
        res = run_backtest(m, seed=0)
        s = summarize(res)
        cum = np.cumprod(1 + res["net"])
        bench_cum = np.cumprod(1 + res["bench"])
        out = pd.DataFrame({"date": res["date"], "portfolio": cum, "benchmark": bench_cum})
        out.to_csv("results/backtest_%s_s0.csv" % m, index=False)
        row = {"model": m}
        row.update(s)
        all_rows.append(row)

    # 汇总表
    cols = ["model", "年化收益", "年化波动", "Sharpe", "基准年化", "年化超额",
            "信息比率", "最大回撤", "月度胜率", "平均换手率", "成本/毛收益"]
    summ = pd.DataFrame(all_rows)[cols]
    summ = summ.sort_values("年化超额", ascending=False)
    print("\n=== 论文对齐口径 6 模型回测汇总（seed=0）===")
    print(summ.to_string(index=False, float_format=lambda x: "%.4f" % x))
    summ.to_csv("results/backtest_summary_close_s0.csv", index=False)
    print("\n汇总已存 results/backtest_summary_close_s0.csv")


if __name__ == "__main__":
    main()
