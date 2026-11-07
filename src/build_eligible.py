# -*- coding: utf-8 -*-
"""
G3 eligible 过滤（股票池）。

核心过滤（两套实验共用）：
  1. 当日属于历史成分（universe.csv 时点成分，防幸存者偏差）
  2. 上市满 60 交易日（累计非NaN close >= 60，兼顾特征回看历史）
  3. 标签存在（label_raw 非 NaN）

论文对齐额外过滤（可交易性，Qlib eligible 口径）：
  4. 非 ST（status_st 快照 forward-fill）
  5. 非停牌（paused == 0）
  6. 非涨跌停（close 不在 high_limit/low_limit 上）
  7. 成交额达标（money >= 阈值）

独立扩展口径：只用核心过滤，可交易性留到回测层（research_protocol 4.2）。

输出：
  data/processed/eligible_core_{universe}.parquet   —— 核心 mask（date × code bool）
  data/processed/eligible_paper_{universe}.parquet  —— 论文对齐 mask
"""

import numpy as np
import pandas as pd

TRADING_DAYS_MIN = 60
MONEY_THRESHOLD = 1e6   # 成交额下限 100 万元（日成交额，可调）


def load_field(field, years=range(2008, 2023)):
    frames = [pd.read_parquet("data/processed/base_%d.parquet" % y)[field] for y in years]
    return pd.concat(frames, axis=0).astype("float32")


def universe_mask(close, universe):
    univ = pd.read_csv("data/raw/universe.csv", dtype=str)
    sub = univ[univ["universe"] == universe]
    month_codes = sub.groupby(sub["date"].str[:7])["code"].apply(set).to_dict()
    months = close.index.strftime("%Y-%m")
    mask = pd.DataFrame(
        [[c in month_codes.get(m, set()) for c in close.columns] for m in months],
        index=close.index, columns=close.columns, dtype=bool,
    )
    return mask


def not_st_mask(close):
    st = pd.read_csv("data/raw/status_st.csv", dtype=str)
    st["date"] = pd.to_datetime(st["date"])
    piv = st.pivot_table(index="date", columns="code", values="is_st", aggfunc="last")
    st_daily = piv.reindex(close.index, method="ffill")
    st_daily = st_daily.reindex(columns=close.columns)
    # is_st 为 '1' 或 'True' 视为 ST
    is_st = st_daily.isin(["1", "True", "true"]).fillna(False)
    return ~is_st


def main():
    close = load_field("close")
    label = pd.read_parquet("data/processed/label_raw.parquet")

    # 核心过滤
    listing = close.notna().cumsum() >= TRADING_DAYS_MIN
    label_ok = label.notna()

    # 可交易性字段
    paused = load_field("paused")
    high_limit = load_field("high_limit")
    low_limit = load_field("low_limit")
    money = load_field("money")

    # 论文对齐额外
    not_st = not_st_mask(close)
    not_paused = (paused == 0) | paused.isna()
    not_limit = ~((close >= high_limit - 0.01) | (close <= low_limit + 0.01))
    liquid = money >= MONEY_THRESHOLD

    for universe in ["csi300", "csi800"]:
        umask = universe_mask(close, universe)
        core = umask & listing & label_ok
        paper = core & not_st & not_paused & not_limit & liquid
        core.to_parquet("data/processed/eligible_core_%s.parquet" % universe)
        paper.to_parquet("data/processed/eligible_paper_%s.parquet" % universe)
        # 统计每日样本数
        print("%s: 核心日均 %d 只 / 论文对齐日均 %d 只" %
              (universe, int(core.sum(axis=1).mean()), int(paper.sum(axis=1).mean())))
        print("   核心总样本 %d, 论文对齐总样本 %d" %
              (int(core.sum().sum()), int(paper.sum().sum())))


if __name__ == "__main__":
    main()
