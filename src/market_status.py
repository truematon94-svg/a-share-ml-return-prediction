# -*- coding: utf-8 -*-
"""
市场状态 63 维（MASTER 论文 Market Status）。

63 = 3 指数(CSI300/CSI500/CSI800) × 21 维。
21 = 当前指数价格(1) + 5 窗口 d ∈ [5,10,20,30,60] × (价格均值、价格标准差、成交量均值、成交量标准差)。

口径：成交量 = volume 字段（论文原文 "market index trading volume"，非成交额 money）。

输入：data/raw/index_daily.csv（date, index_code, close, volume, money）
输出：data/processed/market_status.parquet（index=date, columns=63 维）
"""

import pandas as pd

INDICES = {"000300.XSHG": "csi300", "000905.XSHG": "csi500", "000906.XSHG": "csi800"}
WINDOWS = [5, 10, 20, 30, 60]


def build_market_status():
    df = pd.read_csv("data/raw/index_daily.csv")
    frames = []
    for code, label in INDICES.items():
        sub = df[df["index_code"] == code].sort_values("date").set_index("date")
        sub.index = pd.to_datetime(sub.index)
        cols = {label + "_price": sub["close"]}
        for d in WINDOWS:
            r = lambda s: s.rolling(d, min_periods=1)
            cols["%s_pmean_%d" % (label, d)] = r(sub["close"]).mean()
            cols["%s_pstd_%d" % (label, d)] = r(sub["close"]).std()
            cols["%s_vmean_%d" % (label, d)] = r(sub["volume"]).mean()
            cols["%s_vstd_%d" % (label, d)] = r(sub["volume"]).std()
        frames.append(pd.DataFrame(cols))
    out = pd.concat(frames, axis=1).astype("float32")
    out.to_parquet("data/processed/market_status.parquet")
    print("市场状态完成: %d 天 × %d 维" % out.shape)
    return out


if __name__ == "__main__":
    build_market_status()
