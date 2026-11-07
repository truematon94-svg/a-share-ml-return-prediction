# -*- coding: utf-8 -*-
"""
F1 项目二因子（华泰之四动量/反转因子）——日频滚动适配。

因子（HAlpha 跳过）：
  return_1m/3m/6m/12m        —— 简单 N 月收益
  wgt_return_1m/3m/6m/12m    —— 换手率加权日均收益（weight = volume/factor）
  exp_wgt_return_1m/3m/6m/12m —— 指数衰减换手率加权（τ = N*21/4 交易日）

日频适配：N 月 → N*21 交易日。
换手率等价权重：volume / factor（复权因子消除送股跳增，忠实项目二代码）。

输入：data/processed/base_YYYY.parquet（close/volume/factor）
输出：data/processed/f1_factors.parquet（index=date, columns=MultiIndex(factor, code)）
"""

import numpy as np
import pandas as pd

MONTHS = [1, 3, 6, 12]
TRADING = 21  # 每月交易日


def load_panel_fields(years=None):
    if years is None:
        years = range(2008, 2023)
    close_frames, vol_frames, fac_frames = [], [], []
    for y in years:
        p = pd.read_parquet("data/processed/base_%d.parquet" % y)
        close_frames.append(p["close"])
        vol_frames.append(p["volume"])
        fac_frames.append(p["factor"])
    return (pd.concat(close_frames, axis=0).astype("float32"),
            pd.concat(vol_frames, axis=0).astype("float32"),
            pd.concat(fac_frames, axis=0).astype("float32"))


def compute_f1(close, volume, factor):
    feats = {}
    r = close.pct_change()
    w = volume / factor.replace(0, np.nan)  # 换手率等价权重

    for m in MONTHS:
        W = m * TRADING
        # 简单收益
        feats["return_%dm" % m] = close / close.shift(W) - 1
        # 换手率加权（滚动 W 日）
        num = (w * r).rolling(W, min_periods=1).sum()
        den = w.rolling(W, min_periods=1).sum()
        feats["wgt_return_%dm" % m] = num / den
        # 指数衰减换手率加权（τ = W/4 交易日，等价项目二 exp(-days_back/(N*21/4))）
        tau = W / 4.0
        alpha = 1 - np.exp(-1.0 / tau)
        num_e = (w * r).ewm(alpha=alpha, adjust=False, min_periods=1).mean()
        den_e = w.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
        feats["exp_wgt_return_%dm" % m] = num_e / den_e

    out = pd.concat(feats, axis=1)
    out.columns.names = ["factor", "code"]
    return out


def main():
    close, volume, factor = load_panel_fields()
    out = compute_f1(close, volume, factor)
    out.to_parquet("data/processed/f1_factors.parquet")
    print("F1 因子完成: %d 天 × %d 列（%d 因子 × %d 只）"
          % (out.shape[0], out.shape[1], out.columns.get_level_values(0).nunique(),
             out.columns.get_level_values(1).nunique()))


if __name__ == "__main__":
    main()
