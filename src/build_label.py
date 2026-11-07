# -*- coding: utf-8 -*-
"""
G3 标签：未来 5 日收益 + 逐日截面 Z-score。

label(t) = close[t+5] / close[t+1] - 1   （t+1 收盘 → t+5 收盘，对齐论文，非 t 收盘）

Z-score：每日在股票截面内标准化（只用自己的截面，不跨时间泄漏）。

输入：data/processed/base_YYYY.parquet（close）
输出：
  data/processed/label_raw.parquet  —— 原始 5 日收益（date × code）
  data/processed/label_z.parquet    —— 截面 Z-score（date × code）
"""

import pandas as pd


def load_close(years=None):
    if years is None:
        years = range(2008, 2023)
    frames = [pd.read_parquet("data/processed/base_%d.parquet" % y)["close"] for y in years]
    return pd.concat(frames, axis=0).astype("float32")


def main():
    close = load_close()
    label = close.shift(-5) / close.shift(-1) - 1   # t+5 / t+1 - 1
    mu = label.mean(axis=1)
    sd = label.std(axis=1)
    z = label.sub(mu, axis=0).div(sd, axis=0)
    label.to_parquet("data/processed/label_raw.parquet")
    z.to_parquet("data/processed/label_z.parquet")
    print("标签完成: raw %s, z %s" % (label.shape, z.shape))
    print("label 非NaN占比: %.3f" % label.notna().mean().mean())
    print("z 截面均值≈%.4f 标准差≈%.4f（应为 0 和 1）" % (z.mean().mean(), z.std().mean()))


if __name__ == "__main__":
    main()
