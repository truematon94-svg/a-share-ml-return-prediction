# -*- coding: utf-8 -*-
"""
G2 第 2 步：long 格式股价 CSV -> date × (field, code) 宽面板，存 parquet。

输入：data/raw/stock_daily_YYYY.csv（long: date, code, field, value）
输出：data/processed/base_YYYY.parquet（index=date, columns=MultiIndex(field, code)）

只保留 6 字段 open/high/low/close/volume/money；vwap 在特征构造时由 money/volume 派生。
factor/high_limit/low_limit/paused 暂不加载（后续 eligible 过滤单独处理）。

内存：按年处理 + float32/category dtype 压缩，单年峰值远小于 1G。
运行：python src/build_base_panel.py [年份...]   （不带参数=跑全部 2008-2022）
"""

import os
import sys
import time

import pandas as pd

RAW = "data/raw"
OUT = "data/processed"
NEED = ["open", "high", "low", "close", "volume", "money", "factor"]


def build_year(year):
    f = os.path.join(RAW, "stock_daily_%d.csv" % year)
    out = os.path.join(OUT, "base_%d.parquet" % year)
    if os.path.exists(out):
        print("  %d 已存在，跳过" % year)
        return
    t0 = time.time()
    df = pd.read_csv(
        f,
        usecols=["date", "code", "field", "value"],
        dtype={"date": "category", "code": "category", "field": "category", "value": "float32"},
    )
    df = df[df["field"].isin(NEED)]
    wide = df.pivot_table(index="date", columns=["field", "code"], values="value", observed=True)
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index().sort_index(axis=1)
    # 列索引的 category 级别转成普通字符串，避免 parquet 读回时报错
    wide.columns = pd.MultiIndex.from_arrays(
        [wide.columns.get_level_values(0).astype(str),
         wide.columns.get_level_values(1).astype(str)],
        names=["field", "code"],
    )
    os.makedirs(OUT, exist_ok=True)
    wide.to_parquet(out)
    print("  %d 完成: %d 天 × %d 列, %.1fs, %.1f MB" %
          (year, wide.shape[0], wide.shape[1], time.time() - t0,
           os.path.getsize(out) / 2**20))


def main(years=None):
    if years is None:
        years = range(2008, 2023)
    for y in years:
        build_year(y)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(args if args else None)
