# -*- coding: utf-8 -*-
"""G9锁箱数据本地预处理；所有产物写入data/g9_lock_box。"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from alpha158 import compute_features
from project2_factors import compute_f1


ROOT = Path(__file__).resolve().parents[1]
OLD_RAW = ROOT / "data" / "raw"
OLD_PROCESSED = ROOT / "data" / "processed"
G9 = ROOT / "data" / "g9_lock_box"
RAW = G9 / "raw"
PROCESSED = G9 / "processed"
FEATURES = PROCESSED / "features"
YEARS = range(2008, 2026)
G9_YEARS = (2022, 2023, 2024, 2025)
NEED = ["open", "high", "low", "close", "volume", "money", "factor",
        "high_limit", "low_limit", "paused"]
WINDOWS = (5, 10, 20, 30, 60)
INDICES = {"000300.XSHG": "csi300", "000905.XSHG": "csi500",
           "000906.XSHG": "csi800"}


def _require(path):
    if not Path(path).exists():
        raise FileNotFoundError(path)


def build_base_year(year):
    source = RAW / ("g9_stock_daily_%d.csv" % year)
    output = PROCESSED / ("base_%d.parquet" % year)
    _require(source)
    if output.exists():
        print("已存在，跳过：%s" % output)
        return
    frame = pd.read_csv(
        source,
        usecols=["date", "code", "field", "value"],
        dtype={"date": "category", "code": "category", "field": "category",
               "value": "float32"},
    )
    frame = frame[frame["field"].isin(NEED)]
    wide = frame.pivot_table(index="date", columns=["field", "code"],
                             values="value", observed=True)
    wide.index = pd.to_datetime(wide.index)
    wide.columns = pd.MultiIndex.from_arrays([
        wide.columns.get_level_values(0).astype(str),
        wide.columns.get_level_values(1).astype(str),
    ], names=["field", "code"])
    wide = wide.sort_index().sort_index(axis=1)
    if year == 2022:
        old = pd.read_parquet(OLD_PROCESSED / "base_2022.parquet")
        # G9下载覆盖同名值，同时保留旧历史股票列。
        wide = wide.combine_first(old).sort_index().sort_index(axis=1)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(output)
    print("base %d：%d天 × %d列" % (year, *wide.shape))


def base_path(year):
    return (OLD_PROCESSED / ("base_%d.parquet" % year)
            if year <= 2021 else PROCESSED / ("base_%d.parquet" % year))


def load_field(field):
    parts = []
    for year in YEARS:
        path = base_path(year)
        _require(path)
        parts.append(pd.read_parquet(path)[field])
    return pd.concat(parts).sort_index().astype("float32")


def build_market_status():
    output = PROCESSED / "market_status.parquet"
    if output.exists():
        print("已存在，跳过：%s" % output)
        return
    old = pd.read_csv(OLD_RAW / "index_daily.csv")
    new = pd.read_csv(RAW / "g9_index_daily_2022_2025.csv")
    raw = pd.concat([old, new], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = (raw.sort_values("date")
           .drop_duplicates(["date", "index_code"], keep="last"))
    frames = []
    for code, label in INDICES.items():
        sub = raw[raw["index_code"] == code].set_index("date").sort_index()
        columns = {label + "_price": sub["close"]}
        for window in WINDOWS:
            columns["%s_pmean_%d" % (label, window)] = sub["close"].rolling(window, min_periods=1).mean()
            columns["%s_pstd_%d" % (label, window)] = sub["close"].rolling(window, min_periods=1).std()
            columns["%s_vmean_%d" % (label, window)] = sub["volume"].rolling(window, min_periods=1).mean()
            columns["%s_vstd_%d" % (label, window)] = sub["volume"].rolling(window, min_periods=1).std()
        frames.append(pd.DataFrame(columns))
    result = pd.concat(frames, axis=1).astype("float32")
    result.to_parquet(output)
    print("market：%d天 × %d维，%s—%s" %
          (*result.shape, result.index.min().date(), result.index.max().date()))


def build_labels():
    raw_output = PROCESSED / "label_raw.parquet"
    z_output = PROCESSED / "label_z.parquet"
    if raw_output.exists() and z_output.exists():
        print("标签已存在，跳过")
        return
    close = load_field("close")
    label = close.shift(-5) / close.shift(-1) - 1
    z = label.sub(label.mean(axis=1), axis=0).div(label.std(axis=1), axis=0)
    label.to_parquet(raw_output)
    z.to_parquet(z_output)
    print("label：%d天 × %d只" % label.shape)


def build_universe_and_eligible():
    universe_output = PROCESSED / "universe.csv"
    eligible_output = PROCESSED / "eligible_core_csi300.parquet"
    old = pd.read_csv(OLD_RAW / "universe.csv", dtype=str)
    old = old[old["universe"] == "csi300"]
    new = pd.read_csv(RAW / "g9_universe_2023_2025.csv", dtype=str)
    universe = (pd.concat([old, new], ignore_index=True)
                .drop_duplicates(["date", "universe", "code"]))
    universe.to_csv(universe_output, index=False)
    if eligible_output.exists():
        print("eligible已存在，跳过")
        return
    close = load_field("close")
    label = pd.read_parquet(PROCESSED / "label_raw.parquet").reindex_like(close)
    month_codes = (universe.groupby(universe["date"].str[:7])["code"]
                   .apply(set).to_dict())
    membership = pd.DataFrame(
        [[code in month_codes.get(month, set()) for code in close.columns]
         for month in close.index.strftime("%Y-%m")],
        index=close.index, columns=close.columns, dtype=bool,
    )
    listed_60 = close.notna().cumsum() >= 60
    eligible = membership & listed_60 & label.notna()
    eligible.to_parquet(eligible_output)
    lockbox = eligible.loc["2023":"2025"]
    print("eligible：锁箱期日均%d只，共%d样本" %
          (int(lockbox.sum(axis=1).mean()), int(lockbox.sum().sum())))


def build_f1():
    output = PROCESSED / "f1_factors.parquet"
    if output.exists():
        print("F1已存在，跳过")
        return
    close = load_field("close")
    volume = load_field("volume")
    factor = load_field("factor")
    result = compute_f1(close, volume, factor).astype("float32")
    result.to_parquet(output)
    print("F1：%d天 × %d列" % result.shape)


def build_alpha158(batch_size=50):
    FEATURES.mkdir(parents=True, exist_ok=True)
    fields = {name: load_field(name) for name in
              ("open", "high", "low", "close", "volume", "money")}
    fields["vwap"] = fields["money"] / fields["volume"]
    codes = list(fields["close"].columns)
    for start in range(0, len(codes), batch_size):
        batch = codes[start:start + batch_size]
        output = FEATURES / ("feat_%04d_%04d.parquet" %
                             (start, start + len(batch) - 1))
        if output.exists():
            print("已存在，跳过：%s" % output.name)
            continue
        feat = compute_features(
            fields["open"][batch], fields["high"][batch], fields["low"][batch],
            fields["close"][batch], fields["volume"][batch], fields["vwap"][batch],
        ).astype("float32")
        feat.to_parquet(output)
        print("Alpha158 %d-%d / %d" %
              (start, start + len(batch) - 1, len(codes)), flush=True)


def audit():
    required = [PROCESSED / ("base_%d.parquet" % year) for year in G9_YEARS]
    required += [PROCESSED / name for name in
                 ("market_status.parquet", "label_raw.parquet", "label_z.parquet",
                  "universe.csv", "eligible_core_csi300.parquet", "f1_factors.parquet")]
    missing = [str(path) for path in required if not path.exists()]
    feature_files = sorted(FEATURES.glob("feat_*.parquet"))
    if missing or not feature_files:
        raise FileNotFoundError("缺少产物：%s；Alpha分片=%d" % (missing, len(feature_files)))
    market = pd.read_parquet(PROCESSED / "market_status.parquet")
    label = pd.read_parquet(PROCESSED / "label_z.parquet")
    eligible = pd.read_parquet(PROCESSED / "eligible_core_csi300.parquet")
    f1 = pd.read_parquet(PROCESSED / "f1_factors.parquet")
    common = market.index.intersection(label.index).intersection(eligible.index).intersection(f1.index)
    if common.max() < pd.Timestamp("2025-12-01"):
        raise ValueError("日期未覆盖至2025年末：%s" % common.max())
    print("G9预处理审计通过：%d天，%d个Alpha分片，截止%s" %
          (len(common), len(feature_files), common.max().date()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["base", "panels", "alpha158", "audit", "all"])
    args = parser.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if args.step in ("base", "all"):
        for year in G9_YEARS:
            build_base_year(year)
    if args.step in ("panels", "all"):
        build_market_status()
        build_labels()
        build_universe_and_eligible()
        build_f1()
    if args.step in ("alpha158", "all"):
        build_alpha158()
    if args.step in ("audit", "all"):
        audit()


if __name__ == "__main__":
    main()
