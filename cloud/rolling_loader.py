# -*- coding: utf-8 -*-
"""G8 精简滚动训练的数据层。

三个 expanding-window fold：
  test=2020: train=2008-2018, valid=2019, test=2020
  test=2021: train=2008-2019, valid=2020, test=2021
  test=2022: train=2008-2020, valid=2021, test=2022

训练段和验证段末尾各删除5个交易日，使5日标签不跨入下一段。每个fold的
RobustZScore scaler只使用该fold训练期拟合，并保存到独立缓存目录。
"""

import ast
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cloud.loader import (
    LOOKBACK,
    N_A158,
    N_F1,
    _assemble_stock,
    standardize_3d,
)


TRAIN_START = "2008-01-01"
TEST_YEARS = (2020, 2021, 2022)
EMBARGO = 5
CACHE_VERSION = "rolling_v1"
CACHE_ROOT = "data/cache_rolling"
SCALER_SUBSAMPLE = 5000
SCALER_EPS = 1e-12


def fold_spec(test_year, dates=None):
    if test_year not in TEST_YEARS:
        raise ValueError("test_year必须是 %s" % (TEST_YEARS,))
    if dates is None:
        dates = pd.read_parquet("data/processed/market_status.parquet").index
    dates = pd.DatetimeIndex(dates)

    valid_year = test_year - 1
    train = dates[(dates >= TRAIN_START) & (dates < "%d-01-01" % valid_year)]
    valid = dates[(dates >= "%d-01-01" % valid_year)
                  & (dates <= "%d-12-31" % valid_year)]
    test = dates[(dates >= "%d-01-01" % test_year)
                 & (dates <= "%d-12-31" % test_year)]
    if min(len(train), len(valid), len(test)) <= EMBARGO:
        raise ValueError("fold日期不足：test_year=%d" % test_year)

    train = train[:-EMBARGO]
    valid = valid[:-EMBARGO]
    return {
        "test_year": test_year,
        "train_start": train[0],
        "train_end": train[-1],
        "valid_start": valid[0],
        "valid_end": valid[-1],
        "test_start": test[0],
        "test_end": test[-1],
    }


def _cache_dir(test_year):
    return os.path.join(CACHE_ROOT, "fold_%d" % test_year)


def _read_wide_for_scaler(file):
    table = pq.read_table(file)
    names = table.column_names
    feature_pos = [i for i, name in enumerate(names) if name.startswith("('")]
    index_pos = next(i for i, name in enumerate(names) if not name.startswith("('") )
    dates = pd.to_datetime(table.column(index_pos).to_pandas())
    fields = [ast.literal_eval(names[i])[0] for i in feature_pos]
    # 不能逐列调用 to_numpy：一个分片有约7900列，Python循环会成为主要瓶颈。
    values = table.select(feature_pos).to_pandas(
        split_blocks=False, self_destruct=True, use_threads=True
    ).to_numpy(copy=False)
    return dates, fields, values


def _read_wide_fast(file, n_fields, with_codes=False):
    """批量读取宽表；返回(n_dates,n_fields,n_codes)，可同时返回股票顺序。"""
    table = pq.read_table(file)
    names = table.column_names
    feature_pos = [i for i, name in enumerate(names) if name.startswith("('")]
    codes = None
    if with_codes:
        codes = list(dict.fromkeys(
            ast.literal_eval(names[i])[1] for i in feature_pos))
    values = table.select(feature_pos).to_pandas(
        split_blocks=False, self_destruct=True, use_threads=True
    ).to_numpy(copy=False)
    values = values.reshape(len(values), n_fields, -1)
    return (values, codes) if with_codes else values


def _fit_wide_scaler(files, group, train_start, train_end):
    parts = []
    fields_order = None
    for number, file in enumerate(files, 1):
        print("[rolling scaler] %s %d/%d" %
              (group, number, len(files)), flush=True)
        dates, fields, values = _read_wide_for_scaler(file)
        unique_fields = list(dict.fromkeys(fields))
        if fields_order is None:
            fields_order = unique_fields
        n_fields = len(unique_fields)
        n_codes = values.shape[1] // n_fields
        mask = (dates >= train_start) & (dates <= train_end)
        values = values[mask]
        values = values.reshape(len(values), n_fields, n_codes)
        values = values.transpose(0, 2, 1).reshape(-1, n_fields)
        if len(values) > SCALER_SUBSAMPLE:
            index = np.random.default_rng(0).choice(
                len(values), SCALER_SUBSAMPLE, replace=False)
            values = values[index]
        parts.append(values.astype(np.float64))
    all_values = np.concatenate(parts, axis=0)
    center = np.nanmedian(all_values, axis=0)
    scale = np.nanmedian(np.abs(all_values - center), axis=0)
    scale = (scale + SCALER_EPS) * 1.4826
    return pd.DataFrame({
        "group": group,
        "feature": fields_order,
        "center": center,
        "scale": scale,
    })


def _fit_flat_scaler(file, group, train_start, train_end):
    frame = pd.read_parquet(file).loc[train_start:train_end]
    values = frame.values.astype(np.float64)
    center = np.nanmedian(values, axis=0)
    scale = np.nanmedian(np.abs(values - center), axis=0)
    scale = (scale + SCALER_EPS) * 1.4826
    return pd.DataFrame({
        "group": group,
        "feature": list(frame.columns),
        "center": center,
        "scale": scale,
    })


def _fit_fold_scaler(spec, cache_dir):
    path = os.path.join(cache_dir, "scaler.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)

    train_start = str(spec["train_start"].date())
    train_end = str(spec["train_end"].date())
    feat_files = sorted(glob.glob("data/processed/features/feat_*.parquet"))
    alpha158 = _fit_wide_scaler(
        feat_files, "alpha158", train_start, train_end)
    f1 = _fit_wide_scaler(
        ["data/processed/f1_factors.parquet"], "f1", train_start, train_end)
    market = _fit_flat_scaler(
        "data/processed/market_status.parquet", "market", train_start, train_end)
    out = pd.concat([alpha158, f1, market], ignore_index=True)
    out.to_parquet(path, index=False)
    return out


def _scaler_dict(scaler):
    return {
        group: (part["center"].to_numpy(), part["scale"].to_numpy())
        for group, part in scaler.groupby("group", sort=False)
    }


def _segment_array(dates, spec):
    dates = pd.DatetimeIndex(dates)
    out = np.full(len(dates), "unused", dtype="<U8")
    ranges = {
        "train": (spec["train_start"], spec["train_end"]),
        "valid": (spec["valid_start"], spec["valid_end"]),
        "test": (spec["test_start"], spec["test_end"]),
    }
    for name, (start, end) in ranges.items():
        out[(dates >= start) & (dates <= end)] = name
    return out


def _save_cache(ds, spec, cache_dir):
    for seg in ("train", "valid", "test"):
        for name, idx in (("X_stock", 0), ("X_market", 1), ("y", 2),
                          ("date", 3), ("code", 4)):
            np.save(os.path.join(cache_dir, "%s_%s.npy" % (seg, name)), ds[seg][idx])
    meta = {"version": CACHE_VERSION, "embargo": EMBARGO}
    meta.update({k: str(v.date()) if hasattr(v, "date") else int(v)
                 for k, v in spec.items()})
    with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_cache(test_year):
    cache_dir = _cache_dir(test_year)
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("version") != CACHE_VERSION or meta.get("test_year") != test_year:
        return None

    ds = {}
    names = ("X_stock", "X_market", "y", "date", "code")
    for seg in ("train", "valid", "test"):
        paths = [os.path.join(cache_dir, "%s_%s.npy" % (seg, n)) for n in names]
        if not all(os.path.exists(p) for p in paths):
            return None
        ds[seg] = tuple(np.load(p, mmap_mode="r") for p in paths)
    return ds


def _build_fold(test_year):
    cache_dir = _cache_dir(test_year)
    os.makedirs(cache_dir, exist_ok=True)

    market = pd.read_parquet("data/processed/market_status.parquet")
    spec = fold_spec(test_year, market.index)
    scaler = _scaler_dict(_fit_fold_scaler(spec, cache_dir))
    mkt_z = (market.values - scaler["market"][0]) / scaler["market"][1]
    mkt_z = np.nan_to_num(np.clip(mkt_z, -3.0, 3.0), nan=0.0)

    label = pd.read_parquet("data/processed/label_z.parquet")
    elig = pd.read_parquet("data/processed/eligible_core_csi300.parquet")
    split = _segment_array(market.index, spec)
    f1 = standardize_3d(
        _read_wide_fast("data/processed/f1_factors.parquet", N_F1),
        *scaler["f1"])
    files = sorted(glob.glob("data/processed/features/feat_*.parquet"))

    buckets = {
        seg: {key: [] for key in ("s", "m", "y", "d", "c")}
        for seg in ("train", "valid", "test")
    }
    for fi, file in enumerate(files):
        print("[rolling cache] fold=%d shard=%d/%d" %
              (test_year, fi + 1, len(files)), flush=True)
        raw_a158, codes = _read_wide_fast(file, N_A158, with_codes=True)
        a158 = standardize_3d(raw_a158, *scaler["alpha158"])
        del raw_a158
        n_codes = a158.shape[2]
        start = fi * 50
        feat = np.concatenate([a158, f1[:, :, start:start + n_codes]], axis=1)
        del a158

        for c in range(n_codes):
            code_index = start + c
            seq = _assemble_stock(feat[:, :, c])
            y = label.iloc[LOOKBACK - 1:, code_index].to_numpy().astype(np.float32)
            eligible = elig.iloc[LOOKBACK - 1:, code_index].to_numpy()
            seg_dates = market.index[LOOKBACK - 1:]
            seg_split = split[LOOKBACK - 1:]
            ok = ~np.isnan(y) & eligible

            for seg in ("train", "valid", "test"):
                mask = (seg_split == seg) & ok
                if not mask.any():
                    continue
                b = buckets[seg]
                b["s"].append(seq[mask].astype(np.float32))
                b["m"].append(mkt_z[LOOKBACK - 1:][mask].astype(np.float32))
                b["y"].append(y[mask])
                b["d"].append(seg_dates[mask])
                b["c"].append(np.array([codes[c]] * mask.sum(), dtype="<U16"))

    ds = {}
    for seg, b in buckets.items():
        if not b["s"]:
            raise ValueError("fold %d 的 %s 为空" % (test_year, seg))
        ds[seg] = tuple(np.concatenate(b[key]) for key in ("s", "m", "y", "d", "c"))
    _save_cache(ds, spec, cache_dir)
    return ds


def build_rolling_dataset(test_year, use_cache=True):
    if use_cache:
        ds = _load_cache(test_year)
        if ds is not None:
            print("[rolling_loader] 读取缓存 %s" % _cache_dir(test_year))
            return ds
    print("[rolling_loader] 构建 fold %d（含train-only scaler）..." % test_year)
    _build_fold(test_year)
    return _load_cache(test_year)


if __name__ == "__main__":
    for year in TEST_YEARS:
        print(year, fold_spec(year))
