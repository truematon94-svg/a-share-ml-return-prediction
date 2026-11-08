# -*- coding: utf-8 -*-
"""G9年度扩展窗口数据层；复用G8实现但读取独立锁箱目录。"""

import glob
import os

import numpy as np
import pandas as pd

from cloud import rolling_loader as base


TEST_YEARS = (2023, 2024, 2025)
PROCESSED = "data/g9_lock_box/processed"
FEATURES = PROCESSED + "/features"

# 复用G8已经验证的缓存和日期实现，只替换独立目录与年度。
base.TEST_YEARS = TEST_YEARS
base.CACHE_ROOT = "data/g9_lock_box/cache_rolling"
base.CACHE_VERSION = "g9_lockbox_v1"


def fold_spec(test_year, dates=None):
    """G9独立fold：不调用只支持2020-2022的G8日期检查。"""
    if test_year not in TEST_YEARS:
        raise ValueError("test_year必须是 %s" % (TEST_YEARS,))
    if dates is None:
        dates = pd.read_parquet(PROCESSED + "/market_status.parquet").index
    dates = pd.DatetimeIndex(dates)
    valid_year = test_year - 1
    train = dates[(dates >= base.TRAIN_START)
                  & (dates < "%d-01-01" % valid_year)]
    valid = dates[(dates >= "%d-01-01" % valid_year)
                  & (dates <= "%d-12-31" % valid_year)]
    test = dates[(dates >= "%d-01-01" % test_year)
                 & (dates <= "%d-12-31" % test_year)]
    if min(len(train), len(valid), len(test)) <= base.EMBARGO:
        raise ValueError("G9 fold日期不足：test_year=%d" % test_year)
    train = train[:-base.EMBARGO]
    valid = valid[:-base.EMBARGO]
    return {
        "test_year": test_year,
        "train_start": train[0],
        "train_end": train[-1],
        "valid_start": valid[0],
        "valid_end": valid[-1],
        "test_start": test[0],
        "test_end": test[-1],
    }


def _fit_fold_scaler(spec, cache_dir):
    path = os.path.join(cache_dir, "scaler.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    start = str(spec["train_start"].date())
    end = str(spec["train_end"].date())
    alpha = base._fit_wide_scaler(
        sorted(glob.glob(FEATURES + "/feat_*.parquet")),
        "alpha158", start, end)
    f1 = base._fit_wide_scaler(
        [PROCESSED + "/f1_factors.parquet"], "f1", start, end)
    market = base._fit_flat_scaler(
        PROCESSED + "/market_status.parquet", "market", start, end)
    result = pd.concat([alpha, f1, market], ignore_index=True)
    result.to_parquet(path, index=False)
    return result


def _build_fold(test_year):
    cache_dir = base._cache_dir(test_year)
    os.makedirs(cache_dir, exist_ok=True)
    market = pd.read_parquet(PROCESSED + "/market_status.parquet")
    spec = fold_spec(test_year, market.index)
    scaler = base._scaler_dict(_fit_fold_scaler(spec, cache_dir))
    market_z = np.nan_to_num(
        np.clip((market.values - scaler["market"][0]) / scaler["market"][1], -3, 3),
        nan=0.0,
    )
    label = pd.read_parquet(PROCESSED + "/label_z.parquet")
    eligible = pd.read_parquet(PROCESSED + "/eligible_core_csi300.parquet")
    split = base._segment_array(market.index, spec)
    f1 = base.standardize_3d(
        base._read_wide_fast(PROCESSED + "/f1_factors.parquet", base.N_F1),
        *scaler["f1"],
    )
    files = sorted(glob.glob(FEATURES + "/feat_*.parquet"))
    buckets = {segment: {key: [] for key in ("s", "m", "y", "d", "c")}
               for segment in ("train", "valid", "test")}
    code_offset = 0
    for number, file in enumerate(files, 1):
        print("[G9 cache] fold=%d shard=%d/%d" %
              (test_year, number, len(files)), flush=True)
        raw, codes = base._read_wide_fast(file, base.N_A158, with_codes=True)
        alpha = base.standardize_3d(raw, *scaler["alpha158"])
        count = alpha.shape[2]
        stock = np.concatenate([alpha, f1[:, :, code_offset:code_offset + count]], axis=1)
        del raw, alpha
        for local_index, code in enumerate(codes):
            global_index = code_offset + local_index
            sequence = base._assemble_stock(stock[:, :, local_index])
            y = label.iloc[base.LOOKBACK - 1:, global_index].to_numpy().astype(np.float32)
            ok = (eligible.iloc[base.LOOKBACK - 1:, global_index].to_numpy()
                  & ~np.isnan(y))
            segment_dates = market.index[base.LOOKBACK - 1:]
            segment_split = split[base.LOOKBACK - 1:]
            for segment in ("train", "valid", "test"):
                mask = (segment_split == segment) & ok
                if not mask.any():
                    continue
                bucket = buckets[segment]
                bucket["s"].append(sequence[mask].astype(np.float32))
                bucket["m"].append(market_z[base.LOOKBACK - 1:][mask].astype(np.float32))
                bucket["y"].append(y[mask])
                bucket["d"].append(segment_dates[mask])
                bucket["c"].append(np.array([code] * mask.sum(), dtype="<U16"))
        code_offset += count
    dataset = {}
    for segment, bucket in buckets.items():
        if not bucket["s"]:
            raise ValueError("G9 fold %d 的%s为空" % (test_year, segment))
        dataset[segment] = tuple(np.concatenate(bucket[key])
                                 for key in ("s", "m", "y", "d", "c"))
    base._save_cache(dataset, spec, cache_dir)
    return dataset


base._fit_fold_scaler = _fit_fold_scaler
base._build_fold = _build_fold


def build_rolling_dataset(test_year, use_cache=True):
    return base.build_rolling_dataset(test_year, use_cache)
