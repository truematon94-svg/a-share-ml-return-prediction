# -*- coding: utf-8 -*-
"""
数据加载 + 缓存（memory map）。

第一次从 36 个 Parquet 构造全量序列，缓存成 .npy（mmap 友好），后续直接读缓存。
缓存文件：data/cache/{X_stock,X_market,y,date,code}.npy + meta.txt（版本信息）
口径：标准化(clip[-3,3], NaN->0)、8 天回看、csi300 核心股票池、论文对齐切分。
"""

import ast
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

LOOKBACK = 8
N_A158, N_F1, N_STOCK = 158, 12, 170
N_MARKET = 63
CACHE_DIR = "data/cache"
CACHE_VERSION = "v2"


def load_scaler():
    s = pd.read_parquet("data/processed/scaler.parquet")
    return {g: (grp["center"].to_numpy(), grp["scale"].to_numpy())
            for g, grp in s.groupby("group")}


def read_wide(file, n_fields):
    """读 date × MultiIndex(field, code)，返回 (n_dates, n_fields, n_codes)。"""
    t = pq.read_table(file)
    names = t.column_names
    feat_pos = [i for i, n in enumerate(names) if n.startswith("('")]
    cols = t.columns
    X = np.column_stack([cols[i].to_numpy() for i in feat_pos])
    return X.reshape(len(X), n_fields, -1)


def standardize_3d(arr, center, scale):
    z = (arr - center[None, :, None]) / scale[None, :, None]
    z = np.clip(z, -3.0, 3.0)
    return np.nan_to_num(z, nan=0.0)


def _assemble_stock(f):
    """f:(n_dates,170) -> (n-7, 8, 170)。"""
    w = sliding_window_view(f, LOOKBACK, axis=0)
    return w.transpose(0, 2, 1)


def _codes_of_file(file):
    t = pq.read_table(file)
    names = t.column_names
    seen, out = set(), []
    for n in names:
        if n.startswith("('"):
            c = ast.literal_eval(n)[1]
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def _build_from_parquet(universe):
    scaler = load_scaler()
    files = sorted(glob.glob("data/processed/features/feat_*.parquet"))

    mkt = pd.read_parquet("data/processed/market_status.parquet")
    mkt_z = (mkt.values - scaler["market"][0]) / scaler["market"][1]
    mkt_z = np.clip(mkt_z, -3.0, 3.0)
    mkt_z = np.nan_to_num(mkt_z, nan=0.0)
    label = pd.read_parquet("data/processed/label_z.parquet")
    elig = pd.read_parquet("data/processed/eligible_core_%s.parquet" % universe)
    split = pd.read_csv("data/processed/split_paper.csv", index_col=0).squeeze()
    split.index = pd.to_datetime(split.index)
    split = split.reindex(mkt.index).to_numpy()

    f1 = standardize_3d(read_wide("data/processed/f1_factors.parquet", N_F1), *scaler["f1"])

    data = {k: {"s": [], "m": [], "y": [], "d": [], "c": []}
            for k in ["train", "valid", "test"]}
    for fi, f in enumerate(files):
        codes = _codes_of_file(f)
        a158 = standardize_3d(read_wide(f, N_A158), *scaler["alpha158"])
        nc = a158.shape[2]
        start = fi * 50
        feat = np.concatenate([a158, f1[:, :, start:start + nc]], axis=1)
        for c in range(nc):
            ci = start + c
            seq = _assemble_stock(feat[:, :, c])
            y = label.iloc[LOOKBACK - 1:, ci].to_numpy().astype(np.float32)
            e = elig.iloc[LOOKBACK - 1:, ci].to_numpy()
            s = split[LOOKBACK - 1:]
            ok = ~np.isnan(y) & e
            for seg in ["train", "valid", "test"]:
                m = (s == seg) & ok
                if m.sum():
                    data[seg]["s"].append(seq[m].astype(np.float32))
                    data[seg]["m"].append(mkt_z[LOOKBACK - 1:][m].astype(np.float32))
                    data[seg]["y"].append(y[m])
                    data[seg]["d"].append(mkt.index[LOOKBACK - 1:][m])
                    data[seg]["c"].append(np.array([codes[c]] * m.sum(), dtype="<U16"))

    out = {}
    for seg in ["train", "valid", "test"]:
        if not data[seg]["s"]:
            out[seg] = None
            continue
        out[seg] = (
            np.concatenate(data[seg]["s"]),
            np.concatenate(data[seg]["m"]),
            np.concatenate(data[seg]["y"]),
            np.concatenate(data[seg]["d"]),
            np.concatenate(data[seg]["c"]),
        )
    return out


def _save_cache(ds):
    os.makedirs(CACHE_DIR, exist_ok=True)
    for seg in ["train", "valid", "test"]:
        for name, idx in [("X_stock", 0), ("X_market", 1), ("y", 2), ("date", 3), ("code", 4)]:
            arr = ds[seg][idx] if ds[seg] is not None else np.array([])
            np.save(os.path.join(CACHE_DIR, "%s_%s.npy" % (seg, name)), arr)
    with open(os.path.join(CACHE_DIR, "meta.txt"), "w") as f:
        f.write("version=%s\nsplit=paper_aligned\n" % CACHE_VERSION)


def _load_cache():
    meta_file = os.path.join(CACHE_DIR, "meta.txt")
    if not os.path.exists(meta_file):
        return None
    with open(meta_file) as f:
        if "version=%s" % CACHE_VERSION not in f.read():
            return None
    ds = {}
    for seg in ["train", "valid", "test"]:
        files = {name: os.path.join(CACHE_DIR, "%s_%s.npy" % (seg, name))
                 for name in ["X_stock", "X_market", "y", "date", "code"]}
        if not all(os.path.exists(f) for f in files.values()):
            return None
        ds[seg] = tuple(
            np.load(files[name], mmap_mode="r")
            for name in ["X_stock", "X_market", "y", "date", "code"]
        )
    return ds


def build_dataset(universe="csi300", use_cache=True):
    """返回 dict：train/valid/test 各 (X_stock, X_market, y, date, code)，均为 mmap。"""
    if use_cache:
        ds = _load_cache()
        if ds is not None:
            print("[loader] 读取缓存 data/cache/")
            return ds
    print("[loader] 重新生成缓存 ...")
    ds = _build_from_parquet(universe)
    _save_cache(ds)
    return _load_cache()


if __name__ == "__main__":
    ds = build_dataset()
    for seg, d in ds.items():
        if d is None:
            print(seg, ": 空")
        else:
            print(seg, ": stock", d[0].shape, " code dtype", d[4].dtype, " 天数", len(np.unique(d[3])))
