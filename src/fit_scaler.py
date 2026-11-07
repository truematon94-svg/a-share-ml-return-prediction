# -*- coding: utf-8 -*-
"""
G4 第 1 步：标准化器拟合（train-only，Qlib RobustZScoreNorm 口径）。

对每个特征，在训练段(2008-01-01 ~ 2020-03-31)上算：
  center = median(x)
  scale  = MAD(x) * 1.4826 + EPS     (MAD = median(|x - median|), EPS = 1e-12)

应用时（后续做）： (x - center) / scale，clip 到 [-3, 3]，然后 fillna 0。

三组分别拟合：alpha158(158)、f1(12)、market(63)。
用 pyarrow 读盘（pandas 读 MultiIndex 列在低内存下会爆，pyarrow 不会）。
输出：data/processed/scaler.parquet（group, feature, center, scale）
"""

import ast
import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TRAIN_START = "2008-01-01"
TRAIN_END = "2020-03-31"
EPS = 1e-12
SUBSAMPLE = 5000          # 每个文件抽样数（median/MAD 是稳健统计，抽样足够）


def _read_wide(file):
    """读 date × MultiIndex(field, code) 的 parquet，返回 (日期序列, 字段名列表, 数值矩阵)。"""
    t = pq.read_table(file)
    names = t.column_names
    feat_pos = [i for i, n in enumerate(names) if n.startswith("('")]   # 特征列位置
    idx_pos = next(i for i, n in enumerate(names) if not n.startswith("('"))  # 索引列位置
    dates = pd.to_datetime(t.column(idx_pos).to_pandas())
    fields = [ast.literal_eval(names[i])[0] for i in feat_pos]   # ('KMID','000001.XSHE') -> 'KMID'
    cols = t.columns
    X = np.column_stack([cols[i].to_numpy() for i in feat_pos])   # float32 (n_dates, n_feat_cols)
    return dates, fields, X


def fit_wide(files, group):
    """date × MultiIndex(field, code) 的一组文件 → per-field median/MAD。"""
    parts = []
    fields_order = None
    for f in files:
        dates, fields, X = _read_wide(f)
        # 去重保持出现顺序 → 字段顺序（feature-major）
        seen = []
        for s in fields:
            if s not in seen:
                seen.append(s)
        if fields_order is None:
            fields_order = seen
        n_fields = len(seen)
        n_codes = X.shape[1] // n_fields
        m = (dates >= TRAIN_START) & (dates <= TRAIN_END)
        X = X[m]
        # (n_dates, n_fields*n_codes) -> (n_dates, n_fields, n_codes) -> (n_dates*n_codes, n_fields)
        X = X.reshape(len(X), n_fields, n_codes).transpose(0, 2, 1).reshape(-1, n_fields)
        n = len(X)
        if n > SUBSAMPLE:
            idx = np.random.default_rng(0).choice(n, SUBSAMPLE, replace=False)
            X = X[idx]
        parts.append(X.astype(np.float64))
    Xall = np.concatenate(parts, axis=0)
    center = np.nanmedian(Xall, axis=0)
    scale = np.nanmedian(np.abs(Xall - center), axis=0) + EPS
    scale = scale * 1.4826
    return pd.DataFrame({"group": group, "feature": fields_order,
                         "center": center, "scale": scale})


def fit_flat(file, group):
    """date × n_features 的单个文件（市场状态，63 列，小）→ per-feature median/MAD。"""
    df = pd.read_parquet(file).loc[TRAIN_START:TRAIN_END]
    X = df.values.astype(np.float64)
    center = np.nanmedian(X, axis=0)
    scale = np.nanmedian(np.abs(X - center), axis=0) + EPS
    scale = scale * 1.4826
    return pd.DataFrame({"group": group, "feature": list(df.columns),
                         "center": center, "scale": scale})


def main():
    feat_files = sorted(glob.glob("data/processed/features/feat_*.parquet"))
    alpha158 = fit_wide(feat_files, "alpha158")
    f1 = fit_wide(["data/processed/f1_factors.parquet"], "f1")
    market = fit_flat("data/processed/market_status.parquet", "market")
    out = pd.concat([alpha158, f1, market], ignore_index=True)
    out.to_parquet("data/processed/scaler.parquet", index=False)

    print("标准化器拟合完成：%d 个特征" % len(out))
    print(out.groupby("group").size().to_string())
    print("\ncenter/scale 分布（median / max）:")
    print(out.groupby("group")[["center", "scale"]].agg(["median", "max"]).to_string())
    print("\nscale≈0 的特征数（标准化后退化为常量）: %d" % int((out["scale"] <= 1e-9).sum()))


if __name__ == "__main__":
    main()
