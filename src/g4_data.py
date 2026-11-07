# -*- coding: utf-8 -*-
"""
G4 第 1 步（后半）：标准化应用 + 8 天序列组装。

标准化（Qlib RobustZScoreNorm 口径）：
  z = (x - center) / scale      center=median, scale=MAD*1.4826
  z = clip(z, -3, 3)
  z = fillna(0)

组装：
  每个样本 = 过去 8 天的 170 个特征（158 Alpha158 + 12 F1）+ 当天 63 维市场状态。
  "过去 8 天" = 时间轴上的滑动窗口（sliding_window_view）。
"""

import ast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

LOOKBACK = 8
N_ALPHA158 = 158
N_F1 = 12
N_STOCK = N_ALPHA158 + N_F1   # 170


def load_scaler():
    """加载标准化参数，返回 {group: (feature_list, center_array, scale_array)}。"""
    s = pd.read_parquet("data/processed/scaler.parquet")
    out = {}
    for g, grp in s.groupby("group"):
        out[g] = (grp["feature"].tolist(), grp["center"].to_numpy(), grp["scale"].to_numpy())
    return out


def standardize(arr, center, scale):
    """(x-center)/scale -> clip[-3,3] -> NaN 填 0。arr 最后一维是特征维。"""
    z = (arr - center) / scale
    z = np.clip(z, -3.0, 3.0)
    return np.nan_to_num(z, nan=0.0)


def assemble_stock_sequence(feat, lookback=LOOKBACK):
    """feat: (n_dates, n_features) -> (n_dates-lookback+1, lookback, n_features) 的滑动窗口。"""
    w = sliding_window_view(feat, lookback, axis=0)   # (n-7, n_features, lookback)
    return w.transpose(0, 2, 1)                        # (n-7, lookback, n_features)


def read_code_features(file, n_fields):
    """读 date × MultiIndex(field, code) 的 parquet，返回 (n_dates, n_fields, n_codes)。"""
    t = pq.read_table(file)
    feat_pos = [i for i, n in enumerate(t.column_names) if n.startswith("('")]
    cols = t.columns
    X = np.column_stack([cols[i].to_numpy() for i in feat_pos])   # (n_dates, n_fields*n_codes)
    return X.reshape(len(X), n_fields, -1)                         # (n_dates, n_fields, n_codes)


def demo():
    scaler = load_scaler()

    # 第一只股票（code index 0）的特征
    a158 = read_code_features("data/processed/features/feat_0000_0049.parquet", N_ALPHA158)
    f1 = read_code_features("data/processed/f1_factors.parquet", N_F1)
    a158_code0 = a158[:, :, 0]          # (n_dates, 158)
    f1_code0 = f1[:, :, 0]              # (n_dates, 12)

    feat = np.concatenate([a158_code0, f1_code0], axis=1)          # (n_dates, 170)
    center = np.concatenate([scaler["alpha158"][1], scaler["f1"][1]])
    scale = np.concatenate([scaler["alpha158"][2], scaler["f1"][2]])

    z = standardize(feat, center, scale)                            # (n_dates, 170)
    seq = assemble_stock_sequence(z, LOOKBACK)                      # (n_dates-7, 8, 170)

    print("原始特征 shape:", feat.shape)
    print("标准化后: 范围 [%.2f, %.2f], 无NaN=%s" % (z.min(), z.max(), not np.isnan(z).any()))
    print("8天序列 shape:", seq.shape, "(= n_dates-7 × 8 × 170)")
    print("序列第0天(最早)数值范围 [%.2f, %.2f]" % (seq[0].min(), seq[0].max()))
    print("序列最后一天数值范围 [%.2f, %.2f]" % (seq[-1].min(), seq[-1].max()))

    # 市场状态
    mkt = pd.read_parquet("data/processed/market_status.parquet")
    mkt_z = standardize(mkt.values, scaler["market"][1], scaler["market"][2])
    print("\n市场状态: %d 天 × %d 维, 标准化后范围 [%.2f, %.2f]"
          % (mkt.shape[0], mkt.shape[1], mkt_z.min(), mkt_z.max()))


if __name__ == "__main__":
    demo()
