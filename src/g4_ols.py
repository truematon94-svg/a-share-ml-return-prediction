# -*- coding: utf-8 -*-
"""
G4 第 2 步·第一个模型：OLS 基线（全量）。

全量 1765 只股票 + eligible 过滤(csi300 核心口径) + 标准逐日 RankIC。

流程：
  载入特征 -> 标准化 -> 组装 8 天序列(1423 维) -> 最小二乘拟合(流式累积 XᵀX/Xᵀy)
  -> 预测 -> 逐日截面 Spearman(RankIC) 对天平均。

内存：两趟流式（第一趟累积法方程，第二趟只算预测），不落全量 X。
"""

import ast
import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

LOOKBACK = 8
N_A158, N_F1, N_STOCK = 158, 12, 170
N_MARKET = 63
N_FEAT = LOOKBACK * N_STOCK + N_MARKET   # 1423


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


def assemble(f, mkt_z):
    """f:(n_dates,170) -> X:(n-7,1423)，拼 8 天展平 + 市场状态。"""
    w = sliding_window_view(f, LOOKBACK, axis=0).transpose(0, 2, 1).reshape(-1, LOOKBACK * N_STOCK)
    return np.concatenate([w, mkt_z[LOOKBACK - 1:]], axis=1)


def main():
    scaler = load_scaler()
    files = sorted(glob.glob("data/processed/features/feat_*.parquet"))

    # 小文件
    mkt = pd.read_parquet("data/processed/market_status.parquet")
    mkt_z = (mkt.values - scaler["market"][0]) / scaler["market"][1]
    mkt_z = np.clip(mkt_z, -3.0, 3.0)
    mkt_z = np.nan_to_num(mkt_z, nan=0.0)
    label = pd.read_parquet("data/processed/label_z.parquet")
    elig = pd.read_parquet("data/processed/eligible_core_csi300.parquet")
    split = pd.read_csv("data/processed/split_paper.csv", index_col=0).squeeze()
    split.index = pd.to_datetime(split.index)
    split = split.reindex(mkt.index).to_numpy()
    n_dates = len(mkt)

    # F1（一次性载入、标准化）
    f1 = read_wide("data/processed/f1_factors.parquet", N_F1)
    f1 = standardize_3d(f1, *scaler["f1"])

    # ---- 第一趟：累积法方程（train）----
    XtX = np.zeros((N_FEAT, N_FEAT))
    Xty = np.zeros(N_FEAT)
    sum_x = np.zeros(N_FEAT)
    sum_x2 = np.zeros(N_FEAT)
    n_train = 0
    for fi, f in enumerate(files):
        a158 = standardize_3d(read_wide(f, N_A158), *scaler["alpha158"])
        n_codes = a158.shape[2]
        start = fi * 50
        feat = np.concatenate([a158, f1[:, :, start:start + n_codes]], axis=1)
        del a158
        for c in range(n_codes):
            ci = start + c
            X = assemble(feat[:, :, c], mkt_z)
            y = label.iloc[LOOKBACK - 1:, ci].to_numpy().astype(np.float64)
            e = elig.iloc[LOOKBACK - 1:, ci].to_numpy()
            s = split[LOOKBACK - 1:]
            valid = ~np.isnan(y) & e
            tr = (s == "train") & valid
            Xtr, ytr = X[tr], y[tr]
            XtX += Xtr.T @ Xtr
            Xty += Xtr.T @ ytr
            sum_x += Xtr.sum(axis=0)
            sum_x2 += (Xtr ** 2).sum(axis=0)
            n_train += len(Xtr)
        print("  第一趟 %d/%d 文件" % (fi + 1, len(files)), flush=True)

    # 解 β（删零方差列 + lstsq）
    var = sum_x2 / n_train - (sum_x / n_train) ** 2
    keep = np.where(var > 1e-12)[0]
    beta = np.zeros(N_FEAT)
    beta[keep] = np.linalg.lstsq(XtX[np.ix_(keep, keep)], Xty[keep], rcond=None)[0]
    del XtX, Xty, sum_x, sum_x2

    # ---- 第二趟：预测 valid/test，带日期 ----
    rec = {"date": [], "split": [], "pred": [], "true": []}
    for fi, f in enumerate(files):
        a158 = standardize_3d(read_wide(f, N_A158), *scaler["alpha158"])
        n_codes = a158.shape[2]
        start = fi * 50
        feat = np.concatenate([a158, f1[:, :, start:start + n_codes]], axis=1)
        del a158
        for c in range(n_codes):
            ci = start + c
            X = assemble(feat[:, :, c], mkt_z)
            y = label.iloc[LOOKBACK - 1:, ci].to_numpy().astype(np.float64)
            e = elig.iloc[LOOKBACK - 1:, ci].to_numpy()
            s = split[LOOKBACK - 1:]
            valid = ~np.isnan(y) & e
            m = ((s == "valid") | (s == "test")) & valid
            if m.sum() == 0:
                continue
            pred = X[m] @ beta
            rec["date"].append(mkt.index[LOOKBACK - 1:][m])
            rec["split"].append(s[m])
            rec["pred"].append(pred)
            rec["true"].append(y[m])
        print("  第二趟 %d/%d 文件" % (fi + 1, len(files)), flush=True)

    df = pd.DataFrame({
        "date": np.concatenate(rec["date"]),
        "split": np.concatenate(rec["split"]),
        "pred": np.concatenate(rec["pred"]),
        "true": np.concatenate(rec["true"]),
    })

    def rankic(sub):
        g = sub.groupby("date").apply(
            lambda d: d["pred"].corr(d["true"], method="spearman") if len(d) >= 5 else np.nan)
        return g.mean(), len(g.dropna())

    print("\n=== 全量 OLS 结果（csi300 核心口径）===")
    print("训练样本数: %d, 有效特征维: %d/%d" % (n_train, len(keep), N_FEAT))
    for seg in ["valid", "test"]:
        d = df[df["split"] == seg]
        rk, nd = rankic(d)
        ic = d.groupby("date").apply(
            lambda x: x["pred"].corr(x["true"]) if len(x) >= 5 else np.nan).mean()
        print("%s: %d 样本, %d 天, RankIC=%.4f, IC=%.4f" % (seg, len(d), nd, rk, ic))


if __name__ == "__main__":
    main()
