# -*- coding: utf-8 -*-
"""
G4 第 2 步·第二个模型：Ridge（L2 正则线性回归）。

与 OLS 同构，解 β = (XᵀX + λI)⁻¹ Xᵀy，λ 在 valid 上用 RankIC 网格选优。
流式累积 XᵀX/Xᵀy，内存不落全量 X。
"""

import glob

import numpy as np
import pandas as pd

from g4_ols import (load_scaler, read_wide, standardize_3d, assemble,
                    LOOKBACK, N_A158, N_F1, N_STOCK, N_MARKET, N_FEAT)

LAMBDAS = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]


def rankic_group(df):
    return df.groupby("date").apply(
        lambda d: d["pred"].corr(d["true"], method="spearman") if len(d) >= 5 else np.nan).mean()


def main():
    scaler = load_scaler()
    files = sorted(glob.glob("data/processed/features/feat_*.parquet"))

    mkt = pd.read_parquet("data/processed/market_status.parquet")
    mkt_z = (mkt.values - scaler["market"][0]) / scaler["market"][1]
    mkt_z = np.clip(mkt_z, -3.0, 3.0)
    mkt_z = np.nan_to_num(mkt_z, nan=0.0)
    label = pd.read_parquet("data/processed/label_z.parquet")
    elig = pd.read_parquet("data/processed/eligible_core_csi300.parquet")
    split = pd.read_csv("data/processed/split_paper.csv", index_col=0).squeeze()
    split.index = pd.to_datetime(split.index)
    split = split.reindex(mkt.index).to_numpy()

    f1 = standardize_3d(read_wide("data/processed/f1_factors.parquet", N_F1), *scaler["f1"])

    XtX = np.zeros((N_FEAT, N_FEAT))
    Xty = np.zeros(N_FEAT)
    sum_x = np.zeros(N_FEAT)
    sum_x2 = np.zeros(N_FEAT)
    n_train = 0
    valid_X, valid_y, valid_dates = [], [], []

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
            va = (s == "valid") & valid
            if va.sum():
                valid_X.append(X[va]); valid_y.append(y[va])
                valid_dates.append(mkt.index[LOOKBACK - 1:][va])

    # 删零方差列
    var = sum_x2 / n_train - (sum_x / n_train) ** 2
    keep = np.where(var > 1e-12)[0]
    XtXk, Xtyk = XtX[np.ix_(keep, keep)], Xty[keep]

    vX = np.concatenate(valid_X)[:, keep]
    vy = np.concatenate(valid_y)
    vd = np.concatenate(valid_dates)
    vdf = pd.DataFrame({"date": vd, "true": vy})

    # λ 网格选优（valid RankIC）
    print("λ 网格选优（valid RankIC）:")
    best_lam, best_rk = None, -np.inf
    results = {}
    for lam in LAMBDAS:
        beta = np.linalg.solve(XtXk + lam * np.eye(len(keep)), Xtyk)
        pred = vX @ beta
        d = vdf.copy(); d["pred"] = pred
        rk = rankic_group(d)
        results[lam] = rk
        print("  λ=%.0e  valid RankIC=%.4f" % (lam, rk))
        if rk > best_rk:
            best_rk, best_lam = rk, lam

    beta_best = np.linalg.solve(XtXk + best_lam * np.eye(len(keep)), Xtyk)
    beta_full = np.zeros(N_FEAT)
    beta_full[keep] = beta_best

    # 第二趟：test 预测
    rec = {"date": [], "pred": [], "true": []}
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
            m = (s == "test") & valid
            if m.sum() == 0:
                continue
            rec["date"].append(mkt.index[LOOKBACK - 1:][m])
            rec["pred"].append(X[m] @ beta_full)
            rec["true"].append(y[m])

    df = pd.DataFrame({"date": np.concatenate(rec["date"]),
                       "pred": np.concatenate(rec["pred"]),
                       "true": np.concatenate(rec["true"])})
    rk = rankic_group(df)
    ic = df.groupby("date").apply(
        lambda d: d["pred"].corr(d["true"]) if len(d) >= 5 else np.nan).mean()
    print("\n=== Ridge 结果（csi300 核心口径）===")
    print("训练样本 %d, 有效特征 %d/%d, 最优 λ=%.0e" % (n_train, len(keep), N_FEAT, best_lam))
    print("valid: RankIC=%.4f（最优λ）" % best_rk)
    print("test : %d 样本, %d 天, RankIC=%.4f, IC=%.4f" % (len(df), df["date"].nunique(), rk, ic))


if __name__ == "__main__":
    main()
