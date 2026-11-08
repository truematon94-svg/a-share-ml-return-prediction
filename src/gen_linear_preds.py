# -*- coding: utf-8 -*-
"""
生成 OLS / Ridge 的测试集预测，保存成 pred 文件（供回测用）。

复用 g4_ols 的流式累积，一次算出 OLS(λ=0) 和 Ridge(最优λ) 的 beta，预测 test 并保存。
"""

import glob

import numpy as np
import pandas as pd

from g4_ols import (load_scaler, read_wide, standardize_3d, assemble,
                    LOOKBACK, N_A158, N_F1, N_STOCK, N_MARKET, N_FEAT)

LAMBDAS = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]


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
            ok = ~np.isnan(y) & e
            tr = (s == "train") & ok
            Xtr, ytr = X[tr], y[tr]
            XtX += Xtr.T @ Xtr
            Xty += Xtr.T @ ytr
            sum_x += Xtr.sum(axis=0)
            sum_x2 += (Xtr ** 2).sum(axis=0)
            n_train += len(Xtr)
            va = (s == "valid") & ok
            if va.sum():
                valid_X.append(X[va]); valid_y.append(y[va]); valid_dates.append(mkt.index[LOOKBACK - 1:][va])

    var = sum_x2 / n_train - (sum_x / n_train) ** 2
    keep = np.where(var > 1e-12)[0]
    XtXk, Xtyk = XtX[np.ix_(keep, keep)], Xty[keep]
    vX = np.concatenate(valid_X)[:, keep]
    vy = np.concatenate(valid_y)
    vd = np.concatenate(valid_dates)
    vdf = pd.DataFrame({"date": vd, "true": vy})

    # OLS (λ=0)
    beta_ols = np.linalg.solve(XtXk, Xtyk)
    # Ridge：λ 网格选优（valid RankIC）
    best_lam, best_rk = None, -np.inf
    for lam in LAMBDAS:
        b = np.linalg.solve(XtXk + lam * np.eye(len(keep)), Xtyk)
        pred = vX @ b
        d = vdf.copy(); d["pred"] = pred
        rk = d.groupby("date").apply(lambda x: x["pred"].corr(x["true"], method="spearman") if len(x) >= 5 else np.nan).mean()
        if rk > best_rk:
            best_rk, best_lam = rk, lam
    beta_ridge = np.linalg.solve(XtXk + best_lam * np.eye(len(keep)), Xtyk)
    beta_ols_full = np.zeros(N_FEAT); beta_ols_full[keep] = beta_ols
    beta_ridge_full = np.zeros(N_FEAT); beta_ridge_full[keep] = beta_ridge

    # 第二趟：预测 test，保存
    for name, beta_full in [("ols", beta_ols_full), ("ridge", beta_ridge_full)]:
        rec = {"date": [], "code": [], "y_true": [], "prediction": []}
        for fi, f in enumerate(files):
            codes = _codes(f)
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
                ok = ~np.isnan(y) & e
                m = (s == "test") & ok
                if m.sum():
                    rec["date"].append(mkt.index[LOOKBACK - 1:][m])
                    rec["code"].append(np.array([codes[c]] * m.sum(), dtype="<U16"))
                    rec["y_true"].append(y[m])
                    rec["prediction"].append((X[m] @ beta_full).astype(np.float32))
        df = pd.DataFrame({
            "date": np.concatenate(rec["date"]), "code": np.concatenate(rec["code"]),
            "y_true": np.concatenate(rec["y_true"]), "prediction": np.concatenate(rec["prediction"]),
            "model": name, "seed": 0,
        })
        df.to_parquet("cloud/results_seed0/pred_%s_s0.parquet" % name, index=False)
        print("已存 pred_%s_s0.parquet（%d 行，best λ=%.0e）" % (name, len(df), best_lam))


def _codes(f):
    import ast
    import pyarrow.parquet as pq
    t = pq.read_table(f)
    names = t.column_names
    seen, out = set(), []
    for n in names:
        if n.startswith("('"):
            c = ast.literal_eval(n)[1]
            if c not in seen:
                seen.add(c); out.append(c)
    return out


if __name__ == "__main__":
    main()
