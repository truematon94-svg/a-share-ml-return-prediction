# -*- coding: utf-8 -*-
"""
G4 第 2 步·树模型：LightGBM / XGBoost / 随机森林。

内存方案：X_train 用 numpy memmap 存磁盘、流式填充，不整块占 RAM。
LightGBM / XGBoost 直接读 memmap（内存映射）；随机森林(sklearn)若强制转 float64 会
占双倍内存，若 OOM 则跳过并提示。
"""

import glob
import os

import numpy as np
import pandas as pd

from g4_ols import (load_scaler, read_wide, standardize_3d, assemble,
                    LOOKBACK, N_A158, N_F1, N_STOCK, N_MARKET, N_FEAT)

MMAP_X = "data/processed/_X_train.memmap"
MMAP_Y = "data/processed/_y_train.memmap"


def rankic_group(df):
    ic = df.groupby("date").apply(
        lambda d: d["pred"].corr(d["true"]) if len(d) >= 5 else np.nan).mean()
    rk = df.groupby("date").apply(
        lambda d: d["pred"].corr(d["true"], method="spearman") if len(d) >= 5 else np.nan).mean()
    return ic, rk


def _load_common():
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
    return files, mkt_z, mkt.index, label, elig, split, f1


def _iter_codes(files, f1):
    """逐文件逐代码产出 (feat (n_dates,170), code_idx)。"""
    for fi, f in enumerate(files):
        a158 = standardize_3d(read_wide(f, N_A158), *load_scaler()["alpha158"])
        n_codes = a158.shape[2]
        start = fi * 50
        feat = np.concatenate([a158, f1[:, :, start:start + n_codes]], axis=1)
        del a158
        for c in range(n_codes):
            yield feat[:, :, c], start + c


def build_xy():
    files, mkt_z, dates, label, elig, split, f1 = _load_common()

    # 第一趟：数 train 样本 + 收 valid/test
    n_train = 0
    rec = {"date": [], "seg": [], "X": [], "y": []}
    for feat, ci in _iter_codes(files, f1):
        X = assemble(feat, mkt_z)
        y = label.iloc[LOOKBACK - 1:, ci].to_numpy().astype(np.float32)
        e = elig.iloc[LOOKBACK - 1:, ci].to_numpy()
        s = split[LOOKBACK - 1:]
        ok = ~np.isnan(y) & e
        n_train += int(((s == "train") & ok).sum())
        for seg in ["valid", "test"]:
            m = (s == seg) & ok
            if m.sum():
                rec["date"].append(dates[LOOKBACK - 1:][m])
                rec["seg"].append(seg)
                rec["X"].append(X[m].astype(np.float32))
                rec["y"].append(y[m])

    # memmap 存 train，流式填充
    Xtr = np.memmap(MMAP_X, dtype="float32", mode="w+", shape=(n_train, N_FEAT))
    ytr = np.memmap(MMAP_Y, dtype="float32", mode="w+", shape=(n_train,))
    pos = 0
    for feat, ci in _iter_codes(files, f1):
        X = assemble(feat, mkt_z)
        y = label.iloc[LOOKBACK - 1:, ci].to_numpy().astype(np.float32)
        e = elig.iloc[LOOKBACK - 1:, ci].to_numpy()
        s = split[LOOKBACK - 1:]
        ok = ~np.isnan(y) & e
        m = (s == "train") & ok
        if m.sum():
            Xtr[pos:pos + m.sum()] = X[m]
            ytr[pos:pos + m.sum()] = y[m]
            pos += m.sum()
    Xtr.flush(); ytr.flush()

    d = {}
    for seg in ["valid", "test"]:
        idx = [i for i in range(len(rec["seg"])) if rec["seg"][i] == seg]
        d[seg] = (np.concatenate([rec["X"][i] for i in idx]),
                  np.concatenate([rec["y"][i] for i in idx]),
                  np.concatenate([rec["date"][i] for i in idx]))
    return Xtr, ytr, d


def main():
    print("构建 X/y（memmap 流式，不占内存）...")
    Xtr, ytr, d = build_xy()
    Xva, yva, dva = d["valid"]
    Xte, yte, dte = d["test"]
    print("训练 %d, valid %d, test %d" % (len(ytr), len(yva), len(yte)))

    def report(name, pred):
        ic, rk = rankic_group(pd.DataFrame({"date": dte, "pred": pred, "true": yte}))
        print("%-14s test: IC=%.4f, RankIC=%.4f" % (name, ic, rk))

    # LightGBM（读 memmap）
    import lightgbm as lgb
    tr_ds = lgb.Dataset(Xtr, ytr, free_raw_data=True)
    va_ds = lgb.Dataset(Xva, yva, reference=tr_ds)
    params = {"objective": "regression", "metric": "rmse", "verbosity": -1,
              "learning_rate": 0.05, "num_leaves": 255, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "seed": 0}
    model = lgb.train(params, tr_ds, num_boost_round=500, valid_sets=[va_ds],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    report("LightGBM", model.predict(Xte))
    del model, tr_ds, va_ds

    # XGBoost（读 memmap）
    try:
        import xgboost as xgb
        dtr = xgb.DMatrix(Xtr, ytr)
        dva_ = xgb.DMatrix(Xva, yva)
        dte_ = xgb.DMatrix(Xte)
        xm = xgb.train({"objective": "reg:squarederror", "eta": 0.05, "max_depth": 8,
                        "subsample": 0.8, "colsample_bytree": 0.8, "seed": 0},
                       dtr, num_boost_round=500, evals=[(dva_, "valid")],
                       early_stopping_rounds=50, verbose_eval=False)
        report("XGBoost", xm.predict(dte_))
        del dtr, dva_, dte_, xm
    except ImportError:
        print("XGBoost 未安装")

    # 随机森林（sklearn 可能转 float64 占双倍内存，OOM 则跳过）
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=0)
        rf.fit(Xtr, ytr)
        report("RandomForest", rf.predict(Xte))
    except MemoryError:
        print("RandomForest 内存不足（sklearn 转 float64），跳过")

    os.remove(MMAP_X); os.remove(MMAP_Y)


if __name__ == "__main__":
    main()
