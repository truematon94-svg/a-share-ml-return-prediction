# -*- coding: utf-8 -*-
"""
云端树模型训练：LightGBM / XGBoost / RandomForest，各自独立运行。

统一口径：seed=0、展平 8 天特征(1423维)、valid 早停、test 只在训练后算一次。
设备：--device auto/cpu/cuda（XGBoost 优先 CUDA；LightGBM 小样本测 GPU 回退 CPU；RF 保持 CPU）。

用法：
  python -m cloud.train_tree --model lightgbm --seed 0
  python -m cloud.train_tree --model xgboost --seed 0
  python -m cloud.train_tree --model random_forest --seed 0
  python -m cloud.train_tree --model lightgbm --smoke
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cloud.loader import build_dataset

RESULTS_DIR = "results"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
METRICS_FILE = os.path.join(RESULTS_DIR, "metrics.csv")


def date_groups(dates):
    order = np.argsort(dates)
    sd = dates[order]
    split = np.where(sd[1:] != sd[:-1])[0] + 1
    return [order[a:b] for a, b in zip([0, *split], [*split, len(order)])]


def daily_ic(pred, true, dates):
    ics, rks = [], []
    for g in date_groups(dates):
        if len(g) < 5:
            continue
        p, t = pred[g], true[g]
        ics.append(np.corrcoef(p, t)[0, 1])
        rks.append(spearmanr(p, t).correlation)
    return np.nanmean(ics), np.nanmean(rks)


def save_metrics(model_name, seed, best_iter, valid_ic, valid_rankic, test_ic, test_rankic, seconds):
    row = pd.DataFrame([{
        "model": model_name, "seed": seed, "best_epoch": best_iter, "train_loss": np.nan,
        "valid_ic": valid_ic, "valid_rankic": valid_rankic,
        "test_ic": test_ic, "test_rankic": test_rankic, "train_seconds": seconds,
    }])
    if os.path.exists(METRICS_FILE):
        row.to_csv(METRICS_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(METRICS_FILE, index=False)


def save_prediction(model_name, seed, pred, y_true, dates, codes):
    df = pd.DataFrame({
        "date": dates, "code": codes.astype(str), "y_true": y_true,
        "prediction": pred, "model": model_name, "seed": seed,
    })
    df.to_parquet(os.path.join(RESULTS_DIR, "pred_%s_s%d.parquet" % (model_name, seed)), index=False)


def _flatten(X_stock, X_market):
    return np.concatenate([X_stock.reshape(len(X_stock), -1), X_market], axis=1).astype(np.float32)


def _lightgbm_gpu_ok():
    """小样本测 LightGBM CUDA 可用性。"""
    import lightgbm as lgb
    try:
        d = lgb.Dataset(np.random.rand(100, 10), np.random.rand(100))
        lgb.train({"device_type": "cuda", "max_bin": 63, "verbosity": -1}, d, num_boost_round=1)
        return True
    except Exception as e:
        print("LightGBM CUDA 不可用，回退 CPU：%s" % e)
        return False


def run_lightgbm(Xtr, ytr, Xva, yva, Xte, yte, seed, device, smoke):
    import lightgbm as lgb
    params = {
        "objective": "regression", "metric": "rmse", "verbosity": -1,
        "num_leaves": 63, "learning_rate": 0.05,
        "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1,
        "seed": seed, "num_threads": -1,
    }
    use_cuda = (device == "cuda") or (device == "auto" and _lightgbm_gpu_ok())
    if use_cuda:
        params.update({"device_type": "cuda", "max_bin": 63})
    used = "cuda" if use_cuda else "cpu"

    tr = lgb.Dataset(Xtr, ytr)
    va = lgb.Dataset(Xva, yva, reference=tr)
    n_rounds = 5 if smoke else 200
    callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(20)]
    model = lgb.train(params, tr, num_boost_round=n_rounds, valid_sets=[va], callbacks=callbacks)
    bi = model.best_iteration
    return model, bi, used


def run_xgboost(Xtr, ytr, Xva, yva, Xte, yte, seed, device, smoke):
    import xgboost as xgb
    params = {
        "objective": "reg:squarederror", "max_depth": 6, "eta": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.6, "tree_method": "hist", "seed": seed,
    }
    n_rounds = 5 if smoke else 200
    use_cuda = device != "cpu"
    try:
        if use_cuda:
            params.update({"device": "cuda", "tree_method": "hist"})
        # QuantileDMatrix：valid/test 共享 train 的量化切分（ref=dtr）
        dtr = xgb.QuantileDMatrix(Xtr, ytr)
        dva = xgb.QuantileDMatrix(Xva, yva, ref=dtr)
        dte = xgb.QuantileDMatrix(Xte, ref=dtr)
        model = xgb.train(params, dtr, num_boost_round=n_rounds, evals=[(dva, "valid")],
                          early_stopping_rounds=30, verbose_eval=20)
        used = "cuda" if use_cuda else "cpu"
    except Exception as e:
        if device == "auto":
            print("XGBoost CUDA 不可用，回退 CPU：%s" % e)
            params.update({"device": "cpu", "tree_method": "hist"})
            dtr = xgb.QuantileDMatrix(Xtr, ytr)
            dva = xgb.QuantileDMatrix(Xva, yva, ref=dtr)
            dte = xgb.QuantileDMatrix(Xte, ref=dtr)
            model = xgb.train(params, dtr, num_boost_round=n_rounds, evals=[(dva, "valid")],
                              early_stopping_rounds=30, verbose_eval=20)
            used = "cpu"
        else:
            raise
    return model, model.best_iteration, used, dva, dte


def run_random_forest(Xtr, ytr, Xva, yva, Xte, yte, seed, smoke):
    from sklearn.ensemble import RandomForestRegressor
    if smoke:
        rf = RandomForestRegressor(n_estimators=5, max_depth=6, max_features="sqrt",
                                   min_samples_leaf=20, max_samples=0.1, n_jobs=-1, random_state=seed)
    else:
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, max_features="sqrt",
                                   min_samples_leaf=20, max_samples=0.5, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, ytr)
    return rf, np.nan, "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["lightgbm", "xgboost", "random_forest"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    np.random.seed(args.seed)
    ds = build_dataset()
    Xs_tr, Xm_tr, y_tr, d_tr, c_tr = ds["train"]
    Xs_va, Xm_va, y_va, d_va, c_va = ds["valid"]
    Xs_te, Xm_te, y_te, d_te, c_te = ds["test"]

    if args.smoke:
        # 少量日期做冒烟
        tr_g = date_groups(d_tr)[:30]
        tr_idx = np.concatenate(tr_g)
        va_g = date_groups(d_va)[:10]
        va_idx = np.concatenate(va_g)
        te_g = date_groups(d_te)[:10]
        te_idx = np.concatenate(te_g)
        Xs_tr, Xm_tr, y_tr, d_tr, c_tr = Xs_tr[tr_idx], Xm_tr[tr_idx], y_tr[tr_idx], d_tr[tr_idx], c_tr[tr_idx]
        Xs_va, Xm_va, y_va, d_va, c_va = Xs_va[va_idx], Xm_va[va_idx], y_va[va_idx], d_va[va_idx], c_va[va_idx]
        Xs_te, Xm_te, y_te, d_te, c_te = Xs_te[te_idx], Xm_te[te_idx], y_te[te_idx], d_te[te_idx], c_te[te_idx]

    Xtr = _flatten(Xs_tr, Xm_tr)
    Xva = _flatten(Xs_va, Xm_va)
    Xte = _flatten(Xs_te, Xm_te)

    t0 = time.time()
    if args.model == "lightgbm":
        model, bi, used = run_lightgbm(Xtr, y_tr, Xva, y_va, Xte, y_te, args.seed, args.device, args.smoke)
        pred_va = model.predict(Xva, num_iteration=bi)
        pred_te = model.predict(Xte, num_iteration=bi)
    elif args.model == "xgboost":
        model, bi, used, dva, dte = run_xgboost(Xtr, y_tr, Xva, y_va, Xte, y_te, args.seed, args.device, args.smoke)
        pred_va = model.predict(dva, iteration_range=(0, bi + 1))
        pred_te = model.predict(dte, iteration_range=(0, bi + 1))
    else:
        model, bi, used = run_random_forest(Xtr, y_tr, Xva, y_va, Xte, y_te, args.seed, args.smoke)
        pred_va = model.predict(Xva)
        pred_te = model.predict(Xte)

    ic_v, rk_v = daily_ic(pred_va, y_va, d_va)
    ic_t, rk_t = daily_ic(pred_te, y_te, d_te)
    seconds = time.time() - t0

    if not args.smoke:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        # 保存模型
        if args.model == "lightgbm":
            model.save_model(os.path.join(CHECKPOINT_DIR, "lightgbm_s%d.txt" % args.seed))
        elif args.model == "xgboost":
            model.save_model(os.path.join(CHECKPOINT_DIR, "xgboost_s%d.json" % args.seed))
        else:
            import joblib
            joblib.dump(model, os.path.join(CHECKPOINT_DIR, "random_forest_s%d.joblib" % args.seed))
        save_metrics(args.model, args.seed, bi, ic_v, rk_v, ic_t, rk_t, seconds)
        save_prediction(args.model, args.seed, pred_te, y_te, d_te, c_te)

    print("=== %s seed=%d device=%s best_iter=%.0f: valid IC=%.4f RankIC=%.4f, test IC=%.4f RankIC=%.4f, %.0fs ==="
          % (args.model, args.seed, used, bi, ic_v, rk_v, ic_t, rk_t, seconds))


if __name__ == "__main__":
    main()
