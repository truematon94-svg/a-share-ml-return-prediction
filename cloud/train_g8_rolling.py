# -*- coding: utf-8 -*-
"""G8 精简滚动训练：RandomForest 与 MASTER，三个年度fold，固定seed=0。"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cloud.models import MASTER
from cloud.rolling_loader import TEST_YEARS, build_rolling_dataset, fold_spec
from cloud.train import _predict_by_date, daily_ic, date_groups


def _flatten(x_stock, x_market):
    return np.concatenate(
        [x_stock.reshape(len(x_stock), -1), x_market], axis=1
    ).astype(np.float32)


def _save_metrics(results_dir, row):
    path = os.path.join(results_dir, "metrics.csv")
    new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        duplicate = ((old["model"] == row["model"])
                     & (old["test_year"] == row["test_year"])
                     & (old["seed"] == row["seed"]))
        new = pd.concat([old.loc[~duplicate], new], ignore_index=True)
        new = new.sort_values(["test_year", "model", "seed"])
    new.to_csv(path, index=False)


def _save_prediction(results_dir, model_name, test_year, seed,
                     pred, y_true, dates, codes):
    df = pd.DataFrame({
        "date": dates,
        "code": codes.astype(str),
        "y_true": y_true,
        "prediction": pred,
        "model": model_name,
        "test_year": test_year,
        "seed": seed,
    })
    name = "pred_%s_fold%d_s%d.parquet" % (model_name, test_year, seed)
    df.to_parquet(os.path.join(results_dir, name), index=False)


def _base_row(model_name, test_year, seed, seconds, valid_ic,
              valid_rankic, test_ic, test_rankic, best_epoch):
    spec = fold_spec(test_year)
    return {
        "model": model_name,
        "test_year": test_year,
        "seed": seed,
        "train_start": spec["train_start"].date(),
        "train_end": spec["train_end"].date(),
        "valid_start": spec["valid_start"].date(),
        "valid_end": spec["valid_end"].date(),
        "test_start": spec["test_start"].date(),
        "test_end": spec["test_end"].date(),
        "best_epoch": best_epoch,
        "valid_ic": valid_ic,
        "valid_rankic": valid_rankic,
        "test_ic": test_ic,
        "test_rankic": test_rankic,
        "train_seconds": seconds,
    }


def train_random_forest(ds, test_year, seed, results_dir):
    from sklearn.ensemble import RandomForestRegressor
    import joblib

    xs_tr, xm_tr, y_tr, _, _ = ds["train"]
    xs_va, xm_va, y_va, d_va, _ = ds["valid"]
    xs_te, xm_te, y_te, d_te, c_te = ds["test"]
    print("[rolling] flatten RandomForest fold=%d" % test_year)
    x_tr = _flatten(xs_tr, xm_tr)
    x_va = _flatten(xs_va, xm_va)
    x_te = _flatten(xs_te, xm_te)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        max_features="sqrt",
        min_samples_leaf=20,
        max_samples=0.5,
        n_jobs=-1,
        random_state=seed,
    )
    t0 = time.time()
    model.fit(x_tr, y_tr)
    pred_va = model.predict(x_va)
    pred_te = model.predict(x_te)
    seconds = time.time() - t0
    valid_ic, valid_rankic = daily_ic(pred_va, y_va, d_va)
    test_ic, test_rankic = daily_ic(pred_te, y_te, d_te)

    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    joblib.dump(model, os.path.join(
        checkpoint_dir, "random_forest_fold%d_s%d.joblib" % (test_year, seed)))
    _save_prediction(results_dir, "random_forest", test_year, seed,
                     pred_te, y_te, d_te, c_te)
    row = _base_row("random_forest", test_year, seed, seconds,
                    valid_ic, valid_rankic, test_ic, test_rankic, np.nan)
    _save_metrics(results_dir, row)
    print("=== random_forest fold=%d: valid IC=%.4f RankIC=%.4f, "
          "test IC=%.4f RankIC=%.4f, %.0fs ==="
          % (test_year, valid_ic, valid_rankic, test_ic, test_rankic, seconds))


def train_master(ds, test_year, seed, results_dir,
                 epochs=20, lr=3e-4, patience=5, amp=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = amp and torch.cuda.is_available()

    xs_tr, xm_tr, y_tr, d_tr, _ = ds["train"]
    xs_va, xm_va, y_va, d_va, _ = ds["valid"]
    xs_te, xm_te, y_te, d_te, c_te = ds["test"]
    tr_groups = date_groups(d_tr)

    model = MASTER().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_rankic = -np.inf
    best_epoch = 0
    best_state = None
    best_valid_ic = np.nan
    patience_count = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for gi in np.random.permutation(len(tr_groups)):
            g = tr_groups[gi]
            xs = torch.as_tensor(xs_tr[g], device=device)
            xm = torch.as_tensor(xm_tr[g], device=device)
            yb = torch.as_tensor(y_tr[g], dtype=torch.float32, device=device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = loss_fn(model(xs, xm), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(g)
            n += len(g)

        pred_va = _predict_by_date(
            model, xs_va, xm_va, y_va, d_va, device, use_amp)
        valid_ic, valid_rankic = daily_ic(pred_va, y_va, d_va)
        print("fold=%d epoch=%d train_loss=%.4f valid IC=%.4f RankIC=%.4f"
              % (test_year, epoch, total / n, valid_ic, valid_rankic))

        if valid_rankic > best_rankic:
            best_rankic = valid_rankic
            best_valid_ic = valid_ic
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= patience:
            print("early stopping at epoch %d" % epoch)
            break

    if best_state is None:
        raise ValueError("MASTER未产生有效valid RankIC")
    model.load_state_dict(best_state)
    pred_te = _predict_by_date(
        model, xs_te, xm_te, y_te, d_te, device, use_amp)
    test_ic, test_rankic = daily_ic(pred_te, y_te, d_te)
    seconds = time.time() - t0

    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(best_state, os.path.join(
        checkpoint_dir, "master_fold%d_s%d.pt" % (test_year, seed)))
    _save_prediction(results_dir, "master", test_year, seed,
                     pred_te, y_te, d_te, c_te)
    row = _base_row("master", test_year, seed, seconds,
                    best_valid_ic, best_rankic, test_ic, test_rankic, best_epoch)
    _save_metrics(results_dir, row)
    print("=== master fold=%d seed=%d best_epoch=%d: valid IC=%.4f "
          "RankIC=%.4f, test IC=%.4f RankIC=%.4f, %.0fs ==="
          % (test_year, seed, best_epoch, best_valid_ic, best_rankic,
             test_ic, test_rankic, seconds))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["random_forest", "master"])
    p.add_argument("--test-year", required=True, type=int, choices=TEST_YEARS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", default="results_g8_rolling")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    ds = build_rolling_dataset(args.test_year)
    if args.model == "random_forest":
        train_random_forest(ds, args.test_year, args.seed, args.results_dir)
    else:
        train_master(ds, args.test_year, args.seed, args.results_dir,
                     args.epochs, args.lr, args.patience, not args.no_amp)


if __name__ == "__main__":
    main()
