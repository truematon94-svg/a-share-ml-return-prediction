# -*- coding: utf-8 -*-
"""G8 特征消融：固定 MASTER，比较 F0、F1 与已有 F0+F1。

现有缓存 ``X_stock`` 的排列由 loader.py 冻结为：
  - 前 158 维：F0 / Alpha158
  - 后 12 维：F1 / 项目二量价因子

本脚本只训练缺失的 ``master_f0`` 与 ``master_f1``。完整 F0+F1 的
``master`` 结果直接复用 cloud/results_seed0/1/2，不重复训练。
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cloud.loader import build_dataset
from cloud.models import MASTER
from cloud.train import _predict_by_date, daily_ic, date_groups


FEATURE_GROUPS = {
    "f0": (slice(0, 158), 158),
    "f1": (slice(158, 170), 12),
}


def save_metrics(results_dir, row):
    path = os.path.join(results_dir, "metrics.csv")
    new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        keep = ~((old["model"] == row["model"]) & (old["seed"] == row["seed"]))
        new = pd.concat([old.loc[keep], new], ignore_index=True)
        new = new.sort_values(["model", "seed"])
    new.to_csv(path, index=False)


def save_prediction(results_dir, model_name, seed, pred, y_true, dates, codes):
    df = pd.DataFrame({
        "date": dates,
        "code": codes.astype(str),
        "y_true": y_true,
        "prediction": pred,
        "model": model_name,
        "seed": seed,
    })
    path = os.path.join(results_dir, "pred_%s_s%d.parquet" % (model_name, seed))
    df.to_parquet(path, index=False)


def train_feature_group(feature_group, results_dir="results_g8_features",
                        epochs=20, lr=3e-4, seed=0, amp=True, patience=5):
    feature_slice, n_stock = FEATURE_GROUPS[feature_group]
    model_name = "master_%s" % feature_group

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = amp and torch.cuda.is_available()
    print("device =", device, "| model =", model_name,
          "| features =", n_stock, "| seed =", seed, "| AMP =", use_amp)

    ds = build_dataset()
    Xs_tr_all, Xm_tr, y_tr, d_tr, _ = ds["train"]
    Xs_va_all, Xm_va, y_va, d_va, _ = ds["valid"]
    Xs_te_all, Xm_te, y_te, d_te, c_te = ds["test"]

    if Xs_tr_all.shape[-1] != 170:
        raise ValueError("预期缓存含 170 维 F0+F1，实际为 %d" % Xs_tr_all.shape[-1])

    Xs_tr = Xs_tr_all[..., feature_slice]
    Xs_va = Xs_va_all[..., feature_slice]
    Xs_te = Xs_te_all[..., feature_slice]
    tr_groups = date_groups(d_tr)

    model = MASTER(n_stock=n_stock).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_rankic = -np.inf
    best_epoch = 0
    best_state = None
    best_valid_ic = np.nan
    best_valid_rankic = np.nan
    best_train_loss = np.nan
    patience_cnt = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for gi in np.random.permutation(len(tr_groups)):
            g = tr_groups[gi]
            xs = torch.as_tensor(Xs_tr[g], device=device)
            xm = torch.as_tensor(Xm_tr[g], device=device)
            yb = torch.as_tensor(y_tr[g], dtype=torch.float32, device=device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = loss_fn(model(xs, xm), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(g)
            n += len(g)

        train_loss = total / n
        pv = _predict_by_date(model, Xs_va, Xm_va, y_va, d_va, device, use_amp)
        ic_v, rk_v = daily_ic(pv, y_va, d_va)
        print("epoch %d  train_loss=%.4f  valid IC=%.4f RankIC=%.4f"
              % (epoch, train_loss, ic_v, rk_v))

        if rk_v > best_rankic:
            best_rankic = rk_v
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_valid_ic = ic_v
            best_valid_rankic = rk_v
            best_train_loss = train_loss
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= patience:
            print("early stopping at epoch %d" % epoch)
            break

    if best_state is None or np.isnan(best_rankic):
        raise ValueError("所有 valid RankIC 均为 NaN，请检查数据、标签和标准化")

    model.load_state_dict(best_state)
    pt = _predict_by_date(model, Xs_te, Xm_te, y_te, d_te, device, use_amp)
    ic_t, rk_t = daily_ic(pt, y_te, d_te)
    train_seconds = time.time() - t0

    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(best_state, os.path.join(
        checkpoint_dir, "%s_s%d.pt" % (model_name, seed)))
    save_prediction(results_dir, model_name, seed, pt, y_te, d_te, c_te)
    save_metrics(results_dir, {
        "model": model_name,
        "feature_group": feature_group,
        "n_stock_features": n_stock,
        "seed": seed,
        "best_epoch": best_epoch,
        "train_loss": best_train_loss,
        "valid_ic": best_valid_ic,
        "valid_rankic": best_valid_rankic,
        "test_ic": ic_t,
        "test_rankic": rk_t,
        "train_seconds": train_seconds,
    })

    print("=== %s seed=%d: best_epoch=%d, best_valid RankIC=%.4f, "
          "test IC=%.4f RankIC=%.4f, %.0fs ==="
          % (model_name, seed, best_epoch, best_rankic,
             ic_t, rk_t, train_seconds))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-group", required=True, choices=sorted(FEATURE_GROUPS))
    p.add_argument("--results-dir", default="results_g8_features")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    train_feature_group(args.feature_group, args.results_dir, args.epochs,
                        args.lr, args.seed, not args.no_amp, args.patience)


if __name__ == "__main__":
    main()
