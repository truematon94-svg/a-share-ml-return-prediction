# -*- coding: utf-8 -*-
"""G8：MASTER 两个核心消融的独立训练入口。

该脚本不修改或覆盖 G5 的模型、训练脚本与结果。默认输出到 ``results_g8``。

用法：
  python -m cloud.train_g8 --model master_no_gate --seed 0
  python -m cloud.train_g8 --model master_no_stock --seed 0
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cloud.loader import build_dataset
from cloud.models_g8 import G8_MODELS
from cloud.train import _predict_by_date, daily_ic, date_groups


def save_metrics(results_dir, model_name, seed, best_epoch, best_train_loss,
                 best_valid_ic, best_valid_rankic, test_ic, test_rankic,
                 train_seconds):
    path = os.path.join(results_dir, "metrics.csv")
    row = pd.DataFrame([{
        "model": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "train_loss": best_train_loss,
        "valid_ic": best_valid_ic,
        "valid_rankic": best_valid_rankic,
        "test_ic": test_ic,
        "test_rankic": test_rankic,
        "train_seconds": train_seconds,
    }])
    if os.path.exists(path):
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, index=False)


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


def train_ablation(model_name, results_dir="results_g8", epochs=20,
                   lr=3e-4, seed=0, amp=True, patience=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = amp and torch.cuda.is_available()
    print("device =", device, "| model =", model_name,
          "| seed =", seed, "| AMP =", use_amp)

    ds = build_dataset()
    Xs_tr, Xm_tr, y_tr, d_tr, _ = ds["train"]
    Xs_va, Xm_va, y_va, d_va, _ = ds["valid"]
    Xs_te, Xm_te, y_te, d_te, c_te = ds["test"]
    tr_groups = date_groups(d_tr)

    model = G8_MODELS[model_name]().to(device)
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
    save_metrics(results_dir, model_name, seed, best_epoch, best_train_loss,
                 best_valid_ic, best_valid_rankic, ic_t, rk_t, train_seconds)
    save_prediction(results_dir, model_name, seed, pt, y_te, d_te, c_te)

    print("=== %s seed=%d: best_epoch=%d, best_valid RankIC=%.4f, "
          "test IC=%.4f RankIC=%.4f, %.0fs ==="
          % (model_name, seed, best_epoch, best_rankic,
             ic_t, rk_t, train_seconds))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(G8_MODELS))
    p.add_argument("--results-dir", default="results_g8")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    train_ablation(args.model, args.results_dir, args.epochs, args.lr,
                   args.seed, not args.no_amp, args.patience)


if __name__ == "__main__":
    main()
