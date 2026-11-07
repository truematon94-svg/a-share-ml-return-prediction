# -*- coding: utf-8 -*-
"""
云端神经网络训练：MLP / Transformer / MASTER。

统一口径：seed=0、损失 MSE、按日期批（batch=同日截面）、valid RankIC 选最佳 checkpoint、
early stopping patience=5、AMP 混合精度、test 只在最佳 checkpoint 后算一次。

用法：
  python -m cloud.train --model mlp --epochs 20 --seed 0
  python -m cloud.train --model mlp --smoke            # 冒烟：少量日期 1 epoch，不写正式产物
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from cloud.loader import build_dataset
from cloud.models import MODELS

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


def _predict_by_date(model, Xs, Xm, y, dates, device, use_amp):
    model.eval()
    preds = np.empty(len(y), dtype=np.float32)
    with torch.no_grad():
        for g in date_groups(dates):
            xs = torch.as_tensor(Xs[g], device=device)
            xm = torch.as_tensor(Xm[g], device=device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds[g] = model(xs, xm).cpu().numpy()
    return preds


def save_metrics(model_name, seed, best_epoch, best_train_loss, best_valid_ic, best_valid_rankic,
                 test_ic, test_rankic, train_seconds):
    row = pd.DataFrame([{
        "model": model_name, "seed": seed, "best_epoch": best_epoch,
        "train_loss": best_train_loss, "valid_ic": best_valid_ic, "valid_rankic": best_valid_rankic,
        "test_ic": test_ic, "test_rankic": test_rankic, "train_seconds": train_seconds,
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


def train_model(model_name, epochs=20, lr=3e-4, seed=0, smoke=False, amp=True, patience=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = amp and torch.cuda.is_available()
    print("device =", device, "| model =", model_name, "| seed =", seed, "| AMP =", use_amp)

    ds = build_dataset()
    Xs_tr, Xm_tr, y_tr, d_tr, c_tr = ds["train"]
    Xs_va, Xm_va, y_va, d_va, c_va = ds["valid"]
    Xs_te, Xm_te, y_te, d_te, c_te = ds["test"]

    tr_groups = date_groups(d_tr)
    if smoke:
        tr_groups = tr_groups[:30]
        epochs = 1

    model = MODELS[model_name]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_rankic, best_epoch, best_state = -np.inf, 0, None
    best_valid_ic = best_valid_rankic = best_train_loss = np.nan
    patience_cnt = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for gi in np.random.permutation(len(tr_groups)):   # 打乱日期组顺序，同日股票仍在一起
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
        print("epoch %d  train_loss=%.4f  valid IC=%.4f RankIC=%.4f" % (epoch, train_loss, ic_v, rk_v))

        if rk_v > best_rankic:
            best_rankic, best_epoch = rk_v, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_valid_ic, best_valid_rankic = ic_v, rk_v
            best_train_loss = train_loss
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= patience:
            print("early stopping at epoch %d" % epoch)
            break

    if np.isnan(best_rankic):
        raise ValueError("所有 valid RankIC 均为 NaN，请检查数据/标签/标准化")

    model.load_state_dict(best_state)
    if not smoke:
        pt = _predict_by_date(model, Xs_te, Xm_te, y_te, d_te, device, use_amp)
        ic_t, rk_t = daily_ic(pt, y_te, d_te)
    else:
        ic_t = rk_t = np.nan

    train_seconds = time.time() - t0

    if not smoke:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        torch.save(best_state, os.path.join(CHECKPOINT_DIR, "%s_s%d.pt" % (model_name, seed)))
        save_metrics(model_name, seed, best_epoch, best_train_loss, best_valid_ic, best_valid_rankic,
                     ic_t, rk_t, train_seconds)
        save_prediction(model_name, seed, pt, y_te, d_te, c_te)

    print("=== %s seed=%d: best_epoch=%d, best_valid RankIC=%.4f, test IC=%.4f RankIC=%.4f, %.0fs ==="
          % (model_name, seed, best_epoch, best_rankic, ic_t, rk_t, train_seconds))
    return {"best_epoch": best_epoch, "valid_rankic": best_rankic,
            "test_ic": ic_t, "test_rankic": rk_t}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlp", choices=["mlp", "transformer", "master"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    train_model(args.model, args.epochs, args.lr, args.seed, args.smoke, not args.no_amp)


if __name__ == "__main__":
    main()
