# -*- coding: utf-8 -*-
"""
Alpha158 特征计算（pandas 重写，公式照搬 Qlib 0.9.8）。

158 = 9 kbar + 4 price + 145 rolling（29 算子 × 窗口 [5,10,20,30,60]）。

关键复刻点：
  - 所有滚动算子用 rolling(N, min_periods=1)；
  - Std 是样本标准差（ddof=1）；
  - Rank = rolling.rank(pct=True)；IdxMax/IdxMin = apply(argmax+1)；
  - Slope/Rsquare/Resi = 对位置 x=[1..window]（最新=window）做线性回归，NaN 剔除但保留位置；
    Rsquare 在滚动 std≈0 处置 NaN。

只保留 6 个底层字段 open/high/low/close/volume/vwap（vwap = money/volume 由调用方传入）。
"""

import numpy as np
import pandas as pd

WINDOWS = [5, 10, 20, 30, 60]
EPS = 1e-12


# ---------- 线性回归滚动算子（复刻 Qlib Cython 的 online 算法语义） ----------

def _make_slope(window):
    def f(x):
        n_avail = len(x)
        pos = np.arange(window - n_avail + 1, window + 1, dtype=float)  # 最新=window
        m = ~np.isnan(x)
        if m.sum() < 2:
            return np.nan
        xp = pos[m]; yp = x[m]
        n = len(xp)
        den = n * (xp * xp).sum() - xp.sum() ** 2
        num = n * (xp * yp).sum() - xp.sum() * yp.sum()
        return num / den
    return f


def _make_rsquare(window):
    def f(x):
        n_avail = len(x)
        pos = np.arange(window - n_avail + 1, window + 1, dtype=float)
        m = ~np.isnan(x)
        if m.sum() < 2:
            return np.nan
        xp = pos[m]; yp = x[m]
        n = len(xp)
        num = n * (xp * yp).sum() - xp.sum() * yp.sum()
        den = (n * (xp * xp).sum() - xp.sum() ** 2) * (n * (yp * yp).sum() - yp.sum() ** 2)
        if den <= 0:
            return np.nan
        r = num / np.sqrt(den)
        return r * r
    return f


def _make_resi(window):
    def f(x):
        n_avail = len(x)
        pos = np.arange(window - n_avail + 1, window + 1, dtype=float)
        m = ~np.isnan(x)
        if m.sum() < 2 or np.isnan(x[-1]):
            return np.nan
        xp = pos[m]; yp = x[m]
        n = len(xp)
        slope = (n * (xp * yp).sum() - xp.sum() * yp.sum()) / (n * (xp * xp).sum() - xp.sum() ** 2)
        interp = yp.mean() - slope * xp.mean()
        return x[-1] - (slope * window + interp)
    return f


def _rsquare(close, d):
    rsq = close.rolling(d, min_periods=1).apply(_make_rsquare(d), raw=True)
    rsq[np.isclose(close.rolling(d, min_periods=1).std(), 0, atol=2e-05)] = np.nan
    return rsq


# ---------- 158 特征 ----------

def compute_features(open_, high, low, close, volume, vwap):
    """6 个字段 DataFrame（index=date, columns=code）→ 158 特征 DataFrame（columns=MultiIndex(feature, code)）。"""
    feats = {}
    hl = high - low
    up_edge = np.maximum(open_, close)
    lo_edge = np.minimum(open_, close)

    # kbar (9)
    feats["KMID"] = (close - open_) / open_
    feats["KLEN"] = hl / open_
    feats["KMID2"] = (close - open_) / (hl + EPS)
    feats["KUP"] = (high - up_edge) / open_
    feats["KUP2"] = (high - up_edge) / (hl + EPS)
    feats["KLOW"] = (lo_edge - low) / open_
    feats["KLOW2"] = (lo_edge - low) / (hl + EPS)
    feats["KSFT"] = (2 * close - high - low) / open_
    feats["KSFT2"] = (2 * close - high - low) / (hl + EPS)

    # price (4)
    feats["OPEN0"] = open_ / close
    feats["HIGH0"] = high / close
    feats["LOW0"] = low / close
    feats["VWAP0"] = vwap / close

    # rolling (145)
    ret = close / close.shift(1)          # close[t]/close[t-1]
    logvol = np.log(volume + 1)
    logretvol = np.log(volume / volume.shift(1) + 1)
    chg = close - close.shift(1)
    abschg = chg.abs()
    vchg = volume - volume.shift(1)
    absvchg = vchg.abs()
    wv = (ret - 1).abs() * volume          # |close/close.shift(1)-1| * volume

    for d in WINDOWS:
        r = lambda s: s.rolling(d, min_periods=1)
        feats["ROC%d" % d] = close.shift(d) / close
        feats["MA%d" % d] = r(close).mean() / close
        feats["STD%d" % d] = r(close).std() / close
        feats["BETA%d" % d] = r(close).apply(_make_slope(d), raw=True) / close
        feats["RSQR%d" % d] = _rsquare(close, d)
        feats["RESI%d" % d] = r(close).apply(_make_resi(d), raw=True) / close
        feats["MAX%d" % d] = r(high).max() / close
        feats["MIN%d" % d] = r(low).min() / close
        feats["QTLU%d" % d] = r(close).quantile(0.8) / close
        feats["QTLD%d" % d] = r(close).quantile(0.2) / close
        feats["RANK%d" % d] = r(close).rank(pct=True)

        minlow = r(low).min()
        maxhigh = r(high).max()
        feats["RSV%d" % d] = (close - minlow) / (maxhigh - minlow + EPS)

        imax = r(high).apply(lambda x: np.argmax(x) + 1, raw=True)
        imin = r(low).apply(lambda x: np.argmin(x) + 1, raw=True)
        feats["IMAX%d" % d] = imax / d
        feats["IMIN%d" % d] = imin / d
        feats["IMXD%d" % d] = (imax - imin) / d

        feats["CORR%d" % d] = r(close).corr(logvol)
        feats["CORD%d" % d] = r(ret).corr(logretvol)

        up = (close > close.shift(1)).astype(float)
        dn = (close < close.shift(1)).astype(float)
        feats["CNTP%d" % d] = r(up).mean()
        feats["CNTN%d" % d] = r(dn).mean()
        feats["CNTD%d" % d] = feats["CNTP%d" % d] - feats["CNTN%d" % d]

        feats["SUMP%d" % d] = r(np.maximum(chg, 0)).sum() / (r(abschg).sum() + EPS)
        feats["SUMN%d" % d] = r(np.maximum(-chg, 0)).sum() / (r(abschg).sum() + EPS)
        feats["SUMD%d" % d] = feats["SUMP%d" % d] - feats["SUMN%d" % d]

        feats["VMA%d" % d] = r(volume).mean() / (volume + EPS)
        feats["VSTD%d" % d] = r(volume).std() / (volume + EPS)
        feats["WVMA%d" % d] = r(wv).std() / (r(wv).mean() + EPS)

        feats["VSUMP%d" % d] = r(np.maximum(vchg, 0)).sum() / (r(absvchg).sum() + EPS)
        feats["VSUMN%d" % d] = r(np.maximum(-vchg, 0)).sum() / (r(absvchg).sum() + EPS)
        feats["VSUMD%d" % d] = feats["VSUMP%d" % d] - feats["VSUMN%d" % d]

    out = pd.concat(feats, axis=1)
    out.columns.names = ["feature", "code"]
    return out


# ---------- 批量跑全样本（内存受限：按代码分块） ----------

def load_panel(years=None):
    """加载 base_YYYY.parquet，拼接成 date × (field, code)，并派生 vwap。"""
    if years is None:
        years = range(2008, 2023)
    frames = [pd.read_parquet("03_机器学习股票收益预测/data/processed/base_%d.parquet" % y) for y in years]
    panel = pd.concat(frames, axis=0)
    fields = {}
    for f in ["open", "high", "low", "close", "volume"]:
        fields[f] = panel[f].astype("float32")
    fields["vwap"] = (panel["money"] / panel["volume"]).astype("float32")
    del panel
    return fields


def run_batches(batch_size=50, years=None):
    import os
    fields = load_panel(years)
    codes = list(fields["close"].columns)
    out_dir = "data/processed/features"
    os.makedirs(out_dir, exist_ok=True)
    n = len(codes)
    for i in range(0, n, batch_size):
        batch = codes[i:i + batch_size]
        b = {k: v[batch] for k, v in fields.items()}
        feats = compute_features(b["open"], b["high"], b["low"], b["close"], b["volume"], b["vwap"])
        feats = feats.astype("float32")
        feats.to_parquet(os.path.join(out_dir, "feat_%04d_%04d.parquet" % (i, i + len(batch) - 1)))
        print("  batch %d-%d 完成 (%d 只)" % (i, i + len(batch) - 1, len(batch)), flush=True)
    print("全部完成，共 %d 只，输出到 %s" % (n, out_dir))


if __name__ == "__main__":
    run_batches()
