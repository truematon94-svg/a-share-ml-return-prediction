# -*- coding: utf-8 -*-
"""8 模型 3 种子结果可视化：test RankIC 条形图(mean±std) + valid vs test 过拟合图。"""

import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 品牌中性色
INK = "#0b0b0b"; MUTED = "#898781"; GRID = "#e1e0d9"; SURFACE = "#fcfcfb"
COLOR = {"线性": "#898781", "树": "#1baf7a", "NN": "#2a78d6"}


DISPLAY = {"ols": "OLS", "ridge": "Ridge", "mlp": "MLP", "transformer": "Transformer",
           "master": "MASTER", "lightgbm": "LightGBM", "xgboost": "XGBoost",
           "random_forest": "RandomForest"}
TYPE = {"ols": "线性", "ridge": "线性", "lightgbm": "树", "xgboost": "树",
        "random_forest": "树", "mlp": "NN", "transformer": "NN", "master": "NN"}


def load():
    frames = [pd.read_csv(f) for f in sorted(glob.glob("cloud/results_seed*/metrics.csv"))]
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby("model").agg(
        rankic=("test_rankic", "mean"), rankic_std=("test_rankic", "std"),
        valid_rankic=("valid_rankic", "mean"),
    ).reset_index()
    # 线性基线（单种子，确定性，无 std）
    g = pd.concat([g, pd.DataFrame([
        {"model": "ols", "rankic": 0.0399, "rankic_std": np.nan, "valid_rankic": 0.0539},
        {"model": "ridge", "rankic": 0.0415, "rankic_std": np.nan, "valid_rankic": 0.0567},
    ])], ignore_index=True)
    g["type"] = g["model"].map(TYPE)
    g["label"] = g["model"].map(DISPLAY)
    return g.sort_values("rankic")


def plot_bar(g):
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(g))[::-1]
    colors = [COLOR[t] for t in g["type"]]
    ax.barh(y, g["rankic"], xerr=g["rankic_std"], color=colors, height=0.62,
            capsize=3, error_kw=dict(ecolor=MUTED, linewidth=1))
    for yi, (_, r) in zip(y, g.iterrows()):
        lab = "%.4f" % r["rankic"] if np.isnan(r["rankic_std"]) else "%.4f±%.4f" % (r["rankic"], r["rankic_std"])
        ax.text(r["rankic"] + 0.001, yi, lab, va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(g["label"], fontsize=10, color=INK)
    ax.set_xlabel("test RankIC（逐日截面 Spearman，均值）", color=INK)
    ax.set_title("8 模型样本外 RankIC 对比（3 种子，误差棒=标准差）", fontsize=12, color=INK, loc="left")
    ax.set_xlim(0.02, 0.065)
    ax.tick_params(colors=MUTED)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    # 图例：类型
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=COLOR[k], label=k) for k in ["线性", "树", "NN"]]
    ax.legend(handles=legend, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig("results/model_rankic_bar.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_valid_test(g):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for t in ["线性", "树", "NN"]:
        sub = g[g["type"] == t]
        ax.scatter(sub["valid_rankic"], sub["rankic"], color=COLOR[t], s=70, zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(r["label"], (r["valid_rankic"], r["rankic"]),
                        textcoords="offset points", xytext=(6, 3), fontsize=9, color=INK)
    lims = [0.03, 0.11]
    ax.plot(lims, lims, "--", color=MUTED, linewidth=1, label="y=x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("valid RankIC", color=INK)
    ax.set_ylabel("test RankIC", color=INK)
    ax.set_title("valid vs test RankIC", fontsize=12, color=INK, loc="left")
    ax.tick_params(colors=MUTED)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig("results/model_valid_vs_test.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    g = load()
    plot_bar(g)
    plot_valid_test(g)
    print("已生成：")
    print("  results/model_rankic_bar.png")
    print("  results/model_valid_vs_test.png")


if __name__ == "__main__":
    main()
