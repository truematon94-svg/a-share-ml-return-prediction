# -*- coding: utf-8 -*-
"""
F1 因子与 Alpha158 的相关性可视化。

读入 data/processed/f1_alpha158_corr.csv，输出两张图：
  1. results/f1_maxcorr_bar.png   —— 每个 F1 因子的最大 |相关性|（条形图）
  2. results/f1_corr_heatmap.png  —— 12 F1 因子 × 相关性最高的 15 个 Alpha158 特征（热力图）

配色遵循发散语义：相关 = 蓝(正)/红(负)/灰(零)。
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 品牌中性色（参考 dataviz 规范）
BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def main():
    corr = pd.read_csv("data/processed/f1_alpha158_corr.csv", index_col=0)
    factors = list(corr.index)
    maxcorr = corr.abs().max(axis=1).sort_values(ascending=True)

    # ---- 图 1：条形图（每个因子最大 |相关性|）----
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.barh(maxcorr.index, maxcorr.values, color=BLUE, height=0.62)
    # 高相关（>=0.7）用更深的强调色
    for b, v in zip(bars, maxcorr.values):
        if v >= 0.7:
            b.set_color("#184f95")
    ax.axvline(0.7, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(0.7, len(maxcorr) - 0.3, "  0.7 参考线", color=MUTED, fontsize=9, va="top")
    # 直接标注数值
    for b, v in zip(bars, maxcorr.values):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2, "%.2f" % v,
                va="center", fontsize=9, color=INK)
    ax.set_xlabel("与 Alpha158 的最大 |相关性|", color=INK)
    ax.set_title("F1 因子与 Alpha158 的重叠程度（越低越有增量）", color=INK, fontsize=12, loc="left")
    ax.set_xlim(0, 0.95)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("results/f1_maxcorr_bar.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- 图 2：热力图（12 × top15 特征）----
    top15 = corr.abs().max(axis=0).sort_values(ascending=False).head(15).index.tolist()
    sub = corr[top15].loc[factors]
    # 颜色：发散（蓝正红负灰零）
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f0efec")
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(sub.values, cmap=cmap, vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(top15)))
    ax.set_xticklabels(top15, rotation=45, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels(factors, fontsize=9, color=INK)
    # 数值标注（仅 |corr|>=0.4，避免过密）
    for i in range(len(factors)):
        for j in range(len(top15)):
            v = sub.iloc[i, j]
            if abs(v) >= 0.4:
                ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=7,
                        color="#ffffff" if abs(v) > 0.45 else "#0b0b0b")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("相关系数", color=INK)
    cbar.ax.tick_params(colors=MUTED)
    ax.set_title("F1 因子 × 相关性最高的 15 个 Alpha158 特征（蓝=正相关，红=负相关）",
                 color=INK, fontsize=12, loc="left")
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/f1_corr_heatmap.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    print("图已保存: results/f1_maxcorr_bar.png, results/f1_corr_heatmap.png")
    print("Top15 特征:", top15)


if __name__ == "__main__":
    main()
