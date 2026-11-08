# -*- coding: utf-8 -*-
"""Generate the four publication-ready summary figures from frozen G4-G9 tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#17202a"
MUTED = "#667085"
GRID = "#dfe3e8"
BLUE = "#2563eb"
GREEN = "#059669"
ORANGE = "#ea580c"
RED = "#dc2626"

DISPLAY = {
    "ols": "OLS",
    "ridge": "Ridge",
    "mlp": "MLP",
    "transformer": "Transformer",
    "master": "MASTER",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "master_no_gate": "去市场门控",
    "master_no_stock": "去股票关联",
}


def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_model_comparison():
    df = pd.read_csv(RESULTS / "g8_main_rankic_summary.csv").sort_values("rankic_mean")
    colors = [GREEN if "forest" in m or "gbm" in m or "boost" in m else BLUE for m in df.model]
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    y = np.arange(len(df))
    ax.barh(y, df.rankic_mean, xerr=df.rankic_seed_std.fillna(0), color=colors, alpha=0.92, capsize=3)
    ax.set_yticks(y, [DISPLAY.get(m, m) for m in df.model])
    for yi, (_, row) in zip(y, df.iterrows()):
        ax.text(row.rankic_mean + 0.0008, yi, f"{row.rankic_mean:.4f}", va="center", fontsize=9, color=INK)
    style(ax, "八模型固定切分样本外 RankIC", xlabel="Test RankIC（误差棒为种子间标准差）")
    ax.set_xlim(0, df.rankic_mean.max() + 0.012)
    save(fig, "01_model_rankic.png")


def plot_ablation():
    df = pd.read_csv(RESULTS / "g8_ablation_summary.csv")
    order = ["master", "master_no_gate", "master_no_stock"]
    df = df.set_index("model").loc[order].reset_index()
    labels = ["完整 MASTER", "去市场门控", "去股票关联"]
    colors = [BLUE, ORANGE, RED]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    bars = ax.bar(labels, df.test_rankic_mean, yerr=df.test_rankic_std, color=colors, capsize=4, width=0.62)
    for bar, value in zip(bars, df.test_rankic_mean):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0015, f"{value:.4f}", ha="center", fontsize=10)
    style(ax, "MASTER 结构消融：股票关联模块提供方向性增量", ylabel="Test RankIC")
    ax.set_ylim(0, max(df.test_rankic_mean + df.test_rankic_std) + 0.018)
    save(fig, "02_master_ablation.png")


def plot_rolling():
    df = pd.read_csv(RESULTS / "g8_rolling_summary.csv")
    df = df[df.test_year.astype(str).isin(["2020", "2021", "2022"])].copy()
    df.test_year = df.test_year.astype(int)
    pivot = df.pivot(index="test_year", columns="model", values="test_rankic")
    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor="white")
    ax.plot(pivot.index, pivot.master, marker="o", linewidth=2.3, color=BLUE, label="MASTER")
    ax.plot(pivot.index, pivot.random_forest, marker="o", linewidth=2.3, color=GREEN, label="Random Forest")
    for model, color in [("master", BLUE), ("random_forest", GREEN)]:
        for year, value in pivot[model].items():
            ax.text(year, value + 0.002, f"{value:.3f}", ha="center", fontsize=9, color=color)
    ax.set_xticks(pivot.index)
    ax.legend(frameon=False)
    style(ax, "扩展窗口滚动样本外 RankIC", xlabel="测试年度", ylabel="Test RankIC")
    save(fig, "03_rolling_rankic.png")


def plot_lockbox():
    df = pd.read_csv(RESULTS / "g9_lockbox" / "joinquant_summary.csv")
    labels = ["MASTER", "Random Forest"]
    return_metrics = [
        ("annualized_return", "年化收益", 100),
        ("relative_excess_return", "累计超额收益", 100),
    ]
    ratio_metrics = [
        ("sharpe", "夏普比率", 1),
        ("information_ratio", "信息比率", 1),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), facecolor="white")
    width = 0.34
    for ax, metrics, title, ylabel in [
        (axes[0], return_metrics, "收益表现", "%"),
        (axes[1], ratio_metrics, "风险调整后表现", "比率"),
    ]:
        x = np.arange(len(metrics))
        for i, (model, color) in enumerate(zip(["master", "random_forest"], [BLUE, GREEN])):
            row = df[df.model == model].iloc[0]
            values = [row[key] * scale for key, _, scale in metrics]
            bars = ax.bar(x + (i - 0.5) * width, values, width, color=color, label=labels[i])
            offset = max(values) * 0.035
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x, [label for _, label, _ in metrics])
        style(ax, title, ylabel=ylabel)
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("2023—2025 冻结锁箱：聚宽可交易回测", x=0.07, ha="left", fontsize=14, fontweight="bold", color=INK)
    save(fig, "04_lockbox_performance.png")


def main():
    plot_model_comparison()
    plot_ablation()
    plot_rolling()
    plot_lockbox()
    for path in sorted(OUT.glob("*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
