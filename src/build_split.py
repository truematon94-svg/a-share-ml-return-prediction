# -*- coding: utf-8 -*-
"""
G3 时间切分。

切分 A：论文对齐（已冻结）
  train: 2008-01-01 ~ 2020-03-31
  valid: 2020-04-01 ~ 2020-06-30
  test:  2020-07-01 ~ 2022-12-31

切分 B：滚动样本外（扩展训练 + 年度测试窗）
  train: 从 2008 扩展；valid: 1 年；test: 1 年；段间 5 日 embargo。
  test 年份 2016 ~ 2022，共 7 折。

输出：
  data/processed/split_paper.csv   —— 每个交易日 -> train/valid/test
  data/processed/split_rolling.csv —— 每折 train/valid/test 起止日期
"""

import pandas as pd

PAPER = {
    "train": ("2008-01-01", "2020-03-31"),
    "valid": ("2020-04-01", "2020-06-30"),
    "test": ("2020-07-01", "2022-12-31"),
}

EMBARGO = 5  # 标签持有期（5 交易日），段间隔离


def trading_dates():
    frames = [pd.read_parquet("data/processed/base_%d.parquet" % y) for y in range(2008, 2023)]
    return pd.concat(frames, axis=0).index


def paper_split(dates):
    dates = pd.DatetimeIndex(dates)
    s = pd.Series("train", index=dates, dtype=str)
    for name, (a, b) in PAPER.items():
        if name != "train":
            s[(dates >= a) & (dates <= b)] = name
    return s


def rolling_folds(dates, test_years=range(2016, 2023)):
    dates = pd.DatetimeIndex(dates)
    folds = []
    for Y in test_years:
        test = dates[(dates >= "%d-01-01" % Y) & (dates <= "%d-12-31" % Y)]
        valid = dates[(dates >= "%d-01-01" % (Y - 1)) & (dates <= "%d-12-31" % (Y - 1))]
        train = dates[(dates >= "2008-01-01") & (dates < valid[0])]
        # embargo：train 与 valid 末尾各去掉 5 个交易日
        train = train[: len(train) - EMBARGO]
        valid = valid[: len(valid) - EMBARGO]
        folds.append((train[0], train[-1], valid[0], valid[-1], test[0], test[-1]))
    return folds


def main():
    dates = trading_dates()

    paper = paper_split(dates)
    paper.to_csv("data/processed/split_paper.csv", header=["split"])
    print("=== 论文对齐切分（天数）===")
    print(paper.value_counts().to_string())

    folds = rolling_folds(dates)
    rows = []
    print("\n=== 滚动切分（%d 折）===" % len(folds))
    print("折 | train | valid | test")
    for i, (ta, tb, va, vb, te, tf) in enumerate(folds):
        rows.append({"fold": i, "train_start": ta.date(), "train_end": tb.date(),
                     "valid_start": va.date(), "valid_end": vb.date(),
                     "test_start": te.date(), "test_end": tf.date()})
        print("  %d: %s~%s | %s~%s | %s~%s" % (i, ta.date(), tb.date(), va.date(), vb.date(), te.date(), tf.date()))
    pd.DataFrame(rows).to_csv("data/processed/split_rolling.csv", index=False)


if __name__ == "__main__":
    main()
