# -*- coding: utf-8 -*-
"""G9正式锁箱训练入口：仅RF与MASTER、seed=0。"""

import argparse
import os

from cloud import train_g8_rolling as trainer
from cloud.g9_rolling_loader import TEST_YEARS, build_rolling_dataset, fold_spec


# G8训练函数内部取模块级fold_spec，显式替换为G9冻结年度定义。
trainer.fold_spec = fold_spec
trainer.TEST_YEARS = TEST_YEARS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["random_forest", "master"])
    parser.add_argument("--test-year", required=True, type=int, choices=TEST_YEARS)
    parser.add_argument("--results-dir", default="data/g9_lock_box/results")
    args = parser.parse_args()
    seed = 0
    os.makedirs(args.results_dir, exist_ok=True)
    dataset = build_rolling_dataset(args.test_year)
    if args.model == "random_forest":
        trainer.train_random_forest(dataset, args.test_year, seed, args.results_dir)
    else:
        trainer.train_master(dataset, args.test_year, seed, args.results_dir,
                             epochs=20, lr=3e-4, patience=5, amp=True)


if __name__ == "__main__":
    main()
