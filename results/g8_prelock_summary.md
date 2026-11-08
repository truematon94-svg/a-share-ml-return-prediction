# G8锁箱前分析摘要

本文件由 `python src/g8_prelock_analysis.py` 从既有预测复算；没有重新训练。
RankICIR均为 `日均RankIC / 日度标准差 × sqrt(252)`。配对检验为日度RankIC差的Newey–West检验（lag=5）。

## 主模型

| 模型 | seeds | RankIC | 年化RankICIR |
|---|---:|---:|---:|
| random_forest | 3 | 0.0535 | 5.250 |
| lightgbm | 3 | 0.0479 | 5.056 |
| master | 3 | 0.0457 | 5.009 |
| xgboost | 3 | 0.0444 | 4.684 |
| ridge | 1 | 0.0415 | 4.663 |
| transformer | 3 | 0.0410 | 4.115 |
| mlp | 3 | 0.0372 | 4.148 |
| ols | 1 | 0.0272 | 4.108 |

## 预先约定的主比较

| 比较（A-B） | 日均差 | NW t | p值 |
|---|---:|---:|---:|
| ridge - ols | 0.0143 | 3.67 | 0.0002 |
| random_forest - ridge | 0.0120 | 1.09 | 0.2753 |
| transformer - mlp | 0.0038 | 0.73 | 0.4674 |
| master - transformer | 0.0047 | 0.87 | 0.3862 |
| master - random_forest | -0.0078 | -0.65 | 0.5136 |

> OLS注意：当前 `pred_ols_s0.parquet` 由病态法方程直接 `solve` 生成，
> 复算RankIC为0.0272；稳定 `lstsq` 单次运行记录为0.0399。
> 因此Ridge−OLS配对暂作数值诊断，不用于G9模型选择。

## MASTER消融

| 比较 | 日均差 | NW t | p值 |
|---|---:|---:|---:|
| master_minus_no_gate | -0.0008 | -0.14 | 0.8870 |
| master_minus_no_stock | 0.0067 | 1.11 | 0.2690 |
| master_minus_f0_only | 0.0097 | 1.89 | 0.0590 |
| master_minus_f1_only | -0.0026 | -0.37 | 0.7121 |

## 市场状态（滚动MASTER-RF）

| 状态 | 天数 | RankIC差 | NW t |
|---|---:|---:|---:|
| down_high_vol | 153 | 0.0047 | 0.23 |
| down_low_vol | 207 | 0.0182 | 1.05 |
| up_high_vol | 82 | -0.0078 | -0.32 |
| up_low_vol | 281 | 0.0339 | 2.47 |

## 组合稳健性口径

- 主规则仍为Top-30、5日调仓/持有、基础往返成本0.35%；稳健性结果不用于重新挑选主规则。
- Top-K仅检查10/30/50；成本仅检查0/1/2倍；备选持有期固定为10日。
- 10日结果是同一5日预测信号的执行持有期敏感性，不是重新训练的10日标签模型。
- 本地回测为现有可执行代理；最终锁箱另用聚宽核验逐笔手续费、涨跌停和延迟成交。

完整数字见 `g8_portfolio_robustness.csv`。
