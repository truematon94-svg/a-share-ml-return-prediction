# 数据结构清单（云端交接用）

> 给云端 GPU 机器用。列清每个文件是什么、shape、怎么读。云端代码在 `cloud/` 目录。

## 需要上传到云端的数据（data/processed/）

| 文件 | 内容 | shape | 读取 |
|---|---|---|---|
| `scaler.parquet` | 标准化参数（center/scale） | 233 行 × (group, feature, center, scale) | `pd.read_parquet` |
| `features/feat_*.parquet`（36 个） | Alpha158 特征（158 个） | 3649 天 × (158特征 × 50代码) MultiIndex | `pq.read_table` |
| `f1_factors.parquet` | 项目二因子（12 个） | 3649 × (12 × 1765) MultiIndex | `pq.read_table` |
| `market_status.parquet` | 市场状态（63 维） | 3649 × 63 | `pd.read_parquet` |
| `label_z.parquet` | 标签（截面 Z-score） | 3649 × 1765（date × code） | `pd.read_parquet` |
| `eligible_core_csi300.parquet` | 股票池过滤 mask（核心口径） | 3649 × 1765（bool） | `pd.read_parquet` |
| `eligible_paper_csi300.parquet` | 股票池过滤 mask（论文对齐口径） | 3649 × 1765（bool） | `pd.read_parquet` |
| `split_paper.csv` | 论文对齐切分 | 每行 = date → train/valid/test | `pd.read_csv(index_col=0)` |

## 关键口径（云端代码要遵守）

1. **标准化**：`(x − center)/scale` → clip[−3,3] → NaN 填 0。center/scale 已 fit（train 段），直接读 `scaler.parquet` 用。
2. **8 天回看**：每个样本 = 过去 8 天（t−7..t）的 170 特征 + 当天 63 市场状态 = 1423 维。
3. **标签**：`label_z` 已经是截面 Z-score 后的 5 日收益。
4. **股票池**：主实验用 `eligible_core_csi300`（核心口径）；论文对齐用 `eligible_paper_csi300`。
5. **切分**：`split_paper.csv`（train 2008-2020.3 / valid 2020.4-6 / test 2020.7-2022）。
6. **标签 NaN**：停牌日 label 为 NaN，训练/评估前剔除（DropnaLabel）。

## 特征构成（170 = 158 + 12）

- Alpha158 的 158 个特征：见 `docs/alpha158_spec.md`。
- F1 的 12 个因子：return_1m/3m/6m/12m、wgt_return_1m/3m/6m/12m、exp_wgt_return_1m/3m/6m/12m。

## 代码顺序（feature 维）

Alpha158 文件的列是 `(feature, code)`，feature-major（先 KMID 的 50 个代码，再 KLEN 的 50 个，…）。F1 同理 `(factor, code)`。所以拼成 170 特征时：前 158 列 = Alpha158，后 12 列 = F1。
