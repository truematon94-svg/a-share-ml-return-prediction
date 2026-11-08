# 模型结果记录（G4 · 3 种子）

> 6 个模型（+ 2 个线性基线）样本外预测结果。3 个随机种子（seed 0/1/2），均值±标准差。

## 统一口径

- 股票池：`eligible_core_csi300`；特征 1423 维（8天×170 + 63 市场状态）
- 标准化：RobustZScoreNorm（train-only）；切分：论文对齐
- 评价：逐日截面 IC / RankIC（Spearman），对天平均
- 训练样本：875,697

## 最终排名（test RankIC，均值±标准差）

| 排名 | 模型 | test RankIC | 类型 |
|---|---|---|---|
| 1 | **RandomForest** | **0.0535 ± 0.0026** | 树（非线性） |
| 2 | **LightGBM** | **0.0479 ± 0.0058** | 树（非线性） |
| 3 | **MASTER** | **0.0457 ± 0.0034** | NN（门控+股票间聚合） |
| 4 | XGBoost | 0.0444 ± 0.0116 | 树 |
| 5 | Ridge | 0.0415（单种子，确定性） | 线性+L2 |
| 6 | Transformer | 0.0410 ± 0.0074 | NN |
| 7 | OLS | 0.0399（单种子，确定性） | 线性基线 |
| 8 | MLP | 0.0372 ± 0.0015 | NN |

## 结论（3 种子坐实）

1. **非线性有效，且赢家是树模型**：RF(0.0535)、LightGBM(0.0479) 稳定超线性（OLS 0.0399 / Ridge 0.0415）。
2. **MASTER 是 NN 最好**（0.0457），市场门控+股票间聚合相对 Transformer(0.0410) 有 +0.0047 增量，但不如 RF/LightGBM。
3. **普通 NN 失败**：MLP(0.0372)、Transformer(0.0410) 不超线性基线——H2a/H2b 不成立（负结果）。
4. **XGBoost 方差大**（±0.0116）：seed0 的 0.0561 是运气，3 种子平均只有 0.0444，结论要谨慎。

## 诚实提醒

- OLS/Ridge 是确定性模型（无随机种子），故无 std；其余 6 模型 3 种子。
- 差异显著性未做配对检验（可后续加 block bootstrap / 稳健标准误）。
- 数据缓存 `v2`，checkpoints 在 `cloud/results_seed*/checkpoints/`。

## 详细记录

`cloud/results_seed0/1/2/metrics.csv`（含 best_epoch、valid/test IC/RankIC、训练时长）。
