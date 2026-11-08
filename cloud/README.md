# 云端训练、G8稳健性与G9冻结锁箱

本目录保存项目三的云端训练入口。所有脚本默认从项目根目录以模块方式运行，旧训练入口和冻结结果不因后续实验而改写。

## 环境

```bash
pip install torch pandas numpy scipy pyarrow lightgbm xgboost scikit-learn joblib
```

正式运行环境已验证为 PyTorch + CUDA，神经网络使用GPU；RandomForest主要使用CPU。

## 数据

云端保持以下相对路径：

- `data/processed/features/feat_*.parquet`：36个Alpha158分片；
- `data/processed/f1_factors.parquet`：项目二量价因子；
- `data/processed/market_status.parquet`：市场状态；
- `data/processed/label_z.parquet`：未来5日截面标准化标签；
- `data/processed/eligible_core_csi300.parquet`：历史可用股票池；
- `data/processed/scaler.parquet`、`split_paper.csv`：主实验标准化器和冻结切分。

原始大数据与缓存不纳入Git；预测、指标、配置和必要checkpoint按实验目录保存。

## 入口与冻结结果

| 入口 | 用途 | 结果目录 |
|---|---|---|
| `train.py` | MLP / Transformer / MASTER主实验 | `results_seed0/1/2/` |
| `train_tree.py` | OLS / Ridge / RF / LightGBM / XGBoost | `results_seed0/1/2/` |
| `train_g8.py` + `models_g8.py` | MASTER去门控、去股票间聚合 | `results_g8/` |
| `train_g8_features.py` | F0-only / F1-only特征消融 | `results_g8_features/` |
| `train_g8_rolling.py` + `rolling_loader.py` | RF/MASTER三折扩展窗口 | `results_g8_rolling/` |
| `train_g9_lockbox.py` + `g9_rolling_loader.py` | 2023—2025冻结锁箱年度预测 | `data/g9_lock_box/results/` |

`rolling_loader.py`使用批量PyArrow转换加速读取，但不改变字段、列序、NaN、标准化公式、fold或模型口径。每个fold独立拟合train-only scaler，并在训练段和验证段末尾保留5交易日embargo。

## 已完成实验

- 八模型统一样本外比较；
- 随机模型 seed 0/1/2；
- MASTER两个核心结构消融；
- Alpha158、项目二因子以及二者组合的特征消融；
- RandomForest与MASTER的2020/2021/2022三折扩展窗口训练。
- RandomForest与MASTER的2023/2024/2025一次性冻结锁箱训练与预测。

详细数值和诚实结论见 `../docs/model_results.md`；运行决策见 `../WORKLOG.md`。

滚动统计可在项目根目录复算：

```bash
python src/g8_rolling_analysis.py
python src/g8_rolling_analysis.py --write
```

## 复现纪律

1. 测试集不得参与调参、早停或特征选择；
2. 主实验、消融和滚动实验分别写入独立目录；
3. 不覆盖旧预测和checkpoint；
4. 论文对齐预测与真实可执行回测分开报告；
5. 不因某次测试结果不理想而更换标签、股票池、成交价或信息集。
6. G9完成后不得根据2023—2025结果重新选择模型、种子、Top-K、持有期或成本参数。
