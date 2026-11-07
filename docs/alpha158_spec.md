# Alpha158 特征正本（G2 重写对照）

> 来源：Qlib 0.9.8 `qlib/contrib/data/handler.py` 的 `Alpha158DL.get_feature_config()`。
> 用途：pandas 重写 158 个特征时，以此为准，并用 Qlib 抽样校验一致性。

## 结构：158 = 9 + 4 + 145

| 块 | 数量 | 说明 |
|---|---|---|
| kbar | 9 | 当日 K 线形态（实体/影线/位置） |
| price | 4 | open/high/low/vwap ÷ close |
| rolling | 145 | 29 个算子 × 5 窗口 [5,10,20,30,60] |

## 底层字段（6 个）

open, high, low, close, volume, vwap

- vwap 由我们数据里的 `money / volume` 派生（成交额 ÷ 成交量，日频即 VWAP）。

## 一、kbar（9 个）

| 名称 | 公式 |
|---|---|
| KMID | ($close-$open)/$open |
| KLEN | ($high-$low)/$open |
| KMID2 | ($close-$open)/($high-$low+1e-12) |
| KUP | ($high-Greater($open,$close))/$open |
| KUP2 | ($high-Greater($open,$close))/($high-$low+1e-12) |
| KLOW | (Less($open,$close)-$low)/$open |
| KLOW2 | (Less($open,$close)-$low)/($high-$low+1e-12) |
| KSFT | (2*$close-$high-$low)/$open |
| KSFT2 | (2*$close-$high-$low)/($high-$low+1e-12) |

## 二、price（4 个，窗口只有 [0]）

OPEN0 = $open/$close，HIGH0 = $high/$close，LOW0 = $low/$close，VWAP0 = $vwap/$close

## 三、rolling（29 算子 × 窗口 d ∈ {5,10,20,30,60}）

| # | 名称 | 公式（d=窗口） |
|---|---|---|
| 1 | ROC | Ref($close,d)/$close |
| 2 | MA | Mean($close,d)/$close |
| 3 | STD | Std($close,d)/$close |
| 4 | BETA | Slope($close,d)/$close |
| 5 | RSQR | Rsquare($close,d) |
| 6 | RESI | Resi($close,d)/$close |
| 7 | MAX | Max($high,d)/$close |
| 8 | MIN | Min($low,d)/$close |
| 9 | QTLU | Quantile($close,d,0.8)/$close |
| 10 | QTLD | Quantile($close,d,0.2)/$close |
| 11 | RANK | Rank($close,d) |
| 12 | RSV | ($close-Min($low,d))/(Max($high,d)-Min($low,d)+1e-12) |
| 13 | IMAX | IdxMax($high,d)/d |
| 14 | IMIN | IdxMin($low,d)/d |
| 15 | IMXD | (IdxMax($high,d)-IdxMin($low,d))/d |
| 16 | CORR | Corr($close, Log($volume+1), d) |
| 17 | CORD | Corr($close/Ref($close,1), Log($volume/Ref($volume,1)+1), d) |
| 18 | CNTP | Mean($close>Ref($close,1), d) |
| 19 | CNTN | Mean($close<Ref($close,1), d) |
| 20 | CNTD | Mean($close>Ref($close,1),d)-Mean($close<Ref($close,1),d) |
| 21 | SUMP | Sum(Greater($close-Ref($close,1),0),d)/(Sum(Abs($close-Ref($close,1)),d)+1e-12) |
| 22 | SUMN | Sum(Greater(Ref($close,1)-$close,0),d)/(Sum(Abs($close-Ref($close,1)),d)+1e-12) |
| 23 | SUMD | (Sum(Greater($close-Ref($close,1),0),d)-Sum(Greater(Ref($close,1)-$close,0),d))/(Sum(Abs($close-Ref($close,1)),d)+1e-12) |
| 24 | VMA | Mean($volume,d)/($volume+1e-12) |
| 25 | VSTD | Std($volume,d)/($volume+1e-12) |
| 26 | WVMA | Std(Abs($close/Ref($close,1)-1)*$volume,d)/(Mean(Abs($close/Ref($close,1)-1)*$volume,d)+1e-12) |
| 27 | VSUMP | Sum(Greater($volume-Ref($volume,1),0),d)/(Sum(Abs($volume-Ref($volume,1)),d)+1e-12) |
| 28 | VSUMN | Sum(Greater(Ref($volume,1)-$volume,0),d)/(Sum(Abs($volume-Ref($volume,1)),d)+1e-12) |
| 29 | VSUMD | (Sum(Greater($volume-Ref($volume,1),0),d)-Sum(Greater(Ref($volume,1)-$volume,0),d))/(Sum(Abs($volume-Ref($volume,1)),d)+1e-12) |

## 算子 → pandas 映射

| Qlib 算子 | pandas/numpy 等价 |
|---|---|
| Ref(x, n) | x.shift(n)（n>0=过去，n<0=未来） |
| Mean(x, d) | x.rolling(d).mean() |
| Std(x, d) | x.rolling(d).std()（样本 or 总体，校验时确认） |
| Max(x, d) / Min(x, d) | x.rolling(d).max() / .min() |
| Sum(x, d) | x.rolling(d).sum() |
| Corr(x, y, d) | x.rolling(d).corr(y) |
| Quantile(x, d, q) | x.rolling(d).quantile(q) |
| Greater(a,b) / Less(a,b) | np.maximum(a,b) / np.minimum(a,b) |
| Log / Abs | np.log / np.abs |
| Slope(x, d) | 滚动线性回归斜率（自写） |
| Rsquare(x, d) | 滚动线性回归 R²（自写） |
| Resi(x, d) | 滚动线性回归残差（自写） |
| Rank(x, d) | 窗口内当前值分位（自写） |
| IdxMax(x, d) / IdxMin(x, d) | 距窗口内最高/最低点的天数（自写） |

## 精确语义（从 Qlib 0.9.8 源码核实，重写以此为准）

1. 所有滚动算子：`x.rolling(N, min_periods=1)`（第 1 天就开始算，不是前 N-1 天留 NaN）。
2. `Std` = 样本标准差（pandas 默认 ddof=1）。
3. `Rank` = `x.rolling(N, min_periods=1).rank(pct=True)`（当前值在窗口内分位，Qlib 也是直接调 pandas）。
4. `IdxMax/IdxMin` = `rolling(N, min_periods=1).apply(lambda x: np.argmax(x)+1, raw=True)`（1 起始位置，最新=窗口）。
5. `Quantile` = `rolling(N, min_periods=1).quantile(q)`（pandas 默认线性插值）。
6. `Corr` = `rolling(N, min_periods=1).corr()`，且任一序列滚动 std≈0（<2e-05）处置 NaN。
7. `Greater/Less` = `np.maximum / np.minimum`。
8. `Slope/Rsquare/Resi` = 对位置 x=[1,2,…,window]（最新值 x=window）做线性回归：
   - 窗口内 NaN 值剔除、但其余值仍按原位置编号（非连续）；
   - 有效点数 N<2 时结果为 NaN；
   - slope = (N·Σxy − Σx·Σy) / (N·Σx² − (Σx)²)；
   - rsquare = corr(x,y)²，且 y 的滚动 std≈0 时置 NaN；
   - resi = 最新值 − (slope·window + intercept)，intercept = ȳ − slope·x̄。
   - 等价简化：若窗口内无 NaN，x=[1..N] 连续，resi = y_last − ȳ − slope·(N−1)/2。

## 注意：label 不在 Alpha158 里

Qlib 默认 label 是 `Ref($close,-2)/Ref($close,-1)-1`（2 日），MASTER 论文用 5 日
`Ref($close,-5)/Ref($close,-1)-1`。label 属 G3，此处只用 158 特征，不用 Qlib 默认 label。
