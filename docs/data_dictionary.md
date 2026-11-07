# 数据字典（G0 初稿）

> 记录每个字段的含义、时点、来源。时点是防未来函数的第一道墙。

## 1. 股票行情（日线）

| 字段 | 含义 | 时点 | 来源 |
|---|---|---|---|
| open/high/low/close | 开/高/低/收 | t 日 | get_price(1d) |
| volume | 成交量（股） | t 日 | get_price(1d) |
| money | 成交额（元） | t 日 | get_price(1d) |
| factor | 复权因子 | t 日 | get_price(1d) |
| high_limit/low_limit | 涨停价/跌停价 | t 日 | get_price(1d) |
| paused | 是否停牌 | t 日 | get_price(1d) |

复权口径：前复权 `fq='pre'`（收益口径等价后复权）。

## 2. 股票状态

| 字段 | 含义 | 时点 | 来源 |
|---|---|---|---|
| is_st | 是否 ST | t 日 | get_extras |
| market_cap / circulating_market_cap | 总/流通市值 | t 日 | get_fundamentals |
| industry | 申万一级行业 | t 日 | get_industry |
| start_date | 上市日期 | — | get_security_info |

## 3. 股票池

| 字段 | 含义 | 时点 | 来源 |
|---|---|---|---|
| 沪深300 历史成分 | 主股票池 | date 当日 | get_index_stocks('000300.XSHG', date) |
| 中证800 历史成分 | 扩展股票池 | date 当日 | get_index_stocks('000906.XSHG', date) |

禁止用当前成分回填历史。

## 4. 指数行情（市场状态）

| 字段 | 含义 | 来源 |
|---|---|---|
| 沪深300/中证500/中证800 日线 | 市场状态 63 维的原料 | get_price(指数代码, 1d) |

市场状态 63 维 = 3 指数 × 21 维，21 = 当前价格(1) + 5窗口(5,10,20,30,60)×(价格均值、价格标准差、成交量均值、成交量标准差)。

> 口径更正（G2 查证论文原文）：论文写的是 "market index trading **volume**"（成交量，volume 字段），
> 非成交额（money）。故用 volume，不用 money。

## 5. 标签

`label = close[t+5] / close[t+1] - 1`（t+1 收盘 → t+5 收盘）。

- 信号在 t 日收盘后产生，标签从 t+1 日收盘开始（对齐论文，非 t 日收盘）。
- 论文对齐实验用此口径；独立扩展实验可换 t+1 开盘（更可执行）。
