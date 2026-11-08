# -*- coding: utf-8 -*-
"""G9 聚宽锁箱策略：上传 holdings_*.json 后直接回测。"""

from jqdata import *
import datetime
import json


# 文件上传在聚宽研究目录 /03_ml_prediction_return/ 下。
# 切换模型时只替换文件名：holdings_master.json / holdings_random_forest.json。
HOLDINGS_FILE = "03_ml_prediction_return/holdings_master.json"


def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_slippage(PriceRelatedSlippage(0.001))
    log.set_level("order", "error")
    g.holdings = load_holdings()
    run_daily(rebalance, time="open", reference_security="000300.XSHG")


def load_holdings():
    content = read_file(HOLDINGS_FILE)
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    data = json.loads(content)
    log.info("已载入 %s，%d 个调仓日" % (HOLDINGS_FILE, len(data)))
    return data


def before_trading_start(context):
    # A股印花税：2023-08-28起由千分之一减半。
    tax = 0.0005 if context.current_dt.date() >= datetime.date(2023, 8, 28) else 0.001
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=tax,
        open_commission=0.0003,
        close_commission=0.0003,
        close_today_commission=0,
        min_commission=5,
    ), type="stock")


def rebalance(context):
    date = context.current_dt.strftime("%Y-%m-%d")
    target_list = g.holdings.get(date)
    if target_list is None:
        return

    targets = set(target_list)
    current = get_current_data()
    held = set(context.portfolio.positions.keys())

    # 先卖出非目标股；停牌和跌停时保留，不构造虚假成交。
    for code in sorted(held - targets):
        item = current[code]
        if not item.paused and item.last_price > item.low_limit:
            order_target_value(code, 0)

    if not targets:
        return
    target_value = context.portfolio.total_value / len(targets)
    for code in sorted(targets):
        item = current[code]
        if item.paused or item.last_price >= item.high_limit:
            continue
        order_target_value(code, target_value)

    log.info("%s %s 目标持仓 %d 只" % (date, HOLDINGS_FILE, len(targets)))
