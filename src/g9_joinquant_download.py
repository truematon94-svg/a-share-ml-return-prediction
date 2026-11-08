# -*- coding: utf-8 -*-
"""G9锁箱数据：聚宽研究环境增量下载脚本。

在聚宽「研究」Notebook中整段粘贴运行。输出与2008—2022旧数据分开，
不会覆盖原有CSV。

下载范围：
  - 2022：只作为2023年特征预热期；
  - 2023—2025：正式锁箱期；
  - 主股票池：历史沪深300月度成分；
  - 市场状态：沪深300、中证500、中证800。

推荐分步运行：
  download_universe()
  download_index()
  download_listing()
  download_stock_year(2022)
  download_stock_year(2023)
  download_stock_year(2024)
  download_stock_year(2025)
  audit_outputs()
"""

from jqdata import *

import gc
import os
import time

import pandas as pd


LOCKBOX_START = "2023-01-01"
LOCKBOX_END = "2025-12-31"
WARMUP_START = "2022-01-01"

PRIMARY = "000300.XSHG"
MARKET_INDICES = ["000300.XSHG", "000905.XSHG", "000906.XSHG"]
PRICE_FIELDS = [
    "open", "high", "low", "close", "volume", "money",
    "factor", "high_limit", "low_limit", "paused",
]

STOCK_BATCH = 80
SLEEP_SECONDS = 1.5
PREFIX = "g9_"


def _retry_get_price(codes, start, end, fields, tries=4):
    for attempt in range(tries):
        try:
            return get_price(
                codes,
                start_date=start,
                end_date=end,
                frequency="daily",
                fields=fields,
                fq="pre",
                panel=False,
            )
        except Exception as exc:
            if attempt == tries - 1:
                raise
            wait = 3 * (attempt + 1)
            print("get_price失败，%d秒后重试 %d/%d：%s" %
                  (wait, attempt + 1, tries, exc), flush=True)
            time.sleep(wait)


def _normalise_price_frame(raw):
    """把聚宽panel=False结果统一成 date/code/field/value 长表。"""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["date", "code", "field", "value"])
    frame = raw.copy()
    if "code" not in frame.columns:
        frame = frame.reset_index()
    date_col = next(
        (column for column in ("time", "date", "datetime")
         if column in frame.columns),
        None,
    )
    if date_col is None:
        frame = frame.reset_index()
        date_col = next(
            (column for column in ("time", "date", "datetime")
             if column in frame.columns),
            frame.columns[0],
        )
    if "code" not in frame.columns:
        raise ValueError("get_price返回结果中找不到code列：%s" % list(frame.columns))
    value_columns = [field for field in PRICE_FIELDS if field in frame.columns]
    long = frame.melt(
        id_vars=[date_col, "code"],
        value_vars=value_columns,
        var_name="field",
        value_name="value",
    )
    long = long.rename(columns={date_col: "date"})
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    return long[["date", "code", "field", "value"]]


def _first_trade_day_each_month(start, end):
    days = pd.DatetimeIndex(get_trade_days(start_date=start, end_date=end))
    if len(days) == 0:
        raise ValueError("指定区间没有交易日：%s—%s" % (start, end))
    series = pd.Series(days, index=days)
    return list(series.groupby(days.to_period("M")).first())


def download_universe():
    """下载2023—2025每月首个交易日的历史沪深300成分。"""
    output = PREFIX + "universe_2023_2025.csv"
    if os.path.exists(output):
        print("已存在，跳过：%s" % output)
        return pd.read_csv(output, dtype=str)

    rows = []
    month_days = _first_trade_day_each_month(LOCKBOX_START, LOCKBOX_END)
    for number, date in enumerate(month_days, 1):
        date_string = pd.Timestamp(date).strftime("%Y-%m-%d")
        codes = get_index_stocks(PRIMARY, date=date_string)
        rows.extend({"date": date_string, "universe": "csi300", "code": code}
                    for code in codes)
        print("历史成分 %d/%d：%s，%d只" %
              (number, len(month_days), date_string, len(codes)), flush=True)

    result = pd.DataFrame(rows).drop_duplicates()
    result.to_csv(output, index=False)
    print("完成：%s，%d行，%d个月，去重后%d只股票" %
          (output, len(result), result["date"].nunique(), result["code"].nunique()))
    return result


def _g9_codes():
    universe_file = PREFIX + "universe_2023_2025.csv"
    if not os.path.exists(universe_file):
        download_universe()
    universe = pd.read_csv(universe_file, dtype=str)
    return sorted(universe["code"].dropna().unique())


def download_index():
    """下载2022预热期至2025年末的三条市场指数。"""
    output = PREFIX + "index_daily_2022_2025.csv"
    if os.path.exists(output):
        print("已存在，跳过：%s" % output)
        return
    rows = []
    for code in MARKET_INDICES:
        raw = _retry_get_price(
            code, WARMUP_START, LOCKBOX_END, ["close", "volume", "money"])
        frame = raw.reset_index()
        date_col = next(
            (column for column in ("time", "date", "datetime", "index")
             if column in frame.columns),
            frame.columns[0],
        )
        frame = frame.rename(columns={date_col: "date"})
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        frame["index_code"] = code
        rows.append(frame[["date", "index_code", "close", "volume", "money"]])
        print("指数完成：%s，%d行" % (code, len(frame)), flush=True)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(output, index=False)
    print("完成：%s，%d行" % (output, len(result)))


def download_listing():
    """下载G9股票的上市日期；静态信息单独保存。"""
    output = PREFIX + "status_listing.csv"
    if os.path.exists(output):
        print("已存在，跳过：%s" % output)
        return
    codes = _g9_codes()
    rows = []
    for number, code in enumerate(codes, 1):
        info = get_security_info(code)
        rows.append({
            "code": code,
            "start_date": str(info.start_date) if info is not None else None,
            "end_date": str(info.end_date) if info is not None else None,
        })
        if number % 100 == 0 or number == len(codes):
            print("上市信息 %d/%d" % (number, len(codes)), flush=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print("完成：%s，%d只" % (output, len(rows)))


def download_stock_year(year):
    """按年下载G9股票行情；文件已存在则跳过，可断点续跑。"""
    if year not in (2022, 2023, 2024, 2025):
        raise ValueError("year只能是2022/2023/2024/2025")
    output = PREFIX + "stock_daily_%d.csv" % year
    if os.path.exists(output):
        print("已存在，跳过：%s" % output)
        return

    codes = _g9_codes()
    start = "%d-01-01" % year
    end = "%d-12-31" % year
    first = True
    row_count = 0
    started = time.time()

    for offset in range(0, len(codes), STOCK_BATCH):
        batch = codes[offset:offset + STOCK_BATCH]
        raw = _retry_get_price(batch, start, end, PRICE_FIELDS)
        long = _normalise_price_frame(raw)
        if len(long):
            long.to_csv(output, mode="a", header=first, index=False)
            first = False
            row_count += len(long)
        print("%d：股票 %d/%d，累计%d行" %
              (year, min(offset + len(batch), len(codes)), len(codes), row_count),
              flush=True)
        del raw, long
        gc.collect()
        time.sleep(SLEEP_SECONDS)

    if first:
        raise RuntimeError("%d年没有写出任何数据" % year)
    print("完成：%s，%d行，耗时%.1f分钟" %
          (output, row_count, (time.time() - started) / 60.0))


def audit_outputs():
    """轻量审计输出文件；不加载整份年度CSV到内存。"""
    expected = [
        PREFIX + "universe_2023_2025.csv",
        PREFIX + "index_daily_2022_2025.csv",
        PREFIX + "status_listing.csv",
    ] + [PREFIX + "stock_daily_%d.csv" % year for year in (2022, 2023, 2024, 2025)]

    rows = []
    for filename in expected:
        exists = os.path.exists(filename)
        size_mb = os.path.getsize(filename) / 2**20 if exists else 0.0
        rows.append({"file": filename, "exists": exists, "size_mb": round(size_mb, 2)})
    audit = pd.DataFrame(rows)
    print(audit.to_string(index=False))

    missing = audit.loc[~audit["exists"], "file"].tolist()
    if missing:
        print("\n尚缺文件：%s" % missing)
        return audit

    universe = pd.read_csv(PREFIX + "universe_2023_2025.csv", dtype=str)
    index = pd.read_csv(PREFIX + "index_daily_2022_2025.csv", dtype=str)
    print("\n成分月份：%d，股票去重：%d，日期：%s—%s" %
          (universe["date"].str[:7].nunique(), universe["code"].nunique(),
           universe["date"].min(), universe["date"].max()))
    print("指数代码：%d，日期：%s—%s" %
          (index["index_code"].nunique(), index["date"].min(), index["date"].max()))

    for year in (2022, 2023, 2024, 2025):
        filename = PREFIX + "stock_daily_%d.csv" % year
        header = pd.read_csv(filename, nrows=5)
        expected_columns = {"date", "code", "field", "value"}
        if set(header.columns) != expected_columns:
            raise ValueError("%s列异常：%s" % (filename, list(header.columns)))
        print("%d年列结构正确，文件%.1fMB" %
              (year, os.path.getsize(filename) / 2**20))

    audit.to_csv(PREFIX + "download_audit.csv", index=False)
    print("\nG9聚宽下载文件齐全。请将全部 g9_*.csv 下载到本地。")
    return audit


def main():
    download_universe()
    download_index()
    download_listing()
    for year in (2022, 2023, 2024, 2025):
        download_stock_year(year)
    audit_outputs()


if __name__ == "__main__":
    main()
