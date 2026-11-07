# -*- coding: utf-8 -*-
"""
G1 数据下载脚本（聚宽研究环境运行，整段粘贴到 notebook 跑）。

设计要点（针对聚宽 1G 内存限制）：
  ③ 股票日行情：按代码块分批 get_price（每次 120 只）；panel=False 返回的长表直接
    melt 成 date/code/field/value；每年独立一个 CSV，不在聚宽内存里合并 15 年。
  ④ 状态字段：市值按 500 只一批；snapshot 用交易日（非固定 06-30/12-31）；
    行业每只取一次（申万一级几乎不变）。

断点续跑：
  - index_daily.csv / universe.csv 已存在就跳过；
  - stock_daily_<年>.csv 已存在就跳过该年；
  - 状态字段各 CSV 已存在就跳过对应部分。

运行：整段粘贴后调用 download_stock() / download_status()（或 main() 全量）。
"""

import gc
import os
import time

import pandas as pd

# ===== 参数（与 configs/data.yaml 保持一致）=====
SAMPLE_START = "2008-01-01"
SAMPLE_END = "2022-12-31"
PRIMARY = "000300.XSHG"      # 沪深300（主股票池）
EXTENSION = "000906.XSHG"    # 中证800（扩展股票池）
MARKET_INDICES = ["000300.XSHG", "000905.XSHG", "000906.XSHG"]  # 沪深300/中证500/中证800
PRICE_FIELDS = ["open", "high", "low", "close", "volume", "money",
                "factor", "high_limit", "low_limit", "paused"]

STOCK_BATCH = 120            # 每次 get_price 的股票数（控制内存）
MV_BATCH = 500               # 市值查询每批股票数


def _retry_get_price(codes, start, end, fields, tries=3):
    """get_price 加简单重试，规避偶发接口错误。"""
    for k in range(tries):
        try:
            return get_price(codes, start, end, 'daily', fields, fq='pre', panel=False)
        except Exception as e:
            if k == tries - 1:
                raise
            print("      get_price 重试 %d/%d：%s" % (k + 1, tries, e))
            time.sleep(2 * (k + 1))


# ===== ① 指数行情 =====
def download_index():
    if os.path.exists("index_daily.csv"):
        print("① 指数行情已存在，跳过")
        return
    print("① 下载指数行情...")
    rows = []
    for idx in MARKET_INDICES:
        df = get_price(idx, SAMPLE_START, SAMPLE_END, 'daily',
                       ['close', 'volume', 'money'], fq='pre', panel=False)
        df = df.sort_index()
        df.index = pd.to_datetime(df.index)
        d = df.reset_index().rename(columns={'index': 'date'})
        d['index_code'] = idx
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv("index_daily.csv", index=False)
    print("   指数行情 %d 行" % len(out))


# ===== ② 历史成分（每月）=====
def download_universe():
    if os.path.exists("universe.csv"):
        print("② 历史成分已存在，跳过")
        return
    print("② 下载历史成分（每月）...")
    bench = get_price(PRIMARY, SAMPLE_START, SAMPLE_END, 'daily',
                      ['close'], fq='pre', panel=False)
    bench_close = bench['close'].sort_index()
    bench_close.index = pd.to_datetime(bench_close.index)
    month_first = bench_close.groupby(bench_close.index.strftime('%Y-%m')).head(1).index

    rows = []
    for d in month_first:
        ds = d.strftime('%Y-%m-%d')
        for name, code in [('csi300', PRIMARY), ('csi800', EXTENSION)]:
            for s in get_index_stocks(code, ds):
                rows.append({'date': ds, 'universe': name, 'code': s})
    out = pd.DataFrame(rows)
    out.to_csv("universe.csv", index=False)
    print("   成分记录 %d 行，覆盖 %d 个月" % (len(out), len(month_first)))


# ===== ③ 股票日行情（按年×代码块，逐块落盘）=====
def _to_long(raw, date_col, code_col='code'):
    """把 get_price(panel=False) 的长表 melt 成 date/code/field/value。"""
    value_vars = [f for f in PRICE_FIELDS if f in raw.columns]
    long = raw.melt(id_vars=[date_col, code_col], value_vars=value_vars,
                    var_name='field', value_name='value')
    long = long.rename(columns={date_col: 'date'})
    return long[['date', 'code', 'field', 'value']]


def download_stock(start_year=2008, end_year=2023):
    print("③ 下载股票日行情（%d-%d，按年×代码块，逐块落盘）..." % (start_year, end_year - 1))
    all_codes = sorted(pd.read_csv("universe.csv")['code'].unique())
    print("   成分股去重后 %d 只，每批 %d 只" % (len(all_codes), STOCK_BATCH))

    for y in range(start_year, end_year):
        out_file = "stock_daily_%d.csv" % y
        if os.path.exists(out_file):
            print("   %d 年已存在，跳过" % y)
            continue
        start, end = "%d-01-01" % y, "%d-12-31" % y
        t0 = time.time()
        first = True
        for i in range(0, len(all_codes), STOCK_BATCH):
            batch = all_codes[i:i + STOCK_BATCH]
            raw = _retry_get_price(batch, start, end, PRICE_FIELDS)
            if raw is None or len(raw) == 0:
                continue
            # 统一成含 code 列 + 日期列的长表
            if 'code' not in raw.columns:
                continue
            date_col = next((c for c in ('time', 'date', 'datetime') if c in raw.columns), None)
            if date_col is None:
                raw = raw.reset_index()
                date_col = next((c for c in ('time', 'date', 'datetime') if c in raw.columns),
                                raw.columns[0])
            long = _to_long(raw, date_col)
            long.to_csv(out_file, mode='a', header=first, index=False)
            first = False
            del raw, long
            time.sleep(1.5)   # 批次间隔，避免触发聚宽接口限流（0.2s 太短会被限速）
        gc.collect()          # 每年结束强制回收，防止 notebook 内存只增不减
        print("   %d 年完成，%.1fs -> %s" % (y, time.time() - t0, out_file))

    # 注意：不在聚宽合并所有年份（15 年 2GB 会撑爆 1G 内存），
    # 每年一个 CSV 下载到本地后，再在本地合并。


# ===== ④ 状态字段 =====
def download_status():
    print("④ 下载状态字段...")
    all_codes = list(pd.read_csv("universe.csv")['code'].unique())

    # 上市日期（静态）
    if not os.path.exists("status_listing.csv"):
        listing = {s: get_security_info(s).start_date for s in all_codes}
        pd.DataFrame([{'code': c, 'start_date': str(d)} for c, d in listing.items()]
                     ).to_csv("status_listing.csv", index=False)

    # 交易日历（沪深300）：取每个半年末的最后一个交易日，避免 06-30/12-31 落在周末导致空数据
    bench = get_price(PRIMARY, SAMPLE_START, SAMPLE_END, 'daily', ['close'], fq='pre', panel=False)
    bench_close = bench['close'].sort_index()
    bench_close.index = pd.to_datetime(bench_close.index)
    cal = bench_close.index
    snap_dates = []
    for y in range(2008, 2023):
        for m in [6, 12]:
            month_end = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
            cand = cal[cal <= month_end]
            if len(cand):
                snap_dates.append(cand[-1].strftime('%Y-%m-%d'))

    # is_st（每年 6/12 月末的最后一个交易日）
    if not os.path.exists("status_st.csv"):
        st_rows = []
        for d in snap_dates:
            try:
                st = get_extras('is_st', all_codes, d, d, df=True)
                for c in all_codes:
                    if c in st.columns:
                        st_rows.append({'date': d, 'code': c, 'is_st': st[c].iloc[0]})
            except Exception:
                pass
        pd.DataFrame(st_rows).to_csv("status_st.csv", index=False)

    # 市值（每年 6/12 月末最后一个交易日；500 只一批；手动补 date 列——get_fundamentals 不返回日期）
    if not os.path.exists("status_marketcap.csv"):
        mv_rows = []
        for d in snap_dates:
            for i in range(0, len(all_codes), MV_BATCH):
                sub = all_codes[i:i + MV_BATCH]
                try:
                    mv = get_fundamentals(
                        query(valuation.code, valuation.market_cap,
                              valuation.circulating_market_cap)
                        .filter(valuation.code.in_(sub)), date=d)
                    if len(mv):
                        mv["date"] = d
                        mv_rows.append(mv)
                except Exception:
                    pass
        pd.concat(mv_rows, ignore_index=True).to_csv("status_marketcap.csv", index=False)

    # 行业：每只股票只取一次（申万一级几乎不变），用其首次进池日期。
    # 从"每年×每只 = 3 万次"降到 2 千次，避免又触发限流卡死。
    if not os.path.exists("status_industry.csv"):
        univ = pd.read_csv("universe.csv")
        first_date = univ.groupby('code')['date'].min().to_dict()
        ind_rows = []
        n = len(first_date)
        for i, (c, d) in enumerate(first_date.items(), 1):
            try:
                sw = get_industry(c, d).get('sw_l1', {}).get('industry_name')
                ind_rows.append({'date': d, 'code': c, 'industry_sw_l1': sw})
            except Exception:
                pass
            if i % 500 == 0:
                print("      行业 %d/%d" % (i, n))
            time.sleep(0.2)
        pd.DataFrame(ind_rows).to_csv("status_industry.csv", index=False)

    print("   状态字段已存：listing %d / st %d / marketcap %d / industry %d"
          % tuple(pd.read_csv(f).shape[0] if os.path.exists(f) else 0
                  for f in ["status_listing.csv", "status_st.csv",
                            "status_marketcap.csv", "status_industry.csv"]))


def main():
    download_index()
    download_universe()
    download_stock()
    download_status()
    print("G1 下载完成，把生成的 CSV 下载到本地 data/raw/ 即可。")


if __name__ == "__main__":
    main()
