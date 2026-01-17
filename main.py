import akshare as ak
import pandas as pd
import numpy as np
from ta.trend import ADXIndicator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
import os
import time
import sys
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
import concurrent.futures
import random
import warnings

warnings.filterwarnings('ignore')

# --- 1. 环境与配置 ---
CONFIG = {
    "MIN_AMOUNT": 20000000,   # 最低成交额 2000万
    "MIN_PRICE": 2.5,         # 最低股价
    "MAX_WORKERS": 8,         # 线程数
    "DAYS_LOOKBACK": 200,     # 数据回溯
    "BLACKLIST_DAYS": 30      # 解禁预警天数
}

HISTORY_FILE = "stock_history_log.csv"
HOT_CONCEPTS = [] 
RESTRICTED_LIST = [] 
NORTHBOUND_SET = set() 
MARKET_ENV_TEXT = "?初始化..."

# --- 2. 市场全维情报 ---
def get_market_context():
    global HOT_CONCEPTS, RESTRICTED_LIST, NORTHBOUND_SET, MARKET_ENV_TEXT
    print("?? [1/4] 连接交易所数据中心 (全维扫描)...")
    try:
        next_month = (datetime.now() + timedelta(days=CONFIG["BLACKLIST_DAYS"])).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        df_res = ak.stock_restricted_release_queue_em()
        RESTRICTED_LIST = df_res[(df_res['解禁日期'] >= today) & (df_res['解禁日期'] <= next_month)]['股票代码'].astype(str).tolist()
    except: pass

    try:
        df = ak.stock_board_concept_name_em()
        HOT_CONCEPTS = df.sort_values(by="涨跌幅", ascending=False).head(15)["板块名称"].tolist()
    except: pass

    try:
        df_sh = ak.stock_hsgt_top_10_em(symbol="沪股通")
        df_sz = ak.stock_hsgt_top_10_em(symbol="深股通")
        if df_sh is not None: NORTHBOUND_SET.update(df_sh['代码'].astype(str).tolist())
        if df_sz is not None: NORTHBOUND_SET.update(df_sz['代码'].astype(str).tolist())
    except: pass
    
    try:
        sh = ak.stock_zh_index_daily(symbol="sh000001")
        curr = sh.iloc[-1]
        ma20 = sh['close'].rolling(20).mean().iloc[-1]
        pct = (curr['close'] - sh.iloc[-2]['close']) / sh.iloc[-2]['close'] * 100
        status = "???多头安全" if curr['close'] >= ma20 else "???空头趋势"
        if pct < -1.5: status = "??暴跌风险"
        MARKET_ENV_TEXT = f"上证: {curr['close']:.2f} ({pct:+.2f}%) | {status}"
    except: pass

def get_targets_robust():
    try:
        df = ak.stock_zh_a_spot_em()
        col_map = {"最新价": "price", "成交额": "amount", "代码": "code", "名称": "name", "换手率": "turnover", "市盈率-动态": "pe"}
        df.rename(columns=col_map, inplace=True)
        df = df[df["code"].str.startswith(("60", "00"))]
        df = df[~df['name'].str.contains('ST|退')]
        df = df[(df["price"] >= CONFIG["MIN_PRICE"]) & (df["amount"] > CONFIG["MIN_AMOUNT"])]
        df = df[~df["code"].isin(RESTRICTED_LIST)]
        return df.to_dict('records')
    except: return []

def get_data_with_retry(code, start_date):
    for _ in range(2):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq", timeout=5)
            if df is not None and not df.empty: return df
        except: time.sleep(0.1)
    return None

# --- 3. 核心判定函数 (含跳空缺口) ---
def analyze_kline_health(df_full):
    curr = df_full.iloc[-1]
    prev = df_full.iloc[-2]
    price_range = curr['最高'] - curr['最低']
    if price_range == 0: return "?极小波动", 0
    
    # 跳空缺口判定
    gap_signal = ""
    gap_score = 0
    if curr['最低'] > prev['最高']: gap_signal = "??向上跳空"; gap_score = 40
    elif curr['最高'] < prev['最低']: gap_signal = "??向下跳空"; gap_score = -40

    body_top = max(curr['开盘'], curr['收盘'])
    body_bottom = min(curr['开盘'], curr['收盘'])
    upper_ratio = (curr['最高'] - body_top) / price_range
    lower_ratio = (body_bottom - curr['最低']) / price_range
    
    status = "?普通震荡"; score = 0
    if (curr['收盘'] - curr['开盘']) / price_range > 0.6: status = "??实体强攻"; score = 25
    elif lower_ratio > 0.4: status = "???金针探底"; score = 20
    elif upper_ratio > 0.4: status = "??高位抛压"; score = -10

    if gap_signal: return f"{gap_signal}|{status}", score + gap_score
    return status, score

# --- 4. 选股逻辑核心 (MACD合并显示) ---
def process_stock_logic(df, stock_info):
    if len(df) < 60: return None
    code, name = stock_info['code'], stock_info['name']
    
    close = df["收盘"]
    # 均线
    df["MA5"] = close.rolling(5).mean(); df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean(); df["MA60"] = close.rolling(60).mean()
    df["BIAS20"] = (close - df["MA20"]) / df["MA20"] * 100
    
    # MACD计算
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_Bar"] = (df["DIF"] - df["DEA"]) * 2
    
    curr, prev = df.iloc[-1], df.iloc[-2]

    # --- MACD合并逻辑：金叉 + 动能状态 ---
    bar_trend = ""
    if curr["MACD_Bar"] > 0:
        bar_trend = "??红柱增长" if curr["MACD_Bar"] > prev["MACD_Bar"] else "??红柱缩短"
    else:
        bar_trend = "??绿柱缩短" if curr["MACD_Bar"] > prev["MACD_Bar"] else "??绿柱增长"
    
    is_macd_gold = (prev["DIF"] < prev["DEA"]) and (curr["DIF"] > curr["DEA"])
    macd_final_status = f"?MACD金叉 | {bar_trend}" if is_macd_gold else bar_trend

    # 其他指标
    cmf_ind = ChaikinMoneyFlowIndicator(df["最高"], df["最低"], close, df["成交量"], window=20)
    df["CMF"] = cmf_ind.chaikin_money_flow()
    
    # 策略判定 (选股条件不变)
    signal_type = ""
    if curr["BIAS20"] < -8 and curr["收盘"] > df["MA5"].iloc[-1]: signal_type = "??黄金坑"
    elif curr["收盘"] > curr["MA60"] and curr["CMF"] > 0.15: signal_type = "??机构控盘"
    
    patterns = []
    if curr["MA5"] > curr["MA10"] > curr["MA20"] > curr["MA60"]: patterns.append("??均线多头")
    pattern_str = " ".join(patterns)

    # 严格选股过滤
    if not (signal_type or "均线多头" in pattern_str or is_macd_gold): return None
    if curr["CMF"] < 0: return None # 资金流出不选

    k_status, k_score = analyze_kline_health(df)
    
    return {
        "代码": code, "名称": name, "现价": curr["收盘"],
        "今日涨跌": f"{(curr['收盘']/prev['收盘']-1)*100:+.2f}%",
        "MACD状态": macd_final_status,
        "均线排列": "??多头排列" if "均线多头" in pattern_str else "震荡中",
        "K线形态": k_status, "K线评分": k_score,
        "BIAS乖离": round(curr["BIAS20"], 1), "今日CMF": round(curr["CMF"], 3),
        "信号类型": signal_type, "形态特征": pattern_str,
        "止损价": round(curr["MA20"], 2), "市盈率": stock_info['pe']
    }

# --- 5. 评分与Excel生成 ---
def calculate_score(row):
    score = 50 # 基础分
    if "金叉" in str(row['MACD_Bar' if 'MACD_Bar' in row else 'MACD状态']): score += 40
    if "红柱增长" in str(row['MACD状态']) or "绿柱缩短" in str(row['MACD状态']): score += 15
    if "多头排列" in str(row['均线排列']): score += 30
    score += float(row.get('K线评分', 0))
    return score

def save_and_beautify_final(data_list):
    dt_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"严选_MACD动能增强版_{dt_str}.xlsx"
    df = pd.DataFrame(data_list)
    df["综合评分"] = df.apply(calculate_score, axis=1)
    df = df.sort_values(by="综合评分", ascending=False)
    
    df.to_excel(filename, index=False)
    wb = openpyxl.load_workbook(filename); ws = wb.active
    
    # 美化
    header_font = Font(name='微软雅黑', size=11, bold=True, color="FFFFFF")
    fill_blue = PatternFill("solid", fgColor="4472C4")
    for cell in ws[1]: cell.fill = fill_blue; cell.font = header_font
    
    # 底部全指南与手册
    start_row = ws.max_row + 3
    ws.cell(row=start_row, column=1, value=f"?? {MARKET_ENV_TEXT}").font = Font(size=14, bold=True)
    start_row += 2
    
    cat_font = Font(name='微软雅黑', size=12, bold=True, color="0000FF")
    ws.cell(row=start_row, column=1, value="?? 全指标读图指南").font = cat_font
    start_row += 1
    indicators = [
        ("MACD状态", "??合并显示：[?金叉] 代表趋势反转；[??红增/绿缩] 代表上涨动能正在加强。"),
        ("均线排列", "??多头排列：MA5>10>20>60。这是最强的持股形态，代表全周期走牛。"),
        ("K线形态", "??向上跳空：强力突破信号，通常由重大利好或主力急于拉升引起。"),
        ("CMF资金", "蔡金货币流量：>0代表资金净流入，数值越大主力买入越坚决。")
    ]
    for n, d in indicators:
        ws.cell(row=start_row, column=1, value=n); ws.cell(row=start_row, column=2, value=d)
        start_row += 1
    
    start_row += 1
    ws.cell(row=start_row, column=1, value="?? 五大策略实战手册").font = cat_font
    start_row += 1
    strategies = [
        ("?? 黄金坑", "逻辑：股价大幅偏离均线后的超跌反弹。买点：放量站稳5日线。"),
        ("?? 机构控盘", "逻辑：资金持续流入且均线呈多头主升。买点：沿10日线低吸。"),
        ("?? 缺口突破", "逻辑：以跳空方式突破压力位。买点：缺口当日或回踩不补时。")
    ]
    for n, l in strategies:
        ws.cell(row=start_row, column=1, value=n); ws.cell(row=start_row, column=2, value=l)
        start_row += 1

    wb.save(filename)
    print(f"? 报告已保存: {filename}")

def main():
    get_market_context()
    targets = get_targets_robust()
    if not targets: return
    start_dt = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        future_to_stock = {executor.submit(lambda p: process_stock_logic(get_data_with_retry(p['code'], start_dt), p), r): r['code'] for r in targets}
        for future in concurrent.futures.as_completed(future_to_stock):
            res = future.result()
            if res: results.append(res)
    if results: save_and_beautify_final(results)

if __name__ == "__main__":
    main()
