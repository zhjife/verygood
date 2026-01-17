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
MARKET_ENV_TEXT = "??初始化..."

# --- 2. 市场全维情报 ---
def get_market_context():
    global HOT_CONCEPTS, RESTRICTED_LIST, NORTHBOUND_SET, MARKET_ENV_TEXT
    print("?? [1/4] 连接交易所数据中心...")
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
            if df is not None and not df.empty:
                # 统一列名，防止 Key 错误
                df.rename(columns={"开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount"}, inplace=True)
                return df
        except: time.sleep(0.1)
    return None

# --- 3. 核心判定辅助 ---
def analyze_kline_health(df_full):
    curr = df_full.iloc[-1]
    prev = df_full.iloc[-2]
    price_range = curr['high'] - curr['low']
    if price_range == 0: return "??极小波动", 0
    
    gap_signal = ""
    gap_score = 0
    if curr['low'] > prev['high']: gap_signal = "??向上跳空"; gap_score = 40
    elif curr['high'] < prev['low']: gap_signal = "??向下跳空"; gap_score = -40

    body_top = max(curr['open'], curr['close'])
    body_bottom = min(curr['open'], curr['close'])
    upper_ratio = (curr['high'] - body_top) / price_range
    lower_ratio = (body_bottom - curr['low']) / price_range
    
    status = "??普通震荡"; score = 0
    if (curr['close'] - curr['open']) / price_range > 0.6: status = "??实体强攻"; score = 25
    elif lower_ratio > 0.4: status = "???金针探底"; score = 20
    elif upper_ratio > 0.4: status = "??高位抛压"; score = -10

    if gap_signal: return f"{gap_signal}|{status}", score + gap_score
    return status, score

# --- 4. 核心选股逻辑 ---
def process_stock_logic(df, stock_info):
    if df is None or len(df) < 60: return None
    code, name = stock_info['code'], stock_info['name']
    
    # 1. 计算所有指标 (先计算，后取 curr)
    close = df["close"]
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["BIAS20"] = (close - df["MA20"]) / df["MA20"] * 100
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_Bar"] = (df["DIF"] - df["DEA"]) * 2
    
    # CMF
    cmf_ind = ChaikinMoneyFlowIndicator(df["high"], df["low"], close, df["volume"], window=20)
    df["CMF"] = cmf_ind.chaikin_money_flow()

    # 2. 现在指标计算完毕，获取当前行快照
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # 3. MACD 状态判断 (金叉 + 动能合并)
    bar_trend = ""
    if curr["MACD_Bar"] > 0:
        bar_trend = "??红柱增长" if curr["MACD_Bar"] > prev["MACD_Bar"] else "??红柱缩短"
    else:
        bar_trend = "??绿柱缩短" if curr["MACD_Bar"] > prev["MACD_Bar"] else "??绿柱增长"
    
    is_macd_gold = (prev["DIF"] < prev["DEA"]) and (curr["DIF"] > curr["DEA"])
    macd_final_status = f"?金叉|{bar_trend}" if is_macd_gold else bar_trend

    # 4. 策略筛选条件
    signal_type = ""
    # 策略A: 黄金坑
    if curr["BIAS20"] < -8 and curr["close"] > curr["MA5"]:
        signal_type = "??黄金坑"
    # 策略B: 机构控盘
    elif curr["close"] > curr["MA60"] and curr["CMF"] > 0.15:
        signal_type = "??机构控盘"
    
    # 形态: 多头排列
    is_bull = curr["MA5"] > curr["MA10"] > curr["MA20"] > curr["MA60"]
    
    # 5. 严格过滤: 必须满足策略 或 多头排列 或 MACD金叉
    if not (signal_type or is_bull or is_macd_gold): return None
    if curr["CMF"] < 0: return None # 排除资金净流出的股

    k_status, k_score = analyze_kline_health(df)
    
    return {
        "代码": code, "名称": name, "现价": curr["close"],
        "今日涨跌": f"{(curr['close']/prev['close']-1)*100:+.2f}%",
        "MACD状态": macd_final_status,
        "均线排列": "??多头排列" if is_bull else "震荡中",
        "K线形态": k_status, "K线评分": k_score,
        "BIAS20": round(curr["BIAS20"], 2), 
        "今日CMF": round(curr["CMF"], 3),
        "信号类型": signal_type,
        "止损价": round(curr["MA20"] * 0.98, 2),
        "换手率": stock_info['turnover'],
        "市盈率": stock_info['pe']
    }

# --- 5. 评分与 Excel 增强 ---
def calculate_score(row):
    score = 60 # 基础起步分
    if "金叉" in str(row['MACD状态']): score += 40
    if "红柱增长" in str(row['MACD状态']) or "绿柱缩短" in str(row['MACD状态']): score += 15
    if "多头排列" in str(row['均线排列']): score += 30
    score += float(row.get('K线评分', 0))
    if row['今日CMF'] > 0.2: score += 20
    return score

def save_and_beautify(data_list):
    dt_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"严选_MACD金叉增强版_{dt_str}.xlsx"
    df = pd.DataFrame(data_list)
    df["综合评分"] = df.apply(calculate_score, axis=1)
    df = df.sort_values(by="综合评分", ascending=False)
    
    df.to_excel(filename, index=False)
    wb = openpyxl.load_workbook(filename)
    ws = wb.active
    
    # 样式美化
    header_font = Font(name='微软雅黑', size=11, bold=True, color="FFFFFF")
    fill_blue = PatternFill("solid", fgColor="4472C4")
    for cell in ws[1]:
        cell.fill = fill_blue
        cell.font = header_font

    # 底部全指南
    start_row = ws.max_row + 3
    ws.cell(row=start_row, column=1, value=f"?? {MARKET_ENV_TEXT}").font = Font(size=14, bold=True)
    start_row += 2
    
    cat_font = Font(name='微软雅黑', size=12, bold=True, color="0000FF")
    
    # 读图指南
    ws.cell(row=start_row, column=1, value="?? 全指标读图指南").font = cat_font
    start_row += 1
    indicators = [
        ("MACD状态", "?金叉|XX：代表趋势扭转且动能爆发；红柱增长：多头力度加强；绿柱缩短：止跌信号。"),
        ("均线排列", "??多头排列：MA5>10>20>60。属于最强上升趋势，建议顺势而为。"),
        ("K线形态", "??向上跳空：主力强势抢筹；???金针探底：下方支撑强劲，适合博弈。"),
        ("CMF资金", "蔡金货币流量。>0代表主力正在买入。若评分解析显示资金流入，可靠性更高。")
    ]
    for n, d in indicators:
        ws.cell(row=start_row, column=1, value=n); ws.cell(row=start_row, column=2, value=d)
        start_row += 1
    
    start_row += 1
    # 实战手册
    ws.cell(row=start_row, column=1, value="?? 五大策略实战手册").font = cat_font
    start_row += 1
    strategies = [
        ("?? 黄金坑", "逻辑：股价严重偏离均线后的报复性反弹。买点：放量站稳5日线瞬间。"),
        ("?? 机构控盘", "逻辑：CMF持续高位且均线多头，主力深度介入。买点：缩量回调10日线。"),
        ("?? 金叉共振", "逻辑：MACD金叉 + 均线排列或跳空缺口。买点：信号确认后的次日开盘。")
    ]
    for n, l in strategies:
        ws.cell(row=start_row, column=1, value=n); ws.cell(row=start_row, column=2, value=l)
        start_row += 1

    wb.save(filename)
    print(f"?? 报告已生成: {filename}")

# --- 6. 主程序 ---
def main():
    get_market_context()
    targets = get_targets_robust()
    if not targets: return
    start_dt = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
    
    results = []
    print(f"?? 开始扫描 {len(targets)} 只个股...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        future_to_stock = {executor.submit(lambda p: process_stock_logic(get_data_with_retry(p['code'], start_dt), p), r): r['code'] for r in targets}
        for future in concurrent.futures.as_completed(future_to_stock):
            try:
                res = future.result()
                if res: results.append(res)
            except Exception as e:
                pass

    if results:
        save_and_beautify(results)
    else:
        print("?? 今日未筛选到符合条件的个股")

if __name__ == "__main__":
    main()
