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
MARKET_ENV_TEXT = "初始化..."

# --- 2. 市场全维情报 ---
def get_market_context():
    global HOT_CONCEPTS, RESTRICTED_LIST, NORTHBOUND_SET, MARKET_ENV_TEXT
    print("🚀 [1/4] 连接交易所数据中心 (全维扫描)...")

    # 1. 解禁黑名单
    try:
        next_month = (datetime.now() + timedelta(days=CONFIG["BLACKLIST_DAYS"])).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        df_res = ak.stock_restricted_release_queue_em()
        code_col = next((c for c in df_res.columns if 'code' in c or '代码' in c), None)
        date_col = next((c for c in df_res.columns if 'date' in c or '时间' in c), None)
        if code_col and date_col:
            df_future = df_res[(df_res[date_col] >= today) & (df_res[date_col] <= next_month)]
            RESTRICTED_LIST = df_future[code_col].astype(str).tolist()
            print(f"📡 已拉黑 {len(RESTRICTED_LIST)} 只近期解禁风险股")
    except: pass

    # 2. 市场热点
    try:
        df = ak.stock_board_concept_name_em()
        df = df.sort_values(by="涨跌幅", ascending=False).head(15)
        HOT_CONCEPTS = df["板块名称"].tolist()
        print(f"🔥 今日风口: {HOT_CONCEPTS}")
    except: pass

    # 3. 北向资金
    try:
        df_sh = ak.stock_hsgt_top_10_em(symbol="沪股通")
        df_sz = ak.stock_hsgt_top_10_em(symbol="深股通")
        if df_sh is not None: NORTHBOUND_SET.update(df_sh['代码'].astype(str).tolist())
        if df_sz is not None: NORTHBOUND_SET.update(df_sz['代码'].astype(str).tolist())
    except: pass
    
    # 4. 大盘环境
    try:
        sh = ak.stock_zh_index_daily(symbol="sh000001")
        curr = sh.iloc[-1]
        ma20 = sh['close'].rolling(20).mean().iloc[-1]
        pct = (curr['close'] - sh.iloc[-2]['close']) / sh.iloc[-2]['close'] * 100
        status = "🟢多头安全" if curr['close'] >= ma20 else "🔴空头趋势"
        if pct < -1.5: status = "⚠️暴跌风险"
        MARKET_ENV_TEXT = f"上证: {curr['close']:.2f} ({pct:+.2f}%) | {status}"
        print(f"📊 {MARKET_ENV_TEXT}")
    except: pass

def get_targets_robust():
    print("🔍 [2/4] 全市场扫描与初筛...")
    try:
        df = ak.stock_zh_a_spot_em()
        col_map = {"最新价": "price", "成交额": "amount", "代码": "code", "名称": "name", 
                   "换手率": "turnover", "市盈率-动态": "pe", "市净率": "pb"}
        df.rename(columns=col_map, inplace=True)
        df["price"] = pd.to_numeric(df["price"], errors='coerce')
        df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
        df.dropna(subset=["price", "amount"], inplace=True)
        df = df[df["code"].str.startswith(("60", "00"))]
        df = df[~df['name'].str.contains('ST|退')]
        df = df[df["price"] >= CONFIG["MIN_PRICE"]]
        df = df[df["amount"] > CONFIG["MIN_AMOUNT"]]
        df = df[~df["code"].isin(RESTRICTED_LIST)]
        print(f"✅ 有效标的: {len(df)} 只")
        return df.to_dict('records')
    except: return []

def get_data_with_retry(code, start_date):
    for _ in range(2):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq", timeout=5)
            if df is not None and not df.empty: return df
        except: time.sleep(0.2)
    return None

def get_60m_data_optimized(code):
    try:
        df = ak.stock_zh_a_hist_min_em(symbol=code, period="60", adjust="qfq", timeout=10)
        if df is not None and not df.empty:
            df.rename(columns={"时间":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume"}, inplace=True)
            return df.tail(60)
    except: pass
    return None

def analyze_kline_health(df_full):
    if len(df_full) < 60: return "数据不足", 0
    curr, prev = df_full.iloc[-1], df_full.iloc[-2]
    body_top, body_bottom = max(curr['open'], curr['close']), min(curr['open'], curr['close'])
    price_range = curr['high'] - curr['low']
    if price_range == 0: return "平盘", 0
    
    # 缺口检测
    gap_signal, gap_score = "", 0
    if curr['low'] > prev['high']: gap_signal, gap_score = "向上跳空", 40
    elif curr['high'] < prev['low']: gap_signal, gap_score = "向下跳空", -40

    # 形态识别
    status, score = "普通", 0
    upper_ratio = (curr['high'] - body_top) / price_range
    lower_ratio = (body_bottom - curr['low']) / price_range
    if upper_ratio > 0.4: status, score = "冲高受阻", -10
    elif lower_ratio > 0.4: status, score = "金针探底", 20
    elif (curr['close'] - curr['open']) / price_range > 0.6: status, score = "实体强攻", 25
    
    if gap_signal: return f"{gap_signal}|{status}", score + gap_score
    return status, score

# --- 4. 核心处理逻辑 ---
def process_stock_logic(df, stock_info):
    code, name = stock_info['code'], stock_info['name']
    if len(df) < 100: return None
    df.rename(columns={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount"}, inplace=True)
    
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    df["vwap"] = df["amount"] / volume
    df["pct_chg"] = close.pct_change() * 100
    pct_3day = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100 if len(close) > 4 else 0
    
    # 均线
    df["MA5"], df["MA10"], df["MA20"], df["MA60"] = close.rolling(5).mean(), close.rolling(10).mean(), close.rolling(20).mean(), close.rolling(60).mean()
    df["BIAS20"] = (close - df["MA20"]) / df["MA20"] * 100

    # MACD
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_Bar"] = (df["DIF"] - df["DEA"]) * 2
    
    # KDJ & RSI & CMF
    rsv = (close - low.rolling(9).min()) / (high.rolling(9).max() - low.rolling(9).min()) * 100
    df['J'] = 3 * rsv.ewm(com=2).mean() - 2 * rsv.ewm(com=2).mean().ewm(com=2).mean()
    df['RSI'] = 100 - (100 / (1 + close.diff().clip(lower=0).ewm(com=5).mean() / (-close.diff().clip(upper=0)).ewm(com=5).mean()))
    df["CMF"] = ChaikinMoneyFlowIndicator(high, low, close, volume).chaikin_money_flow()
    
    curr, prev = df.iloc[-1], df.iloc[-2]
    
    # --- 过滤器与策略 ---
    if curr["J"] > 105 or curr["CMF"] < 0.05 or curr["MACD_Bar"] <= prev["MACD_Bar"]: return None

    signal_type = ""
    if (prev["BIAS20"] < -8 or prev["RSI"] < 25) and curr["close"] > df["MA5"].iloc[-1]: signal_type = "🌟黄金坑"
    elif curr["close"] > curr["MA60"] and curr["CMF"] > 0.15: signal_type = "🏦机构控盘"
    elif BollingerBands(close).bollinger_wband().iloc[-1] < 12: signal_type = "🌀底部变盘"

    # 形态增强
    patterns = []
    if curr["MA5"] > curr["MA10"] > curr["MA20"] > curr["MA60"]: patterns.append("📈均线多头")
    if (close.pct_change().tail(20).clip(lower=0).sum()) > ((-close.pct_change().tail(20).clip(upper=0)).sum() * 2): patterns.append("🔴红肥绿瘦")
    pattern_str = " ".join(patterns)

    # MACD 状态 (本轮核心)
    macd_cross = "MACD金叉" if (prev["DIF"] <= prev["DEA"] and curr["DIF"] > curr["DEA"]) else ("多头" if curr["DIF"] > curr["DEA"] else "空头")
    macd_warn = "空中加油" if (curr["DIF"] > 0 and curr["MACD_Bar"] > prev["MACD_Bar"] and prev["MACD_Bar"] > 0) else ""
    macd_final = f"{macd_cross}|{'红增' if curr['MACD_Bar']>0 else '绿缩'}{'|'+macd_warn if macd_warn else ''}"

    # 判定入选条件
    is_gold = (prev["J"] < 10 and curr["J"] > 10) or (prev["DIF"] <= prev["DEA"] and curr["DIF"] > curr["DEA"])
    if not (signal_type or pattern_str or is_gold): return None

    k_status, k_score = analyze_kline_health(df)
    
    # 60分钟
    status_60m = "震荡"
    df60 = get_60m_data_optimized(code)
    if df60 is not None:
        d60 = df60["close"].ewm(span=12).mean() - df60["close"].ewm(span=26).mean()
        status_60m = "60分金叉" if d60.iloc[-1] > d60.ewm(span=9).mean().iloc[-1] and d60.iloc[-2] <= d60.ewm(span=9).mean().iloc[-2] else ("60分多头" if d60.iloc[-1] > 0 else "回调")

    return {
        "代码": code, "名称": name, "现价": curr["close"], "今日涨跌": f"{curr['pct_chg']:+.2f}%", "3日涨跌": f"{pct_3day:+.2f}%",
        "K线形态": k_status, "K线评分": k_score, "60分状态": status_60m, "BIAS20": round(curr["BIAS20"], 1),
        "连续": "", "信号类型": signal_type, "形态特征": pattern_str, "MACD状态": macd_final,
        "今日CMF": round(curr["CMF"], 3), "昨日CMF": round(prev["CMF"], 3), "前日CMF": round(df["CMF"].iloc[-3], 3),
        "建议挂单": round(curr["close"], 2), "止损价": round(curr["MA20"], 2), "换手率": stock_info.get('turnover', 0), "市盈率": stock_info.get('pe', 0)
    }

# --- 评分系统 ---
def calculate_score_details(row):
    score, details = 0, []
    # 大盘
    if "多头" in MARKET_ENV_TEXT: score += 10; details.append("大盘多头+10")
    elif "空头" in MARKET_ENV_TEXT: score -= 15; details.append("大盘空头-15")
    # 技术面
    score += float(row['K线评分']); details.append(f"形态分{row['K线评分']:+}")
    if "金叉" in row['60分状态']: score += 80; details.append("60分金叉+80")
    if "均线多头" in row['形态特征']: score += 30; details.append("均线多头+30")
    if "MACD金叉" in row['MACD状态']: score += 25; details.append("MACD金叉+25")
    # 资金
    if row['今日CMF'] > row['昨日CMF']: score += 15; details.append("资金流入+15")
    if float(row['市盈率']) > 0 and float(row['市盈率']) < 30: score += 20; details.append("绩优+20")
    return score, " | ".join(details)

def update_history(results):
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        hist = pd.read_csv(HISTORY_FILE) if os.path.exists(HISTORY_FILE) else pd.DataFrame(columns=["date", "code"])
        hist['date'] = hist['date'].astype(str)
        sorted_dates = sorted(hist['date'].unique(), reverse=True)
        for r in results:
            streak = 1
            for d in sorted_dates:
                if not hist[(hist['date']==d) & (hist['code']==r['代码'])].empty: streak += 1
                else: break
            r['连续'] = f"{streak}连" if streak > 1 else "首榜"
        new_data = pd.DataFrame([{"date": today, "code": r['代码']} for r in results])
        pd.concat([hist, new_data]).to_csv(HISTORY_FILE, index=False)
    except: pass
    return results

# --- 5. 增强版 Excel 保存与美化 ---
def save_and_beautify(results):
    if not results: return print("🌑 今日无标的符合条件")
    dt = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"严选_全维度增强版_{dt}.xlsx"
    
    df = pd.DataFrame(results)
    scored = df.apply(calculate_score_details, axis=1)
    df["综合评分"], df["评分解析"] = [x[0] for x in scored], [x[1] for x in scored]
    
    cols = ["代码", "名称", "综合评分", "评分解析", "现价", "今日涨跌", "3日涨跌", "K线形态", "60分状态", 
            "BIAS20", "连续", "信号类型", "形态特征", "MACD状态", "今日CMF", "昨日CMF", "前日CMF", 
            "建议挂单", "止损价", "换手率", "市盈率"]
    df = df[cols].sort_values(by="综合评分", ascending=False)
    df.to_excel(filename, index=False)
    
    wb = openpyxl.load_workbook(filename)
    ws = wb.active
    # 样式定义
    header_fill = PatternFill("solid", fgColor="4472C4")
    yellow_fill = PatternFill("solid", fgColor="FFF2CC")
    red_font = Font(color="FF0000", bold=True)
    green_font = Font(color="008000")
    
    for cell in ws[1]:
        cell.fill, cell.font = header_fill, Font(color="FFFFFF", bold=True)
    
    for row in ws.iter_rows(min_row=2):
        # 评分高亮
        if row[2].value >= 120: row[2].fill = PatternFill("solid", fgColor="FFC7CE")
        # MACD状态标色
        if "金叉" in str(row[13].value): row[13].fill, row[13].font = yellow_fill, red_font
        # 涨跌标色
        if "+" in str(row[5].value): row[5].font = red_font
        elif "-" in str(row[5].value): row[5].font = green_font
        # 均线多头标色
        if "均线多头" in str(row[12].value): row[12].font = red_font

    # --- 底部文档恢复 ---
    ws.column_dimensions['D'].width = 50
    ws.column_dimensions['N'].width = 25
    curr_row = ws.max_row + 3
    
    # 1. 大盘看板
    env_cell = ws.cell(row=curr_row, column=1, value=f"📊 {MARKET_ENV_TEXT}")
    env_cell.font = Font(size=14, bold=True, color="FFFFFF")
    env_cell.fill = PatternFill("solid", fgColor="008000") if "多头" in MARKET_ENV_TEXT else PatternFill("solid", fgColor="FFA500")
    ws.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=21)
    
    # 2. 实战手册
    curr_row += 2
    ws.cell(row=curr_row, column=1, value="📘 五大策略实战手册").font = Font(size=12, bold=True, color="0000FF")
    curr_row += 1
    strategies = [
        ("🌟 黄金坑", "核心逻辑：深跌后缩量企稳，BIAS20 < -8，今日站稳MA5。适合左侧抄底。", "现价买入，以前日低点止损。"),
        ("🏦 机构控盘", "核心逻辑：CMF > 0.15，趋势向上且有机构持续吸筹。适合主升浪持有。", "沿10日均线持股，跌破止损。"),
        ("🌀 底部变盘", "核心逻辑：布林带宽极度收口（<12），即将选择方向。配合MACD金叉买入。", "放量突破布林上轨瞬间介入。")
    ]
    for n, l, a in strategies:
        ws.cell(row=curr_row, column=1, value=n).font = Font(bold=True)
        ws.cell(row=curr_row, column=2, value=l)
        ws.cell(row=curr_row, column=3, value=a)
        ws.merge_cells(start_row=curr_row, start_column=3, end_row=curr_row, end_column=10)
        curr_row += 1

    # 3. 读图指南
    curr_row += 1
    ws.cell(row=curr_row, column=1, value="📒 全指标读图指南").font = Font(size=12, bold=True, color="0000FF")
    curr_row += 1
    guides = [
        ("评分解析", "全逻辑白盒展示，清晰告知为何加分（如：均线多头+30）。"),
        ("MACD状态", "新增“金叉/多头/空头”及“空中加油”判定，高亮显示最佳买点。"),
        ("K线形态", "新增“向上跳空”权重分，缺口代表主力进攻决心。"),
        ("CMF三日", "主力资金流。若 [前 < 昨 < 今] 代表主力不计成本抢筹。")
    ]
    for n, g in guides:
        ws.cell(row=curr_row, column=1, value=n).font = Font(bold=True)
        ws.cell(row=curr_row, column=2, value=g)
        ws.merge_cells(start_row=curr_row, start_column=2, end_row=curr_row, end_column=10)
        curr_row += 1

    wb.save(filename)
    print(f"✨ 增强全功能版报告已保存: {filename}")

def main():
    get_market_context()
    targets = get_targets_robust()
    if not targets: return
    start_dt = (datetime.now() - timedelta(days=CONFIG["DAYS_LOOKBACK"])).strftime("%Y%m%d")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        future_to_stock = {executor.submit(lambda r: process_stock_logic(get_data_with_retry(r['code'], start_dt), r), target): target['code'] for target in targets}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_stock)):
            if i % 100 == 0: print(f"进度: {i}/{len(targets)}...")
            res = future.result()
            if res: results.append(res)
    if results:
        results = update_history(results)
        save_and_beautify(results)

if __name__ == "__main__":
    main()
