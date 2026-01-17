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
    "BLACKLIST_DAYS": 30,     # 解禁预警天数
    "IS_MANUAL": True         # 【重要】True: 手动测试(不计连板次数) | False: 自动执行(计连板，检测节假日)
}

HISTORY_FILE = "stock_history_log.csv"
HOT_CONCEPTS = [] 
RESTRICTED_LIST = [] 
NORTHBOUND_SET = set() 
MARKET_ENV_TEXT = "?初始化..."

# --- 2. 交易日判定 (自动模式专用) ---
def is_china_trade_day():
    """判断今天是否为中国股市交易日"""
    if CONFIG["IS_MANUAL"]: return True # 手动模式跳过判定
    try:
        today = datetime.now().strftime("%Y%m%d")
        df_trade_days = ak.tool_trade_date_hist_sina()
        trade_days = [d.strftime("%Y%m%d") for d in df_trade_days['trade_date']]
        return today in trade_days
    except:
        # 降级方案：判断周六日
        return datetime.now().weekday() < 5

# --- 3. 市场全维情报 ---
def get_market_context():
    global HOT_CONCEPTS, RESTRICTED_LIST, NORTHBOUND_SET, MARKET_ENV_TEXT
    print("?? [1/4] 连接交易所数据中心 (全维扫描开始)...")

    # 1. 解禁黑名单
    try:
        next_month = (datetime.now() + timedelta(days=CONFIG["BLACKLIST_DAYS"])).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        df_res = ak.stock_restricted_release_queue_em()
        RESTRICTED_LIST = df_res[(df_res['解禁日期'] >= today) & (df_res['解禁日期'] <= next_month)]['股票代码'].tolist()
    except: pass

    # 2. 市场热点
    try:
        df = ak.stock_board_concept_name_em()
        HOT_CONCEPTS = df.sort_values(by="涨跌幅", ascending=False).head(15)["板块名称"].tolist()
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
        status = "???多头安全" if curr['close'] >= ma20 else "???空头趋势"
        if pct < -1.5: status = "??暴跌风险"
        MARKET_ENV_TEXT = f"上证: {curr['close']:.2f} ({pct:+.2f}%) | {status}"
        print(f"?? {MARKET_ENV_TEXT}")
    except: pass

def get_targets_robust():
    try:
        df = ak.stock_zh_a_spot_em()
        col_map = {"最新价": "price", "成交额": "amount", "代码": "code", "名称": "name", 
                   "换手率": "turnover", "市盈率-动态": "pe"}
        df.rename(columns=col_map, inplace=True)
        df = df[df["code"].str.startswith(("60", "00"))]
        df = df[~df['name'].str.contains('ST|退')]
        df = df[(df["price"] >= CONFIG["MIN_PRICE"]) & (df["amount"] > CONFIG["MIN_AMOUNT"])]
        df = df[~df["code"].isin(RESTRICTED_LIST)]
        return df.to_dict('records')
    except: return []

def analyze_kline_health(df_full):
    if len(df_full) < 60: return "?数据不足", 0
    curr = df_full.iloc[-1]
    prev = df_full.iloc[-2]
    
    # 跳空缺口检测
    gap_type = ""
    gap_score = 0
    if curr['low'] > prev['high']: gap_type = "??向上跳空"; gap_score = 40
    elif curr['high'] < prev['low']: gap_type = "??向下跳空"; gap_score = -40

    body_top = max(curr['open'], curr['close'])
    body_bottom = min(curr['open'], curr['close'])
    price_range = curr['high'] - curr['low']
    if price_range == 0: return "?极小波动", 0
    
    upper_ratio = (curr['high'] - body_top) / price_range
    lower_ratio = (body_bottom - curr['low']) / price_range
    
    status = ""
    score = 0
    if upper_ratio > 0.4: status, score = "??冲高受阻", -10
    elif lower_ratio > 0.4: status, score = "???金针探底", 20
    elif (curr['close'] - curr['open']) / price_range > 0.6: status, score = "??大阳突破", 30
    else: status, score = "震荡整理", 0
    
    final_status = f"{gap_type}|{status}" if gap_type else status
    return final_status, score + gap_score

# --- 4. 核心逻辑 ---
def process_stock_logic(df, stock_info):
    code = stock_info['code']
    name = stock_info['name']
    pe = stock_info.get('pe', 0)
    turnover = stock_info.get('turnover', 0)

    if len(df) < 100: return None
    df.rename(columns={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount"}, inplace=True)
    
    close = df["close"]
    # 均线
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    
    # 涨跌序列 (最近3天)
    df["pct"] = close.pct_change() * 100
    p_seq = df["pct"].tail(3).tolist()
    seq_str = " | ".join([f"{x:+.2f}%" for x in p_seq])

    # MACD计算
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_Bar"] = (df["DIF"] - df["DEA"]) * 2
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # MACD状态逻辑
    is_macd_gold = (prev["DIF"] < prev["DEA"]) and (curr["DIF"] > curr["DEA"])
    macd_status = "??MACD金叉" if is_macd_gold else ("??红柱增" if curr["MACD_Bar"] > prev["MACD_Bar"] > 0 else "??绿柱缩" if curr["MACD_Bar"] > prev["MACD_Bar"] else "空头趋势")
    if curr["DIF"] > 0 and curr["DEA"] > 0 and is_macd_gold: macd_status = "?空中加油(强)"

    # CMF/KDJ 过滤 (保持原有严选逻辑)
    cmf_ind = ChaikinMoneyFlowIndicator(df["high"], df["low"], close, df["volume"], window=20)
    df["CMF"] = cmf_ind.chaikin_money_flow()
    if df["CMF"].iloc[-1] < 0.05 or df["CMF"].iloc[-1] <= df["CMF"].iloc[-2]: return None

    # 策略判定
    signal_type = ""
    if curr["close"] > curr["MA5"] > curr["MA10"] > curr["MA20"] > curr["MA60"]: signal_type = "??均线多头"
    elif is_macd_gold and curr["close"] > curr["MA20"]: signal_type = "??金叉突破"
    elif curr["CMF"] > 0.2: signal_type = "??主力强吸"
    
    if not signal_type: return None

    k_status, k_score = analyze_kline_health(df)

    return {
        "代码": code, "名称": name, "现价": curr["close"],
        "3日涨跌序列": seq_str,
        "K线形态": k_status, "K线评分": k_score,
        "均线排列": "??四线多头" if curr["MA5"] > curr["MA10"] > curr["MA20"] > curr["MA60"] else "交织震荡",
        "MACD状态": macd_status,
        "BIAS20": round((curr['close']-curr['MA20'])/curr['MA20']*100, 2),
        "信号类型": signal_type,
        "今日CMF": round(curr["CMF"], 3),
        "昨日CMF": round(prev["CMF"], 3),
        "连续": "", # 占位
        "市盈率": pe, "换手率": turnover
    }

# --- 5. 评分与历史 ---
def calculate_score_enhanced(row):
    score = 60
    details = []
    if "金叉" in row["MACD_Bar" if "MACD_Bar" in row else "MACD状态"]: score += 30; details.append("MACD金叉+30")
    if "跳空" in row["K线形态"]: score += 40; details.append("向上缺口+40")
    if "多头" in row["均线排列"]: score += 30; details.append("均线多头+30")
    if row["今日CMF"] > row["昨日CMF"]: score += 15; details.append("资金流入+15")
    return score, " | ".join(details)

def update_history_with_mode(results):
    today_str = datetime.now().strftime("%Y-%m-%d")
    # 如果是手动模式，不读写文件，不计连板
    if CONFIG["IS_MANUAL"]:
        for res in results: res["连续"] = "测试模式"
        return results

    try:
        if os.path.exists(HISTORY_FILE): hist_df = pd.read_csv(HISTORY_FILE)
        else: hist_df = pd.DataFrame(columns=["date", "code"])
        hist_df['date'] = hist_df['date'].astype(str)
    except: hist_df = pd.DataFrame(columns=["date", "code"])

    hist_df = hist_df[hist_df['date'] != today_str]
    sorted_dates = sorted(hist_df['date'].unique(), reverse=True)
    new_rows = []
    
    for res in results:
        code = res['代码']
        streak = 1
        for d in sorted_dates:
            if not hist_df[(hist_df['date'] == d) & (hist_df['code'] == str(code))].empty: streak += 1
            else: break
        res['连续'] = f"??{streak}天入选" if streak >= 2 else "首榜"
        new_rows.append({"date": today_str, "code": str(code)})

    pd.concat([hist_df, pd.DataFrame(new_rows)], ignore_index=True).to_csv(HISTORY_FILE, index=False)
    return results

# --- 6. Excel美化与指南 ---
def save_and_beautify_final(results):
    dt_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"严选_MACD金叉版_{dt_str}.xlsx"
    if not results: return
    
    df = pd.DataFrame(results)
    score_res = df.apply(calculate_score_enhanced, axis=1)
    df["综合评分"] = [x[0] for x in score_res]
    df["评分解析"] = [x[1] for x in score_res]
    
    cols = ["代码", "名称", "综合评分", "评分解析", "现价", "3日涨跌序列", "MACD状态", "K线形态", 
            "均线排列", "信号类型", "连续", "BIAS20", "今日CMF", "市盈率"]
    df = df[cols].sort_values(by="综合评分", ascending=False)
    df.to_excel(filename, index=False)
    
    wb = openpyxl.load_workbook(filename)
    ws = wb.active
    
    # 美化
    fill_blue = PatternFill("solid", fgColor="4472C4")
    font_white = Font(color="FFFFFF", bold=True)
    for cell in ws[1]: cell.fill = fill_blue; cell.font = font_white
    
    # 底部指南
    start_row = ws.max_row + 3
    ws.cell(row=start_row, column=1, value=f"?? 市场环境：{MARKET_ENV_TEXT}").font = Font(size=14, bold=True, color="FF0000")
    ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=10)
    
    # 手册内容
    start_row += 2
    guide_title = ws.cell(row=start_row, column=1, value="?? 全指标读图指南 & 策略实战手册")
    guide_title.font = Font(size=12, bold=True, color="0000FF")
    
    instructions = [
        ("MACD状态", "??金叉：水上金叉为极强爆发信号；空中加油：回调不破 DEA 再次发散。"),
        ("3日涨跌序列", "展示最近3天的单日波动。若出现[负|负|正]代表止跌企稳；[正|正|正]为趋势加速。"),
        ("K线形态", "??向上跳空：最强进攻信号，主力不计成本扫货；???金针探底：底部强力托盘。"),
        ("均线排列", "四线多头代表 MA5>MA10>MA20>MA60，属于标准的主升浪单边行情。"),
        ("模式说明", f"当前模式：{'手动模式(测试不计天数)' if CONFIG['IS_MANUAL'] else '自动模式(严格计连板)'}"),
        ("风控提醒", "BIAS20 > 12% 属于超买区，建议等回调再入；止损位建议设在 MA20 支撑线。")
    ]
    
    for i, (item, desc) in enumerate(instructions):
        ws.cell(row=start_row+1+i, column=1, value=item).font = Font(bold=True)
        ws.cell(row=start_row+1+i, column=2, value=desc)
        ws.merge_cells(start_row=start_row+1+i, start_column=2, end_row=start_row+1+i, end_column=10)

    # 自动宽度
    ws.column_dimensions['A'].width = 10; ws.column_dimensions['F'].width = 25
    ws.column_dimensions['D'].width = 40; ws.column_dimensions['G'].width = 15
    
    wb.save(filename)
    print(f"? 报告生成完毕: {filename}")

def main():
    print(f"=== 严选启动 (模式: {'手动' if CONFIG['IS_MANUAL'] else '自动'}) ===")
    
    if not is_china_trade_day():
        print("?? 今天是非交易日/节假日，程序自动退出。")
        return

    get_market_context()
    targets = get_targets_robust()
    if not targets: return

    start_dt = (datetime.now() - timedelta(days=CONFIG["DAYS_LOOKBACK"])).strftime("%Y%m%d")
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        future_to_stock = {executor.submit(lambda p: process_stock_logic(get_data_with_retry(p['code'], start_dt), p), r): r for r in targets}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_stock)):
            if i % 100 == 0: print(f"扫描进度: {i}/{len(targets)}...")
            try:
                res = future.result()
                if res: results.append(res)
            except: pass

    if results:
        results = update_history_with_mode(results)
        save_and_beautify_final(results)

if __name__ == "__main__":
    main()
