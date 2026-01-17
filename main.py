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
    "MANUAL_MODE": False,     # 【新增】手动模式开关：True时不开更新历史记录(不计连板)
    "MAX_CAP": 200,           # 【新增】市值上限（亿元），用于中小市值评分
    "MIN_CAP": 30             # 【新增】市值下限（亿元）
}

HISTORY_FILE = "stock_history_log.csv"
HOT_CONCEPTS = [] 
RESTRICTED_LIST = [] 
NORTHBOUND_SET = set() 
MARKET_ENV_TEXT = "?初始化..."

# --- 2. 辅助功能：节假日判定 ---
def is_trade_day():
    """判断今天是否为中国股市交易日"""
    try:
        today = datetime.now().strftime("%Y%m%d")
        df_trade_days = ak.tool_trade_date_hist_sina()
        trade_days = [d.strftime("%Y%m%d") for d in df_trade_days['trade_date']]
        return today in trade_days
    except:
        # 如果获取失败，降级为判断周六日
        return datetime.now().weekday() < 5

# --- 3. 市场全维情报 ---
def get_market_context():
    global HOT_CONCEPTS, RESTRICTED_LIST, NORTHBOUND_SET, MARKET_ENV_TEXT
    print("?? [1/4] 连接交易所数据中心 (新增市值与缺口监测)...")

    # 1. 解禁黑名单
    try:
        next_month = (datetime.now() + timedelta(days=CONFIG["BLACKLIST_DAYS"])).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        df_res = ak.stock_restricted_release_queue_em()
        cols = df_res.columns.tolist()
        code_col = next((c for c in cols if 'code' in c or '代码' in c), None)
        date_col = next((c for c in cols if 'date' in c or '时间' in c), None)
        if code_col and date_col:
            df_future = df_res[(df_res[date_col] >= today) & (df_res[date_col] <= next_month)]
            RESTRICTED_LIST = df_future[code_col].astype(str).tolist()
    except: pass

    # 2. 市场热点
    try:
        df = ak.stock_board_concept_name_em()
        df = df.sort_values(by="涨跌幅", ascending=False).head(15)
        HOT_CONCEPTS = df["板块名称"].tolist()
    except: pass

    # 3. 北向资金 (陆股通标的动态)
    try:
        df_sh = ak.stock_hsgt_list_em(symbol="沪股通")
        df_sz = ak.stock_hsgt_list_em(symbol="深股通")
        NORTHBOUND_SET.update(df_sh['代码'].astype(str).tolist())
        NORTHBOUND_SET.update(df_sz['代码'].astype(str).tolist())
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
    except: pass

def get_targets_robust():
    print(">>> [2/4] 全市场扫描 (同步获取市值数据)...")
    try:
        df = ak.stock_zh_a_spot_em()
        # 增加‘总市值’列
        col_map = {"最新价": "price", "成交额": "amount", "代码": "code", "名称": "name", 
                   "换手率": "turnover", "市盈率-动态": "pe", "市净率": "pb", "总市值": "mkt_cap"}
        df.rename(columns=col_map, inplace=True)
        
        df["price"] = pd.to_numeric(df["price"], errors='coerce')
        df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
        df["mkt_cap"] = pd.to_numeric(df["mkt_cap"], errors='coerce') / 100000000 # 转为亿元
        
        df = df[df["code"].str.startswith(("60", "00"))]
        df = df[~df['name'].str.contains('ST|退')]
        df = df[df["price"] >= CONFIG["MIN_PRICE"]]
        df = df[df["amount"] > CONFIG["MIN_AMOUNT"]]
        df = df[~df["code"].isin(RESTRICTED_LIST)]
        
        return df.to_dict('records')
    except Exception as e:
        print(f"?? 异常: {e}")
        return []

def get_data_with_retry(code, start_date):
    for _ in range(2):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq", timeout=5)
            if df is not None and not df.empty: return df
        except: time.sleep(0.2)
    return None

def analyze_kline_health(df_full):
    if len(df_full) < 60: return "?数据不足", 0
    curr = df_full.iloc[-1]
    prev = df_full.iloc[-2]
    
    # 缺口检测
    is_gap_up = curr['low'] > prev['high']
    gap_text = "??向上跳空" if is_gap_up else ""
    
    body_top = max(curr['open'], curr['close'])
    body_bottom = min(curr['open'], curr['close'])
    price_range = curr['high'] - curr['low']
    if price_range == 0: return "?极小波动", 0
    
    upper_ratio = (curr['high'] - body_top) / price_range
    lower_ratio = (body_bottom - curr['low']) / price_range
    
    # 基础分
    score = 50 if is_gap_up else 0
    status = gap_text if is_gap_up else "普通"

    if upper_ratio > 0.4:
        status = "??冲高回落"; score -= 10
    elif lower_ratio > 0.4:
        status = "???金针探底"; score += 20
    elif (curr['close'] - curr['open']) / price_range > 0.6:
        status = "??大阳突破"; score += 30
        
    return status, score

# --- 4. 核心逻辑 ---
def process_stock_logic(df, stock_info):
    code = stock_info['code']
    name = stock_info['name']
    pe = stock_info.get('pe', 0)
    turnover = stock_info.get('turnover', 0)
    mkt_cap = stock_info.get('mkt_cap', 0)

    if len(df) < 100: return None
    
    df.rename(columns={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount"}, inplace=True)
    
    close = df["close"]
    # 均线系统
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    
    # 【新增】均线多头排列判定
    is_bull_align = (df["MA5"].iloc[-1] > df["MA10"].iloc[-1] > df["MA20"].iloc[-1] > df["MA60"].iloc[-1])
    
    # 【新增】连续3天涨跌值
    df["pct"] = close.pct_change() * 100
    last_3_pcts = df["pct"].tail(3).tolist()
    pct_str = " | ".join([f"{x:+.2f}%" for x in last_3_pcts])

    # 缺口
    is_gap_up = df["low"].iloc[-1] > df["high"].iloc[-2]

    # 指标计算 (MACD, KDJ, RSI, CMF)
    # ... (保持原有的MACD, KDJ, RSI, CMF逻辑不变) ...
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_Bar"] = (df["DIF"] - df["DEA"]) * 2
    
    # CMF
    cmf_ind = ChaikinMoneyFlowIndicator(df["high"], df["low"], close, df["volume"], window=20)
    df["CMF"] = cmf_ind.chaikin_money_flow()
    
    # 策略初筛
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    signal_type = ""
    # 简化策略判定
    if is_gap_up and curr['pct'] > 2: signal_type = "??缺口突破"
    elif is_bull_align and curr['pct'] > 0: signal_type = "??趋势主升"
    elif curr['CMF'] > 0.15: signal_type = "??机构潜伏"
    else: return None # 无核心信号不录入

    # 构造理由与风险
    reasons = [signal_type]
    risks = []
    if is_bull_align: reasons.append("均线多头")
    if is_gap_up: reasons.append("向上缺口")
    if mkt_cap < CONFIG["MAX_CAP"]: reasons.append("中小市值")
    
    if curr['pct'] > 7: risks.append("追高风险")
    if pe < 0: risks.append("业绩亏损")
    if (curr['close'] - df["MA20"].iloc[-1])/df["MA20"].iloc[-1] > 0.15: risks.append("乖离过大")
    
    k_status, k_score = analyze_kline_health(df)

    return {
        "代码": code, "名称": name, "现价": curr["close"],
        "3日独立涨跌": pct_str,
        "K线形态": k_status, "K线评分": k_score,
        "均线排列": "??多头排列" if is_bull_align else "交织震荡",
        "缺口理论": "??向上跳空" if is_gap_up else "--",
        "总市值(亿)": round(mkt_cap, 2),
        "北向动态": "??陆股通标" if code in NORTHBOUND_SET else "--",
        "信号类型": signal_type,
        "选股理由": " + ".join(reasons),
        "风险提示": " | ".join(risks) if risks else "暂无明显风险",
        "今日CMF": round(curr["CMF"], 3),
        "昨日CMF": round(prev["CMF"], 3),
        "市盈率": pe,
        "BIAS20": round((curr['close'] - df["MA20"].iloc[-1])/df["MA20"].iloc[-1]*100, 2)
    }

# --- 5. 评分系统 ---
def calculate_score_enhanced(row):
    score = 60 # 初始分
    details = []
    
    # 1. 缺口加分
    if "跳空" in str(row['缺口理论']):
        score += 50; details.append("向上缺口+50")
    
    # 2. 均线排列加分
    if "多头" in str(row['均线排列']):
        score += 30; details.append("均线多头+30")
    
    # 3. 市值评分
    mkt_cap = row['总市值(亿)']
    if CONFIG["MIN_CAP"] < mkt_cap < CONFIG["MAX_CAP"]:
        score += 20; details.append("中小市值+20")
    
    # 4. 资金面
    if row['今日CMF'] > row['昨日CMF']:
        score += 15; details.append("资金流入+15")
    
    # 5. 北向
    if "陆股通" in str(row['北向动态']):
        score += 10; details.append("北向标的+10")

    # 6. 基本面
    pe = row['市盈率']
    if 0 < pe < 30:
        score += 15; details.append("绩优低估+15")
    elif pe < 0:
        score -= 20; details.append("亏损减分-20")

    return score, " | ".join(details)

def update_history_safe(current_results):
    """仅在非手动模式且是交易日时更新历史记录"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # 如果是手动模式，或今天不是交易日（节假日），则不计入连板逻辑
    if CONFIG["MANUAL_MODE"] or not is_trade_day():
        print("?? [提示] 手动模式或非交易日，跳过连板历史统计。")
        for res in current_results: res['连续'] = "测试模式"
        return current_results

    try:
        if os.path.exists(HISTORY_FILE):
            hist_df = pd.read_csv(HISTORY_FILE)
            hist_df['date'] = hist_df['date'].astype(str)
        else: hist_df = pd.DataFrame(columns=["date", "code"])
    except: hist_df = pd.DataFrame(columns=["date", "code"])

    hist_df = hist_df[hist_df['date'] != today_str]
    sorted_dates = sorted(hist_df['date'].unique(), reverse=True)
    processed_results = []
    new_rows = []
    
    for res in current_results:
        code = res['代码']
        streak = 1
        for d in sorted_dates:
            if not hist_df[(hist_df['date'] == d) & (hist_df['code'] == str(code))].empty: streak += 1
            else: break
        res['连续'] = f"??{streak}天入选" if streak >= 2 else "首日"
        processed_results.append(res)
        new_rows.append({"date": today_str, "code": str(code)})

    if new_rows: 
        hist_df = pd.concat([hist_df, pd.DataFrame(new_rows)], ignore_index=True)
        hist_df.to_csv(HISTORY_FILE, index=False)
    return processed_results

# --- 6. 保存与美化 ---
def save_and_beautify_enhanced(data_list):
    dt_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"严选_全维度评分_{dt_str}.xlsx"
    
    if not data_list:
        print("?? 今日无标的入选"); return
    
    df = pd.DataFrame(data_list)
    res = df.apply(calculate_score_enhanced, axis=1)
    df["综合评分"] = [x[0] for x in res]
    df["评分解析"] = [x[1] for x in res]
    
    cols = ["代码", "名称", "综合评分", "评分解析", "选股理由", "风险提示", "现价", "3日独立涨跌", 
            "均线排列", "缺口理论", "总市值(亿)", "北向动态", "K线形态", "连续", 
            "信号类型", "今日CMF", "昨日CMF", "市盈率", "BIAS20"]
    
    df = df[cols].sort_values(by="综合评分", ascending=False)
    df.to_excel(filename, index=False)
    
    # Excel美化逻辑 (保持与原版相似但适配新列宽)
    wb = openpyxl.load_workbook(filename)
    ws = wb.active
    fill_blue = PatternFill("solid", fgColor="4472C4")
    header_font = Font(name='微软雅黑', size=11, bold=True, color="FFFFFF")
    
    for cell in ws[1]:
        cell.fill = fill_blue; cell.font = header_font
        
    # 调整列宽
    ws.column_dimensions['D'].width = 40 # 评分解析
    ws.column_dimensions['E'].width = 30 # 选股理由
    ws.column_dimensions['F'].width = 30 # 风险提示
    ws.column_dimensions['H'].width = 25 # 3日涨跌
    
    wb.save(filename)
    print(f"? 增强版报告已生成: {filename}")

def main():
    print(f"=== A股严选 (增强评分版) | 模式: {'手动' if CONFIG['MANUAL_MODE'] else '自动'} ===")
    if not is_trade_day():
        print("?? 今天是非交易日，建议使用手动模式进行历史复盘。")
    
    get_market_context()
    targets = get_targets_robust()
    if not targets: return

    start_dt = (datetime.now() - timedelta(days=CONFIG["DAYS_LOOKBACK"])).strftime("%Y%m%d")
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        future_to_stock = {executor.submit(lambda p: analyze_one_stock(p, start_dt), r): r['code'] for r in targets}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_stock)):
            if i % 100 == 0: print(f"进度: {i}/{len(targets)}...")
            try:
                res = future.result()
                if res: results.append(res)
            except: pass

    if results:
        results = update_history_safe(results)
        save_and_beautify_enhanced(results)

def analyze_one_stock(stock_info, start_dt):
    try:
        df = get_data_with_retry(stock_info['code'], start_dt)
        if df is None: return None
        return process_stock_logic(df, stock_info)
    except: return None

if __name__ == "__main__":
    main()
