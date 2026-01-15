import akshare as ak
import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import StochasticOscillator, RSIIndicator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ta.volatility import BollingerBands, AverageTrueRange
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

# --- 1. 全局配置 ---
CONFIG = {
    "MIN_AMOUNT": 30000000,    # 3000万成交额
    "MIN_PRICE": 3.0,          # 最低股价
    "MAX_WORKERS": 10,         # 线程数
    "DAYS_LOOKBACK": 200,      # 数据回溯
    "RISK_MONEY": 2000,        # 单笔风险金 (小白如果不改，默认亏损承受额为2000元)
    "BLACKLIST_DAYS": 30       # 解禁预警
}

HOT_CONCEPTS = [] 
RESTRICTED_LIST = [] 
MARKET_ENV_TEXT = "⏳正在初始化..."

# --- 2. 市场情报 ---
def get_market_context():
    global HOT_CONCEPTS, RESTRICTED_LIST, MARKET_ENV_TEXT
    print("📡 [1/4] 连接交易所数据中心...")

    # 解禁排雷
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
            print(f"✅ 已拉黑 {len(RESTRICTED_LIST)} 只解禁风险股")
    except: pass

    # 热点
    try:
        df = ak.stock_board_concept_name_em()
        df = df.sort_values(by="涨跌幅", ascending=False).head(15)
        HOT_CONCEPTS = df["板块名称"].tolist()
        print(f"🔥 今日风口: {HOT_CONCEPTS}")
    except: pass

    # 大盘
    try:
        sh = ak.stock_zh_index_daily(symbol="sh000001")
        curr = sh.iloc[-1]
        ma20 = sh['close'].rolling(20).mean().iloc[-1]
        pct = (curr['close'] - sh.iloc[-2]['close']) / sh.iloc[-2]['close'] * 100
        trend = "🐂多头" if curr['close'] > ma20 else "🐻空头"
        MARKET_ENV_TEXT = f"上证: {curr['close']:.2f} ({pct:+.2f}%) | 趋势:{trend}"
        print(f"🌍 {MARKET_ENV_TEXT}")
    except: pass

# --- 3. 选股初筛 ---
def get_targets_robust():
    print(">>> [2/4] 全市场扫描与初筛...")
    try:
        df = ak.stock_zh_a_spot_em()
        col_map = {"最新价": "price", "成交额": "amount", "代码": "code", "名称": "name", 
                   "换手率": "turnover", "市盈率-动态": "pe", "市净率": "pb"}
        df.rename(columns=col_map, inplace=True)
        for c in ["price", "amount", "turnover", "pe", "pb"]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df.dropna(subset=["price", "amount"], inplace=True)
        df = df[df["code"].str.startswith(("60", "00"))]
        df = df[~df['name'].str.contains('ST|退')]
        df = df[df["price"] >= CONFIG["MIN_PRICE"]]
        df = df[df["amount"] > CONFIG["MIN_AMOUNT"]]
        df = df[df["turnover"] >= 1.0] 
        df = df[df["pb"] <= 20]
        df = df[~df["code"].isin(RESTRICTED_LIST)]
        
        print(f"✅ 有效标的: {len(df)} 只")
        return df.to_dict('records')
    except: return []

# --- 4. 核心逻辑 ---
def get_data_safe(code):
    time.sleep(random.uniform(0.01, 0.05))
    start_dt = (datetime.now() - timedelta(days=CONFIG["DAYS_LOOKBACK"])).strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_dt, adjust="qfq", timeout=5)
        if df is None or df.empty: return None
        return df
    except: return None

def get_60m_data(code):
    try:
        df = ak.stock_zh_a_hist_min_em(symbol=code, period="60", adjust="qfq")
        if df is None or df.empty: return None
        return df.tail(40)
    except: return None

# K线形态分析
def analyze_kline_patterns(df):
    patterns = []
    curr = df.iloc[-1]
    
    if curr['pct_chg'] > 9.5 and curr['volume'] < df['volume'].tail(5).mean():
        patterns.append("🔒缩量板")
    
    body_top = max(curr['open'], curr['close'])
    price_range = curr['high'] - curr['low']
    if price_range > 0:
        if (curr['high'] - body_top) / price_range > 0.4 and curr['close'] > curr['open']:
            patterns.append("☝️仙人指路")
        body_bottom = min(curr['open'], curr['close'])
        if (body_bottom - curr['low']) / price_range > 0.4:
            patterns.append("🛡️金针探底")

    vol_up = df[df['close']>df['open']].tail(20)['volume'].sum()
    vol_down = df[df['close']<df['open']].tail(20)['volume'].sum()
    if vol_up > vol_down * 1.5:
        patterns.append("🟥红肥")

    return " ".join(patterns)

def analyze_stock(stock_info):
    code = stock_info['code']
    name = stock_info['name']
    
    df = get_data_safe(code)
    if df is None or len(df) < 100: return None
    
    rename_dict = {"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume"}
    df.rename(columns={k:v for k,v in rename_dict.items() if k in df.columns}, inplace=True)
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    # 指标计算
    df["pct_chg"] = close.pct_change() * 100
    df["MA20"] = close.rolling(20).mean()
    df["ATR"] = AverageTrueRange(high, low, close, window=14).average_true_range()
    
    # 1. BIAS / RSI / KDJ
    df["BIAS20"] = (close - df["MA20"]) / df["MA20"] * 100
    df["RSI"] = RSIIndicator(close, window=14).rsi()
    kdj = StochasticOscillator(high, low, close)
    df["J"] = kdj.stoch() * 3 - kdj.stoch_signal() * 2
    
    # 2. 布林
    bb = BollingerBands(close, window=20)
    df["BB_W"] = bb.bollinger_wband()
    df["BB_Up"] = bb.bollinger_hband()
    df["BB_PctB"] = bb.bollinger_pband()
    df["BB_Low"] = bb.bollinger_lband()
    
    # 3. 资金
    df["OBV"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["OBV_MA"] = df["OBV"].rolling(10).mean()
    df["CMF"] = ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()
    
    # 4. MACD
    macd = MACD(close)
    df["MACD_Bar"] = macd.macd_diff()
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    is_limit_up = curr["close"] >= round(prev["close"] * 1.095, 2)
    turnover = stock_info['turnover']

    # --- 铁血过滤 ---
    if curr["close"] < prev["close"] * 0.91: return None 
    if turnover > 25 and not is_limit_up: return None 
    if not is_limit_up:
        if curr["OBV"] <= curr["OBV_MA"] or curr["OBV"] <= prev["OBV"]: return None
        if curr["MACD_Bar"] <= prev["MACD_Bar"]: return None

    # --- 策略匹配 ---
    signal = ""
    base_score = 0
    stop_loss = 0
    
    if prev["BIAS20"] < -8 and curr["MACD_Bar"] < 0:
        signal = "⚱️黄金坑"; base_score = 70; stop_loss = curr["low"]
    elif curr["CMF"] > 0.1 and curr["close"] > curr["MA20"] and curr["MACD_Bar"] > 0:
        signal = "🏦机构控盘"; base_score = 75; stop_loss = curr["MA20"]
    elif (close.pct_change().tail(20) > 0.095).any() and turnover < 10:
         if abs(curr["close"] - curr["MA20"])/curr["MA20"] < 0.05:
            signal = "🐉龙回头"; base_score = 80; stop_loss = df["BB_Low"].iloc[-1]
    elif df["BB_W"].iloc[-5:].mean() < 15 and curr["OBV"] > df["OBV"].iloc[-10:].max():
        signal = "🚀底部异动"; base_score = 75; stop_loss = curr["open"]

    if not signal: return None

    # --- 加分项 ---
    extra_score = 0
    
    # 60分钟
    status_60m = "⚪"
    try:
        df_60 = get_60m_data(code)
        if df_60 is not None and len(df_60) > 20:
            c60 = df_60["close"]
            m60 = MACD(c60)
            dif60, dea60 = m60.macd(), m60.macd_signal()
            if dif60.iloc[-2] < dea60.iloc[-2] and dif60.iloc[-1] > dea60.iloc[-1]:
                status_60m = "✅60分金叉"; extra_score += 30
            elif dif60.iloc[-1] > dea60.iloc[-1]:
                status_60m = "🚀60分多头"; extra_score += 10
            else:
                status_60m = "⚠️60分回调"; extra_score -= 10
    except: pass
    
    # 筹码分布
    chip_dist = ""
    if close.tail(60).std() / close.tail(60).mean() < 0.15:
        chip_dist = "🏆筹码密集"; extra_score += 10
    
    # 布林状态
    bb_status = ""
    if curr["BB_PctB"] > 1.0: bb_status = "🚀突破上轨"
    elif curr["BB_W"] < 12: bb_status = "↔️极度收口"
    
    # 热点
    news = ""
    try:
        news_df = ak.stock_news_em(symbol=code)
        if not news_df.empty: news = news_df.iloc[0]['新闻标题']
    except: pass
    concept_match = next((hot for hot in HOT_CONCEPTS if hot in news), "")
    if concept_match: extra_score += 15

    # 资金加速
    cmf_3days = df["CMF"].tail(3).values
    cmf_accelerating = (len(cmf_3days) == 3 and cmf_3days[2] > cmf_3days[1] > cmf_3days[0])
    if cmf_accelerating: extra_score += 25
    
    total_score = base_score + extra_score
    
    # 数据格式化
    cmf_str = " | ".join([f"{c:.2f}" for c in cmf_3days])
    if cmf_accelerating: cmf_str = f"🔺{cmf_str}"
    pct_3days = df["pct_chg"].tail(3).values
    pct_str = " | ".join([f"{p:+.1f}%" for p in pct_3days])

    # 仓位
    atr_stop = curr["close"] - 2.5 * curr["ATR"]
    final_stop = max(stop_loss, atr_stop)
    rec_shares = int(CONFIG["RISK_MONEY"] / max(curr["close"] - final_stop, 0.05) / 100) * 100
    
    return {
        "代码": code, "名称": name, "评分": total_score, "信号": signal,
        "现价": curr["close"], "今日涨跌": f"{curr['pct_chg']:+.2f}%",
        "建议": "买入" if total_score > 90 else "观察",
        "建议仓位": max(rec_shares, 100), "止损价": round(final_stop, 2),
        "60分状态": status_60m, 
        "BIAS乖离": round(curr["BIAS20"], 1),
        "布林状态": bb_status,
        "RSI指标": round(curr["RSI"], 1), "J值": round(curr["J"], 1),
        "筹码分布": chip_dist,
        "MACD形态": "🔴红柱增长" if curr["MACD_Bar"]>0 else "🟢绿柱缩短",
        "近3日CMF": cmf_str, "CMF加速": cmf_accelerating,
        "换手率": turnover, "形态特征": analyze_kline_patterns(df),
        "OBV状态": "🚀流入", "热点": f"🔥{concept_match}" if concept_match else "",
        "市盈率": stock_info.get('pe', '')
    }

# --- 5. Excel 导出 (核心美化) ---
def save_excel(results):
    if not results: return
    dt_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"严选_保姆级操作版_{dt_str}.xlsx"
    
    df = pd.DataFrame(results)
    df.sort_values(by="评分", ascending=False, inplace=True)
    
    # 20列
    cols = ["代码", "名称", "评分", "信号", "建议", "现价", "今日涨跌", 
            "建议仓位", "止损价", "60分状态", "BIAS乖离", "布林状态", 
            "RSI指标", "J值", "筹码分布", "MACD形态", "近3日CMF", 
            "换手率", "OBV状态", "热点"]
    for c in cols: 
        if c not in df.columns: df[c] = ""
    
    cmf_acc_dict = {row['代码']: row.get('CMF加速', False) for _, row in df.iterrows()}
    
    df = df[cols]
    df.to_excel(filename, index=False)
    
    wb = openpyxl.load_workbook(filename)
    ws = wb.active
    ws.title = "严选池"
    
    # 样式
    header_font = Font(name='微软雅黑', size=11, bold=True, color="FFFFFF")
    font_red = Font(name='微软雅黑', color="FF0000", bold=True)
    font_green = Font(name='微软雅黑', color="008000", bold=True)
    font_blue = Font(name='微软雅黑', color="0000FF", bold=True)
    fill_header = PatternFill("solid", fgColor="2F75B5")
    fill_red = PatternFill("solid", fgColor="FFC7CE")
    fill_yellow = PatternFill("solid", fgColor="FFF2CC")
    
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal='center')
        
    for row in ws.iter_rows(min_row=2):
        code_val = str(row[0].value)
        for cell in row:
            cell.alignment = Alignment(horizontal='center')
            cell.border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))
            
        # 评分
        if float(row[2].value) >= 90: row[2].fill = fill_red; row[2].font = font_red
        
        # 涨跌颜色
        if "+" in str(row[6].value): row[6].font = font_red
        elif "-" in str(row[6].value): row[6].font = font_green
        
        # 60分状态
        if "金叉" in str(row[9].value): row[9].fill = fill_yellow; row[9].font = font_red
        elif "回调" in str(row[9].value): row[9].font = font_green

        # 乖离率 (高亮危险和机会)
        try:
            bias = float(row[10].value)
            if bias > 12: row[10].font = font_red # 过热
            elif bias < -8: row[10].font = font_green # 黄金坑
        except: pass

        # CMF加速
        if cmf_acc_dict.get(code_val, False):
            row[16].fill = fill_yellow; row[16].font = font_red

    # 列宽
    ws.column_dimensions['Q'].width = 20 # CMF
    ws.column_dimensions['L'].width = 15 

    # ==========================================
    # 📖 保姆级实战说明书 (Human-Readable Manual)
    # ==========================================
    end_row = ws.max_row + 3
    
    # 1. 红绿灯
    env_cell = ws.cell(row=end_row, column=1, value=f"🚥 第一步：看大盘红绿灯 ({MARKET_ENV_TEXT})")
    env_cell.font = Font(size=14, bold=True, color="FFFFFF")
    if "暴跌" in MARKET_ENV_TEXT: env_cell.fill = PatternFill("solid", fgColor="FF0000")
    elif "安全" in MARKET_ENV_TEXT: env_cell.fill = PatternFill("solid", fgColor="008000")
    else: env_cell.fill = PatternFill("solid", fgColor="FFA500")
    ws.merge_cells(start_row=end_row, start_column=1, end_row=end_row, end_column=20)
    end_row += 2

    # 2. 选股口诀
    ws.cell(row=end_row, column=1, value="🔍 第二步：选股口诀 (只看前排)").font = Font(size=12, bold=True)
    end_row += 1
    
    rules = [
        ("🟥 红底红字", "系统评分>90的极品股，优先看。"),
        ("🟨 黄底提醒", "代表强力信号：'60分金叉'(即刻买入) 或 'CMF加速'(主力抢筹)。"),
        ("🟩 绿字提醒", "代表风险或等待：'60分回调'(下午再看) 或 'BIAS<-8'(超跌反弹)。")
    ]
    for title, desc in rules:
        ws.cell(row=end_row, column=1, value=title).font = Font(bold=True)
        ws.cell(row=end_row, column=2, value=desc)
        ws.merge_cells(start_row=end_row, start_column=2, end_row=end_row, end_column=20)
        end_row += 1
    end_row += 1

    # 3. 大白话指标字典
    ws.cell(row=end_row, column=1, value="📖 第三步：看不懂指标？看这里").font = Font(size=12, bold=True)
    end_row += 1
    
    dicts = [
        ("BIAS乖离", "通俗解释：'股价是不是跑得太远了'。负数很大(绿色)说明跌过头了，可以抄底；正数很大(红色)说明涨过头了，别追。"),
        ("60分状态", "通俗解释：'现在能不能动手'。✅金叉=现在买；⚠️回调=再等等。这是防止你买在当天最高点。"),
        ("建议仓位", "通俗解释：'买多少股'。系统算好了，按这个买，就算止损也只亏小钱。"),
        ("近3日CMF", "通俗解释：'主力资金进场了吗'。带🔺符号且标黄，说明主力这三天在疯狂买入。"),
        ("筹码密集", "通俗解释：'上方有没有人被套'。密集说明没套牢盘，拉升容易。"),
        ("RSI / J值", "通俗解释：'强弱尺子'。数值>80/100是超买(太热了)，<20/0是超卖(太冷了)。")
    ]
    for title, desc in dicts:
        ws.cell(row=end_row, column=1, value=title).font = Font(bold=True)
        ws.cell(row=end_row, column=2, value=desc)
        ws.merge_cells(start_row=end_row, start_column=2, end_row=end_row, end_column=20)
        end_row += 1

    # 4. 止损铁律
    final_cell = ws.cell(row=end_row, column=1, value="⛔ 风控铁律：收盘价如果跌破【止损价】，必须无条件卖出！")
    final_cell.font = Font(color="FF0000", bold=True, size=12)
    ws.merge_cells(start_row=end_row, start_column=1, end_row=end_row, end_column=20)

    wb.save(filename)
    print(f"\n🚀 保姆级战报已生成: {filename}")

def main():
    print(f"=== A股严选 v11.0 (终极保姆教学版) ===")
    get_market_context()
    target_list = get_targets_robust()
    if not target_list: return
    
    print(f"\n>>> [3/4] 深度全维计算...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        future_to_stock = {executor.submit(analyze_stock, t): t['code'] for t in target_list}
        count = 0
        for future in concurrent.futures.as_completed(future_to_stock):
            count += 1
            if count % 50 == 0: print(f"进度: {count}/{len(target_list)}...")
            try:
                res = future.result()
                if res: results.append(res)
            except: pass
            
    print(f"\n>>> [4/4] 生成战报...")
    save_excel(results)

if __name__ == "__main__":
    main()
