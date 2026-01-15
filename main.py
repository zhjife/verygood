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
    "MAX_WORKERS": 6,          # 🔥 降低线程数至6，大幅提高60分钟数据获取成功率
    "DAYS_LOOKBACK": 200,      # 数据回溯
    "RISK_MONEY": 2000,        # 单笔风险金
    "BLACKLIST_DAYS": 30       # 解禁预警
}

HISTORY_FILE = "stock_history_log.csv"
HOT_CONCEPTS = [] 
RESTRICTED_LIST = [] 
NORTHBOUND_SET = set() 
MARKET_ENV_TEXT = "⏳正在初始化..."

# --- 2. 市场情报 ---
def get_market_context():
    global HOT_CONCEPTS, RESTRICTED_LIST, MARKET_ENV_TEXT, NORTHBOUND_SET
    print("📡 [1/4] 连接交易所数据中心...")

    # 1. 解禁排雷
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

    # 2. 热点
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
        print(f"💰 北向活跃资金: 已锁定 {len(NORTHBOUND_SET)} 只重点股")
    except: pass

    # 4. 大盘
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

# --- 4. 核心逻辑 (增强稳定性) ---
def get_data_safe(code):
    # 增加随机延迟，防止封IP
    time.sleep(random.uniform(0.1, 0.3)) 
    start_dt = (datetime.now() - timedelta(days=CONFIG["DAYS_LOOKBACK"])).strftime("%Y%m%d")
    
    # 增加重试机制
    for _ in range(3):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_dt, adjust="qfq", timeout=5)
            if df is not None and not df.empty: return df
        except: 
            time.sleep(0.5) # 失败后歇一会再试
    return None

def get_60m_data(code):
    """
    🔥 增强版60分钟数据获取
    包含重试机制和随机延迟，解决'数据不足'问题
    """
    for _ in range(3): # 最多重试3次
        try:
            time.sleep(random.uniform(0.1, 0.4)) # 每次请求前随机等待
            df = ak.stock_zh_a_hist_min_em(symbol=code, period="60", adjust="qfq", timeout=3)
            if df is not None and not df.empty:
                return df.tail(40)
        except:
            time.sleep(0.5) # 休息一下再试
    return None

def analyze_kline_health(df_full):
    if len(df_full) < 60: return "⚪数据不足", 0
    curr = df_full.iloc[-1]
    
    body_top = max(curr['open'], curr['close'])
    price_range = curr['high'] - curr['low']
    if price_range == 0: return "⚪极小波动", 0
    
    upper_ratio = (curr['high'] - body_top) / price_range
    vol_ratio = curr['volume'] / df_full['volume'].tail(5).mean()
    trend_up = curr['close'] > df_full['close'].tail(20).mean()

    if upper_ratio > 0.4:
        if vol_ratio > 2.0: return "⚠️高位抛压", -30
        elif not trend_up: return "📉冲高受阻", -10
        elif curr['close'] >= curr['open']: return "☝️仙人指路", 15
    elif (min(curr['open'], curr['close']) - curr['low']) / price_range > 0.4:
        if curr['low'] <= df_full['close'].tail(20).mean(): return "🛡️金针探底", 20
        return "⚓底部承接", 15
    elif (curr['close'] - curr['open']) / price_range > 0.6:
        prev_open = df_full['open'].iloc[-2]
        if curr['close'] > prev_open: return "⚡阳包阴", 25
        return "💪实体强攻", 10
            
    return "⚪普通震荡", 0

def analyze_stock(stock_info):
    code = stock_info['code']
    name = stock_info['name']
    
    df = get_data_safe(code)
    if df is None or len(df) < 100: return None
    
    rename_dict = {"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount"}
    df.rename(columns={k:v for k,v in rename_dict.items() if k in df.columns}, inplace=True)
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    df["pct_chg"] = close.pct_change() * 100
    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["ATR"] = AverageTrueRange(high, low, close, window=14).average_true_range()
    df["BIAS20"] = (close - df["MA20"]) / df["MA20"] * 100
    df["RSI"] = RSIIndicator(close, window=14).rsi()
    kdj = StochasticOscillator(high, low, close)
    df["J"] = kdj.stoch() * 3 - kdj.stoch_signal() * 2
    
    bb = BollingerBands(close, window=20)
    df["BB_W"] = bb.bollinger_wband()
    df["BB_Up"] = bb.bollinger_hband()
    df["BB_PctB"] = bb.bollinger_pband()
    df["BB_Low"] = bb.bollinger_lband()
    
    df["OBV"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["OBV_MA"] = df["OBV"].rolling(10).mean()
    df["CMF"] = ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()
    df["vwap"] = df["amount"] / volume
    
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
    suggest_buy = curr["close"]
    
    if prev["BIAS20"] < -8 and curr["MACD_Bar"] < 0:
        signal = "⚱️黄金坑"; base_score = 70; stop_loss = curr["low"]
    elif curr["CMF"] > 0.1 and curr["close"] > curr["MA20"] and curr["MACD_Bar"] > 0:
        signal = "🏦机构控盘"; base_score = 75; stop_loss = curr["MA20"]
        suggest_buy = round(curr["vwap"], 2)
    elif (close.pct_change().tail(20) > 0.095).any() and turnover < 10:
         if abs(curr["close"] - curr["MA20"])/curr["MA20"] < 0.05:
            signal = "🐉龙回头"; base_score = 80; stop_loss = df["BB_Low"].iloc[-1]
            suggest_buy = round(df["MA20"].iloc[-1], 2)
    elif df["BB_W"].iloc[-5:].mean() < 15 and curr["OBV"] > df["OBV"].iloc[-10:].max():
        signal = "🚀底部异动"; base_score = 75; stop_loss = curr["open"]

    if not signal: return None

    kline_status, kline_score = analyze_kline_health(df)

    # --- 加分项 ---
    extra_score = 0
    resonance_list = []
    
    # 60分钟状态 (带重试机制)
    status_60m = "⏳数据不足" # 默认值，如果获取失败则显示此值
    try:
        df_60 = get_60m_data(code)
        if df_60 is not None and len(df_60) > 20:
            c60 = df_60["close"]
            m60 = MACD(c60)
            dif60, dea60 = m60.macd(), m60.macd_signal()
            if dif60.iloc[-2] < dea60.iloc[-2] and dif60.iloc[-1] > dea60.iloc[-1]:
                status_60m = "✅60分金叉"; extra_score += 30; resonance_list.append("60分共振")
            elif dif60.iloc[-1] > dea60.iloc[-1]:
                status_60m = "🚀60分多头"; extra_score += 10
            else:
                status_60m = "⚠️60分回调"; extra_score -= 10
        elif df_60 is None:
            # 如果实在获取不到，不扣分，给一个中性状态
            status_60m = "⚪获取超时"
    except: pass
    
    # 北向
    is_northbound = "否"
    if code in NORTHBOUND_SET:
        is_northbound = "💰外资重仓"; extra_score += 20; resonance_list.append("北向")

    # 布林
    bb_status = ""
    if curr["BB_PctB"] > 1.0: bb_status = "🚀突破上轨"
    elif curr["BB_W"] < 12: bb_status = "↔️极度收口"; resonance_list.append("变盘节点")
    
    # 热点
    news = ""
    try:
        news_df = ak.stock_news_em(symbol=code)
        if not news_df.empty: news = news_df.iloc[0]['新闻标题']
    except: pass
    concept_match = next((hot for hot in HOT_CONCEPTS if hot in news), "")
    if concept_match: extra_score += 15; resonance_list.append("热点")

    # 资金加速
    cmf_3days = df["CMF"].tail(3).values
    cmf_accelerating = (len(cmf_3days) == 3 and cmf_3days[2] > cmf_3days[1] > cmf_3days[0])
    if cmf_accelerating: extra_score += 25; resonance_list.append("资金加速")
    
    total_score = base_score + extra_score + kline_score
    
    cmf_str = " | ".join([f"{c:.2f}" for c in cmf_3days])
    if cmf_accelerating: cmf_str = f"🔺{cmf_str}"
    
    pct_3days = df["pct_chg"].tail(3).values
    pct_str = " | ".join([f"{p:+.1f}%" for p in pct_3days])
    
    atr_stop = curr["close"] - 2.5 * curr["ATR"]
    final_stop = max(stop_loss, atr_stop)
    rec_shares = int(CONFIG["RISK_MONEY"] / max(curr["close"] - final_stop, 0.05) / 100) * 100
    
    patterns = []
    if close.tail(60).std() / close.tail(60).mean() < 0.15: patterns.append("🏆筹码密集")
    if is_limit_up and turnover < 5: patterns.append("🔒缩量板")
    
    return {
        "代码": code, "名称": name, "评分": total_score, "信号": signal,
        "现价": curr["close"], "建议挂单": suggest_buy,
        "建议": "买入" if total_score > 90 else "观察",
        "建议仓位": max(rec_shares, 100), "止损价": round(final_stop, 2),
        "60分状态": status_60m, 
        "K线形态": kline_status, "K线评分": kline_score,
        "共振因子": "+".join(resonance_list),
        "BIAS乖离": round(curr["BIAS20"], 1), "布林状态": bb_status,
        "RSI指标": round(curr["RSI"], 1), "J值": round(curr["J"], 1),
        "MACD形态": "🔴红柱增长" if curr["MACD_Bar"]>0 else "🟢绿柱缩短",
        "近3日CMF": cmf_str, "CMF加速": cmf_accelerating,
        "近3日涨幅": pct_str,
        "换手率": turnover, "形态特征": " ".join(patterns),
        "OBV状态": "🚀流入", "热点": f"🔥{concept_match}" if concept_match else "",
        "北向资金": is_northbound, "市盈率": stock_info.get('pe', ''),
        "今日涨跌": f"{curr['pct_chg']:+.2f}%"
    }

# --- 5. 历史记录 ---
def update_history(current_results):
    today_str = datetime.now().strftime("%Y-%m-%d")
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
        code = str(res['代码'])
        streak = 1
        for d in sorted_dates:
            if not hist_df[(hist_df['date'] == d) & (hist_df['code'] == code)].empty: streak += 1
            else: break
            
        res['连续'] = f"🔥{streak}连" if streak >= 2 else "首榜"
        processed_results.append(res)
        new_rows.append({"date": today_str, "code": code})

    if new_rows: 
        hist_df = pd.concat([hist_df, pd.DataFrame(new_rows)], ignore_index=True)
        hist_df.to_csv(HISTORY_FILE, index=False)
        
    return processed_results

# --- 6. Excel 导出 ---
def save_excel(results):
    if not results: return
    dt_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"严选_v18稳定版_{dt_str}.xlsx"
    
    df = pd.DataFrame(results)
    df.sort_values(by="评分", ascending=False, inplace=True)
    
    cols = ["代码", "名称", "评分", "信号", "建议", "建议挂单", "现价", "今日涨跌", "近3日涨幅",
            "建议仓位", "止损价", "连续", "60分状态", "K线形态", "共振因子",
            "BIAS乖离", "布林状态", "RSI指标", "J值", "MACD形态", 
            "近3日CMF", "形态特征", "换手率", "OBV状态", "北向资金", "热点", "市盈率", "K线评分"]
            
    for c in cols: 
        if c not in df.columns: df[c] = ""
    
    cmf_acc_dict = {row['代码']: row.get('CMF加速', False) for _, row in df.iterrows()}
    
    df = df[cols]
    df.to_excel(filename, index=False)
    
    wb = openpyxl.load_workbook(filename)
    ws = wb.active
    ws.title = "严选池"
    
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
            
        if float(row[2].value) >= 90: row[2].fill = fill_red; row[2].font = font_red
        row[5].font = font_blue 
        if "连" in str(row[11].value): row[11].font = font_red; row[11].fill = fill_yellow
        if "金叉" in str(row[12].value): row[12].fill = fill_yellow; row[12].font = font_red
        elif "回调" in str(row[12].value): row[12].font = font_green
        k_val = str(row[13].value)
        if "仙人" in k_val or "阳包阴" in k_val: row[13].font = font_red
        elif "抛压" in k_val: row[13].font = font_green
        if cmf_acc_dict.get(code_val, False): row[20].fill = fill_yellow; row[20].font = font_red
        if "外资" in str(row[24].value): row[24].font = font_red; row[24].fill = fill_yellow

    ws.column_dimensions['I'].width = 22 
    ws.column_dimensions['U'].width = 22 

    # ==========================================
    # 📖 终极指标详解 (小白必读)
    # ==========================================
    end_row = ws.max_row + 3
    
    env_cell = ws.cell(row=end_row, column=1, value=f"🚥 环境: {MARKET_ENV_TEXT}")
    env_cell.font = Font(size=14, bold=True, color="FFFFFF")
    if "暴跌" in MARKET_ENV_TEXT: env_cell.fill = PatternFill("solid", fgColor="FF0000")
    elif "安全" in MARKET_ENV_TEXT: env_cell.fill = PatternFill("solid", fgColor="008000")
    else: env_cell.fill = PatternFill("solid", fgColor="FFA500")
    ws.merge_cells(start_row=end_row, start_column=1, end_row=end_row, end_column=28)
    end_row += 2

    ws.cell(row=end_row, column=1, value="📚 全指标操作说明书 (小白必读)").font = Font(size=12, bold=True)
    end_row += 1
    
    guides = [
        ("评分/连续", "分越高越好。🔥3连代表真龙。"),
        ("建议挂单", "【重要】不要只看现价。这是系统算出的最佳买点。"),
        ("60分状态", "✅金叉=现在买；⚠️回调=等下午买；⏳数据不足=网络波动，可参考日线。"),
        ("北向资金", "💰外资重仓：代表聪明钱(Smart Money)在关注，基本面通常较好。"),
        ("K线形态", "这是单日检查。'☝️仙人指路'是上涨中继，'⚠️高位抛压'要小心。"),
        ("共振因子", "列出了加分项，越多越好。"),
        ("BIAS/RSI/J", "绿色数值(负很多)是机会，红色数值(正很多)是风险。"),
        ("近3日CMF", "带🔺标黄代表主力资金连续3天加速抢筹。"),
        ("建议仓位", "系统算好的安全股数，照做即可。"),
        ("止损价", "收盘跌破此价，必须卖出！")
    ]
    for title, desc in guides:
        ws.cell(row=end_row, column=1, value=title).font = Font(bold=True)
        ws.cell(row=end_row, column=2, value=desc)
        ws.merge_cells(start_row=end_row, start_column=2, end_row=end_row, end_column=28)
        end_row += 1

    wb.save(filename)
    print(f"\n🚀 v18.0 稳定版战报已生成: {filename}")

def main():
    print(f"=== A股严选 v18.0 (网络稳定增强版) ===")
    get_market_context()
    target_list = get_targets_robust()
    if not target_list: return
    
    print(f"\n>>> [3/4] 深度全维计算 (稳定抓取模式)...")
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
            
    print(f"\n>>> [4/4] 更新历史记录并生成战报...")
    results = update_history(results)
    save_excel(results)

if __name__ == "__main__":
    main()
