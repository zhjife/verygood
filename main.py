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
            print(f"??? 已拉黑 {len(RESTRICTED_LIST)} 只近期解禁风险股")
    except: pass

    # 2. 市场热点
    try:
        df = ak.stock_board_concept_name_em()
        df = df.sort_values(by="涨跌幅", ascending=False).head(15)
        HOT_CONCEPTS = df["板块名称"].tolist()
        print(f"?? 今日风口: {HOT_CONCEPTS}")
    except: pass

    # 3. 北向资金
    try:
        df_sh = ak.stock_hsgt_top_10_em(symbol="沪股通")
        df_sz = ak.stock_hsgt_top_10_em(symbol="深股通")
        if df_sh is not None: NORTHBOUND_SET.update(df_sh['代码'].astype(str).tolist())
        if df_sz is not None: NORTHBOUND_SET.update(df_sz['代码'].astype(str).tolist())
        print(f"?? 北向重仓: {len(NORTHBOUND_SET)} 只")
    except: pass
    
    # 4. 大盘环境
    try:
        sh = ak.stock_zh_index_daily(symbol="sh000001")
        curr = sh.iloc[-1]
        ma20 = sh['close'].rolling(20).mean().iloc[-1]
        pct = (curr['close'] - sh.iloc[-2]['close']) / sh.iloc[-2]['close'] * 100
        
        status = ""
        if pct < -1.5: status = "??暴跌风险"
        elif curr['close'] < ma20: status = "???空头趋势"
        else: status = "???多头安全"
        
        MARKET_ENV_TEXT = f"上证: {curr['close']:.2f} ({pct:+.2f}%) | {status}"
        print(f"?? {MARKET_ENV_TEXT}")
    except: pass

def get_targets_robust():
    print(">>> [2/4] 全市场扫描与初筛...")
    try:
        df = ak.stock_zh_a_spot_em()
        col_map = {"最新价": "price", "成交额": "amount", "代码": "code", "名称": "name", 
                   "换手率": "turnover", "市盈率-动态": "pe", "市净率": "pb"}
        df.rename(columns=col_map, inplace=True)
        
        df["price"] = pd.to_numeric(df["price"], errors='coerce')
        df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
        df["turnover"] = pd.to_numeric(df["turnover"], errors='coerce')
        df.dropna(subset=["price", "amount"], inplace=True)
        
        df = df[df["code"].str.startswith(("60", "00"))]
        df = df[~df['name'].str.contains('ST|退')]
        df = df[df["price"] >= CONFIG["MIN_PRICE"]]
        df = df[df["amount"] > CONFIG["MIN_AMOUNT"]]
        df = df[~df["code"].isin(RESTRICTED_LIST)]
        
        print(f"? 有效标的: {len(df)} 只 (已剔除风险股)")
        return df.to_dict('records')
    except Exception as e:
        print(f"?? 异常: {e}")
        return []

def get_data_with_retry(code, start_date):
    time.sleep(random.uniform(0.01, 0.05)) 
    for _ in range(2):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq", timeout=5)
            if df is not None and not df.empty: return df
        except: time.sleep(0.2)
    return None

def get_60m_data_optimized(code):
    for attempt in range(3):
        try:
            time.sleep(random.uniform(0.1, 0.4))
            try:
                df = ak.stock_zh_a_hist_min_em(symbol=code, period="60", adjust="qfq", timeout=10)
            except:
                df = ak.stock_zh_a_hist_min_em(symbol=code, period="60", adjust="", timeout=10)
                
            if df is not None and not df.empty:
                df.rename(columns={"时间":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume"}, inplace=True)
                return df.tail(60) 
        except:
            time.sleep(1) 
    return None

def get_stock_catalysts(code):
    try:
        news_df = ak.stock_news_em(symbol=code)
        if not news_df.empty:
            return news_df.iloc[0]['新闻标题']
    except: pass
    return ""

def analyze_kline_health(df_full):
    if len(df_full) < 60: return "?数据不足", 0
    curr = df_full.iloc[-1]
    prev = df_full.iloc[-2]
    
    body_top = max(curr['open'], curr['close'])
    body_bottom = min(curr['open'], curr['close'])
    price_range = curr['high'] - curr['low']
    if price_range == 0: return "?极小波动", 0
    
    upper_ratio = (curr['high'] - body_top) / price_range
    lower_ratio = (body_bottom - curr['low']) / price_range
    
    high_60 = df_full['high'].tail(60).max()
    low_60 = df_full['low'].tail(60).min()
    rp = (curr['close'] - low_60) / (high_60 - low_60 + 0.0001)
    
    vol_ratio = curr['volume'] / df_full['volume'].tail(5).mean()
    trend_up = curr['close'] > df_full['close'].tail(20).mean()

    # --- 跳空缺口检测 ---
    gap_signal = ""
    gap_score = 0
    if curr['low'] > prev['high']:
        gap_signal = "??向上跳空"
        gap_score = 40
    elif curr['high'] < prev['low']:
        gap_signal = "??向下跳空"
        gap_score = -40

    res_status = ""
    res_score = 0
    if upper_ratio > 0.4:
        if rp > 0.8 and vol_ratio > 2.0: res_status, res_score = "??高位抛压", -30
        elif not trend_up and curr['close'] < curr['open']: res_status, res_score = "??冲高受阻", -10
        elif rp < 0.6 and vol_ratio < 1.5 and curr['close'] >= curr['open']: res_status, res_score = "??仙人指路", 15
        else: res_status, res_score = "?上影震荡", 0
    elif lower_ratio > 0.4:
        if not trend_up and curr['close'] < df_full['close'].iloc[-2]: res_status, res_score = "??下跌中继", -20
        elif curr['low'] <= df_full['close'].tail(20).mean(): res_status, res_score = "???金针探底", 20
        elif rp < 0.2: res_status, res_score = "?底部承接", 15
        else: res_status, res_score = "?下影震荡", 5
    elif (curr['close'] - curr['open']) / price_range > 0.6:
        prev_open = df_full['open'].iloc[-2]
        if curr['close'] > prev_open: res_status, res_score = "?阳包阴", 25
        else: res_status, res_score = "??实体强攻", 10
    elif (curr['open'] - curr['close']) / price_range > 0.6:
        if vol_ratio > 2.0: res_status, res_score = "??放量杀跌", -20
        else: res_status, res_score = "??阴线调整", -5
    else:
        if vol_ratio < 0.6: res_status, res_score = "?缩量十字", 5
        else: res_status, res_score = "?普通震荡", 0
    
    if gap_signal:
        return f"{gap_signal}|{res_status}", res_score + gap_score
    return res_status, res_score

# --- 4. 核心逻辑 ---
def process_stock_logic(df, stock_info):
    code = stock_info['code']
    name = stock_info['name']
    pe = stock_info.get('pe', 0)
    turnover = stock_info.get('turnover', 0)

    if len(df) < 100: return None
    
    rename_dict = {"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount"}
    col_map = {k:v for k,v in rename_dict.items() if k in df.columns}
    df.rename(columns=col_map, inplace=True)
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    df["vwap"] = df["amount"] / volume if "amount" in df.columns else (high + low + close) / 3

    df["pct_chg"] = close.pct_change() * 100
    today_pct = df["pct_chg"].iloc[-1]
    pct_3day = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100 if len(close) > 4 else 0
    
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["BIAS20"] = (close - df["MA20"]) / df["MA20"] * 100

    bb_ind = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb_ind.bollinger_hband()
    df["BB_Lower"] = bb_ind.bollinger_lband()
    df["BB_Width"] = bb_ind.bollinger_wband()
    df["BB_PctB"] = bb_ind.bollinger_pband()

    # 指标计算
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_Bar"] = (df["DIF"] - df["DEA"]) * 2
    
    low_9 = low.rolling(9, min_periods=9).min()
    high_9 = high.rolling(9, min_periods=9).max()
    rsv = (close - low_9) / (high_9 - low_9) * 100
    rsv = rsv.fillna(50)
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=5, adjust=False).mean()
    ema_down = down.ewm(com=5, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    obv_ind = OnBalanceVolumeIndicator(close, volume)
    df["OBV"] = obv_ind.on_balance_volume()
    df["OBV_MA10"] = df["OBV"].rolling(10).mean()
    
    cmf_ind = ChaikinMoneyFlowIndicator(high, low, close, volume, window=20)
    df["CMF"] = cmf_ind.chaikin_money_flow()
    df["ADX"] = ADXIndicator(high, low, close, window=14).adx()

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev_2 = df.iloc[-3]

    # --- 熔断过滤 ---
    has_zt = (df["pct_chg"].tail(30) > 9.5).sum() >= 1
    is_today_limit = curr["close"] >= round(prev["close"] * 1.095, 2)
    
    if turnover > 25 and not is_today_limit: return None
    if curr["J"] > 105: return None 
    if curr["OBV"] <= curr["OBV_MA10"]: return None
    if curr["CMF"] < 0.05: return None
    if curr["CMF"] <= prev["CMF"]: return None
    if curr["MACD_Bar"] <= prev["MACD_Bar"]: return None 

    # --- 策略信号判定 ---
    signal_type = ""
    suggest_buy = curr["close"]
    stop_loss = curr["MA20"]
    
    if (prev["BIAS20"] < -8) or (prev["RSI"] < 20):
        if (curr["close"] > curr["MA5"]) and (curr["pct_chg"] > 1.5):
            signal_type = "??黄金坑(企稳)"; stop_loss = round(curr["low"] * 0.98, 2)
    
    if not signal_type and has_zt and curr["close"] > curr["MA60"]:
        vol_ratio = curr["volume"] / df["volume"].tail(5).mean()
        if vol_ratio < 0.85 and -8.0 < curr["BIAS20"] < 8.0:
            signal_type = "??龙回头"
    
    if not signal_type and curr["close"] > curr["MA60"] and curr["CMF"] > 0.1 and curr["ADX"] > 25:
        signal_type = "??机构控盘"
    
    if not signal_type and curr["close"] < curr["MA60"] * 1.2 and curr["BB_Width"] < 12:
        signal_type = "?底部变盘"

    # --- 形态 ---
    chip_signal = ""
    high_120 = df["high"].tail(120).max()
    low_120 = df["low"].tail(120).min()
    current_pos = (curr["close"] - low_120) / (high_120 - low_120 + 0.001)
    if current_pos < 0.4:
        volatility = df["close"].tail(60).std() / df["close"].tail(60).mean()
        if volatility < 0.15: chip_signal = "??筹码密集" 

    patterns = []
    if curr["MA5"] > curr["MA10"] > curr["MA20"] > curr["MA60"]:
        patterns.append("??均线多头")
    vol_up = df[df['close']>df['open']].tail(20)['volume'].sum()
    vol_down = df[df['close']<df['open']].tail(20)['volume'].sum()
    if vol_up > vol_down * 2.0 and curr["close"] > curr["MA20"]: patterns.append("??红肥绿瘦")
    
    recent_5 = df.tail(5)
    if (recent_5['close'] > recent_5['MA5']).all() and (recent_5['pct_chg'].abs() < 4.0).all():
        patterns.append("??蚂蚁上树")
    pattern_str = " ".join(patterns)

    # --- MACD 详细状态判定 ---
    is_macd_gold = (prev["DIF"] < prev["DEA"]) and (curr["DIF"] > curr["DEA"])
    is_kdj_gold = (prev["J"] < prev["K"]) and (curr["J"] > curr["K"]) and (curr["J"] < 80)
    
    macd_detail = ""
    if is_macd_gold:
        macd_detail = "??MACD金叉"
    elif curr["DIF"] > curr["DEA"]:
        if curr["MACD_Bar"] > prev["MACD_Bar"]:
            macd_detail = "??红柱增量"
        else:
            macd_detail = "??强势运行"
        if curr["DIF"] > 0 and prev["MACD_Bar"] < curr["MACD_Bar"] and prev["MACD_Bar"] > 0:
            macd_detail = "?空中加油"
    else:
        macd_detail = "??绿柱缩短" if curr["MACD_Bar"] > prev["MACD_Bar"] else "??空头控制"

    if signal_type != "??黄金坑(企稳)":
        if not (is_macd_gold or is_kdj_gold or (curr["DIF"] > curr["DEA"] and curr["MACD_Bar"] > prev["MACD_Bar"])): 
            return None

    # --- 最终校验 ---
    if not (bool(signal_type) or bool(chip_signal or pattern_str)): return None

    kline_status, kline_score = analyze_kline_health(df)

    # 60分钟
    status_60m = "?数据不足"
    try:
        df_60 = get_60m_data_optimized(code)
        if df_60 is not None and len(df_60) > 20:
            c60 = df_60["close"]
            d60 = c60.ewm(span=12, adjust=False).mean() - c60.ewm(span=26, adjust=False).mean()
            e60 = d60.ewm(span=9, adjust=False).mean()
            if d60.iloc[-2] < e60.iloc[-2] and d60.iloc[-1] > e60.iloc[-1]: status_60m = "?60分金叉"
            elif d60.iloc[-1] > e60.iloc[-1]: status_60m = "??60分多头"
            else: status_60m = "??60分回调"
    except: pass

    # 共振
    reasons = []
    if signal_type: reasons.append("策略触发")
    if is_macd_gold: reasons.append("MACD金叉")
    if is_kdj_gold: reasons.append("KDJ金叉")
    if code in NORTHBOUND_SET: reasons.append("外资重仓")
    resonance_str = "+".join(reasons)

    return {
        "代码": code, "名称": name, "现价": curr["close"],
        "今日涨跌": f"{today_pct:+.2f}%", "3日涨跌": f"{pct_3day:+.2f}%",
        "K线形态": kline_status, "K线评分": kline_score,
        "60分状态": status_60m, "BIAS乖离": round(curr["BIAS20"], 1),
        "连续": "", "共振因子": resonance_str,
        "信号类型": signal_type, "热门概念": "",
        "OBV状态": "??健康流入",
        "筹码分布": chip_signal, "形态特征": pattern_str,
        "MACD状态": macd_detail, "布林状态": "??突破上轨" if curr["BB_PctB"] > 1.0 else "",
        "今日CMF": round(curr["CMF"], 3), "昨日CMF": round(prev["CMF"], 3), "前日CMF": round(prev_2["CMF"], 3),
        "RSI指标": round(curr["RSI"], 1), "J值": round(curr["J"], 1),
        "建议挂单": suggest_buy, "止损价": stop_loss,
        "换手率": turnover, "市盈率": pe
    }

# --- 评分与详情生成 ---
def calculate_score_and_details(row):
    score = 0
    details = []
    
    # 环境
    trend_str = str(MARKET_ENV_TEXT)
    if "暴跌" in trend_str: score -= 50; details.append("大盘暴跌-50")
    elif "多头" in trend_str: score += 10; details.append("大盘多头+10")
    
    # 技术分
    k_score = float(row.get('K线评分', 0))
    if k_score != 0: score += k_score; details.append(f"K线形态{k_score:+}")
    
    s60 = str(row.get('60分状态', ''))
    if "金叉" in s60: score += 100; details.append("60分金叉+100")
    elif "多头" in s60: score += 60; details.append("60分多头+60")
    
    # MACD 专项分
    macd_st = str(row.get('MACD状态', ''))
    if "金叉" in macd_st: score += 40; details.append("MACD金叉+40")
    elif "加油" in macd_st: score += 30; details.append("空中加油+30")
    
    # 资金
    try:
        c1, c2 = float(row.get('今日CMF', 0)), float(row.get('昨日CMF', 0))
        if c1 > c2: score += 15; details.append("资金流入+15")
    except: pass
    
    # 形态
    patterns = str(row.get('形态特征', ''))
    if "均线多头" in patterns: score += 30; details.append("均线多头+30")
    
    return score, " | ".join(details)

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
        code = res['代码']
        streak = 1
        for d in sorted_dates:
            if not hist_df[(hist_df['date'] == d) & (hist_df['code'] == str(code))].empty: streak += 1
            else: break
        res['连续'] = f"??{streak}连" if streak >= 2 else "首榜"
        processed_results.append(res)
        new_rows.append({"date": today_str, "code": str(code)})

    if new_rows: hist_df = pd.concat([hist_df, pd.DataFrame(new_rows)], ignore_index=True)
    try: hist_df.to_csv(HISTORY_FILE, index=False)
    except: pass
    return processed_results

def save_and_beautify(data_list):
    dt_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"严选_MACD金叉增强版_{dt_str}.xlsx"
    if not data_list: return
    
    df = pd.DataFrame(data_list)
    res = df.apply(calculate_score_and_details, axis=1)
    df["综合评分"] = [x[0] for x in res]; df["评分解析"] = [x[1] for x in res]
    
    cols = ["代码", "名称", "综合评分", "评分解析", "现价", "今日涨跌", "3日涨跌", "K线形态", "60分状态", 
            "BIAS乖离", "连续", "共振因子", "信号类型", "MACD状态", "形态特征", "今日CMF", "市盈率", "止损价"]
    df = df[cols].sort_values(by="综合评分", ascending=False)
    df.to_excel(filename, index=False)
    
    wb = openpyxl.load_workbook(filename); ws = wb.active
    header_font = Font(name='微软雅黑', size=11, bold=True, color="FFFFFF")
    fill_blue = PatternFill("solid", fgColor="4472C4")
    font_red = Font(color="FF0000", bold=True)
    
    for cell in ws[1]: cell.fill = fill_blue; cell.font = header_font
    for row in ws.iter_rows(min_row=2):
        row[3].alignment = Alignment(horizontal='left'); row[3].font = Font(size=9)
        if "金叉" in str(row[13].value): row[13].font = font_red
        if "均线多头" in str(row[14].value): row[14].font = font_red

    ws.column_dimensions['D'].width = 50; ws.column_dimensions['H'].width = 25
    
    # --- 指南与手册 (详细版) ---
    sr = ws.max_row + 3
    env_cell = ws.cell(row=sr, column=1, value=f"?? {MARKET_ENV_TEXT}")
    env_cell.font = Font(size=14, bold=True, color="FFFFFF")
    env_cell.fill = PatternFill("solid", fgColor="008000") if "多头" in MARKET_ENV_TEXT else PatternFill("solid", fgColor="FFA500")
    ws.merge_cells(start_row=sr, start_column=1, end_row=sr, end_column=18); sr += 2

    cat_font = Font(name='微软雅黑', size=12, bold=True, color="0000FF")
    ws.cell(row=sr, column=1, value="?? 五大策略实战手册 (高手进阶版)").font = cat_font; sr += 1
    strategies = [
        ("?? 黄金坑", "逻辑：股价经历大幅杀跌后缩量见底，BIAS跌破-8后，首次放量站上5日线。特点：左侧反转，盈亏比极高。", "操作：现价买入，止损设在坑底最低点。目标MA60。"),
        ("?? 龙回头", "逻辑：前期有30天内涨停板记录，主力已深度介入。回调至MA20或MA60缩量企稳。特点：二次爆发力强。", "操作：分批建仓，若放量反包前一阴线则重仓追击。"),
        ("?? 机构控盘", "逻辑：CMF吸筹指标>0.1且ADX趋势指标>25，股价维持MA60上方。特点：机构抱团，走势稳健。", "操作：沿MA10或MA20持股，不破不卖，防守位设在MA20。"),
        ("?? 极度超跌", "逻辑：RSI跌至20以下极度卖空区，且60分钟MACD出现背离金叉。特点：短线抢反弹利器。", "操作：快进快出，获利5%-10%即主动止盈，严格执行止损。"),
        ("? MACD金叉", "逻辑：DIF上穿DEA，且伴随跳空缺口或倍量阳线。特点：多头行情正式确立。", "操作：金叉当日介入，若金叉位置处于零轴上方（空中加油），爆发力更强。")
    ]
    for n, l, a in strategies:
        ws.cell(row=sr, column=1, value=n).font = Font(bold=True)
        ws.cell(row=sr, column=2, value=l); ws.cell(row=sr, column=3, value=a)
        ws.merge_cells(start_row=sr, start_column=3, end_row=sr, end_column=10); sr += 1
    sr += 1
    
    ws.cell(row=sr, column=1, value="?? 全指标读图指南 (必读)").font = cat_font; sr += 1
    indicators = [
        ("MACD状态", "??金叉：日线级别买入信号；??空中加油：强势主升延续；??绿柱缩短：底部回升初期。"),
        ("60分状态", "短线择时核心。?金叉代表1-3日内的上涨动力；??多头代表目前处于日内强势阶段。建议等60分不为'回调'时买。"),
        ("CMF吸筹", "蔡金资金流指标。今日>昨日>前日 代表主力正在不计成本加速抢筹。数值>0.1为强庄。"),
        ("BIAS/RSI", "风控指标。BIAS > 12 代表短线严重超买（防诱多）；RSI < 20 代表严重超跌（挖坑区）。"),
        ("跳空缺口", "K线形态列显示'??向上跳空'，是多头进攻的强烈宣告，缺口不补，强势不止。"),
        ("均线排列", "形态特征显示'??均线多头'，代表5/10/20/60均线顺向向上排列，是标准的牛股模板。"),
        ("止损价", "这是系统的风控铁律。收盘价跌破此价格，代表技术面支撑彻底失效，应坚决离场。")
    ]
    for n, d in indicators:
        ws.cell(row=sr, column=1, value=n).font = Font(bold=True)
        ws.cell(row=sr, column=2, value=d)
        ws.merge_cells(start_row=sr, start_column=2, end_row=sr, end_column=10); sr += 1
    
    wb.save(filename)
    print(f"? 增强版报告已生成: {filename}")

def analyze_one_stock(stock_info, start_dt):
    try:
        df = get_data_with_retry(stock_info['code'], start_dt)
        if df is None: return None
        return process_stock_logic(df, stock_info)
    except: return None

def main():
    get_market_context()
    targets = get_targets_robust()
    if not targets: return
    s_dt = (datetime.now() - timedelta(days=CONFIG["DAYS_LOOKBACK"])).strftime("%Y%m%d")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        futs = {executor.submit(analyze_one_stock, r, s_dt): r['code'] for r in targets}
        for f in concurrent.futures.as_completed(futs):
            res = f.result()
            if res: results.append(res)
    if results: results = update_history(results); save_and_beautify(results)

if __name__ == "__main__":
    main()
