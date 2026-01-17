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

    # --- [新增] 跳空缺口逻辑 ---
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
    df["MA10"] = close.rolling(10).mean() # 新增用于多头排列
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["BIAS20"] = (close - df["MA20"]) / df["MA20"] * 100

    bb_ind = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb_ind.bollinger_hband()
    df["BB_Lower"] = bb_ind.bollinger_lband()
    df["BB_Width"] = bb_ind.bollinger_wband()
    df["BB_PctB"] = bb_ind.bollinger_pband()

    # 指标 (国产算法)
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

    # --- 策略 ---
    signal_type = ""
    suggest_buy = curr["close"]
    stop_loss = curr["MA20"]
    
    # 策略A: 黄金坑
    is_deep_dip = (prev["BIAS20"] < -8) or (prev["RSI"] < 20)
    is_reversal = (curr["close"] > curr["MA5"]) and (curr["pct_chg"] > 1.5)
    if is_deep_dip and is_reversal:
        signal_type = "??黄金坑(企稳)"; stop_loss = round(curr["low"] * 0.98, 2)
    
    # 策略B: 龙回头 (量能优化)
    if not signal_type and has_zt and curr["close"] > curr["MA60"]:
        vol_ratio = curr["volume"] / df["volume"].tail(5).mean()
        if vol_ratio < 0.85: # 缩量
            if -8.0 < curr["BIAS20"] < 8.0 and curr["close"] > df["BB_Lower"].iloc[-1]:
                signal_type = "??龙回头"; stop_loss = min(prev["low"], df["BB_Lower"].iloc[-1])
    
    # 策略C: 机构控盘
    if not signal_type and curr["close"] > curr["MA60"] and curr["CMF"] > 0.1 and curr["ADX"] > 25:
        signal_type = "??机构控盘"; suggest_buy = round(curr["vwap"], 2)
    
    # 策略D: 底部变盘
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
    # --- [新增] 均线多头判定 ---
    if curr["MA5"] > curr["MA10"] > curr["MA20"] > curr["MA60"]:
        patterns.append("??均线多头")

    vol_up = df[df['close']>df['open']].tail(20)['volume'].sum()
    vol_down = df[df['close']<df['open']].tail(20)['volume'].sum()
    if vol_up > vol_down * 2.0 and curr["close"] > curr["MA20"]: patterns.append("??红肥绿瘦")
    if (prev['close'] < prev['open']) and (curr['close'] > curr['open']) and (curr['close'] > prev['open']): patterns.append("?N字反包")
    
    recent_5 = df.tail(5)
    if (recent_5['close'] > recent_5['MA5']).all() and (recent_5['pct_chg'].abs() < 4.0).all() and (recent_5['close'].iloc[-1] > recent_5['close'].iloc[0]):
        patterns.append("??蚂蚁上树")
    pattern_str = " ".join(patterns)

    is_macd_gold = (prev["DIF"] < prev["DEA"]) and (curr["DIF"] > curr["DEA"])
    is_kdj_gold = (prev["J"] < prev["K"]) and (curr["J"] > curr["K"]) and (curr["J"] < 80)
    
    if signal_type != "??黄金坑(企稳)":
        if not (is_macd_gold or is_kdj_gold): return None

    # --- 检查 ---
    has_strategy = bool(signal_type)
    has_resonance = bool(chip_signal or pattern_str) 
    if not (has_strategy or has_resonance): return None

    kline_status, kline_score = analyze_kline_health(df)

    # --- 60分钟 ---
    status_60m = "?数据不足"
    try:
        df_60 = get_60m_data_optimized(code)
        if df_60 is not None and len(df_60) > 20:
            close_60 = df_60["close"]
            ema12_60 = close_60.ewm(span=12, adjust=False).mean()
            ema26_60 = close_60.ewm(span=26, adjust=False).mean()
            dif_60 = ema12_60 - ema26_60
            dea_60 = dif_60.ewm(span=9, adjust=False).mean()
            d_curr, e_curr = dif_60.iloc[-1], dea_60.iloc[-1]
            d_prev, e_prev = dif_60.iloc[-2], dea_60.iloc[-2]
            
            if d_prev < e_prev and d_curr > e_curr:
                status_60m = "?60分金叉"
            elif d_curr > e_curr: status_60m = "??60分多头"
            elif d_curr < e_curr: status_60m = "??60分回调"
            else: status_60m = "?60分震荡"
        else:
            status_60m = "?获取失败"
    except Exception as e: 
        status_60m = "??计算异常"

    # --- 组装 ---
    cross_status = ""
    if is_macd_gold and is_kdj_gold: cross_status = "?双金叉"
    elif is_macd_gold: cross_status = "??MACD金叉"
    elif is_kdj_gold: cross_status = "??KDJ金叉"
    elif signal_type == "??黄金坑(企稳)": cross_status = "??绿柱缩短"

    reasons = []
    if signal_type: reasons.append("策略")
    if has_resonance: reasons.append("筹/形共振")
    if cross_status == "?双金叉": reasons.append("双金叉")
    if code in NORTHBOUND_SET: reasons.append("外资重仓")
    resonance_str = "+".join(reasons)

    news_title = get_stock_catalysts(code)
    hot_matched = ""
    for hot in HOT_CONCEPTS:
        if hot in news_title: hot_matched = hot; break
    display_concept = f"??{hot_matched}" if hot_matched else ""

    macd_warn = "?空中加油" if (curr["DIF"]>curr["DEA"] and curr["DIF"]>0 and curr["MACD_Bar"]>prev["MACD_Bar"]) else ""
    bar_trend = "??红增" if curr["MACD_Bar"] > 0 else "??绿缩"
    final_macd = f"{bar_trend}|{macd_warn if macd_warn else cross_status}"
    bb_state = "??突破上轨" if curr["BB_PctB"] > 1.0 else ("??极度收口" if curr["BB_Width"] < 12 else "")

    return {
        "代码": code, "名称": name, "现价": curr["close"],
        "今日涨跌": f"{today_pct:+.2f}%", "3日涨跌": f"{pct_3day:+.2f}%",
        "K线形态": kline_status, "K线评分": kline_score,
        "60分状态": status_60m, "BIAS乖离": round(curr["BIAS20"], 1),
        "连续": "", "共振因子": resonance_str,
        "信号类型": signal_type, "热门概念": display_concept,
        "OBV状态": "??健康流入",
        "筹码分布": chip_signal, "形态特征": pattern_str,
        "MACD状态": final_macd, "布林状态": bb_state,
        "今日CMF": round(curr["CMF"], 3), "昨日CMF": round(prev["CMF"], 3), "前日CMF": round(prev_2["CMF"], 3),
        "RSI指标": round(curr["RSI"], 1), "J值": round(curr["J"], 1),
        "建议挂单": suggest_buy, "止损价": stop_loss,
        "换手率": turnover, "市盈率": pe
    }

# --- 评分与详情生成 (白盒化) ---
def calculate_score_and_details(row):
    score = 0
    details = []
    
    # 1. 大盘环境熔断 (Beta)
    trend_str = str(MARKET_ENV_TEXT)
    if "暴跌" in trend_str: 
        score -= 50; details.append("??大盘暴跌-50")
    elif "空头" in trend_str: 
        score -= 15; details.append("???大盘空头-15")
    elif "多头" in trend_str: 
        score += 10; details.append("???大盘多头+10")
    
    # 2. 技术面评分 (含跳空缺口分)
    k_score = float(row.get('K线评分', 0))
    if k_score != 0: 
        score += k_score; details.append(f"K线形态分{k_score:+}")
    
    s60 = str(row.get('60分状态', ''))
    if "金叉" in s60: 
        score += 100; details.append("?60分金叉+100")
    elif "多头" in s60: 
        score += 80; details.append("??60分多头+80")
    elif "回调" in s60: 
        score -= 20; details.append("??60分回调-20")
    
    # 3. 趋势与连板
    streak = str(row.get('连续', ''))
    if "3连" in streak or "4连" in streak: 
        score += 50; details.append("??连板强势+50")
    elif "2连" in streak: 
        score += 30; details.append("??2连板+30")
    
    # 4. 资金面
    try:
        c1, c2, c3 = float(row.get('今日CMF', 0)), float(row.get('昨日CMF', 0)), float(row.get('前日CMF', 0))
        if c1 > c2 > c3: 
            score += 30; details.append("??资金加速+30")
        elif c1 > c2: 
            score += 10; details.append("资金流入+10")
    except: pass
    
    if "外资" in str(row.get('共振因子', '')): 
        score += 25; details.append("??北向重仓+25")
        
    # 5. 量价结构 (含多头排列分)
    patterns = str(row.get('形态特征', ''))
    if "红肥" in patterns: 
        score += 15; details.append("??红肥绿瘦+15")
    if "均线多头" in patterns:
        score += 30; details.append("??均线多头+30")
    
    # 6. 信号加成
    if "黄金坑" in str(row.get('信号类型', '')): 
        score += 20; details.append("??黄金坑+20")
    if "双金叉" in str(row.get('共振因子', '')): 
        score += 15; details.append("?双金叉+15")
    if "??" in str(row.get('热门概念', '')): 
        score += 15; details.append("??蹭热点+15")
    
    # 7. 基本面估值 (PE)
    try:
        pe = float(row.get('市盈率', 0))
        if 0 < pe < 25: 
            score += 25; details.append("??绩优低估+25")
        elif 25 <= pe < 50: 
            score += 10; details.append("??估值合理+10")
        elif pe < 0: 
            score -= 20; details.append("?业绩亏损-20")
        elif pe > 150: 
            score -= 15; details.append("??估值过高-15")
    except: pass
    
    # 8. 乖离率风控 (BIAS)
    try:
        bias = float(row.get('BIAS乖离', 0))
        if bias > 18: 
            score -= 40; details.append("??乖离过大-40")
    except: pass

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
        code = res['code'] if 'code' in res else res['代码']
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
    filename = f"严选_透明评分最终版_{dt_str}.xlsx"
    
    if not data_list:
        pd.DataFrame([["无股入选 (条件严苛)"]]).to_excel(filename)
        print("?? 今日无标的入选")
        return filename

    df = pd.DataFrame(data_list)
    res = df.apply(calculate_score_and_details, axis=1)
    df["综合评分"] = [x[0] for x in res]
    df["评分解析"] = [x[1] for x in res]
    
    cols = ["代码", "名称", "综合评分", "评分解析", "现价", "今日涨跌", "3日涨跌", "K线形态", "60分状态", 
            "BIAS乖离", "连续", "共振因子", "信号类型", "热门概念", "OBV状态", "今日CMF", 
            "昨日CMF", "前日CMF", "筹码分布", "形态特征", "MACD状态", "布林状态", 
            "RSI指标", "J值", "建议挂单", "止损价", "换手率", "市盈率"]
    for c in cols:
        if c not in df.columns: df[c] = ""
    df = df[cols]
    df.sort_values(by=["综合评分"], ascending=False, inplace=True)
    df.to_excel(filename, index=False)
    
    wb = openpyxl.load_workbook(filename)
    ws = wb.active
    
    header_font = Font(name='微软雅黑', size=11, bold=True, color="FFFFFF")
    fill_blue = PatternFill("solid", fgColor="4472C4")
    font_red = Font(color="FF0000", bold=True)
    font_green = Font(color="008000", bold=True)
    font_purple = Font(color="800080", bold=True)
    fill_yellow = PatternFill("solid", fgColor="FFF2CC")
    
    for cell in ws[1]:
        cell.fill = fill_blue
        cell.font = header_font
    
    for row in ws.iter_rows(min_row=2):
        if row[2].value and float(row[2].value) >= 150: row[2].fill = PatternFill("solid", fgColor="FFC7CE") 
        row[3].alignment = Alignment(horizontal='left')
        row[3].font = Font(size=9)

        for idx in [5, 6]: 
            val = str(row[idx].value)
            if "+" in val: row[idx].font = font_red
            elif "-" in val: row[idx].font = font_green
        
        k_val = str(row[7].value)
        if "强攻" in k_val or "仙人" in k_val or "跳空" in k_val: row[7].font = font_red
        elif "护盘" in k_val: row[7].font = font_purple
        elif "抛压" in k_val: row[7].font = font_green; row[7].fill = fill_yellow

        if "金叉" in str(row[8].value): row[8].font = font_red; row[8].fill = fill_yellow
        elif "回调" in str(row[8].value): row[8].font = font_green

        bias_val = row[9].value
        if isinstance(bias_val, (int, float)):
            if bias_val < -8: row[9].font = font_green; row[9].fill = fill_yellow
            elif bias_val > 12: row[9].font = font_red

        if "连" in str(row[10].value): row[10].font = font_red; row[10].fill = fill_yellow
        if "外资" in str(row[11].value): row[11].font = font_red; row[11].fill = fill_yellow
        if "流入" in str(row[14].value): row[14].font = font_red
        if "红增" in str(row[20].value): row[20].font = font_red
        if "均线多头" in str(row[19].value): row[19].font = font_red
        
        try:
            c1, c2, c3 = float(row[15].value), float(row[16].value), float(row[17].value)
            row[15].font = font_red
            if c1 > c2 > c3:
                row[15].fill = fill_yellow; row[16].font = font_red; row[17].font = font_red
        except: pass

        if "蚂蚁" in str(row[19].value): row[19].font = font_purple
        if "红肥" in str(row[19].value): row[19].font = font_red

    # 调整列宽
    ws.column_dimensions['D'].width = 50 
    ws.column_dimensions['H'].width = 25 
    ws.column_dimensions['I'].width = 15
    ws.column_dimensions['L'].width = 25
    
    # --- [重点恢复] 底部大盘看板、策略手册和指南 ---
    start_row = ws.max_row + 3
    
    # 大盘环境
    env_cell = ws.cell(row=start_row, column=1, value=f"?? {MARKET_ENV_TEXT}")
    env_cell.font = Font(size=14, bold=True, color="FFFFFF")
    if "多头" in MARKET_ENV_TEXT: env_cell.fill = PatternFill("solid", fgColor="008000")
    else: env_cell.fill = PatternFill("solid", fgColor="FFA500")
    ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=26)
    start_row += 2

    cat_font = Font(name='微软雅黑', size=12, bold=True, color="0000FF")
    text_font = Font(name='微软雅黑', size=10)
    
    # 策略手册
    ws.cell(row=start_row, column=1, value="?? 五大策略实战手册 (透明评分版)").font = cat_font
    start_row += 1
    strategies = [
        ("?? 黄金坑", "【核心逻辑】深跌(BIAS<-8)后，今日放量阳线站稳MA5。左侧反转第一天。", "【买卖点】现价买入。止损设在前日最低点。"),
        ("?? 龙回头", "【核心逻辑】前期妖股回调至生命线(MA60/MA20)附近，极致缩量。", "【买卖点】在'建议挂单'价位低吸。跌破布林下轨止损。"),
        ("?? 机构控盘", "【核心逻辑】CMF>0.1(强吸筹) + ADX趋势向上 + 均线多头。", "【买卖点】沿5日线/10日线持股。"),
        ("?? 极度超跌", "【核心逻辑】RSI(6)<20 或 底背离，且资金未流出。", "【买卖点】左侧分批买入，反弹5-10%即止盈。"),
        ("? 底部变盘", "【核心逻辑】布林带宽<12(极度收口) + 资金异动。", "【买卖点】放量突破布林上轨瞬间追击。")
    ]
    for name, logic, action in strategies:
        ws.cell(row=start_row, column=1, value=name).font = Font(bold=True)
        ws.cell(row=start_row, column=2, value=logic).font = text_font
        ws.cell(row=start_row, column=3, value=action).font = text_font
        ws.merge_cells(start_row=start_row, start_column=3, end_row=start_row, end_column=10)
        start_row += 1
    start_row += 1
    
    # 指标指南
    ws.cell(row=start_row, column=1, value="?? 全指标读图指南").font = cat_font
    start_row += 1
    indicators = [
        ("评分解析", "?? 逻辑全透明：显示详细的加分/减分理由，一眼看懂为何该股高分。"),
        ("均线多头", "?? 新增高权分(+30)：MA5>MA10>MA20>MA60，代表该股处于标准主升趋势。"),
        ("跳空缺口", "?? 新增高权分(+40)：向上跳空代表主力进攻意愿极强，是极佳的突破信号。"),
        ("K线形态", "??实体强攻：多头强势；???金针探底：主力托底；??仙人指路：上涨中继。"),
        ("60分状态", "?金叉(黄底)：日内最佳买点；??多头(红字)：顺势持股；??回调(绿字)：短线震荡。"),
        ("CMF三日", "主力吸筹指标。若[前<昨<今]且标黄，代表资金加速抢筹，爆发力最强。"),
        ("BIAS乖离", "<-8%：黄金坑买入区； >12%：谨防短线冲高回落风险。"),
        ("止损价", "? 风控铁律！收盘价跌破此价格，说明技术逻辑破坏，必须无条件离场。")
    ]
    for name, desc in indicators:
        ws.cell(row=start_row, column=1, value=name).font = Font(bold=True)
        ws.cell(row=start_row, column=2, value=desc).font = text_font
        ws.merge_cells(start_row=start_row, start_column=2, end_row=start_row, end_column=10)
        start_row += 1

    wb.save(filename)
    print(f"? 增强版结果已保存: {filename}")
    return filename

def analyze_one_stock(stock_info, start_dt):
    try:
        df = get_data_with_retry(stock_info['code'], start_dt)
        if df is None: return None
        return process_stock_logic(df, stock_info)
    except: return None

def main():
    print("=== A股严选 (逻辑增强+全图表版) ===")
    get_market_context() 
    start_time = time.time()
    targets = get_targets_robust() 
    if not targets: return

    start_dt = (datetime.now() - timedelta(days=CONFIG["DAYS_LOOKBACK"])).strftime("%Y%m%d")
    
    print(f"?? 待扫描: {len(targets)} 只 | 启动 {CONFIG['MAX_WORKERS']} 线程...")
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        future_to_stock = {executor.submit(analyze_one_stock, r, start_dt): r['code'] for r in targets}
        count = 0
        total = len(targets)
        for future in concurrent.futures.as_completed(future_to_stock):
            count += 1
            if count % 100 == 0: print(f"进度: {count}/{total} ...")
            try:
                res = future.result()
                if res: results.append(res)
            except: pass

    if results: results = update_history(results)
    print(f"\n耗时: {int(time.time() - start_time)}秒 | 选中 {len(results)} 只")
    save_and_beautify(results)

if __name__ == "__main__":
    main()
