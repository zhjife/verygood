import akshare as ak
import pandas as pd
import numpy as np
from ta.trend import ADXIndicator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
import time
import concurrent.futures
import random
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

# --- 全局统计计数器 (诊断核心) ---
DEBUG_STATS = Counter()
FAIL_EXAMPLES = {} # 记录失败样本

# --- 配置 ---
CONFIG = {
    "MIN_AMOUNT": 20000000,
    "MIN_PRICE": 2.5,
    "MAX_WORKERS": 8,
    "DAYS_LOOKBACK": 250,
    "BLACKLIST_DAYS": 30
}

RESTRICTED_LIST = [] 

# --- 1. 基础数据 ---
def get_market_context():
    global RESTRICTED_LIST
    print("📡 [1/2] 正在获取解禁名单(防雷)...")
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

def get_targets_robust():
    print(">>> [2/2] 获取全市场标的并进行初筛...")
    try:
        df = ak.stock_zh_a_spot_em()
        col_map = {"最新价": "price", "成交额": "amount", "代码": "code", "名称": "name", 
                   "换手率": "turnover", "市盈率-动态": "pe", "总市值": "mktcap"}
        df.rename(columns=col_map, inplace=True)
        
        # 记录原始数量
        DEBUG_STATS['0. 全市场总数'] = len(df)
        
        df["price"] = pd.to_numeric(df["price"], errors='coerce')
        df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
        df["turnover"] = pd.to_numeric(df["turnover"], errors='coerce')
        
        df.dropna(subset=["price", "amount"], inplace=True)
        
        # 逐步过滤并记录
        df = df[df["code"].str.startswith(("60", "00"))]
        df = df[~df['name'].str.contains('ST|退')]
        DEBUG_STATS['1. 剔除ST/科创/北交'] = DEBUG_STATS['0. 全市场总数'] - len(df)
        
        temp_len = len(df)
        df = df[df["price"] >= CONFIG["MIN_PRICE"]]
        DEBUG_STATS['2. 剔除低价股(<2.5)'] = temp_len - len(df)
        
        temp_len = len(df)
        df = df[df["amount"] > CONFIG["MIN_AMOUNT"]]
        DEBUG_STATS['3. 剔除成交额低(<2000万)'] = temp_len - len(df)
        
        temp_len = len(df)
        df = df[~df["code"].isin(RESTRICTED_LIST)]
        DEBUG_STATS['4. 剔除解禁风险股'] = temp_len - len(df)
        
        print(f"✅ 进入深度扫描标的: {len(df)} 只")
        return df.to_dict('records')
    except Exception as e:
        print(f"⚠️ 异常: {e}")
        return []

def get_data_with_retry(code, start_date):
    time.sleep(random.uniform(0.001, 0.01)) 
    for _ in range(2):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq", timeout=3)
            if df is not None and not df.empty: return df
        except: time.sleep(0.1)
    return None

# --- 2. 核心诊断逻辑 ---
def process_stock_logic(df, stock_info):
    code = stock_info['code']
    name = stock_info['name']
    turnover = stock_info.get('turnover', 0)

    # 1. 数据长度检查
    if len(df) < 120: 
        DEBUG_STATS['A. 数据不足120天'] += 1
        return None
    
    rename_dict = {"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount"}
    col_map = {k:v for k,v in rename_dict.items() if k in df.columns}
    df.rename(columns=col_map, inplace=True)
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    # 计算指标
    df["pct_chg"] = close.pct_change() * 100
    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["BIAS20"] = (close - df["MA20"]) / df["MA20"] * 100
    
    bb = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_Bar"] = (df["DIF"] - df["DEA"]) * 2
    
    # KDJ
    low_9 = low.rolling(9, min_periods=9).min()
    high_9 = high.rolling(9, min_periods=9).max()
    rsv = (close - low_9) / (high_9 - low_9) * 100
    rsv = rsv.fillna(50)
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # OBV & CMF & ADX
    df["OBV"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["OBV_MA10"] = df["OBV"].rolling(10).mean()
    df["CMF"] = ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()
    df["ADX"] = ADXIndicator(high, low, close, window=14).adx()

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # --- 🔍 熔断诊断区 (Fail Fast Diagnostics) ---
    
    # 1. 换手率出货检查
    has_zt = (df["pct_chg"].tail(30) > 9.5).sum() >= 1
    is_today_limit = curr["close"] >= round(prev["close"] * 1.095, 2)
    if turnover > 25 and not is_today_limit: 
        DEBUG_STATS['B. 换手过高且非涨停'] += 1
        return None
    
    # 2. 追高风险
    if curr["J"] > 105: 
        DEBUG_STATS['C. J值过高(超买)'] += 1
        if random.random() < 0.01: FAIL_EXAMPLES['J值高'] = f"{name}: {curr['J']:.1f}"
        return None 
    
    # 3. 资金流向 (OBV)
    if curr["OBV"] <= curr["OBV_MA10"]: 
        DEBUG_STATS['D. OBV趋势向下(资金流出)'] += 1
        return None

    # 4. 资金强度 (CMF) - 这是一个强过滤
    if curr["CMF"] < 0.05: 
        DEBUG_STATS['E. CMF资金强度不足(<0.05)'] += 1
        if random.random() < 0.005: FAIL_EXAMPLES['CMF弱'] = f"{name}: {curr['CMF']:.3f}"
        return None
    
    # 5. 资金加速 (CMF Acceleration)
    if curr["CMF"] <= prev["CMF"]: 
        DEBUG_STATS['F. CMF未加速(资金衰退)'] += 1
        return None
        
    # 6. 动能 (MACD)
    if curr["MACD_Bar"] <= prev["MACD_Bar"]: 
        DEBUG_STATS['G. MACD动能减弱'] += 1
        return None 

    # --- 🔍 策略匹配诊断区 ---
    signal_type = ""
    
    # 策略A: 黄金坑
    is_deep_dip = (prev["BIAS20"] < -8) 
    is_reversal = (curr["close"] > curr["MA5"]) and (curr["pct_chg"] > 1.5)
    if is_deep_dip and is_reversal: signal_type = "黄金坑"
    
    # 策略B: 龙回头
    if not signal_type and has_zt and curr["close"] > curr["MA60"]:
        vol_ratio = curr["volume"] / df["volume"].tail(5).mean()
        if vol_ratio < 0.85: 
            if -8.0 < curr["BIAS20"] < 8.0 and curr["close"] > df["BB_Lower"].iloc[-1]:
                signal_type = "龙回头"
    
    # 策略C: 机构控盘
    if not signal_type and curr["close"] > curr["MA60"] and curr["CMF"] > 0.1 and curr["ADX"] > 25:
        signal_type = "机构控盘"
    
    # 策略D: 底部变盘
    if not signal_type and curr["close"] < curr["MA60"] * 1.2 and curr["BB_Width"] < 12:
        signal_type = "底部变盘"

    if not signal_type:
        DEBUG_STATS['H. 通过指标但未匹配策略'] += 1
        # 记录一些“好苗子”但没匹配上策略的，看看是不是策略太严
        if random.random() < 0.01: FAIL_EXAMPLES['无策略'] = f"{name}: CMF={curr['CMF']:.2f}, ADX={curr['ADX']:.1f}"
        return None
        
    # 金叉检查
    is_macd_gold = (prev["DIF"] < prev["DEA"]) and (curr["DIF"] > curr["DEA"])
    is_kdj_gold = (prev["J"] < prev["K"]) and (curr["J"] > curr["K"]) and (curr["J"] < 80)
    
    if signal_type != "黄金坑":
        if not (is_macd_gold or is_kdj_gold): 
            DEBUG_STATS['I. 缺少金叉共振'] += 1
            return None

    DEBUG_STATS['✅ 成功入选'] += 1
    return {"code": code, "name": name, "signal": signal_type}

def analyze_one_stock(stock_info, start_dt):
    try:
        df = get_data_with_retry(stock_info['code'], start_dt)
        if df is None: 
            DEBUG_STATS['X. 数据获取失败'] += 1
            return None
        return process_stock_logic(df, stock_info)
    except: 
        DEBUG_STATS['X. 运行异常'] += 1
        return None

def main():
    print("=== 🛡️ A股严选·选股漏斗诊断工具 ===")
    print("正在进行快速扫描，请稍候...")
    
    get_market_context()
    targets = get_targets_robust()
    
    start_dt = (datetime.now() - timedelta(days=CONFIG["DAYS_LOOKBACK"])).strftime("%Y%m%d")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        future_to_stock = {executor.submit(analyze_one_stock, r, start_dt): r['code'] for r in targets}
        count = 0
        total = len(targets)
        for future in concurrent.futures.as_completed(future_to_stock):
            count += 1
            if count % 200 == 0: print(f"进度: {count}/{total}...")
            future.result()

    # --- 🖨️ 打印诊断报告 ---
    print("\n" + "="*40)
    print("📊 选股漏斗诊断报告 (Funnel Report)")
    print("="*40)
    
    # 按键名排序打印
    keys = sorted(DEBUG_STATS.keys())
    for k in keys:
        count = DEBUG_STATS[k]
        print(f"{k.ljust(25)}: {count} 只")
    
    print("-" * 40)
    print("💡 典型失败样本 (Sample Failures):")
    for k, v in FAIL_EXAMPLES.items():
        print(f"  [{k}]: {v}")
        
    print("="*40)
    
    # 给出优化建议
    print("\n🩺 医生建议:")
    if DEBUG_STATS['D. OBV趋势向下(资金流出)'] > len(targets) * 0.4:
        print("🔴 市场资金面较差：大量股票资金在流出。建议：不做或只做'龙回头'低吸。")
    if DEBUG_STATS['E. CMF资金强度不足(<0.05)'] > len(targets) * 0.5:
        print("🔴 主力活跃度低：CMF过滤太严。建议：将CMF阈值从0.05降低到0.02或0。")
    if DEBUG_STATS['G. MACD动能减弱'] > len(targets) * 0.4:
        print("🔴 动能衰退期：大量股票MACD红柱缩短。建议：耐心等待回调结束。")
    if DEBUG_STATS['H. 通过指标但未匹配策略'] > 100:
        print("🟡 策略太死板：很多股票指标不错但没套进模型。建议：放宽'ADX>25'或'BIAS'限制。")
    if DEBUG_STATS['✅ 成功入选'] == 0:
        print("❌ 当前无股入选。请尝试修改代码中的以下阈值：")
        print("   1. process_stock_logic 中: if curr['CMF'] < 0.05 -> 改为 < 0")
        print("   2. process_stock_logic 中: if curr['OBV'] <= curr['OBV_MA10'] -> 注释掉")
        print("   3. 找回 get_targets_robust 中: price >= 2.5 (是否过滤了低价妖股?)")

if __name__ == "__main__":
    main()
