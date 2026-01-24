# -*- coding: utf-8 -*-
"""
A股游资·天眼系统 (Ultimate Full-Armor Stable / 最终全装甲·网络稳定版)
版本: v2.0 Refined
优化内容: 指数退避重试、向量化计算、全局异常熔断、内存缓存
"""

import akshare as ak
import pandas as pd
import numpy as np
import time
import concurrent.futures
from datetime import datetime, timedelta
from tqdm import tqdm
from colorama import init, Fore, Style, Back
import warnings
import random
import sys
import http.client
import requests
import functools

# 初始化
init(autoreset=True)
warnings.filterwarnings('ignore')

# ==========================================
# 0. 全局作战配置 (Battle Configuration)
# ==========================================
class BattleConfig:
    # --- 基础筛选 (Funnel) ---
    MIN_CAP = 15 * 10**8       # 最小流通市值 15亿
    MAX_CAP = 400 * 10**8      # 最大流通市值 400亿 (容纳中军)
    MIN_PRICE = 3.0            # 最低价
    MAX_PRICE = 90.0           # 最高价
    
    # --- 活跃度门槛 ---
    FILTER_PCT_CHG = 2.0       # 涨幅 > 2% (捕捉起爆点，不过滤太多)
    FILTER_TURNOVER = 4.5      # 换手 > 4.5% (游资票必须活跃)
    
    # --- 系统参数 ---
    HISTORY_DAYS = 60          # K线回溯天数
    MAX_WORKERS = 8            # 分析引擎并发线程数
    FILE_NAME = f"Dragon_FullArmor_{datetime.now().strftime('%Y%m%d')}.xlsx"

# ==========================================
# 0.1 核心工具链 (Core Toolchain)
# ==========================================
def retry_robust(max_retries=3, base_delay=1.0, backoff_factor=2.0):
    """
    [新增] 指数退避重试装饰器
    功能：在网络请求失败时，按 1s -> 2s -> 4s 的节奏重试，并增加随机抖动防止惊群效应。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        # 增加 0-50% 的随机抖动
                        sleep_time = delay * (1 + random.random() * 0.5)
                        time.sleep(sleep_time)
                        delay *= backoff_factor
            # 重试耗尽，静默失败（符合原有逻辑），返回None或抛出特定异常
            # print(Fore.RED + f"    [API失败] {func.__name__}: {last_exception}")
            return None
        return wrapper
    return decorator

# ==========================================
# 1. 舆情风控哨兵 (News Sentry)
# ==========================================
class NewsSentry:
    """
    [优化] 增加缓存机制，优化字符串匹配算法。
    """
    NEGATIVE_KEYWORDS = [
        "立案", "调查", "违规", "警示", "减持", "亏损", "大幅下降", 
        "无法表示意见", "ST", "退市", "诉讼", "冻结", "留置", "黑天鹅"
    ]
    
    _cache = {} # 类级别缓存，防止同个代码重复请求

    @staticmethod
    @retry_robust(max_retries=2, base_delay=0.5)
    def check_news(code):
        # 1. 检查缓存
        if code in NewsSentry._cache:
            return NewsSentry._cache[code]

        try:
            df = ak.stock_news_em(symbol=code)
            if df is None or df.empty:
                return False, "无近期资讯"
            
            # 2. 向量化文本检查 (性能优化)
            # 将最近10条标题合并为一个大字符串进行搜索，比循环快
            recent_titles = df.head(10)['新闻标题'].astype(str).tolist()
            combined_text = " ".join(recent_titles)
            
            risk_msgs = []
            for kw in NewsSentry.NEGATIVE_KEYWORDS:
                if kw in combined_text:
                    risk_msgs.append(kw)
            
            if risk_msgs:
                # 去重
                unique_risks = sorted(list(set(risk_msgs)))
                result = (True, f"⚠️利空含:{','.join(unique_risks)}")
            else:
                result = (False, "舆情平稳")
            
            # 3. 写入缓存
            NewsSentry._cache[code] = result
            return result
            
        except:
            return False, "资讯接口跳过"

# ==========================================
# 2. 龙虎榜基因雷达 (Dragon-Tiger Radar)
# ==========================================
class DragonTigerRadar:
    """
    扫描最近3天的龙虎榜，建立游资基因库。
    """
    def __init__(self):
        self.lhb_stocks = set()

    def scan(self):
        print(Fore.MAGENTA + ">>> [3/8] 扫描游资龙虎榜基因...")
        try:
            for i in range(3): # 追溯3天
                d = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
                self._fetch_daily_lhb(d)
                
            print(Fore.GREEN + f"    ✅ 基因库构建完毕，收录 {len(self.lhb_stocks)} 只游资票")
        except Exception as e:
            print(Fore.YELLOW + f"    ⚠️ 龙虎榜接口波动(非致命): {e}")

    @retry_robust(max_retries=2, base_delay=0.5)
    def _fetch_daily_lhb(self, date_str):
        """内部辅助方法，带重试"""
        try:
            df = ak.stock_lhb_detail_daily_sina(date=date_str)
            if df is not None and not df.empty:
                codes = df['代码'].astype(str).tolist()
                self.lhb_stocks.update(codes)
        except:
            raise ValueError("LHB fetch failed") # 抛出异常以触发重试

    def has_gene(self, code):
        return code in self.lhb_stocks

# ==========================================
# 3. 热点与龙头锚定雷达 (Hot Concept & Leader)
# ==========================================
class HotConceptRadar:
    """
    扫描全市场热点，并锁定每个板块的【当前龙头】作为参照物。
    """
    def __init__(self):
        self.stock_concept_map = {}   # {个股代码: 概念名称}
        self.concept_leader_map = {}  # {概念名称: "龙头名(涨幅%)"}

    def scan(self):
        print(Fore.MAGENTA + ">>> [4/8] 扫描顶级热点 & 锁定板块龙头...")
        try:
            df_board = ak.stock_board_concept_name_em()
            noise = ["昨日", "连板", "首板", "涨停", "融资", "融券", "转债", "ST", "板块", "指数", "深股通", "沪股通"]
            mask = ~df_board['板块名称'].str.contains("|".join(noise))
            df_top = df_board[mask].sort_values(by="涨跌幅", ascending=False).head(10)
            hot_list = df_top['板块名称'].tolist()
            
            print(Fore.MAGENTA + f"    🔥 顶级风口: {hot_list[:6]}...")
            
            print(Fore.CYAN + "    ⚡ 正在精密扫描热点 (已开启限流保护模式)...")
            
            # 使用 ThreadPoolExecutor 并结合 retry 机制
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                futures = [ex.submit(self._fetch_constituents_safe, t) for t in hot_list]
                for f in concurrent.futures.as_completed(futures):
                    c_name, codes, l_info = f.result()
                    self.concept_leader_map[c_name] = l_info
                    for code in codes:
                        if code not in self.stock_concept_map: 
                            self.stock_concept_map[code] = []
                        self.stock_concept_map[code].append(c_name)
                        
            print(Fore.GREEN + f"    ✅ 龙头锚定完毕 (示例: {list(self.concept_leader_map.items())[0] if self.concept_leader_map else '无'})")
            
        except Exception as e:
            print(Fore.RED + f"    ⚠️ 热点雷达波动: {e}")

    @retry_robust(max_retries=2, base_delay=1.0)
    def _fetch_constituents_safe(self, name):
        """带重试的热点成分股获取"""
        try:
            df = ak.stock_board_concept_cons_em(symbol=name)
            if df is not None and not df.empty:
                leader_info = "未知"
                if '涨跌幅' in df.columns:
                    df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
                    df.sort_values(by='涨跌幅', ascending=False, inplace=True)
                    top_stock = df.iloc[0]
                    leader_info = f"{top_stock['名称']}({top_stock['涨跌幅']}%)"
                return name, df['代码'].tolist(), leader_info
            return name, [], "-"
        except Exception:
            raise ValueError("Concept fetch failed")

    def get_info(self, code):
        concepts = self.stock_concept_map.get(code, [])
        if not concepts: return False, "-", "-"
        main_concept = concepts[0]
        leader_info = self.concept_leader_map.get(main_concept, "-")
        return True, main_concept, leader_info

# ==========================================
# 4. 市场哨兵 (Market Sentry)
# ==========================================
class MarketSentry:
    @staticmethod
    @retry_robust(max_retries=2, base_delay=0.5)
    def check_market():
        print(Fore.MAGENTA + ">>> [2/8] 侦测大盘环境...")
        try:
            df = ak.stock_zh_index_daily(symbol="sh000001")
            if df is None or df.empty: raise ValueError("Index data missing")
            
            today = df.iloc[-1]
            pct = (today['close'] - today['open']) / today['open'] * 100
            
            if pct < -1.5:
                print(Fore.RED + f"    ⚠️ 警告：大盘暴跌 ({round(pct,2)}%)，已启动【防御模式】(只看硬板)。")
                BattleConfig.FILTER_PCT_CHG = 5.0
            else:
                print(Fore.GREEN + f"    ✅ 大盘环境正常 ({round(pct,2)}%)。")
        except:
            print(Fore.YELLOW + "    ⚠️ 大盘数据获取失败，默认正常模式。")

# ==========================================
# 5. 核心分析引擎 (Identity Engine)
# ==========================================
class IdentityEngine:
    def __init__(self, concept_radar, lhb_radar):
        self.concept_radar = concept_radar
        self.lhb_radar = lhb_radar

    @retry_robust(max_retries=3, base_delay=0.3)
    def get_kline(self, code):
        """[优化] 获取K线数据，集成重试与异常处理"""
        end = datetime.now().strftime("%Y%m%d")
        # 多取几天防止数据缺失
        start = (datetime.now() - timedelta(days=BattleConfig.HISTORY_DAYS + 10)).strftime("%Y%m%d")
        
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
        if df is not None and not df.empty:
            df.rename(columns={'日期':'date','开盘':'open','收盘':'close','最高':'high',
                               '最低':'low','成交量':'volume','成交额':'amount','涨跌幅':'pct_chg'}, inplace=True)
            return df
        raise ValueError("Empty K-line") # 触发重试

    def calculate_cmf(self, df):
        """[优化] 计算 CMF (向量化计算，极速版)"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df['volume']
            
            # 向量化操作
            range_hl = (high - low)
            # 避免除以0，替换为极小值
            range_hl = range_hl.replace(0, 0.01)
            
            mf_vol = (((close - low) - (high - close)) / range_hl) * volume
            
            # 使用 rolling sum 计算20日累积
            cmf_val = mf_vol.rolling(20).sum() / volume.rolling(20).sum()
            
            val = cmf_val.iloc[-1]
            return 0.0 if (np.isnan(val) or np.isinf(val)) else val
        except: 
            return 0.0

    def check_overheat(self, df, turnover):
        """情绪过热熔断器"""
        try:
            close = df['close']; pct_chg = df['pct_chg']
            # 1. RSI极度超买 (向量化)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(6).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
            # 避免 loss 为 0
            loss = loss.replace(0, 0.01)
            rsi = 100 - (100 / (1 + gain / loss))
            if rsi.iloc[-1] > 90: return True, "RSI超买"
            
            # 2. 加速赶顶
            today = df.iloc[-1]
            upper_s = today['high'] - max(today['open'], today['close'])
            body = abs(today['close'] - today['open'])
            if pct_chg.tail(3).sum() > 25.0 and (upper_s > body * 2):
                return True, "加速赶顶"
                
            return False, ""
        except: return False, ""

    def analyze(self, snapshot_row):
        code = snapshot_row['code']
        name = snapshot_row['name']
        
        # --- 1. 获取K线数据 ---
        df = self.get_kline(code)
        if df is None or len(df) < 30: return None
        
        today = df.iloc[-1]
        prev = df.iloc[-2]
        close = today['close']
        high = today['high']
        open_p = today['open']
        volume = today['volume']
        amount = today['amount']
        pct_chg = today['pct_chg']
        
        turnover = snapshot_row['turnover']
        vol_ratio = snapshot_row.get('量比', 0)
        cmf_val = self.calculate_cmf(df)
        
        # --- 2. 风险风控 (Defense) ---
        is_risk = False
        risk_msg = []
        score = 60
        features = []
        
        # A. 炸板/烂板检测
        if high >= prev['close'] * 1.095 and (high - close) / close > 0.03:
            is_risk = True; risk_msg.append("炸板/烂板")
            
        # B. 乖离率过大
        ma5 = df['close'].rolling(5).mean().iloc[-1]
        if ma5 > 0 and (close - ma5) / ma5 > 0.18:
            is_risk = True; risk_msg.append("乖离率大")
            
        # C. 均价压制
        vwap = amount / volume if volume > 0 else close
        if close < vwap * 0.985 and pct_chg < 9.8:
            is_risk = True; risk_msg.append("均价压制")
            
        # D. 情绪过热熔断
        is_oh, oh_msg = self.check_overheat(df, turnover)
        if is_oh: is_risk = True; risk_msg.append(oh_msg)

        # --- 3. 机会挖掘 (Offense) ---
        
        # A. 竞价与开盘
        if vol_ratio > 8.0: score += 15; features.append(f"竞价抢筹(量比{vol_ratio})")
        
        # B. 弱转强
        open_pct = (open_p - prev['close']) / prev['close'] * 100
        if prev['pct_chg'] < 3.0 and 2.0 < open_pct < 6.0:
            score += 20; features.append("🔥弱转强")
            
        # C. 基因
        limit_ups = len(df[df['pct_chg'] > 9.5].tail(20))
        if limit_ups > 0: score += 10; features.append(f"妖股({limit_ups}板)")
        if self.lhb_radar.has_gene(code): score += 20; features.append("🐉龙虎榜")
        
        # D. 资金 (CMF)
        if cmf_val > 0.15: score += 15; features.append("主力锁仓")
        elif cmf_val < -0.1: score -= 15; features.append("资金流出")
        
        # E. 热点
        is_hot, concept_name, leader_info = self.concept_radar.get_info(code)
        if is_hot:
            score += 25
            if name in leader_info: # 自己是龙头
                features.append(f"🔥板块龙头:{concept_name}")
                leader_display = "★本机★"
            else:
                features.append(f"热点:{concept_name}")
                leader_display = leader_info
        else:
            leader_display = "-"

        # --- 4. 舆情排雷 (Lazy Check) ---
        # 仅当分数足够高且无其他风险时，才请求舆情接口，节省网络资源
        news_msg = "平稳"
        if score > 80 and not is_risk:
            has_bad_news, n_msg = NewsSentry.check_news(code)
            if has_bad_news:
                is_risk = True
                risk_msg.append(n_msg)
                score -= 100
            news_msg = n_msg

        # --- 5. 最终裁决 ---
        if is_risk:
            score -= 100
            features.insert(0, f"⚠️{'/'.join(risk_msg)}")
        
        identity = "🐕跟风"
        advice = "观察"
        
        if is_risk: identity = "💀陷阱"; advice = "回避"
        elif score >= 110: identity = "🐲真龙 (T0)"; advice = "扫板/锁仓"
        elif "弱转强" in features and score >= 90: identity = "🚀接力 (T1)"; advice = "竞价跟随"
        elif cmf_val > 0.1 and not is_risk: identity = "💰趋势 (T1)"; advice = "低吸"
        else: identity = "🦊套利 (T2)"; advice = "快进快出"

        if score < 55 and not is_risk: return None
        
        return {
            "代码": code, "名称": name, "身份": identity, "建议": advice,
            "总分": score, 
            "板块龙头": leader_display, 
            "舆情风控": news_msg,
            "涨幅%": pct_chg, "换手%": turnover, "量比": vol_ratio,
            "CMF": round(cmf_val, 3), "特征": " | ".join(features)
        }

# ==========================================
# 6. 指挥官 (Commander)
# ==========================================
class Commander:
    def get_snapshot_robust(self):
        """
        [快照优先策略]
        最先执行，确保在网络状态最好、IP无污点时拉取大数据。
        """
        max_retries = 6 
        for attempt in range(max_retries):
            print(Fore.CYAN + f">>> [1/8] 获取全市场快照 (战术尝试 {attempt + 1}/{max_retries})...")
            
            if attempt > 0: time.sleep(random.uniform(2.0, 4.0))
            
            try:
                # 方案 A: 分层切片拉取 (优先)
                print(Fore.CYAN + "    ⚡ 启动分战区切片拉取模式 (降低负载)...")
                df_sh = ak.stock_sh_a_spot_em(); time.sleep(0.5)
                df_sz = ak.stock_sz_a_spot_em(); time.sleep(0.5)
                df_bj = ak.stock_bj_a_spot_em()
                df = pd.concat([df_sh, df_sz, df_bj], ignore_index=True)
                
            except Exception as split_err:
                print(Fore.YELLOW + f"    ⚠️ 分层拉取阻碍，启动降级方案...")
                # 方案 B: 降级单次拉取
                try:
                    time.sleep(2)
                    df = ak.stock_zh_a_spot_em()
                except Exception as mono_err:
                    print(Fore.RED + f"    ❌ 降级方案失败: {mono_err}")
                    continue 

            if df is not None and not df.empty and len(df) > 1000:
                rename_map = {
                    '代码':'code', '名称':'name', '最新价':'close', 
                    '涨跌幅':'pct_chg', '换手率':'turnover', 
                    '流通市值':'circ_mv', '量比':'量比'
                }
                df.rename(columns=rename_map, inplace=True)
                
                cols_to_numeric = ['close','pct_chg','turnover','circ_mv','量比']
                for c in cols_to_numeric:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
                
                print(Fore.GREEN + f"    ✅ 成功获取 {len(df)} 只股票数据！")
                return df
            else:
                print(Fore.YELLOW + "    ⚠️ 数据不完整，准备重试...")
        
        print(Fore.RED + "❌ 致命错误：无法获取行情数据。")
        return None

    def generate_excel(self, df_res):
        """生成带说明书和格式化的Excel"""
        try:
            with pd.ExcelWriter(BattleConfig.FILE_NAME, engine='xlsxwriter') as writer:
                df_res.to_excel(writer, sheet_name='真龙榜', index=False)
                
                manual_data = {
                    '关键列名': ['身份', '板块龙头', '舆情风控', '量比 (9:25专用)', 'CMF (14:30专用)', '特征-弱转强', '特征-炸板'],
                    '实战含义': [
                        '【真龙T0】: 确定性最高，热点+资金+龙虎榜共振；【陷阱】: 无论涨多好，坚决不买。',
                        '锚定效应。如果龙头涨停，你的跟风票才安全；如果龙头跳水，你的票要先跑。',
                        '一票否决。含“立案、调查”等字眼，大概率第二天跌停。',
                        '竞价抢筹指标。> 5.0 表示主力急不可耐；> 10 表示极度一致。',
                        '主力意图指标。> 0.15 表示主力锁仓；< 0 表示主力流出。',
                        '最强游资信号。昨日弱势，今日高开爆量，往往是连板起点。',
                        '最强风险信号。摸过涨停但没封住，次日大概率核按钮。'
                    ]
                }
                pd.DataFrame(manual_data).to_excel(writer, sheet_name='实战说明书', index=False)
                
                wb = writer.book
                ws = writer.sheets['真龙榜']
                fmt_bad = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
                ws.conditional_format('C2:C150', {'type': 'text', 'criteria': 'containing', 'value': '陷阱', 'format': fmt_bad})
                ws.conditional_format('G2:G150', {'type': 'text', 'criteria': 'containing', 'value': '利空', 'format': fmt_bad})
                fmt_good = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
                ws.conditional_format('C2:C150', {'type': 'text', 'criteria': 'containing', 'value': '真龙', 'format': fmt_good})
        except Exception as e:
            print(Fore.RED + f"Excel生成出错: {e}")

    def run(self):
        print(Fore.GREEN + f"=== 🐲 A股游资·天眼系统 (Snapshot-First / v2.0 Refined) ===")
        print(Fore.YELLOW + f"🕒 当前时间: {datetime.now().strftime('%H:%M:%S')}")

        # STEP 1: 获取快照
        df = self.get_snapshot_robust()
        if df is None: return

        # STEP 2: 战术冷却
        print(Fore.YELLOW + "\n>>> ❄️ 核心数据获取完毕，战术冷却 5 秒 (释放连接)...")
        time.sleep(5)
        print("    ✅ 网络通道重置完毕。\n")

        # STEP 3 & 4: 启动雷达
        MarketSentry.check_market()
        lhb = DragonTigerRadar()
        lhb.scan()
        concept = HotConceptRadar()
        concept.scan()

        # STEP 5: 漏斗筛选
        print(Fore.CYAN + ">>> [5/8] 漏斗筛选...")
        mask = (
            (~df['name'].str.contains('ST|退|C|U')) & 
            (~df['code'].str.startswith(('8','4','92'))) &
            (df['close'].between(BattleConfig.MIN_PRICE, BattleConfig.MAX_PRICE)) &
            (df['circ_mv'].between(BattleConfig.MIN_CAP, BattleConfig.MAX_CAP)) &
            (df['pct_chg'] >= BattleConfig.FILTER_PCT_CHG) &
            (df['turnover'] >= BattleConfig.FILTER_TURNOVER) &
            (df['量比'] > 0.8)
        )
        candidates = df[mask].copy()
        print(Fore.YELLOW + f"    📉 入围: {len(candidates)} 只")

        # STEP 6: 深度运算 (并发优化版)
        print(Fore.CYAN + ">>> [6/8] 深度运算 (资金+风控+舆情+龙头锚定)...")
        engine = IdentityEngine(concept, lhb)
        results = []
        
        target_rows = candidates.sort_values(by='量比', ascending=False).head(150)
        tasks = [row.to_dict() for _, row in target_rows.iterrows()]
        
        # 优化进度条显示
        pbar = tqdm(total=len(tasks), desc="    ⚡ 分析进度", unit="股", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=BattleConfig.MAX_WORKERS) as ex:
            futures = {ex.submit(engine.analyze, task): task for task in tasks}
            for f in concurrent.futures.as_completed(futures):
                try:
                    # 增加30秒超时，防止线程挂死
                    res = f.result(timeout=30)
                    if res: results.append(res)
                except concurrent.futures.TimeoutError:
                    # 超时忽略，不打印错误以免刷屏
                    pass 
                except Exception:
                    pass
                finally:
                    pbar.update(1)
        pbar.close()

        # STEP 7: 导出
        print(Fore.CYAN + f">>> [7/8] 生成战报: {BattleConfig.FILE_NAME}")
        if results:
            df_res = pd.DataFrame(results)
            df_res.sort_values(by='总分', ascending=False, inplace=True)
            cols = ['代码','名称','身份','建议','板块龙头','舆情风控','总分','涨幅%','量比','CMF','特征']
            final_cols = [c for c in cols if c in df_res.columns]
            df_res = df_res[final_cols]
            self.generate_excel(df_res)
            print(Fore.GREEN + f"✅ 成功! 请打开 Excel 查看【实战说明书】")
            print(df_res[['名称','身份','板块龙头','特征']].head(5).to_string(index=False))
        else:
            print(Fore.RED + "❌ 无有效标的。")

if __name__ == "__main__":
    Commander().run()
