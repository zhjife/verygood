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
import os
import requests # 引入requests以处理底层异常

# ==========================================
# 0. 战备配置 (Battle Config)
# ==========================================
init(autoreset=True)
warnings.filterwarnings('ignore')

class BattleConfig:
    # 基础门槛
    MIN_CAP = 12 * 10**8
    MAX_CAP = 1200 * 10**8 
    MIN_PRICE = 2.0
    MAX_PRICE = 130.0
    
    # --- [A] 进攻模式 (真龙标准) ---
    STRICT_PCT_CHG = 3.5       
    STRICT_TURNOVER = 3.8      
    
    # --- [B] 防守模式 (冰点标准) ---
    LOOSE_PCT_CHG = 0.5        
    LOOSE_TURNOVER = 1.0       
    
    HISTORY_DAYS = 250
    MAX_WORKERS = 8 
    FILE_NAME = f"Titan_Dragon_Eye_Retry_{datetime.now().strftime('%Y%m%d')}.xlsx"

# ==========================================
# 1. 泰坦雷达 (Titan Radar - Enhanced Retry)
# ==========================================
class TitanRadar:
    """
    全维溯源：[金]资金流 | [业]行业势 | [概]概念风
    *增加：板块获取时的重试机制，防止漏掉热点*
    """
    def __init__(self):
        self.hot_stock_map = {} 
        self.active_sources = []

    def _retry_fetch(self, func, retries=3, delay=1):
        """通用重试装饰器"""
        for i in range(retries):
            try:
                return func()
            except Exception as e:
                if i == retries - 1: return None # 最后一次失败返回None
                time.sleep(delay)
        return None

    def scan_market(self):
        print(Fore.MAGENTA + ">>> [1/5] 启动真龙雷达 (全维溯源 + 网络硬化)...")
        targets = [] 

        # --- A. 资金源 (机构战场) ---
        def get_funds():
            df = ak.stock_market_fund_flow()
            return df.sort_values(by="今日主力净流入", ascending=False).head(5)
        
        df_fund = self._retry_fetch(get_funds)
        if df_fund is not None:
            for _, row in df_fund.iterrows():
                targets.append((row['名称'], 50, "[金]"))
        else:
            print(Fore.RED + "    ⚠️ 资金流接口多次请求失败，已跳过")

        # --- B. 行业源 (板块贝塔) ---
        def get_industry():
            df = ak.stock_board_industry_name_em()
            return df.sort_values(by="涨跌幅", ascending=False).head(5)

        df_ind = self._retry_fetch(get_industry)
        if df_ind is not None:
            for _, row in df_ind.iterrows():
                targets.append((row['板块名称'], 40, "[业]"))

        # --- C. 题材源 (游资战场) ---
        def get_concepts():
            df = ak.stock_board_concept_name_em()
            noise = ["昨日", "连板", "首板", "涨停", "融资", "融券", "转债", "ST", "标普", "指数", "高股息", "破净", "增持", "深股通", "沪股通", "AB股", "AH股", "同花顺", "MSCI"]
            mask = ~df['板块名称'].str.contains("|".join(noise))
            return df[mask].sort_values(by="涨跌幅", ascending=False).head(15)

        df_con = self._retry_fetch(get_concepts)
        if df_con is not None:
            for i, (_, row) in enumerate(df_con.iterrows()):
                name = row['板块名称']
                if i < 3: score = 45     
                elif i < 8: score = 25   
                else: score = 15         
                targets.append((name, score, "[概]"))
        
        self.active_sources = [f"{t[2]}{t[0]}" for t in targets]
        print(Fore.MAGENTA + f"    🎯 锁定源头: {self.active_sources[:6]}... (共{len(targets)}个)")

        # --- D. 倒排索引 (Inverted Index with Retry) ---
        print(Fore.MAGENTA + "    📥 构建内存索引 (含并发重试)...")
        
        def fetch_cons(t):
            name, score, type_ = t
            # 内部定义重试逻辑
            for attempt in range(3):
                try:
                    time.sleep(random.uniform(0.1, 0.3)) # 随机延迟
                    if "[金]" in type_ or "[业]" in type_:
                        df = ak.stock_board_industry_cons_em(symbol=name)
                    else:
                        df = ak.stock_board_concept_cons_em(symbol=name)
                    return name, score, type_, df['代码'].tolist()
                except:
                    # 尝试互查兜底
                    try:
                        df = ak.stock_board_concept_cons_em(symbol=name)
                        return name, score, type_, df['代码'].tolist()
                    except:
                        time.sleep(1) # 失败等待
                        continue
            return name, 0, "", [] # 彻底失败

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(fetch_cons, t) for t in targets]
            for f in concurrent.futures.as_completed(futures):
                name, score, type_, codes = f.result()
                for code in codes:
                    if code not in self.hot_stock_map:
                        self.hot_stock_map[code] = {'score': 0, 'sources': set()}
                    curr = self.hot_stock_map[code]['score']
                    self.hot_stock_map[code]['score'] = min(curr + score, 95) 
                    self.hot_stock_map[code]['sources'].add(f"{type_}{name}")

    def check(self, code):
        if code in self.hot_stock_map:
            d = self.hot_stock_map[code]
            return d['score'], list(d['sources'])
        return 0, []

# ==========================================
# 2. 静态知识库 (Static Knowledge)
# ==========================================
class StaticKnowledge:
    # 补充API可能缺失的常识性关联
    THEME_DICT = {
        "低空经济": ["飞行汽车", "eVTOL", "无人机", "万丰", "中信海直", "宗申", "设计"],
        "华为链": ["华为", "海思", "鸿蒙", "欧拉", "昇腾", "常山", "润和", "软通", "拓维"],
        "AI算力": ["CPO", "光模块", "液冷", "英伟达", "铜连接", "工业富联", "寒武纪", "中际"],
        "固态电池": ["固态", "硫化物", "清陶", "赣锋", "宁德", "有研", "紫江"],
        "并购重组": ["重组", "股权转让", "借壳", "双成", "银之杰", "光智", "电投"],
        "大金融": ["证券", "互联金融", "东方财富", "同花顺", "中信", "指南针"]
    }

    @staticmethod
    def match(name):
        hits = []
        for theme, kws in StaticKnowledge.THEME_DICT.items():
            for kw in kws:
                if kw in name:
                    hits.append(f"[静]{theme}")
                    break 
        return hits

# ==========================================
# 3. 身份判别引擎 (Identity Engine)
# ==========================================
class IdentityEngine:
    def __init__(self, radar):
        self.radar = radar

    def get_kline(self, code):
        """
        获取K线，增加强力重试机制
        """
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=BattleConfig.HISTORY_DAYS)).strftime("%Y%m%d")
        
        for attempt in range(4): # 提升到4次重试
            try:
                # 动态延迟：重试次数越多，等待越久
                sleep_time = random.uniform(0.1, 0.3) + (attempt * 0.5)
                time.sleep(sleep_time)
                
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
                if df is not None and not df.empty:
                    df.rename(columns={'日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'volume', '涨跌幅':'pct_chg'}, inplace=True)
                    return df
            except Exception:
                continue
        return None

    def analyze(self, base_info, is_strict_mode):
        code = base_info['code']
        name = base_info['name']
        
        # --- A. K线数据获取 ---
        df = self.get_kline(code)
        if df is None or len(df) < 30: return None
        
        close = df['close'].values
        curr = close[-1]
        
        # 均线计算
        ma_list = {}
        for w in [5, 10, 20, 60]:
            if len(close) >= w:
                ma_list[w] = pd.Series(close).rolling(w).mean().values[-1]
            else: ma_list[w] = 0
        ma60 = ma_list.get(60, 0)
        ma20 = ma_list.get(20, 0)
        ma10 = ma_list.get(10, 0)
        ma5 = ma_list.get(5, 0)

        # --- B. 技术铁律 (The Filter) ---
        tech_reasons = []
        
        # 1. 趋势一票否决
        if ma60 > 0 and curr < ma60: return None
        
        # 2. 攻击形态
        is_bull_trend = (ma5 > ma10)
        is_breakout = (curr > ma20) and (df['open'].values[-1] < ma20)
        
        if is_strict_mode:
            if not (is_bull_trend or is_breakout): return None
        else:
            if ma20 > 0 and curr < ma20: return None
        
        if is_bull_trend: tech_reasons.append("多头排列")
        if is_breakout: tech_reasons.append("一阳穿线")

        # --- C. 源头溯源 ---
        dyn_score, dyn_sources = self.radar.check(code)
        static_sources = StaticKnowledge.match(name)
        all_sources = list(set(dyn_sources + static_sources))
        
        # --- D. 股性与分数 ---
        tech_score = 60
        
        limit_ups = len(df[df['pct_chg'] > 9.5].tail(15))
        if limit_ups >= 2: 
            tech_score += 20; tech_reasons.append(f"妖股基因({limit_ups}板)")
        
        h120 = df['high'].iloc[-120:].max()
        if (h120 - curr) / curr < 0.05: 
            tech_score += 20; tech_reasons.append("突破新高")
            
        vol_ma5 = pd.Series(df['volume'].values).rolling(5).mean().values[-1]
        if vol_ma5 > 0 and (df['volume'].values[-1] / vol_ma5) > 1.2:
            tech_score += 5; tech_reasons.append("放量")
        
        # --- E. 身份认定 ---
        total_score = tech_score + dyn_score + (len(static_sources)*10)
        
        score_threshold = 85 if is_strict_mode else 70
        
        if dyn_score == 0 and len(static_sources) == 0 and total_score < score_threshold:
            return None
        
        if total_score < 70: return None
        
        identity = "跟风 (T3)"
        advice = "观察"
        
        has_fund = any("[金]" in s for s in all_sources)
        has_concept = any("[概]" in s for s in all_sources)
        
        if total_score >= 95 and has_concept and (has_fund or limit_ups >= 1):
            identity = "🐲真龙 (T0)"
            advice = "锁仓/抢筹"
        elif has_fund and base_info['circ_mv'] > 80 * 10**8:
            identity = "🐢中军 (T1)"
            advice = "均线低吸"
        elif has_concept and (limit_ups >= 1 or "突破新高" in tech_reasons):
            identity = "🚀先锋 (T1)"
            advice = "打板/半路"
        elif "突破新高" in tech_reasons:
            identity = "💰趋势龙 (T2)"
            advice = "5日线跟随"
        elif not is_strict_mode:
            identity = "🛡️防守 (T3)"
            advice = "低吸套利"

        return {
            "代码": code, "名称": name,
            "身份": identity,
            "结论": advice,
            "总分": total_score,
            "上涨源头": ",".join(all_sources) if all_sources else "-",
            "技术特征": "|".join(tech_reasons),
            "涨幅%": base_info['pct_chg'],
            "换手%": base_info['turnover']
        }

# ==========================================
# 4. 指挥中枢 (Commander - Network Hardened)
# ==========================================
class Commander:
    
    def get_snapshot_safe(self):
        """
        [网络硬化核心]
        1. 循环重试 10 次
        2. 指数退避 (Sleep时间递增)
        3. 备用接口切换
        """
        print(Fore.CYAN + ">>> [2/5] 获取全市场快照 (硬化模式)...")
        
        # 阶段一：尝试东财接口 (Deadly Persistence)
        for i in range(1, 11):
            try:
                print(Fore.YELLOW + f"    ⚡ 正在连接东财服务器 (第 {i}/10 次)...")
                df = ak.stock_zh_a_spot_em()
                
                # 校验数据
                df.rename(columns={'代码':'code', '名称':'name', '最新价':'close', '涨跌幅':'pct_chg', 
                                  '换手率':'turnover', '总市值':'total_mv', '流通市值':'circ_mv'}, inplace=True)
                for c in ['close', 'pct_chg', 'turnover', 'circ_mv']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                
                if len(df) > 3000:
                    print(Fore.GREEN + f"    ✅ 连接成功，接收 {len(df)} 条数据")
                    return df
            except Exception as e:
                wait = 2 + i # 递增等待时间
                print(Fore.RED + f"    ❌ 连接失败: {str(e)[:40]}... 等待 {wait} 秒")
                time.sleep(wait)
        
        # 阶段二：东财彻底失败，切换新浪 (Fallback)
        print(Fore.MAGENTA + "    ⚠️ 东财线路熔断，紧急切换 [新浪财经] 线路...")
        try:
            df = ak.stock_zh_a_spot() 
            # 新浪列名适配
            rename_map = {'symbol':'code', 'name':'name', 'trade':'close', 'pricechangepercent':'pct_chg', 
                          'turnoverratio':'turnover', 'mktcap':'total_mv', 'nmc':'circ_mv'}
            # 简单校验
            if 'trade' in df.columns:
                df.rename(columns=rename_map, inplace=True)
                print(Fore.GREEN + f"    ✅ 新浪接口接入成功")
                return df
        except Exception as e:
            print(Fore.RED + f"    ❌ 备用线路也失败: {e}")
            
        return None

    def run(self):
        print(Fore.GREEN + "=== 🐲 A股游资·真龙天眼 (网络硬化版) ===")
        print(Fore.WHITE + "架构：T0-T3身份 | 全维溯源 | 自动降级 | 死磕重试")
        
        radar = TitanRadar()
        radar.scan_market()
        
        # 使用硬化后的快照获取
        df = self.get_snapshot_safe()
        
        if df is None or df.empty:
            print(Fore.RED + "❌ 致命错误：全网断连，无法获取行情。"); self.save_empty(); return

        print(Fore.CYAN + ">>> [3/5] 执行自适应漏斗...")
        
        # 确保列名存在
        required = ['code', 'name', 'close', 'pct_chg', 'turnover', 'circ_mv']
        for c in required:
            if c not in df.columns:
                print(Fore.RED + f"❌ 数据缺列: {c}"); self.save_empty(); return

        # 0. 基础池
        base_mask = (
            (~df['name'].str.contains('ST|退|C|U')) & 
            (df['close'].between(BattleConfig.MIN_PRICE, BattleConfig.MAX_PRICE)) &
            (df['circ_mv'].between(BattleConfig.MIN_CAP, BattleConfig.MAX_CAP))
        )
        base_pool = df[base_mask].copy()
        print(Fore.WHITE + f"    [INFO] 基础池: {len(base_pool)} 只")
        
        # 1. 尝试[进攻模式]
        strict_mask = (
            (base_pool['pct_chg'] >= BattleConfig.STRICT_PCT_CHG) & 
            (base_pool['turnover'] >= BattleConfig.STRICT_TURNOVER)
        )
        candidates = base_pool[strict_mask].copy()
        IS_STRICT = True 
        
        # 2. 自动降级判断
        if len(candidates) < 5:
            print(Fore.YELLOW + f"    ⚠️ 目标过少({len(candidates)})，切换 [防守模式]...")
            loose_mask = (
                (base_pool['pct_chg'] >= BattleConfig.LOOSE_PCT_CHG) & 
                (base_pool['turnover'] >= BattleConfig.LOOSE_TURNOVER)
            )
            candidates = base_pool[loose_mask].copy()
            IS_STRICT = False
        else:
            print(Fore.GREEN + f"    ⚔️ 市场火热，维持 [进攻模式]")

        candidates = candidates.sort_values(by='turnover', ascending=False).head(150)
        print(Fore.YELLOW + f"    📉 入围深度分析: {len(candidates)} 只")
        
        if len(candidates) == 0:
            print(Fore.RED + "❌ 市场极度冰点，无标的。"); self.save_empty(); return

        # 4. 深度分析
        engine = IdentityEngine(radar)
        results = []
        tasks = [row.to_dict() for _, row in candidates.iterrows()]
        
        print(Fore.CYAN + f">>> [4/5] 深度运算 (模式: {'Strict' if IS_STRICT else 'Loose'})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=BattleConfig.MAX_WORKERS) as ex:
            futures = [ex.submit(engine.analyze, task, IS_STRICT) for task in tasks]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
                res = f.result()
                if res: results.append(res)

        # 5. 导出
        print(Fore.CYAN + f">>> [5/5] 导出: {BattleConfig.FILE_NAME}")
        if results:
            results.sort(key=lambda x: x['总分'], reverse=True)
            df_res = pd.DataFrame(results[:40])
            
            cols = ["代码", "名称", "身份", "结论", "总分", "上涨源头", "技术特征", "涨幅%", "换手%"]
            df_res = df_res[[c for c in cols if c in df_res.columns]]
            
            df_res.to_excel(BattleConfig.FILE_NAME, index=False)
            print(Fore.GREEN + f"✅ 成功锁定 {len(df_res)} 只标的。")
            print(df_res[['名称', '身份', '结论', '上涨源头']].head(5).to_string(index=False))
        else:
            print(Fore.RED + "❌ 分析后无结果"); self.save_empty()

    def save_empty(self):
        pd.DataFrame(columns=["Info"]).to_excel(BattleConfig.FILE_NAME)

if __name__ == "__main__":
    Commander().run()
