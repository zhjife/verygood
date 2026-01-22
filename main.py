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

# ==========================================
# 0. 战备配置 (Battle Config)
# ==========================================
init(autoreset=True)
warnings.filterwarnings('ignore')

class BattleConfig:
    # 基础门槛
    MIN_CAP = 12 * 10**8
    MAX_CAP = 1200 * 10**8 # 涵盖大盘中军
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
    FILE_NAME = f"Titan_Dragon_Eye_{datetime.now().strftime('%Y%m%d')}.xlsx"

# ==========================================
# 1. 泰坦雷达 (Titan Radar)
# ==========================================
class TitanRadar:
    """
    全维溯源：
    [金] = 行业资金流 (机构)
    [业] = 行业涨幅榜 (贝塔)
    [概] = 概念涨幅榜 (游资)
    """
    def __init__(self):
        self.hot_stock_map = {} 
        self.active_sources = []

    def scan_market(self):
        print(Fore.MAGENTA + ">>> [1/5] 启动真龙雷达 (全维溯源)...")
        targets = [] 

        # A. 资金源 (机构战场)
        try:
            df_fund = ak.stock_market_fund_flow()
            df_fund = df_fund.sort_values(by="今日主力净流入", ascending=False).head(5)
            for _, row in df_fund.iterrows():
                targets.append((row['名称'], 50, "[金]")) # 50分
        except: pass

        # B. 行业源 (板块贝塔)
        try:
            df_ind = ak.stock_board_industry_name_em()
            df_ind = df_ind.sort_values(by="涨跌幅", ascending=False).head(5)
            for _, row in df_ind.iterrows():
                targets.append((row['板块名称'], 40, "[业]")) # 40分
        except: pass

        # C. 题材源 (游资战场)
        try:
            df_con = ak.stock_board_concept_name_em()
            # 完整去噪
            noise = ["昨日", "连板", "首板", "涨停", "融资", "融券", "转债", "ST", "标普", "指数", "高股息", "破净", "增持", "深股通", "沪股通", "AB股", "AH股", "同花顺", "MSCI"]
            mask = ~df_con['板块名称'].str.contains("|".join(noise))
            df_con = df_con[mask].sort_values(by="涨跌幅", ascending=False).head(15)
            for i, (_, row) in enumerate(df_con.iterrows()):
                name = row['板块名称']
                if i < 3: score = 45     # 龙一龙二龙三
                elif i < 8: score = 25   # 中排
                else: score = 15         # 后排
                targets.append((name, score, "[概]"))
        except: pass
        
        self.active_sources = [f"{t[2]}{t[0]}" for t in targets]
        print(Fore.MAGENTA + f"    🎯 锁定源头: {self.active_sources[:6]}... (共{len(targets)}个)")

        # D. 倒排索引 (Inverted Index)
        print(Fore.MAGENTA + "    📥 构建内存索引...")
        
        def fetch_cons(t):
            name, score, type_ = t
            try:
                if "[金]" in type_ or "[业]" in type_:
                    df = ak.stock_board_industry_cons_em(symbol=name)
                else:
                    df = ak.stock_board_concept_cons_em(symbol=name)
                return name, score, type_, df['代码'].tolist()
            except:
                try: 
                    df = ak.stock_board_concept_cons_em(symbol=name)
                    return name, score, type_, df['代码'].tolist()
                except: return name, 0, "", []

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(fetch_cons, t) for t in targets]
            for f in concurrent.futures.as_completed(futures):
                name, score, type_, codes = f.result()
                for code in codes:
                    if code not in self.hot_stock_map:
                        self.hot_stock_map[code] = {'score': 0, 'sources': set()}
                    curr = self.hot_stock_map[code]['score']
                    self.hot_stock_map[code]['score'] = min(curr + score, 95) # 上限95
                    self.hot_stock_map[code]['sources'].add(f"{type_}{name}")

    def check(self, code):
        if code in self.hot_stock_map:
            d = self.hot_stock_map[code]
            return d['score'], list(d['sources'])
        return 0, []

# ==========================================
# 2. 静态知识库 (Static Backup)
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
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=BattleConfig.HISTORY_DAYS)).strftime("%Y%m%d")
        for _ in range(2):
            try:
                time.sleep(random.uniform(0.01, 0.05))
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
                if df is not None and not df.empty:
                    df.rename(columns={'日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'volume', '涨跌幅':'pct_chg'}, inplace=True)
                    return df
            except: pass
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
        
        # 1. 趋势一票否决 (生命线)
        if ma60 > 0 and curr < ma60: return None
        
        # 2. 攻击形态 (根据模式切换)
        is_bull_trend = (ma5 > ma10)
        is_breakout = (curr > ma20) and (df['open'].values[-1] < ma20)
        
        if is_strict_mode:
            # 严格模式：必须 多头排列 OR 强势突破
            if not (is_bull_trend or is_breakout): return None
        else:
            # 宽松模式：只要站稳20日线
            if ma20 > 0 and curr < ma20: return None
        
        if is_bull_trend: tech_reasons.append("多头排列")
        if is_breakout: tech_reasons.append("一阳穿线")

        # --- C. 源头溯源 (Source Tracing) ---
        dyn_score, dyn_sources = self.radar.check(code)
        static_sources = StaticKnowledge.match(name)
        all_sources = list(set(dyn_sources + static_sources))
        
        # --- D. 股性与分数 (DNA) ---
        tech_score = 60
        
        # 1. 妖股记忆
        limit_ups = len(df[df['pct_chg'] > 9.5].tail(15))
        if limit_ups >= 2: 
            tech_score += 20; tech_reasons.append(f"妖股基因({limit_ups}板)")
        
        # 2. 突破新高
        h120 = df['high'].iloc[-120:].max()
        if (h120 - curr) / curr < 0.05: 
            tech_score += 20; tech_reasons.append("突破新高")
            
        # 3. 量能确认
        vol_ma5 = pd.Series(df['volume'].values).rolling(5).mean().values[-1]
        if vol_ma5 > 0 and (df['volume'].values[-1] / vol_ma5) > 1.2:
            tech_score += 5; tech_reasons.append("放量")
        
        # --- E. 身份认定 (Identity Definition) ---
        total_score = tech_score + dyn_score + (len(static_sources)*10)
        
        # 筛选门槛 (严格模式下保持高标准)
        score_threshold = 85 if is_strict_mode else 70
        
        # 如果既无动态源头，也无静态源头，且分数不够，剔除
        if dyn_score == 0 and len(static_sources) == 0 and total_score < score_threshold:
            return None
        
        # 最终及格线
        if total_score < 70: return None
        
        # === 核心：身份定义 ===
        identity = "跟风 (T3)"
        advice = "观察"
        
        has_fund = any("[金]" in s for s in all_sources)
        has_concept = any("[概]" in s for s in all_sources)
        
        # T0: 真龙 (三维共振)
        if total_score >= 95 and has_concept and (has_fund or limit_ups >= 1):
            identity = "🐲真龙 (T0)"
            advice = "锁仓/抢筹"
            
        # T1: 中军 (资金驱动)
        elif has_fund and base_info['circ_mv'] > 80 * 10**8:
            identity = "🐢中军 (T1)"
            advice = "均线低吸"
            
        # T1: 先锋 (情绪驱动)
        elif has_concept and (limit_ups >= 1 or "突破新高" in tech_reasons):
            identity = "🚀先锋 (T1)"
            advice = "打板/半路"
            
        # T2: 趋势 (技术驱动)
        elif "突破新高" in tech_reasons:
            identity = "💰趋势龙 (T2)"
            advice = "5日线跟随"
            
        # T3: 宽松模式下的防守
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
# 4. 指挥中枢 (Auto-Scaling Commander)
# ==========================================
class Commander:
    def run(self):
        print(Fore.GREEN + "=== 🐲 A股游资·真龙天眼 (Titan: Dragon Eye Ultimate) ===")
        print(Fore.WHITE + "包含：身份定义(T0-T3) | 源头溯源 | 铁血风控 | 自动降级")
        
        radar = TitanRadar()
        radar.scan_market()
        
        print(Fore.CYAN + ">>> [2/5] 获取快照...")
        try:
            df = ak.stock_zh_a_spot_em()
            df.rename(columns={'代码':'code', '名称':'name', '最新价':'close', '涨跌幅':'pct_chg', 
                              '换手率':'turnover', '总市值':'total_mv', '流通市值':'circ_mv'}, inplace=True)
            for c in ['close', 'pct_chg', 'turnover', 'circ_mv']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        except Exception as e:
            print(Fore.RED + f"❌ 快照失败: {e}"); self.save_empty(); return

        print(Fore.CYAN + ">>> [3/5] 执行自适应漏斗...")
        
        # 0. 基础池
        base_mask = (
            (~df['name'].str.contains('ST|退|C|U')) & 
            (df['close'].between(BattleConfig.MIN_PRICE, BattleConfig.MAX_PRICE)) &
            (df['circ_mv'].between(BattleConfig.MIN_CAP, BattleConfig.MAX_CAP))
        )
        base_pool = df[base_mask].copy()
        print(Fore.WHITE + f"    [INFO] 基础池: {len(base_pool)} 只")
        
        # 1. 尝试[进攻模式] (Strict)
        strict_mask = (
            (base_pool['pct_chg'] >= BattleConfig.STRICT_PCT_CHG) & 
            (base_pool['turnover'] >= BattleConfig.STRICT_TURNOVER)
        )
        candidates = base_pool[strict_mask].copy()
        IS_STRICT = True 
        
        # 2. 自动降级判断
        if len(candidates) < 5:
            print(Fore.YELLOW + f"    ⚠️ 进攻目标过少({len(candidates)})，切换至 [防守模式]...")
            print(Fore.YELLOW + f"       标准降级: 涨幅>{BattleConfig.LOOSE_PCT_CHG}%, 换手>{BattleConfig.LOOSE_TURNOVER}%")
            
            loose_mask = (
                (base_pool['pct_chg'] >= BattleConfig.LOOSE_PCT_CHG) & 
                (base_pool['turnover'] >= BattleConfig.LOOSE_TURNOVER)
            )
            candidates = base_pool[loose_mask].copy()
            IS_STRICT = False
        else:
            print(Fore.GREEN + f"    ⚔️ 市场火热，维持 [进攻模式] (真龙标准)")

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
            # 排序：优先看总分
            results.sort(key=lambda x: x['总分'], reverse=True)
            df_res = pd.DataFrame(results[:40])
            
            # 完美复刻真龙天眼的输出列
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
