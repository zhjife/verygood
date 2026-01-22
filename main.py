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
# 0. 战备配置
# ==========================================
init(autoreset=True)
warnings.filterwarnings('ignore')

class BattleConfig:
    MIN_CAP = 18 * 10**8
    MAX_CAP = 1000 * 10**8 # 放宽上限以容纳中军
    MIN_PRICE = 3.0
    MAX_PRICE = 120.0
    FILTER_PCT_CHG = 3.5       
    FILTER_TURNOVER = 3.8      
    HISTORY_DAYS = 250
    MAX_WORKERS = 8 
    FILE_NAME = f"Dragon_Eye_{datetime.now().strftime('%Y%m%d')}.xlsx"

# ==========================================
# 1. 全维共振雷达 (Source Tracer)
# ==========================================
class ResonanceRadar:
    """
    负责寻找上涨源头，并构建倒排索引。
    区分：[金]资金流、[业]行业势、[概]情绪口
    """
    def __init__(self):
        # {code: {'score': int, 'sources': set()}}
        self.hot_stock_map = {} 
        self.active_sources = []

    def scan_market(self):
        print(Fore.MAGENTA + ">>> [1/5] 启动真龙雷达 (资金/行业/题材 三维扫描)...")
        targets = [] # (Name, Score, Type)

        # --- A. 资金源 (机构战场) ---
        try:
            df_fund = ak.stock_market_fund_flow()
            df_fund = df_fund.sort_values(by="今日主力净流入", ascending=False).head(5)
            for _, row in df_fund.iterrows():
                targets.append((row['名称'], 50, "[金]")) # 50分高权重
        except: pass

        # --- B. 行业源 (板块轮动) ---
        try:
            df_ind = ak.stock_board_industry_name_em()
            df_ind = df_ind.sort_values(by="涨跌幅", ascending=False).head(5)
            for _, row in df_ind.iterrows():
                targets.append((row['板块名称'], 40, "[业]"))
        except: pass

        # --- C. 题材源 (游资战场) ---
        try:
            df_con = ak.stock_board_concept_name_em()
            noise = ["昨日", "连板", "首板", "涨停", "融资", "融券", "转债", "ST", "标普", "指数", "高股息", "破净", "增持", "深股通", "沪股通", "AB股", "AH股"]
            mask = ~df_con['板块名称'].str.contains("|".join(noise))
            df_con = df_con[mask].sort_values(by="涨跌幅", ascending=False).head(15)
            
            for i, (_, row) in enumerate(df_con.iterrows()):
                name = row['板块名称']
                # 龙一板块给高分
                if i < 3: score = 45     
                elif i < 8: score = 25   
                else: score = 15         
                targets.append((name, score, "[概]"))
        except: pass
        
        # 记录源头
        self.active_sources = [f"{t[2]}{t[0]}" for t in targets]
        print(Fore.MAGENTA + f"    🎯 核心源头: {self.active_sources[:8]}... (共{len(targets)}个)")

        # --- D. 倒排索引构建 (精准匹配) ---
        print(Fore.MAGENTA + "    📥 正在溯源成分股...")
        
        def fetch_cons(t):
            name, score, type_ = t
            try:
                if "[金]" in type_ or "[业]" in type_:
                    df = ak.stock_board_industry_cons_em(symbol=name)
                else:
                    df = ak.stock_board_concept_cons_em(symbol=name)
                return name, score, type_, df['代码'].tolist()
            except:
                try: # 容错兜底
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
                    
                    # 叠加分数 (上限90)
                    curr = self.hot_stock_map[code]['score']
                    self.hot_stock_map[code]['score'] = min(curr + score, 90)
                    # 记录源头标签
                    self.hot_stock_map[code]['sources'].add(f"{type_}{name}")
                    
        print(Fore.GREEN + f"    ✅ 索引构建完毕，覆盖 {len(self.hot_stock_map)} 只活跃股")

    def check(self, code):
        if code in self.hot_stock_map:
            d = self.hot_stock_map[code]
            return d['score'], list(d['sources'])
        return 0, []

# ==========================================
# 2. 静态知识库 (Static Backup)
# ==========================================
class StaticKnowledge:
    # 补充常识
    THEME_DICT = {
        "低空经济": ["飞行汽车", "eVTOL", "无人机", "万丰", "中信海直", "宗申"],
        "华为链": ["华为", "海思", "鸿蒙", "欧拉", "昇腾", "常山", "润和"],
        "AI算力": ["CPO", "光模块", "液冷", "英伟达", "铜连接", "工业富联", "寒武纪"],
        "固态电池": ["固态", "硫化物", "清陶", "赣锋", "宁德"],
        "并购重组": ["重组", "股权转让", "借壳", "双成", "银之杰"],
        "大金融": ["证券", "互联金融", "东方财富", "同花顺", "中信"]
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
        for _ in range(3):
            try:
                time.sleep(random.uniform(0.01, 0.05))
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
                if df is not None and not df.empty:
                    df.rename(columns={'日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'volume', '涨跌幅':'pct_chg'}, inplace=True)
                    return df
            except: pass
        return None

    def analyze(self, base_info):
        code = base_info['code']
        name = base_info['name']
        
        # --- A. 技术铁律 (Survival) ---
        df = self.get_kline(code)
        if df is None or len(df) < 60: return None
        
        close = df['close'].values
        ma5, ma10, ma20, ma60 = [pd.Series(close).rolling(w).mean().values for w in [5,10,20,60]]
        curr = close[-1]
        
        # 1. 趋势一票否决
        if curr < ma60[-1]: return None
        # 2. 形态必须具有攻击性
        if not ((ma5[-1] > ma10[-1]) or (curr > ma20[-1] and df['open'].values[-1] < ma20[-1])):
            return None

        # --- B. 源头溯源 (Source Analysis) ---
        dyn_score, dyn_sources = self.radar.check(code)
        static_sources = StaticKnowledge.match(name)
        all_sources = list(set(dyn_sources + static_sources))
        
        # --- C. 股性基因 (DNA) ---
        tech_score = 60
        reasons = []
        
        # 1. 妖股记忆 (涨停数)
        limit_ups = len(df[df['pct_chg'] > 9.5].tail(15))
        if limit_ups >= 2: tech_score += 20; reasons.append(f"妖股基因({limit_ups}板)")
        
        # 2. 突破新高
        h120 = df['high'].iloc[-120:].max()
        if (h120 - curr) / curr < 0.05: tech_score += 20; reasons.append("突破新高")
        
        # 3. 量能配合
        vol_ma5 = pd.Series(df['volume'].values).rolling(5).mean().values[-1]
        if vol_ma5 > 0 and (df['volume'].values[-1] / vol_ma5) > 1.2: tech_score += 5
        
        # --- D. 身份认定 (Identity Definition) ---
        # 计算总分
        total_score = tech_score + dyn_score + (len(static_sources)*10)
        
        # 筛选门槛
        if dyn_score == 0 and len(static_sources) == 0 and total_score < 90: return None
        if total_score < 75: return None
        
        # 核心逻辑：定义身份
        identity = "🐕跟风"
        advice = "观察"
        
        # 判定逻辑：
        has_fund = any("[金]" in s for s in all_sources)
        has_concept = any("[概]" in s for s in all_sources)
        is_high_score = total_score >= 100
        
        if is_high_score and has_concept and has_fund:
            identity = "🐲真龙 (T0)"
            advice = "锁仓/抢筹"
        elif has_fund and base_info['circ_mv'] > 100 * 10**8: # 资金驱动且盘子大
            identity = "🐢中军 (T1)"
            advice = "均线低吸"
        elif has_concept and limit_ups >= 1: # 概念驱动且有涨停
            identity = "🚀先锋 (T1)"
            advice = "打板/半路"
        elif "新高" in reasons:
            identity = "💰趋势龙 (T2)"
            advice = "五日线跟随"
        else:
            identity = "🦊套利 (T3)"
            advice = "快进快出"

        return {
            "代码": code, "名称": name,
            "身份": identity,
            "结论": advice,
            "总分": total_score,
            "上涨源头": ",".join(all_sources) if all_sources else "-",
            "技术特征": "|".join(reasons),
            "涨幅%": base_info['pct_chg'],
            "换手%": base_info['turnover']
        }

# ==========================================
# 4. 指挥中枢
# ==========================================
class Commander:
    def run(self):
        print(Fore.GREEN + "=== 🐲 A股游资·真龙天眼系统 (Titan: Dragon Eye) ===")
        print(Fore.WHITE + "核心功能：上涨溯源 + 身份认定 + 结论输出")
        
        # 1. 启动雷达
        radar = ResonanceRadar()
        radar.scan_market()
        
        # 2. 快照
        print(Fore.CYAN + ">>> [2/5] 获取快照...")
        try:
            df = ak.stock_zh_a_spot_em()
            df.rename(columns={'代码':'code', '名称':'name', '最新价':'close', '涨跌幅':'pct_chg', 
                              '换手率':'turnover', '总市值':'total_mv', '流通市值':'circ_mv'}, inplace=True)
            for c in ['close', 'pct_chg', 'turnover', 'circ_mv']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        except: self.save_empty(); return

        # 3. 漏斗 (Adaptive Auto-Lowering)
        # 修改说明：增加了循环降级机制，如果选不出来，自动降低换手率标准
        print(Fore.CYAN + f">>> [3/5] 执行漏斗 (初始标准: 换手>{BattleConfig.FILTER_TURNOVER}%)...")
        
        current_turnover_threshold = BattleConfig.FILTER_TURNOVER
        min_turnover_limit = 1.0 # 最低底线，防止选出死股
        candidates = pd.DataFrame()
        
        while True:
            mask = (
                (~df['name'].str.contains('ST|退|C|U')) & 
                (df['close'].between(BattleConfig.MIN_PRICE, BattleConfig.MAX_PRICE)) &
                (df['circ_mv'].between(BattleConfig.MIN_CAP, BattleConfig.MAX_CAP)) &
                (df['pct_chg'] >= BattleConfig.FILTER_PCT_CHG) & 
                (df['turnover'] >= current_turnover_threshold) # 使用动态阈值
            )
            candidates = df[mask].copy().sort_values(by='turnover', ascending=False).head(150)
            
            if len(candidates) > 0:
                print(Fore.YELLOW + f"    📉 最终使用标准(换手>={current_turnover_threshold:.1f}%) 入围: {len(candidates)} 只")
                break
            
            # 如果没选到股，降低标准
            print(Fore.RED + f"    ⚠️ 换手率 {current_turnover_threshold:.1f}% 无符合标的，正在降级搜索...")
            current_turnover_threshold -= 0.8 # 每次降低0.8
            
            # 触底检测
            if current_turnover_threshold < min_turnover_limit:
                print(Fore.RED + "    ❌ 已降至最低标准，仍无标的，今日建议空仓。")
                break
        
        if len(candidates) == 0: self.save_empty(); return

        # 4. 深度分析
        engine = IdentityEngine(radar)
        results = []
        tasks = [row.to_dict() for _, row in candidates.iterrows()]
        
        print(Fore.CYAN + f">>> [4/5] 深度运算 (Workers: {BattleConfig.MAX_WORKERS})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=BattleConfig.MAX_WORKERS) as ex:
            futures = [ex.submit(engine.analyze, task) for task in tasks]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
                res = f.result()
                if res: results.append(res)

        # 5. 导出
        print(Fore.CYAN + f">>> [5/5] 导出: {BattleConfig.FILE_NAME}")
        if results:
            # 排序：优先看身份等级 (T0 > T1)，其次看总分
            # 这里的trick是给身份加个前缀排序，或者自定义排序
            # 简单起见，按总分降序即可，因为真龙分通常最高
            results.sort(key=lambda x: x['总分'], reverse=True)
            df_res = pd.DataFrame(results[:35])
            
            # 格式化输出
            cols = ["代码", "名称", "身份", "结论", "总分", "上涨源头", "技术特征", "涨幅%", "换手%"]
            df_res = df_res[[c for c in cols if c in df_res.columns]]
            
            df_res.to_excel(BattleConfig.FILE_NAME, index=False)
            print(Fore.GREEN + f"✅ 成功锁定 {len(df_res)} 只核心标的。")
            print(Fore.WHITE + "\n🔥 Top 5 核心真龙:")
            print(df_res[['名称', '身份', '结论', '上涨源头']].head(5).to_string(index=False))
        else:
            print(Fore.RED + "❌ 无标的"); self.save_empty()

    def save_empty(self):
        pd.DataFrame(columns=["Info"]).to_excel(BattleConfig.FILE_NAME)

if __name__ == "__main__":
    Commander().run()
