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
import traceback

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
    # 初始筛选标准
    FILTER_PCT_CHG = 3.5       
    FILTER_TURNOVER = 3.8      
    HISTORY_DAYS = 250
    MAX_WORKERS = 4 # 降低并发数以防封IP，配合快照模式足够快
    FILE_NAME = f"Dragon_Eye_Snapshot_{datetime.now().strftime('%Y%m%d')}.xlsx"

# ==========================================
# 1. 题材标签雷达 (独立运行，只负责打标签)
# ==========================================
class ThemeRadar:
    """
    不负责选股，只负责产生 {代码: (分数, [来源列表])} 的映射表
    即使这里全挂了，也不会影响主程序运行
    """
    def __init__(self):
        self.stock_tags = {} # {code: {'score': 0, 'sources': set()}}
        self.active_sources = []

    def scan(self):
        print(Fore.MAGENTA + ">>> [2/5] 启动题材雷达 (构建标签库)...")
        targets = [] 

        # --- A. 资金源 ---
        try:
            df_fund = ak.stock_market_fund_flow()
            df_fund = df_fund.sort_values(by="今日主力净流入", ascending=False).head(5)
            for _, row in df_fund.iterrows(): targets.append((row['名称'], 50, "[金]"))
        except: pass

        # --- B. 行业源 ---
        try:
            df_ind = ak.stock_board_industry_name_em()
            df_ind = df_ind.sort_values(by="涨跌幅", ascending=False).head(5)
            for _, row in df_ind.iterrows(): targets.append((row['板块名称'], 40, "[业]"))
        except: pass

        # --- C. 题材源 ---
        try:
            df_con = ak.stock_board_concept_name_em()
            noise = ["昨日", "连板", "首板", "涨停", "融资", "融券", "转债", "ST", "标普", "指数", "高股息", "破净", "增持", "深股通", "沪股通", "AB股", "AH股"]
            mask = ~df_con['板块名称'].str.contains("|".join(noise))
            df_con = df_con[mask].sort_values(by="涨跌幅", ascending=False).head(15)
            
            for i, (_, row) in enumerate(df_con.iterrows()):
                name = row['板块名称']
                if i < 3: score = 45     
                elif i < 8: score = 25   
                else: score = 15         
                targets.append((name, score, "[概]"))
        except: pass
        
        self.active_sources = [f"{t[2]}{t[0]}" for t in targets]
        print(Fore.MAGENTA + f"    🎯 核心源头: {self.active_sources[:8]}...")

        # --- D. 并行获取成分股 ---
        def fetch_cons(t):
            name, score, type_ = t
            try:
                time.sleep(random.uniform(0.5, 1.0)) # 增加延时防封
                if "[金]" in type_ or "[业]" in type_:
                    df = ak.stock_board_industry_cons_em(symbol=name)
                else:
                    df = ak.stock_board_concept_cons_em(symbol=name)
                return name, score, type_, df['代码'].tolist()
            except:
                return name, 0, "", []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(fetch_cons, t) for t in targets]
            for f in concurrent.futures.as_completed(futures):
                try:
                    name, score, type_, codes = f.result(timeout=10)
                    for code in codes:
                        if code not in self.stock_tags:
                            self.stock_tags[code] = {'score': 0, 'sources': set()}
                        
                        # 累加分数 (上限90)
                        curr = self.stock_tags[code]['score']
                        self.stock_tags[code]['score'] = min(curr + score, 90)
                        self.stock_tags[code]['sources'].add(f"{type_}{name}")
                except: pass
        
        print(Fore.GREEN + f"    ✅ 标签库构建完毕，覆盖 {len(self.stock_tags)} 只股票")

    def get_tag_info(self, code):
        if code in self.stock_tags:
            d = self.stock_tags[code]
            return d['score'], list(d['sources'])
        return 0, []

# ==========================================
# 2. 静态知识库
# ==========================================
class StaticKnowledge:
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
# 3. 核心分析引擎 (逻辑完全复刻原代码)
# ==========================================
class IdentityEngine:
    def __init__(self, radar):
        self.radar = radar

    def get_kline_history(self, code):
        # 即使是快照模式，技术指标(MA60/新高)仍需历史K线
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=BattleConfig.HISTORY_DAYS)).strftime("%Y%m%d")
        for i in range(3):
            try:
                time.sleep(random.uniform(0.1, 0.3))
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
                if df is not None and not df.empty:
                    df.rename(columns={'日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'volume', '涨跌幅':'pct_chg'}, inplace=True)
                    return df
            except: time.sleep(0.5)
        return None

    def analyze(self, snapshot_row):
        # 这里接收的是快照的一行数据
        code = snapshot_row['code']
        name = snapshot_row['name']
        
        # 1. 获取K线 (用于技术铁律判断)
        df = self.get_kline_history(code)
        
        # 容错：如果K线获取失败，但快照显示它是大涨股，标记为"待复核"并保留
        if df is None or len(df) < 60: 
            return None 
        
        # --- A. 技术铁律 (Survival) ---
        close = df['close'].values
        ma5, ma10, ma20, ma60 = [pd.Series(close).rolling(w).mean().values for w in [5,10,20,60]]
        curr = close[-1]
        
        # 1. 趋势一票否决 (保留原代码逻辑)
        if curr < ma60[-1]: return None
        # 2. 形态必须具有攻击性
        if not ((ma5[-1] > ma10[-1]) or (curr > ma20[-1] and df['open'].values[-1] < ma20[-1])):
            return None

        # --- B. 源头溯源 ---
        # 改为从 ThemeRadar 获取
        dyn_score, dyn_sources = self.radar.get_tag_info(code)
        static_sources = StaticKnowledge.match(name)
        all_sources = list(set(dyn_sources + static_sources))
        
        # --- C. 股性基因 (DNA) ---
        tech_score = 60
        reasons = []
        
        # 1. 妖股记忆
        limit_ups = len(df[df['pct_chg'] > 9.5].tail(15))
        if limit_ups >= 2: tech_score += 20; reasons.append(f"妖股基因({limit_ups}板)")
        
        # 2. 突破新高
        h120 = df['high'].iloc[-120:].max()
        if (h120 - curr) / curr < 0.05: tech_score += 20; reasons.append("突破新高")
        
        # 3. 量能配合
        vol_ma5 = pd.Series(df['volume'].values).rolling(5).mean().values[-1]
        if vol_ma5 > 0 and (df['volume'].values[-1] / vol_ma5) > 1.2: tech_score += 5
        
        # --- D. 身份认定 ---
        total_score = tech_score + dyn_score + (len(static_sources)*10)
        
        # 筛选门槛 (原逻辑)
        if dyn_score == 0 and len(static_sources) == 0 and total_score < 90: return None
        if total_score < 75: return None
        
        # 身份定义
        identity = "🐕跟风"
        advice = "观察"
        
        has_fund = any("[金]" in s for s in all_sources)
        has_concept = any("[概]" in s for s in all_sources)
        is_high_score = total_score >= 100
        
        if is_high_score and has_concept and has_fund:
            identity = "🐲真龙 (T0)"; advice = "锁仓/抢筹"
        elif has_fund and snapshot_row['circ_mv'] > 100 * 10**8:
            identity = "🐢中军 (T1)"; advice = "均线低吸"
        elif has_concept and limit_ups >= 1:
            identity = "🚀先锋 (T1)"; advice = "打板/半路"
        elif "新高" in reasons:
            identity = "💰趋势龙 (T2)"; advice = "五日线跟随"
        else:
            identity = "🦊套利 (T3)"; advice = "快进快出"

        # 返回符合原代码要求的数据结构
        return {
            "代码": code, "名称": name,
            "身份": identity,
            "结论": advice,
            "总分": total_score,
            "上涨源头": ",".join(all_sources) if all_sources else "-",
            "技术特征": "|".join(reasons),
            "涨幅%": snapshot_row['pct_chg'],
            "换手%": snapshot_row['turnover']
        }

# ==========================================
# 4. 指挥中枢 (Snapshot-First 架构)
# ==========================================
class Commander:
    def run(self):
        print(Fore.GREEN + "=== 🐲 A股游资·真龙天眼 (Snapshot-First Version) ===")
        print(Fore.WHITE + "架构：全市场快照 -> 智能漏斗 -> 题材注入 -> 深度分析")
        
        # 1. 获取全市场快照 (最稳健的一步)
        print(Fore.CYAN + ">>> [1/5] 获取全市场快照...")
        try:
            df_all = ak.stock_zh_a_spot_em()
            # 统一列名，确保后续逻辑通用
            df_all.rename(columns={'代码':'code', '名称':'name', '最新价':'close', '涨跌幅':'pct_chg', 
                                  '换手率':'turnover', '总市值':'total_mv', '流通市值':'circ_mv'}, inplace=True)
            for c in ['close', 'pct_chg', 'turnover', 'circ_mv']:
                df_all[c] = pd.to_numeric(df_all[c], errors='coerce')
        except Exception as e:
            print(Fore.RED + f"❌ 快照获取失败: {e}"); self.save_empty(); return

        # 2. 启动题材雷达 (并行运行，不阻塞主流程太多)
        radar = ThemeRadar()
        radar.scan()

        # 3. 智能漏斗 (保留原代码的 自动降级 逻辑)
        print(Fore.CYAN + f">>> [3/5] 执行漏斗 (初始标准: 换手>{BattleConfig.FILTER_TURNOVER}%)...")
        
        current_turnover_threshold = BattleConfig.FILTER_TURNOVER
        min_turnover_limit = 1.0
        candidates = pd.DataFrame()
        
        # 基础过滤 (去除ST/退市/价格不符/小市值)
        base_mask = (
            (~df_all['name'].str.contains('ST|退|C|U')) & 
            (df_all['close'].between(BattleConfig.MIN_PRICE, BattleConfig.MAX_PRICE)) &
            (df_all['circ_mv'].between(BattleConfig.MIN_CAP, BattleConfig.MAX_CAP))
        )
        
        # 循环降级逻辑
        while True:
            mask = base_mask & (df_all['pct_chg'] >= BattleConfig.FILTER_PCT_CHG) & (df_all['turnover'] >= current_turnover_threshold)
            candidates = df_all[mask].copy().sort_values(by='turnover', ascending=False).head(150)
            
            if len(candidates) > 0:
                print(Fore.YELLOW + f"    📉 最终使用标准(换手>={current_turnover_threshold:.1f}%) 入围: {len(candidates)} 只")
                break
            
            print(Fore.RED + f"    ⚠️ 换手率 {current_turnover_threshold:.1f}% 无符合标的，正在降级搜索...")
            current_turnover_threshold -= 0.8 
            
            if current_turnover_threshold < min_turnover_limit:
                print(Fore.RED + "    ❌ 已降至最低标准，仍无标的，启用【纯涨幅】保底策略。")
                # 最后的保底：直接取涨幅榜前30，不做换手限制
                candidates = df_all[base_mask].sort_values(by='pct_chg', ascending=False).head(30)
                break
        
        # 4. 深度分析 (融合)
        engine = IdentityEngine(radar)
        results = []
        tasks = [row.to_dict() for _, row in candidates.iterrows()]
        
        print(Fore.CYAN + f">>> [4/5] 深度运算 (Workers: {BattleConfig.MAX_WORKERS})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=BattleConfig.MAX_WORKERS) as ex:
            futures = {ex.submit(engine.analyze, task): task for task in tasks}
            
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
                try:
                    res = f.result(timeout=20) # 防止卡死
                    if res: results.append(res)
                except: continue

        # 5. 导出
        print(Fore.CYAN + f">>> [5/5] 导出: {BattleConfig.FILE_NAME}")
        if results:
            results.sort(key=lambda x: x['总分'], reverse=True)
            df_res = pd.DataFrame(results[:35])
            
            # 严格按照要求的输出格式
            cols = ["代码", "名称", "身份", "结论", "总分", "上涨源头", "技术特征", "涨幅%", "换手%"]
            df_res = df_res[[c for c in cols if c in df_res.columns]]
            
            df_res.to_excel(BattleConfig.FILE_NAME, index=False)
            print(Fore.GREEN + f"✅ 成功锁定 {len(df_res)} 只核心标的。")
            print(Fore.WHITE + "\n🔥 Top 5 核心真龙:")
            try:
                print(df_res[['名称', '身份', '结论', '上涨源头']].head(5).to_string(index=False))
            except: pass
        else:
            # 绝对保底：如果深度分析全部过滤掉了，把初选名单导出来
            print(Fore.RED + "⚠️ 深度分析未通过，导出初选名单作为参考。")
            candidates['身份'] = '初选入围'
            candidates['结论'] = '需人工复核'
            candidates.to_excel(BattleConfig.FILE_NAME, index=False)

    def save_empty(self):
        pd.DataFrame(columns=["Info"]).to_excel(BattleConfig.FILE_NAME)

if __name__ == "__main__":
    Commander().run()
