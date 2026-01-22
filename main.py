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
    MAX_CAP = 1000 * 10**8 
    MIN_PRICE = 3.0
    MAX_PRICE = 120.0
    FILTER_PCT_CHG = 3.5       
    FILTER_TURNOVER = 3.8      
    HISTORY_DAYS = 250
    # ★ 改动1：降速，防止被接口封IP导致进度条卡死
    MAX_WORKERS = 4  
    FILE_NAME = f"Dragon_Eye_{datetime.now().strftime('%Y%m%d')}.xlsx"
    IS_FREEZING_POINT = False 

# ==========================================
# 1. 全维共振雷达
# ==========================================
class ResonanceRadar:
    def __init__(self):
        self.hot_stock_map = {} 
        self.active_sources = []

    def scan_market(self):
        print(Fore.MAGENTA + ">>> [1/5] 启动真龙雷达 (降速模式)...")
        targets = [] 

        # 简单的容错获取
        try:
            df_fund = ak.stock_market_fund_flow()
            df_fund = df_fund.sort_values(by="今日主力净流入", ascending=False).head(5)
            for _, row in df_fund.iterrows(): targets.append((row['名称'], 50, "[金]")) 
        except: pass

        try:
            df_ind = ak.stock_board_industry_name_em()
            df_ind = df_ind.sort_values(by="涨跌幅", ascending=False).head(5)
            for _, row in df_ind.iterrows(): targets.append((row['板块名称'], 40, "[业]"))
        except: pass

        try:
            df_con = ak.stock_board_concept_name_em()
            noise = ["昨日", "连板", "首板", "涨停", "融资", "融券", "转债", "ST", "标普", "指数", "高股息", "破净", "增持", "深股通", "沪股通", "AB股", "AH股"]
            mask = ~df_con['板块名称'].str.contains("|".join(noise))
            df_con = df_con[mask].sort_values(by="涨跌幅", ascending=False).head(15)
            for i, (_, row) in enumerate(df_con.iterrows()):
                name = row['板块名称']
                score = 45 if i < 3 else (25 if i < 8 else 15)
                targets.append((name, score, "[概]"))
        except: pass
        
        self.active_sources = [f"{t[2]}{t[0]}" for t in targets]
        print(Fore.MAGENTA + f"    🎯 核心源头: {self.active_sources[:8]}...")
        
        # 溯源
        def fetch_cons(t):
            name, score, type_ = t
            try:
                time.sleep(random.uniform(0.5, 1.0)) # 增加延迟
                if "[金]" in type_ or "[业]" in type_:
                    df = ak.stock_board_industry_cons_em(symbol=name)
                else:
                    df = ak.stock_board_concept_cons_em(symbol=name)
                return name, score, type_, df['代码'].tolist()
            except:
                return name, 0, "", []

        # 减少并发
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(fetch_cons, t) for t in targets]
            for f in concurrent.futures.as_completed(futures):
                try:
                    name, score, type_, codes = f.result(timeout=10)
                    for code in codes:
                        if code not in self.hot_stock_map:
                            self.hot_stock_map[code] = {'score': 0, 'sources': set()}
                        curr = self.hot_stock_map[code]['score']
                        self.hot_stock_map[code]['score'] = min(curr + score, 90)
                        self.hot_stock_map[code]['sources'].add(f"{type_}{name}")
                except: continue
                    
        print(Fore.GREEN + f"    ✅ 索引构建完毕，覆盖 {len(self.hot_stock_map)} 只活跃股")

    def check(self, code):
        if code in self.hot_stock_map:
            d = self.hot_stock_map[code]
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
# 3. 身份判别引擎 (高鲁棒版)
# ==========================================
class IdentityEngine:
    def __init__(self, radar):
        self.radar = radar

    def get_kline(self, code):
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=BattleConfig.HISTORY_DAYS)).strftime("%Y%m%d")
        # 增加重试次数，处理网络波动
        for i in range(3):
            try:
                # 动态延迟，越往后延迟越长
                time.sleep(random.uniform(0.2, 0.5) * (i + 1))
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
                if df is not None and not df.empty:
                    df.rename(columns={'日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'volume', '涨跌幅':'pct_chg'}, inplace=True)
                    return df
            except: 
                pass
        return None

    def analyze(self, base_info):
        try:
            code = base_info['code']
            name = base_info['name']
            
            # --- A. 技术铁律 ---
            df = self.get_kline(code)
            
            # 如果获取不到数据，不要直接丢弃，而是标记为"待人工复核"
            if df is None or len(df) < 60: 
                # ★ 改动2：如果处于冰点模式且数据获取失败，返回一个保底结果
                if BattleConfig.IS_FREEZING_POINT:
                    return {
                        "代码": code, "名称": name, "身份": "❓未知(数据缺失)", 
                        "结论": "需人工看盘", "总分": 50, "上涨源头": "数据获取失败", 
                        "技术特征": "-", "涨幅%": base_info['pct_chg'], "换手%": base_info['turnover']
                    }
                return None
            
            close = df['close'].values
            ma5, ma10, ma20, ma60 = [pd.Series(close).rolling(w).mean().values for w in [5,10,20,60]]
            curr = close[-1]
            
            # 冰点宽松逻辑
            if BattleConfig.IS_FREEZING_POINT:
                # 只要没跌破MA20太多，或者今天是放量大阳线，就放行
                is_strong_today = (df['pct_chg'].iloc[-1] > 4.0)
                if curr < ma20[-1] and not is_strong_today: return None 
            else:
                if curr < ma60[-1]: return None
                if not ((ma5[-1] > ma10[-1]) or (curr > ma20[-1])): return None

            # --- B. 评分逻辑 ---
            dyn_score, dyn_sources = self.radar.check(code)
            static_sources = StaticKnowledge.match(name)
            all_sources = list(set(dyn_sources + static_sources))
            
            tech_score = 60
            reasons = []
            
            limit_ups = len(df[df['pct_chg'] > 9.5].tail(15))
            if limit_ups >= 2: tech_score += 20; reasons.append(f"妖股基因({limit_ups}板)")
            
            h120 = df['high'].iloc[-120:].max()
            if (h120 - curr) / curr < 0.15: # 进一步放宽
                tech_score += 20; reasons.append("接近新高")
            
            vol_ma5 = pd.Series(df['volume'].values).rolling(5).mean().values[-1]
            if vol_ma5 > 0 and (df['volume'].values[-1] / vol_ma5) > 1.2: tech_score += 5
            
            total_score = tech_score + dyn_score + (len(static_sources)*10)
            
            # 冰点模式下，大幅降低门槛，只要是活口就行
            threshold = 55 if BattleConfig.IS_FREEZING_POINT else 75
            
            if total_score < threshold: return None
            
            # 身份定义
            identity = "🐕跟风"
            advice = "观察"
            
            has_fund = any("[金]" in s for s in all_sources)
            has_concept = any("[概]" in s for s in all_sources)
            
            if total_score >= 100: identity = "🐲真龙 (T0)"; advice = "锁仓/抢筹"
            elif has_fund and base_info['circ_mv'] > 100 * 10**8: identity = "🐢中军 (T1)"; advice = "均线低吸"
            elif has_concept and limit_ups >= 1: identity = "🚀先锋 (T1)"; advice = "打板/半路"
            elif "新高" in reasons: identity = "💰趋势龙 (T2)"; advice = "五日线跟随"
            else: identity = "🦊套利 (T3)"; advice = "快进快出"

            return {
                "代码": code, "名称": name,
                "身份": identity, "结论": advice,
                "总分": total_score,
                "上涨源头": ",".join(all_sources) if all_sources else "-",
                "技术特征": "|".join(reasons),
                "涨幅%": base_info['pct_chg'],
                "换手%": base_info['turnover']
            }
        except Exception as e:
            return None

# ==========================================
# 4. 指挥中枢
# ==========================================
class Commander:
    def run(self):
        print(Fore.GREEN + "=== 🐲 A股游资·真龙天眼系统 (Titan: Dragon Eye) ===")
        
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
        except: self.save_fallback(pd.DataFrame(), "快照获取失败"); return

        # 3. 漏斗
        print(Fore.CYAN + f">>> [3/5] 执行漏斗 (初始标准: 换手>{BattleConfig.FILTER_TURNOVER}%)...")
        current_turnover = BattleConfig.FILTER_TURNOVER
        min_limit = 1.0 
        candidates = pd.DataFrame()
        
        while True:
            mask = (
                (~df['name'].str.contains('ST|退|C|U')) & 
                (df['close'].between(BattleConfig.MIN_PRICE, BattleConfig.MAX_PRICE)) &
                (df['circ_mv'].between(BattleConfig.MIN_CAP, BattleConfig.MAX_CAP)) &
                (df['pct_chg'] >= BattleConfig.FILTER_PCT_CHG) & 
                (df['turnover'] >= current_turnover) 
            )
            candidates = df[mask].copy().sort_values(by='turnover', ascending=False).head(150)
            
            if len(candidates) > 0:
                print(Fore.YELLOW + f"    📉 标准(换手>={current_turnover:.1f}%) 入围: {len(candidates)} 只")
                break
            
            print(Fore.RED + f"    ⚠️ 换手率 {current_turnover:.1f}% 无标的，降级...")
            current_turnover -= 0.8
            BattleConfig.IS_FREEZING_POINT = True 
            
            if current_turnover < min_limit:
                print(Fore.RED + "    ❌ 已降至最低标准，强制使用全市场涨幅前列作为备选。")
                # 最后的保底：如果换手率实在选不出，就硬选涨幅榜
                candidates = df.sort_values(by='pct_chg', ascending=False).head(20)
                BattleConfig.IS_FREEZING_POINT = True
                break
        
        if len(candidates) == 0: self.save_fallback(df.head(10), "全市场无符合条件"); return

        # 4. 深度分析
        engine = IdentityEngine(radar)
        results = []
        tasks = [row.to_dict() for _, row in candidates.iterrows()]
        
        print(Fore.CYAN + f">>> [4/5] 深度运算 (Workers: {BattleConfig.MAX_WORKERS}) [冰点:{BattleConfig.IS_FREEZING_POINT}]...")
        
        # 使用更稳健的循环
        with concurrent.futures.ThreadPoolExecutor(max_workers=BattleConfig.MAX_WORKERS) as ex:
            futures = {ex.submit(engine.analyze, task): task for task in tasks}
            
            # 增加总超时保护，防止卡死
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
                try:
                    # 30秒超时，如果卡住直接跳过，保证进度条走完
                    res = f.result(timeout=30) 
                    if res: results.append(res)
                except Exception:
                    # 任何错误都忽略，保证程序不崩
                    continue

        # 5. 导出 (★ 核心保底逻辑)
        print(Fore.CYAN + f">>> [5/5] 导出: {BattleConfig.FILE_NAME}")
        
        if results:
            results.sort(key=lambda x: x['总分'], reverse=True)
            df_res = pd.DataFrame(results[:35])
            self.save_excel(df_res)
        else:
            # ★ 如果深度扫描结果为空，强行保存初选名单，绝不给空文件
            print(Fore.RED + "⚠️ 深度扫描无结果，启动保底存档模式...")
            fallback_data = candidates.copy()
            fallback_data['备注'] = "初选入围-深度扫描未通过或数据缺失"
            self.save_excel(fallback_data)

    def save_excel(self, df):
        try:
            df.to_excel(BattleConfig.FILE_NAME, index=False)
            print(Fore.GREEN + f"✅ 成功导出 {len(df)} 条数据。")
            if '身份' in df.columns:
                 print(df[['名称', '身份', '结论']].head(5).to_string(index=False))
        except Exception as e:
            print(Fore.RED + f"❌ 保存失败: {e}")

    def save_fallback(self, df, reason):
        df['Reason'] = reason
        df.to_excel(BattleConfig.FILE_NAME)

if __name__ == "__main__":
    Commander().run()
