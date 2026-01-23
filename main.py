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

init(autoreset=True)
warnings.filterwarnings('ignore')

class BattleConfig:
    MIN_CAP = 15 * 10**8
    MAX_CAP = 2000 * 10**8
    MIN_PRICE = 3.0
    MAX_PRICE = 130.0
    FILTER_PCT_CHG = 3.0       
    FILTER_TURNOVER = 3.0      
    HISTORY_DAYS = 250
    MAX_WORKERS = 4 
    FILE_NAME = f"Dragon_Eye_ProMax_{datetime.now().strftime('%Y%m%d')}.xlsx"
    IS_FREEZING_POINT = False 

# ==========================================
# 1. 动态热点雷达
# ==========================================
class HotConceptRadar:
    def __init__(self):
        self.dynamic_map = {} 

    def scan(self):
        print(Fore.MAGENTA + ">>> [2/6] 启动热点概念雷达...")
        try:
            df_board = ak.stock_board_concept_name_em()
            noise = ["昨日", "连板", "首板", "涨停", "融资", "融券", "转债", "ST", "标普", "指数", "高股息", "破净", "增持", "深股通", "沪股通", "AB股", "AH股", "含可转债", "板块"]
            mask = ~df_board['板块名称'].str.contains("|".join(noise))
            df_top = df_board[mask].sort_values(by="涨跌幅", ascending=False).head(8)
            
            targets = df_top['板块名称'].tolist()
            print(Fore.MAGENTA + f"    🔥 今日突发热点: {targets}")
            
            def fetch_cons(name):
                try:
                    time.sleep(random.uniform(0.5, 0.8))
                    df = ak.stock_board_concept_cons_em(symbol=name)
                    return name, df['代码'].tolist()
                except: return name, []

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                futures = [ex.submit(fetch_cons, t) for t in targets]
                for f in concurrent.futures.as_completed(futures):
                    try:
                        name, codes = f.result(timeout=10)
                        for code in codes:
                            if code not in self.dynamic_map: self.dynamic_map[code] = []
                            self.dynamic_map[code].append(f"[🔥热]{name}")
                    except: pass
            
            print(Fore.GREEN + f"    ✅ 动态热点库构建完毕，覆盖 {len(self.dynamic_map)} 只股票")

        except Exception as e:
            print(Fore.RED + f"    ⚠️ 热点雷达接口波动: {e}")

    def get_dynamic_tags(self, code):
        return self.dynamic_map.get(code, [])

# ==========================================
# 2. 板块资金雷达 (增强版：支持5日兜底)
# ==========================================
class SectorFundRadar:
    def __init__(self):
        self.hot_sectors = {} 

    def scan(self):
        print(Fore.MAGENTA + ">>> [3/6] 启动行业资金雷达...")
        
        def _get_sector_flow(period="今日"):
            try:
                print(Fore.MAGENTA + f"    ⏳ 尝试拉取 [{period}] 行业资金流...")
                df = ak.stock_sector_fund_flow_rank(indicator=period, sector_type="行业")
                if df is None or df.empty: return None
                
                # 模糊匹配列名
                flow_col = next((c for c in df.columns if "净流入" in c or "净额" in c), None)
                name_col = next((c for c in df.columns if "名称" in c), None)
                
                if flow_col and name_col:
                    df[flow_col] = pd.to_numeric(df[flow_col], errors='coerce').fillna(0)
                    df = df[df[flow_col] > 0].sort_values(by=flow_col, ascending=False).head(15)
                    return df, flow_col, name_col
                return None
            except: return None

        # 策略：先拿今日，如果今日为空（盘后清算中），则拿5日
        res = _get_sector_flow("今日")
        if not res:
            print(Fore.YELLOW + "    ⚠️ 今日资金流数据暂缺，切换为 [5日] 资金流兜底...")
            res = _get_sector_flow("5日")
        
        if res:
            df_top, flow_col, name_col = res
            print(Fore.MAGENTA + f"    🔥 资金主力进攻行业: {df_top[name_col].head(5).tolist()}")
            
            for _, row in df_top.iterrows():
                name = row[name_col]
                flow_val = round(row[flow_col] / 100000000, 2) # 亿
                self.hot_sectors[name] = flow_val
        else:
            print(Fore.RED + "    ❌ 行业资金数据完全获取失败")

    def check_is_hot(self, industry_name):
        if not industry_name: return False, 0
        for hot_name, flow in self.hot_sectors.items():
            # 宽松匹配：只要包含即可
            if hot_name in industry_name or industry_name in hot_name:
                return True, flow
        return False, 0

# ==========================================
# 3. 静态知识库
# ==========================================
class StaticKnowledge:
    THEME_DICT = {
        "低空/飞行": ["飞行", "eVTOL", "无人机", "万丰", "中信海直", "宗申", "航天"],
        "华为/鸿蒙": ["华为", "海思", "鸿蒙", "常山", "润和", "软通", "拓维"],
        "AI/算力": ["CPO", "光模块", "液冷", "英伟达", "工业富联", "寒武纪", "中际", "浪潮"],
        "芯片/半导体": ["芯片", "半导体", "光刻", "存储", "中芯", "北方华创", "海光", "韦尔"],
        "固态电池": ["固态", "硫化物", "清陶", "赣锋", "宁德", "粤桂", "有研"],
        "重组/金融": ["重组", "证券", "互联金融", "东方财富", "同花顺", "银之杰", "赢时胜"],
        "机器人": ["机器人", "减速器", "鸣志", "绿的", "赛力斯", "柯力"],
        "消费电子": ["消费电子", "手机", "苹果", "立讯", "歌尔", "光弘"],
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
# 4. 个股深度查询
# ==========================================
class StockProfiler:
    @staticmethod
    def get_industry(code):
        try:
            info = ak.stock_individual_info_em(symbol=code)
            industry = ""
            for _, row in info.iterrows():
                if row['item'] == '行业': industry = row['value']; break
            return industry
        except: return ""

# ==========================================
# 5. 核心分析引擎 (三保险资金获取)
# ==========================================
class IdentityEngine:
    def __init__(self, sector_radar, concept_radar):
        self.sector_radar = sector_radar
        self.concept_radar = concept_radar

    def get_kline_history(self, code):
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=BattleConfig.HISTORY_DAYS)).strftime("%Y%m%d")
        for _ in range(3):
            try:
                time.sleep(random.uniform(0.1, 0.2))
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
                if df is not None and not df.empty:
                    df.rename(columns={'日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'volume', '涨跌幅':'pct_chg'}, inplace=True)
                    return df
            except: time.sleep(0.5)
        return None

    def get_realtime_fund_flow(self, code):
        """
        【ProMax版】三保险获取主力资金流，确保不为空
        """
        # -----------------------------------------------------------
        # 优先方案: stock_financial_flow_data_dict (字典接口，最稳)
        # -----------------------------------------------------------
        try:
            data_dict = ak.stock_financial_flow_data_dict(symbol=code)
            # 这个接口返回类似: {'主力净流入': 1234.5, ...}
            if data_dict and isinstance(data_dict, dict):
                # 尝试不同的 key 组合
                for key in ['主力净流入', '今日主力净流入', '主力净额']:
                    if key in data_dict:
                         val = float(data_dict[key])
                         if val != 0: return val
        except: pass

        # -----------------------------------------------------------
        # 备选方案: stock_individual_spot_em (实时快照)
        # -----------------------------------------------------------
        try:
            df_spot = ak.stock_individual_spot_em(symbol=code)
            if df_spot is not None and not df_spot.empty:
                for col in df_spot.columns:
                    # 匹配 "主力" + "净流入"
                    if "主力" in col and ("净流入" in col or "净额" in col):
                        val = df_spot[col].values[0] # 确保取值
                        return float(val)
        except: pass

        # -----------------------------------------------------------
        # 兜底方案: stock_individual_fund_flow (历史数据)
        # -----------------------------------------------------------
        try:
            df_flow = ak.stock_individual_fund_flow(symbol=code)
            if df_flow is not None and not df_flow.empty:
                target_col = None
                for col in df_flow.columns:
                    if "主力" in col and ("净流入" in col or "净额" in col) and "占比" not in col:
                        target_col = col
                        break
                if target_col and '日期' in df_flow.columns:
                    df_flow['日期'] = pd.to_datetime(df_flow['日期'])
                    df_flow.sort_values('日期', ascending=False, inplace=True)
                    return float(df_flow.iloc[0][target_col])
        except: pass
        
        return 0.0

    def analyze(self, snapshot_row):
        code = snapshot_row['code']
        name = snapshot_row['name']
        
        # 1. 获取K线
        df = self.get_kline_history(code)
        if df is None or len(df) < 60: return None 
        
        # 2. 获取主力资金流 (三保险)
        net_flow = self.get_realtime_fund_flow(code)
        
        # --- 基础技术指标 ---
        close = df['close'].values
        ma5 = pd.Series(close).rolling(5).mean().values
        ma10 = pd.Series(close).rolling(10).mean().values
        ma20 = pd.Series(close).rolling(20).mean().values
        ma60 = pd.Series(close).rolling(60).mean().values
        curr = close[-1]
        
        # A. 铁血逻辑
        if not BattleConfig.IS_FREEZING_POINT:
            if curr < ma60[-1]: return None
            if not ((ma5[-1] > ma10[-1]) or (curr > ma20[-1])): return None
        else:
            if curr < ma5[-1] and snapshot_row['pct_chg'] < 5.0: return None

        # B. 题材/行业/资金
        industry = StockProfiler.get_industry(code)
        is_hot_sector, sector_flow = self.sector_radar.check_is_hot(industry)
        static_sources = StaticKnowledge.match(name)
        dynamic_sources = self.concept_radar.get_dynamic_tags(code)
        
        all_sources = list(set(static_sources + dynamic_sources))
        if industry: all_sources.append(f"[业]{industry}")
        
        hot_sector_str = "否"
        if is_hot_sector:
            all_sources.append("[🔥行业风口]")
            hot_sector_str = f"是 (流入{sector_flow}亿)"

        # C. 股性
        tech_score = 60
        reasons = []
        limit_ups = len(df[df['pct_chg'] > 9.5].tail(20))
        if limit_ups >= 2: tech_score += 20; reasons.append(f"妖股基因({limit_ups}板)")
        h120 = df['high'].iloc[-120:].max()
        if (h120 - curr) / curr < 0.05: tech_score += 20; reasons.append("突破新高")
        
        # D. 资金与出货
        turnover = snapshot_row['turnover']
        pct_chg = snapshot_row['pct_chg']
        
        # 严格的资金格式化 (解决空白问题)
        flow_str = "-"
        if net_flow is not None and net_flow != 0:
            val = round(net_flow/100000000, 2)
            if abs(val) >= 1: flow_str = f"{val}亿"
            else: flow_str = f"{round(net_flow/10000, 0)}万"
        
        is_shipping = False
        warning_msg = ""
        if turnover > 15: 
            if net_flow < -30000000:
                is_shipping = True; warning_msg = "⚠️高换手出货"; tech_score -= 30
            elif pct_chg < 2.0:
                is_shipping = True; warning_msg = "⚠️高位滞涨"; tech_score -= 15

        if net_flow > 50000000:
            tech_score += 15
            reasons.append("主力抢筹")
            
        if is_hot_sector: tech_score += 25
        if len(dynamic_sources) > 0: tech_score += 20

        # E. 身份
        total_score = tech_score + (len(static_sources) * 20)
        threshold = 60 if BattleConfig.IS_FREEZING_POINT else 70
        if total_score < threshold: return None
        
        identity = "🐕跟风"
        advice = "观察"
        
        has_strong_theme = (is_hot_sector or len(dynamic_sources) > 0 or len(static_sources) > 0)
        
        if is_shipping:
            identity = warning_msg; advice = "回避/卖出"; total_score = 50
        elif total_score >= 100 and has_strong_theme:
            identity = "🐲真龙 (T0)"; advice = "锁仓/抢筹"
        elif is_hot_sector and snapshot_row['circ_mv'] > 100 * 10**8:
            identity = "🐢中军 (T1)"; advice = "均线低吸"
        elif has_strong_theme and limit_ups >= 1:
            identity = "🚀先锋 (T1)"; advice = "打板/半路"
        elif "新高" in reasons:
            identity = "💰趋势龙 (T2)"; advice = "五日线跟随"
        else:
            identity = "🦊套利 (T3)"; advice = "快进快出"

        return {
            "代码": code, "名称": name,
            "身份": identity, "结论": advice,
            "总分": total_score,
            "是否主线": hot_sector_str,
            "所属行业": industry if industry else "-",
            "主力净额": flow_str,
            "上涨源头": ",".join(all_sources),
            "技术特征": "|".join(reasons),
            "涨幅%": pct_chg, "换手%": turnover
        }

# ==========================================
# 6. 指挥中枢
# ==========================================
class Commander:
    def run(self):
        print(Fore.GREEN + "=== 🐲 A股游资·真龙天眼 (ProMax - 三保险资金版) ===")
        
        # 1. 快照
        print(Fore.CYAN + ">>> [1/6] 获取全市场基础快照...")
        try:
            df_all = ak.stock_zh_a_spot_em()
            spot_map = {
                '代码':'code', '名称':'name', '最新价':'close', '涨跌幅':'pct_chg', 
                '换手率':'turnover', '总市值':'total_mv', '流通市值':'circ_mv'
            }
            df_all.rename(columns=spot_map, inplace=True)
            for c in ['close', 'pct_chg', 'turnover', 'circ_mv']:
                df_all[c] = pd.to_numeric(df_all[c], errors='coerce')
            print(Fore.GREEN + "    ✅ 基础数据获取成功")
        except Exception as e:
            print(Fore.RED + f"❌ 快照失败: {e}"); return

        # 2. 启动两大雷达
        concept_radar = HotConceptRadar()
        concept_radar.scan()
        sector_radar = SectorFundRadar()
        sector_radar.scan()

        # 3. 漏斗
        print(Fore.CYAN + f">>> [4/6] 执行漏斗筛选...")
        current_turnover = BattleConfig.FILTER_TURNOVER
        candidates = pd.DataFrame()
        
        base_mask = (
            (~df_all['name'].str.contains('ST|退|C|U')) & 
            (df_all['close'].between(BattleConfig.MIN_PRICE, BattleConfig.MAX_PRICE)) &
            (df_all['circ_mv'].between(BattleConfig.MIN_CAP, BattleConfig.MAX_CAP))
        )
        
        while True:
            mask = base_mask & (df_all['pct_chg'] >= BattleConfig.FILTER_PCT_CHG) & (df_all['turnover'] >= current_turnover)
            candidates = df_all[mask].copy().sort_values(by='turnover', ascending=False).head(150)
            if len(candidates) > 0:
                print(Fore.YELLOW + f"    📉 入围: {len(candidates)} 只 (换手>={current_turnover:.1f}%)")
                break
            current_turnover -= 0.8 
            BattleConfig.IS_FREEZING_POINT = True 
            if current_turnover < 1.0:
                print(Fore.RED + "    ❌ 降至最低标准，启用保底策略。")
                candidates = df_all[base_mask].sort_values(by='pct_chg', ascending=False).head(30)
                break
        
        # 4. 深度分析
        print(Fore.CYAN + f">>> [5/6] 深度分析 & 逐个拉取主力资金 (ProMax模式)...")
        engine = IdentityEngine(sector_radar, concept_radar)
        results = []
        tasks = [row.to_dict() for _, row in candidates.iterrows()]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=BattleConfig.MAX_WORKERS) as ex:
            futures = {ex.submit(engine.analyze, task): task for task in tasks}
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
                try:
                    res = f.result(timeout=15)
                    if res: results.append(res)
                except: continue

        # 5. 导出
        print(Fore.CYAN + f">>> [6/6] 导出: {BattleConfig.FILE_NAME}")
        if results:
            results.sort(key=lambda x: x['总分'], reverse=True)
            df_res = pd.DataFrame(results[:40])
            cols = ["代码", "名称", "身份", "结论", "总分", "是否主线", "所属行业", "主力净额", "上涨源头", "技术特征", "涨幅%", "换手%"]
            df_res = df_res[[c for c in cols if c in df_res.columns]]
            df_res.to_excel(BattleConfig.FILE_NAME, index=False)
            print(Fore.GREEN + f"✅ 成功! 文件: {BattleConfig.FILE_NAME}")
            try:
                # 打印预览，重点看 '是否主线' 和 '主力净额'
                print(df_res[['名称', '是否主线', '主力净额', '技术特征']].head(10).to_string(index=False))
            except: pass
        else:
            candidates.to_excel(BattleConfig.FILE_NAME, index=False)

if __name__ == "__main__":
    Commander().run()
