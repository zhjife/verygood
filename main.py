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
    FILE_NAME = f"Dragon_Eye_Dynamic_{datetime.now().strftime('%Y%m%d')}.xlsx"
    IS_FREEZING_POINT = False 

# ==========================================
# 1. 动态热点雷达 (新增：解决静态库滞后问题)
# ==========================================
class HotConceptRadar:
    """
    只抓取全市场涨幅 Top 8 的概念板块，获取其成分股。
    用于捕捉"静态库"里没有的新热点。
    """
    def __init__(self):
        self.dynamic_map = {} # {code: ['[热]Sora', '[热]量子']}

    def scan(self):
        print(Fore.MAGENTA + ">>> [2/6] 启动热点概念雷达 (捕捉市场突发热点)...")
        try:
            # 1. 获取概念涨幅榜
            df_board = ak.stock_board_concept_name_em()
            # 剔除垃圾板块
            noise = ["昨日", "连板", "首板", "涨停", "融资", "融券", "转债", "ST", "标普", "指数", "高股息", "破净", "增持", "深股通", "沪股通", "AB股", "AH股", "含可转债", "板块"]
            mask = ~df_board['板块名称'].str.contains("|".join(noise))
            # 只取前 8 名，减少网络请求压力，防止被封
            df_top = df_board[mask].sort_values(by="涨跌幅", ascending=False).head(8)
            
            targets = df_top['板块名称'].tolist()
            print(Fore.MAGENTA + f"    🔥 今日突发热点: {targets}")
            
            # 2. 抓取成分股
            def fetch_cons(name):
                try:
                    time.sleep(random.uniform(0.5, 1.0)) # 必须延时
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
            print(Fore.RED + f"    ⚠️ 热点雷达接口波动，自动切换回纯静态模式: {e}")

    def get_dynamic_tags(self, code):
        return self.dynamic_map.get(code, [])

# ==========================================
# 2. 板块资金雷达 (行业维度)
# ==========================================
class SectorFundRadar:
    def __init__(self):
        self.hot_sectors = {} 

    def scan(self):
        print(Fore.MAGENTA + ">>> [3/6] 启动行业资金雷达...")
        try:
            df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业")
            df['今日主力净流入'] = pd.to_numeric(df['今日主力净流入'], errors='coerce')
            df_top = df[df['今日主力净流入'] > 0].sort_values(by='今日主力净流入', ascending=False).head(15)
            for _, row in df_top.iterrows():
                name = row['名称']
                flow_val = round(row['今日主力净流入'] / 100000000, 2)
                self.hot_sectors[name] = flow_val
        except: pass

    def check_is_hot(self, industry_name):
        for hot_name, flow in self.hot_sectors.items():
            if hot_name in industry_name or industry_name in hot_name:
                return True, flow
        return False, 0

# ==========================================
# 3. 静态知识库 (保底底座)
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
# 5. 核心分析引擎
# ==========================================
class IdentityEngine:
    def __init__(self, sector_radar, concept_radar):
        self.sector_radar = sector_radar
        self.concept_radar = concept_radar

    def get_kline_history(self, code):
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
        code = snapshot_row['code']
        name = snapshot_row['name']
        
        df = self.get_kline_history(code)
        if df is None or len(df) < 60: return None 
        
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

        # B. 题材/行业/资金 综合画像
        # 1. 行业资金校验
        industry = StockProfiler.get_industry(code)
        is_hot_sector, sector_flow = self.sector_radar.check_is_hot(industry)
        
        # 2. 静态匹配
        static_sources = StaticKnowledge.match(name)
        
        # 3. 动态热点匹配 (新增)
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
        net_flow = snapshot_row.get('net_flow', 0)
        turnover = snapshot_row['turnover']
        pct_chg = snapshot_row['pct_chg']
        
        flow_str = "-"
        if net_flow:
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

        if net_flow > 50000000: tech_score += 15; reasons.append("主力抢筹")
        if is_hot_sector: tech_score += 25 # 行业风口加分
        if len(dynamic_sources) > 0: tech_score += 20 # 动态热点加分

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
        print(Fore.GREEN + "=== 🐲 A股游资·真龙天眼 (动态热点+行业风口双驱版) ===")
        
        # 1. 快照
        print(Fore.CYAN + ">>> [1/6] 获取全市场快照...")
        try:
            df_all = ak.stock_zh_a_spot_em()
            df_all.rename(columns={
                '代码':'code', '名称':'name', '最新价':'close', '涨跌幅':'pct_chg', 
                '换手率':'turnover', '总市值':'total_mv', '流通市值':'circ_mv', 
                '主力净流入':'net_flow'
            }, inplace=True)
            for c in ['close', 'pct_chg', 'turnover', 'circ_mv', 'net_flow']:
                df_all[c] = pd.to_numeric(df_all[c], errors='coerce')
        except Exception as e:
            print(Fore.RED + f"❌ 快照失败: {e}"); return

        # 2. 启动两大雷达 (动态概念 + 行业资金)
        concept_radar = HotConceptRadar()
        concept_radar.scan()
        
        sector_radar = SectorFundRadar()
        sector_radar.scan()

        # 3. 漏斗
        print(Fore.CYAN + f">>> [4/6] 执行漏斗 (初始标准: 换手>{BattleConfig.FILTER_TURNOVER}%)...")
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
        print(Fore.CYAN + f">>> [5/6] 深度分析...")
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
                print(df_res[['名称', '身份', '是否主线', '上涨源头']].head(5).to_string(index=False))
            except: pass
        else:
            candidates.to_excel(BattleConfig.FILE_NAME, index=False)

if __name__ == "__main__":
    Commander().run()
