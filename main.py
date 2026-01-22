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
    MIN_CAP = 15 * 10**8
    MAX_CAP = 2000 * 10**8
    MIN_PRICE = 3.0
    MAX_PRICE = 130.0
    FILTER_PCT_CHG = 3.0       
    FILTER_TURNOVER = 3.0      
    HISTORY_DAYS = 250
    MAX_WORKERS = 4 
    FILE_NAME = f"Dragon_Eye_Final_{datetime.now().strftime('%Y%m%d')}.xlsx"
    IS_FREEZING_POINT = False 

# ==========================================
# 1. 超级静态知识库 (扩充版)
# ==========================================
class StaticKnowledge:
    # 包含了市场上绝大多数热门题材，确保"上涨源头"不为空
    THEME_DICT = {
        "低空/飞行": ["飞行", "eVTOL", "无人机", "万丰", "中信海直", "宗申", "深城交", "航天"],
        "华为/鸿蒙": ["华为", "海思", "鸿蒙", "常山", "润和", "软通", "拓维", "诚迈"],
        "AI/算力": ["CPO", "光模块", "液冷", "英伟达", "工业富联", "寒武纪", "中际", "新易盛", "浪潮"],
        "芯片/半导体": ["芯片", "半导体", "光刻", "存储", "中芯", "北方华创", "海光", "韦尔"],
        "固态电池": ["固态", "硫化物", "清陶", "赣锋", "宁德", "粤桂", "当升", "有研"],
        "重组/金融": ["重组", "证券", "互联金融", "东方财富", "同花顺", "银之杰", "赢时胜", "指南针"],
        "机器人": ["机器人", "减速器", "执行器", "鸣志", "绿的", "赛力斯", "柯力"],
        "消费电子": ["消费电子", "手机", "苹果", "立讯", "歌尔", "福日", "光弘"],
        "新能源车": ["汽车", "比亚迪", "赛力斯", "江淮", "长安", "零部件"],
        "军工": ["军工", "航天", "导弹", "卫星", "中航", "北方"],
        "医药": ["医药", "创新药", "恒瑞", "药明", "片仔癀"],
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
# 2. 个股深度查询 (解决源头为空的问题)
# ==========================================
class StockProfiler:
    """
    专门负责查询单只股票的行业和概念，替代不稳定的板块接口
    """
    @staticmethod
    def get_profile(code):
        try:
            # 获取个股的行业信息（比抓整个板块要稳定得多）
            # 注意：Akshare没有直接查个股所属概念的简单接口，这里主要靠行业和静态库
            # 我们可以尝试用 stock_individual_info_em
            info = ak.stock_individual_info_em(symbol=code)
            # info 是一个 DataFrame，通常包含 '行业' 字段
            industry = ""
            for _, row in info.iterrows():
                if row['item'] == '行业':
                    industry = row['value']
                    break
            return f"[业]{industry}" if industry else ""
        except:
            return ""

# ==========================================
# 3. 核心分析引擎 (含出货判定)
# ==========================================
class IdentityEngine:
    def __init__(self):
        pass

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
        
        # 1. 基础 K 线
        df = self.get_kline_history(code)
        if df is None or len(df) < 60: return None 
        
        close = df['close'].values
        ma5 = pd.Series(close).rolling(5).mean().values
        ma10 = pd.Series(close).rolling(10).mean().values
        ma20 = pd.Series(close).rolling(20).mean().values
        ma60 = pd.Series(close).rolling(60).mean().values
        curr = close[-1]
        
        # --- A. 铁血逻辑 (Survival) ---
        if not BattleConfig.IS_FREEZING_POINT:
            # 正常时期：必须站上生命线
            if curr < ma60[-1]: return None
            if not ((ma5[-1] > ma10[-1]) or (curr > ma20[-1])): return None
        else:
            # 冰点时期：放宽限制
            if curr < ma5[-1] and snapshot_row['pct_chg'] < 5.0: return None

        # --- B. 源头填充 (解决为空问题) ---
        # 1. 静态匹配
        static_sources = StaticKnowledge.match(name)
        # 2. 动态查询 (个股行业)
        ind_source = StockProfiler.get_profile(code)
        
        all_sources = list(set(static_sources))
        if ind_source: all_sources.append(ind_source)
        
        # 如果还是为空，尝试从名称猜
        if not all_sources:
            if "科技" in name: all_sources.append("[猜]科技")
            elif "药" in name: all_sources.append("[猜]医药")
            else: all_sources.append("[业]其他")

        # --- C. 股性评分 ---
        tech_score = 60
        reasons = []
        
        # 妖股基因
        limit_ups = len(df[df['pct_chg'] > 9.5].tail(20))
        if limit_ups >= 2: tech_score += 20; reasons.append(f"妖股基因({limit_ups}板)")
        
        # 突破新高
        h120 = df['high'].iloc[-120:].max()
        if (h120 - curr) / curr < 0.05: tech_score += 20; reasons.append("突破新高")
        
        # --- D. 资金与出货判定 (新增) ---
        net_flow = snapshot_row.get('net_flow', 0)
        turnover = snapshot_row['turnover']
        pct_chg = snapshot_row['pct_chg']
        
        # 资金流展示
        flow_str = "-"
        if net_flow:
            val = round(net_flow/100000000, 2)
            if abs(val) >= 1: flow_str = f"{val}亿"
            else: flow_str = f"{round(net_flow/10000, 0)}万"
        
        # ★ 关键逻辑：判断是不是出货 ★
        is_shipping = False
        warning_msg = ""
        
        if turnover > 15: # 高换手
            if net_flow < -30000000: # 流出超过3000万
                is_shipping = True
                warning_msg = "⚠️高换手出货"
                tech_score -= 30 # 大幅扣分
            elif pct_chg < 2.0: # 换手巨大但涨不动
                is_shipping = True
                warning_msg = "⚠️高位滞涨"
                tech_score -= 15

        # 主力加分
        if net_flow > 50000000: # 流入超5000万
            tech_score += 15
            reasons.append("主力抢筹")
        
        # --- E. 身份认定 ---
        # 动态分主要靠静态库命中数
        dyn_score = len(static_sources) * 20
        total_score = tech_score + dyn_score
        
        # 门槛
        threshold = 60 if BattleConfig.IS_FREEZING_POINT else 70
        if total_score < threshold: return None
        
        identity = "🐕跟风"
        advice = "观察"
        
        has_big_fund = (net_flow > 80000000)
        has_theme = (len(static_sources) > 0)
        
        # 身份定义逻辑
        if is_shipping:
            identity = warning_msg # 直接覆盖身份显示为警告
            advice = "回避/卖出"
            total_score = 50 # 强制低分
        elif total_score >= 100 and has_theme:
            identity = "🐲真龙 (T0)"; advice = "锁仓/抢筹"
        elif has_big_fund and snapshot_row['circ_mv'] > 100 * 10**8:
            identity = "🐢中军 (T1)"; advice = "均线低吸"
        elif has_theme and limit_ups >= 1:
            identity = "🚀先锋 (T1)"; advice = "打板/半路"
        elif "新高" in reasons:
            identity = "💰趋势龙 (T2)"; advice = "五日线跟随"
        else:
            identity = "🦊套利 (T3)"; advice = "快进快出"

        return {
            "代码": code, "名称": name,
            "身份": identity, "结论": advice,
            "总分": total_score,
            "主力净额": flow_str,
            "所属行业": ind_source if ind_source else "-",
            "上涨源头": ",".join(all_sources),
            "技术特征": "|".join(reasons),
            "涨幅%": pct_chg,
            "换手%": turnover
        }

# ==========================================
# 4. 指挥中枢
# ==========================================
class Commander:
    def run(self):
        print(Fore.GREEN + "=== 🐲 A股游资·真龙天眼 (最终修正版) ===")
        
        print(Fore.CYAN + ">>> [1/4] 获取全市场快照...")
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

        print(Fore.CYAN + f">>> [2/4] 执行漏斗 (初始标准: 换手>{BattleConfig.FILTER_TURNOVER}%)...")
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
        
        print(Fore.CYAN + f">>> [3/4] 深度分析 (个股查证 + 出货识别)...")
        engine = IdentityEngine()
        results = []
        tasks = [row.to_dict() for _, row in candidates.iterrows()]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=BattleConfig.MAX_WORKERS) as ex:
            futures = {ex.submit(engine.analyze, task): task for task in tasks}
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
                try:
                    res = f.result(timeout=15)
                    if res: results.append(res)
                except: continue

        print(Fore.CYAN + f">>> [4/4] 导出: {BattleConfig.FILE_NAME}")
        if results:
            results.sort(key=lambda x: x['总分'], reverse=True)
            df_res = pd.DataFrame(results[:40])
            
            # 确保列齐全
            cols = ["代码", "名称", "身份", "结论", "总分", "主力净额", "上涨源头", "所属行业", "技术特征", "涨幅%", "换手%"]
            df_res = df_res[[c for c in cols if c in df_res.columns]]
            
            df_res.to_excel(BattleConfig.FILE_NAME, index=False)
            print(Fore.GREEN + f"✅ 成功! 文件: {BattleConfig.FILE_NAME}")
            try:
                print(df_res[['名称', '身份', '主力净额', '上涨源头']].head(5).to_string(index=False))
            except: pass
        else:
            print(Fore.RED + "⚠️ 无结果，导出初选名单。")
            candidates.to_excel(BattleConfig.FILE_NAME, index=False)

if __name__ == "__main__":
    Commander().run()
