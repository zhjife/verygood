# -*- coding: utf-8 -*-
"""
A股游资·天眼系统 (God Mode / 最终全装甲版)
功能：CMF资金算法 + 竞价弱转强 + 尾盘风控 + 舆情排雷 + 龙头锚定
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

# 初始化
init(autoreset=True)
warnings.filterwarnings('ignore')

# ==========================================
# 0. 全局作战配置 (Battle Configuration)
# ==========================================
class BattleConfig:
    # --- 基础漏斗 (Funnel) ---
    MIN_CAP = 15 * 10**8       # 最小流通市值 15亿
    MAX_CAP = 400 * 10**8      # 最大流通市值 400亿 (容纳中军)
    MIN_PRICE = 3.0            # 最低价
    MAX_PRICE = 90.0           # 最高价
    
    # --- 活跃度门槛 ---
    FILTER_PCT_CHG = 2.0       # 涨幅 > 2% (捕捉起爆点)
    FILTER_TURNOVER = 4.5      # 换手 > 4.5% (游资票必须活跃)
    
    # --- 系统参数 ---
    HISTORY_DAYS = 60          # K线回溯天数
    MAX_WORKERS = 8            # 并发线程数
    FILE_NAME = f"Dragon_GodMode_{datetime.now().strftime('%Y%m%d')}.xlsx"

# ==========================================
# 1. 舆情风控哨兵 (News Sentry)
# ==========================================
class NewsSentry:
    """
    全网搜索个股资讯，进行关键词排雷。
    只在股票通过技术面筛选后才触发，节省资源。
    """
    NEGATIVE_KEYWORDS = [
        "立案", "调查", "违规", "警示", "减持", "亏损", "大幅下降", 
        "无法表示意见", "ST", "退市", "诉讼", "冻结", "平仓", "黑天鹅", "留置"
    ]
    
    @staticmethod
    def check_news(code):
        try:
            # 随机延迟，防止请求过快
            time.sleep(random.uniform(0.1, 0.3))
            # 获取个股新闻
            df = ak.stock_news_em(symbol=code)
            if df is None or df.empty:
                return False, "无近期资讯"
            
            # 取最近 10 条标题
            recent_news = df.head(10)['新闻标题'].tolist()
            risk_msgs = []
            
            for title in recent_news:
                for kw in NewsSentry.NEGATIVE_KEYWORDS:
                    if kw in title:
                        if kw not in str(risk_msgs):
                            risk_msgs.append(kw)
            
            if risk_msgs:
                return True, f"⚠️利空含:{','.join(risk_msgs)}"
            
            return False, "舆情平稳"
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
        print(Fore.MAGENTA + ">>> [1/7] 扫描游资龙虎榜基因...")
        try:
            for i in range(3): # 追溯3天
                d = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
                try:
                    df = ak.stock_lhb_detail_daily_sina(date=d)
                    if df is not None and not df.empty:
                        codes = df['代码'].astype(str).tolist()
                        self.lhb_stocks.update(codes)
                except: pass
            print(Fore.GREEN + f"    ✅ 基因库构建完毕，收录 {len(self.lhb_stocks)} 只游资票")
        except Exception as e:
            print(Fore.YELLOW + f"    ⚠️ 龙虎榜接口波动: {e}")

    def has_gene(self, code):
        return code in self.lhb_stocks

# ==========================================
# 3. 热点与龙头锚定雷达 (Hot Concept & Leader Radar)
# ==========================================
class HotConceptRadar:
    """
    扫描全市场热点，并锁定每个板块的【当前龙头】作为参照物。
    """
    def __init__(self):
        self.stock_concept_map = {}   # {个股代码: 概念名称}
        self.concept_leader_map = {}  # {概念名称: "龙头名(涨幅%)"}

    def scan(self):
        print(Fore.MAGENTA + ">>> [2/7] 扫描顶级热点 & 锁定板块龙头...")
        try:
            df_board = ak.stock_board_concept_name_em()
            # 过滤干扰项
            noise = ["昨日", "连板", "首板", "涨停", "融资", "融券", "转债", "ST", "板块", "指数", "深股通", "沪股通"]
            mask = ~df_board['板块名称'].str.contains("|".join(noise))
            # 取涨幅前 10 的核心板块
            df_top = df_board[mask].sort_values(by="涨跌幅", ascending=False).head(10)
            hot_list = df_top['板块名称'].tolist()
            
            print(Fore.MAGENTA + f"    🔥 顶级风口: {hot_list[:6]}")
            
            # 定义获取成分股的函数
            def fetch_constituents(name):
                try:
                    df = ak.stock_board_concept_cons_em(symbol=name)
                    if df is not None and not df.empty:
                        # 尝试寻找龙头 (涨幅第一)
                        leader_info = "未知"
                        if '涨跌幅' in df.columns:
                            df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
                            df.sort_values(by='涨跌幅', ascending=False, inplace=True)
                            top_stock = df.iloc[0]
                            leader_info = f"{top_stock['名称']}({top_stock['涨跌幅']}%)"
                        
                        return name, df['代码'].tolist(), leader_info
                    return name, [], "-"
                except: return name, [], "-"
            
            # 多线程抓取
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                futures = [ex.submit(fetch_constituents, t) for t in hot_list]
                for f in concurrent.futures.as_completed(futures):
                    c_name, codes, l_info = f.result()
                    self.concept_leader_map[c_name] = l_info
                    for code in codes:
                        if code not in self.stock_concept_map: self.stock_concept_map[code] = []
                        self.stock_concept_map[code].append(c_name)
                        
            print(Fore.GREEN + f"    ✅ 龙头锚定完毕 (示例: {list(self.concept_leader_map.items())[0]})")
            
        except Exception as e:
            print(Fore.RED + f"    ⚠️ 热点雷达波动: {e}")

    def get_info(self, code):
        """返回: (是否热点, 概念名, 龙头信息)"""
        concepts = self.stock_concept_map.get(code, [])
        if not concepts: return False, "-", "-"
        main_concept = concepts[0] # 取第一个主要概念
        leader_info = self.concept_leader_map.get(main_concept, "-")
        return True, main_concept, leader_info

# ==========================================
# 4. 市场哨兵 (Market Sentry)
# ==========================================
class MarketSentry:
    """大盘环境风控，暴跌时自动收紧策略"""
    @staticmethod
    def check_market():
        try:
            df = ak.stock_zh_index_daily(symbol="sh000001")
            today = df.iloc[-1]
            pct = (today['close'] - today['open']) / today['open'] * 100
            
            if pct < -1.5:
                print(Fore.RED + f"⚠️ 警告：大盘暴跌 ({round(pct,2)}%)，已启动【防御模式】。")
                BattleConfig.FILTER_PCT_CHG = 5.0 # 提高门槛，只看硬板
            else:
                print(Fore.GREEN + f"✅ 大盘环境正常 ({round(pct,2)}%)。")
        except:
            pass

# ==========================================
# 5. 核心分析引擎 (Identity Engine)
# ==========================================
class IdentityEngine:
    def __init__(self, concept_radar, lhb_radar):
        self.concept_radar = concept_radar
        self.lhb_radar = lhb_radar

    def get_kline(self, code):
        """获取K线数据，带重试机制"""
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=BattleConfig.HISTORY_DAYS)).strftime("%Y%m%d")
        for _ in range(3):
            try:
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
                if df is not None and not df.empty:
                    df.rename(columns={'日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'volume','成交额':'amount','涨跌幅':'pct_chg'}, inplace=True)
                    return df
            except: time.sleep(0.1)
        return None

    def calculate_cmf(self, df):
        """计算 CMF 资金流指标"""
        try:
            high = df['high']; low = df['low']; close = df['close']; volume = df['volume']
            range_hl = (high - low).replace(0, 0.01) # 防止除0
            mf_vol = (((close - low) - (high - close)) / range_hl) * volume
            cmf = mf_vol.rolling(20).sum() / volume.rolling(20).sum()
            return cmf.iloc[-1]
        except: return 0.0

    def check_overheat(self, df, turnover):
        """情绪过热熔断器"""
        try:
            close = df['close']; pct_chg = df['pct_chg']
            # 1. RSI极度超买
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(6).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
            rsi = 100 - (100 / (1 + gain / loss))
            if rsi.iloc[-1] > 90: return True, "RSI超买"
            
            # 2. 加速赶顶 (高位放量滞涨/星线)
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
        
        # --- 1. 获取数据 ---
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
        
        # A. 炸板/烂板检测 (Touch Limit but Failed)
        # 假设涨停是10% (近似), 触及涨停但回落 > 3%
        if high >= prev['close'] * 1.095 and (high - close) / close > 0.03:
            is_risk = True; risk_msg.append("炸板/烂板")
            
        # B. 乖离率过大
        ma5 = df['close'].rolling(5).mean().iloc[-1]
        if (close - ma5) / ma5 > 0.18:
            is_risk = True; risk_msg.append("乖离率大")
            
        # C. 均价压制 (VWAP Pressure)
        vwap = amount / volume if volume > 0 else close
        if close < vwap * 0.985 and pct_chg < 9.8:
            is_risk = True; risk_msg.append("均价压制")
            
        # D. 情绪过热熔断
        is_oh, oh_msg = self.check_overheat(df, turnover)
        if is_oh: is_risk = True; risk_msg.append(oh_msg)

        # --- 3. 机会挖掘 (Offense) ---
        
        # A. 竞价与开盘 (Auction)
        if vol_ratio > 8.0: score += 15; features.append(f"竞价抢筹(量比{vol_ratio})")
        
        # B. 弱转强 (Weak to Strong)
        open_pct = (open_p - prev['close']) / prev['close'] * 100
        if prev['pct_chg'] < 3.0 and 2.0 < open_pct < 6.0:
            score += 20; features.append("🔥弱转强")
            
        # C. 基因 (Genes)
        limit_ups = len(df[df['pct_chg'] > 9.5].tail(20))
        if limit_ups > 0: score += 10; features.append(f"妖股({limit_ups}板)")
        if self.lhb_radar.has_gene(code): score += 20; features.append("🐉龙虎榜")
        
        # D. 资金 (Money Flow)
        if cmf_val > 0.15: score += 15; features.append("主力锁仓")
        elif cmf_val < -0.1: score -= 15; features.append("资金流出")
        
        # E. 热点 (Hot Concept)
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

        # --- 4. 舆情排雷 (仅对优质股检查) ---
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

        # 过滤低分杂毛 (保留高分 或 有风险提示的)
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
    def generate_excel(self, df_res):
        """生成带说明书和格式化的Excel"""
        with pd.ExcelWriter(BattleConfig.FILE_NAME, engine='xlsxwriter') as writer:
            df_res.to_excel(writer, sheet_name='真龙榜', index=False)
            
            # 使用说明书
            manual_data = {
                '关键列名': ['身份', '板块龙头', '舆情风控', '量比 (9:25专用)', 'CMF (14:30专用)', '特征-弱转强', '特征-炸板'],
                '实战含义': [
                    '【真龙T0】: 确定性最高，热点+资金+龙虎榜共振；【陷阱】: 无论涨多好，坚决不买，有货快跑。',
                    '锚定效应。如果龙头涨停，你的跟风票才安全；如果龙头跳水，你的票要先跑。',
                    '一票否决。如果含“立案、调查”等字眼，大概率第二天跌停，切勿火中取栗。',
                    '竞价抢筹指标。> 5.0 表示主力急不可耐；> 10 表示极度一致。配合“弱转强”使用。',
                    '主力意图指标。> 0.15 表示主力锁仓（买的多卖的少）；< 0 表示主力流出。',
                    '最强游资信号。昨日弱势，今日高开爆量，往往是连板起点。',
                    '最强风险信号。摸过涨停但没封住，套牢盘巨大，次日大概率核按钮。'
                ]
            }
            pd.DataFrame(manual_data).to_excel(writer, sheet_name='实战说明书', index=False)
            
            # 格式美化
            wb = writer.book
            ws = writer.sheets['真龙榜']
            
            # 红色高亮利空/陷阱
            fmt_bad = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
            ws.conditional_format('C2:C150', {'type': 'text', 'criteria': 'containing', 'value': '陷阱', 'format': fmt_bad})
            ws.conditional_format('G2:G150', {'type': 'text', 'criteria': 'containing', 'value': '利空', 'format': fmt_bad})
            
            # 绿色高亮真龙
            fmt_good = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
            ws.conditional_format('C2:C150', {'type': 'text', 'criteria': 'containing', 'value': '真龙', 'format': fmt_good})

    def run(self):
        print(Fore.GREEN + f"=== 🐲 A股游资·天眼系统 (Ultimate Full-Armor) ===")
        
        # --- 智能时间感知 ---
        now_t = datetime.now().time()
        t_925 = datetime.strptime("09:25", "%H:%M").time()
        t_1030 = datetime.strptime("10:30", "%H:%M").time()
        t_1430 = datetime.strptime("14:30", "%H:%M").time()
        
        print(Fore.YELLOW + f"🕒 当前时间: {now_t.strftime('%H:%M:%S')}")
        if t_925 <= now_t < t_1030:
            print(Fore.RED + "🔥 [竞价/早盘模式] 战术：找【量比>5】且【弱转强】的票，关注【板块龙头】走势。")
        elif now_t >= t_1430:
            print(Fore.BLUE + "🛡️ [尾盘/复盘模式] 战术：剔除【陷阱】(炸板/均价压制)，潜伏【CMF>0.15】的真龙。")
        else:
            print(Fore.WHITE + "☕ [盘中震荡] 战术：多看少动，等待尾盘信号。")

        # 1. 启动雷达
        MarketSentry.check_market()
        lhb = DragonTigerRadar(); lhb.scan()
        concept = HotConceptRadar(); concept.scan()
        
        # 2. 获取快照
        print(Fore.CYAN + ">>> [3/7] 全市场快照 & 竞价数据...")
        try:
            df = ak.stock_zh_a_spot_em()
            df.rename(columns={'代码':'code','名称':'name','最新价':'close','涨跌幅':'pct_chg','换手率':'turnover','流通市值':'circ_mv','量比':'量比'}, inplace=True)
            for c in ['close','pct_chg','turnover','circ_mv','量比']: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        except Exception as e:
            print(Fore.RED + f"❌ 数据获取失败: {e}"); return

        # 3. 漏斗
        print(Fore.CYAN + ">>> [4/7] 漏斗筛选...")
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

        # 4. 深度运算
        print(Fore.CYAN + ">>> [5/7] 深度运算 (资金+风控+舆情+龙头锚定)...")
        engine = IdentityEngine(concept, lhb)
        results = []
        # 优先处理量比高的，取前120只
        tasks = [row.to_dict() for _, row in candidates.sort_values(by='量比', ascending=False).head(120).iterrows()]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=BattleConfig.MAX_WORKERS) as ex:
            futures = {ex.submit(engine.analyze, task): task for task in tasks}
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
                try:
                    res = f.result(timeout=20)
                    if res: results.append(res)
                except: continue

        # 5. 导出
        print(Fore.CYAN + f">>> [6/7] 生成战报: {BattleConfig.FILE_NAME}")
        if results:
            df_res = pd.DataFrame(results)
            df_res.sort_values(by='总分', ascending=False, inplace=True)
            
            # 整理列顺序
            cols = ['代码','名称','身份','建议','板块龙头','舆情风控','总分','涨幅%','量比','CMF','特征']
            df_res = df_res[cols]
            
            self.generate_excel(df_res)
            print(Fore.GREEN + f"✅ 成功! 请打开 Excel 查看【实战说明书】")
            print(df_res[['名称','身份','板块龙头','特征']].head(5).to_string(index=False))
        else:
            print(Fore.RED + "❌ 无有效标的。")

if __name__ == "__main__":
    Commander().run()
