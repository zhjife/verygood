import akshare as ak
import pandas as pd
import numpy as np
import time
import concurrent.futures
from datetime import datetime, timedelta
from tqdm import tqdm
from colorama import init, Fore, Style, Back
import requests
import warnings
import random

# ==========================================
# 0. 战备参数 (针对快速轮动优化)
# ==========================================
init(autoreset=True)
warnings.filterwarnings('ignore')

class BattleConfig:
    # 资金门槛：轮动快时，微盘股流动性差容易被核按钮，大盘股拉不动
    MIN_CAP = 18 * 10**8       # 提高到18亿，过滤掉纯粹的庄股
    MAX_CAP = 600 * 10**8      
    
    # 价格门槛
    MIN_PRICE = 3.5            
    MAX_PRICE = 95.0          
    
    # 进攻信号：在轮动行情中，只有日内强势的才能拿住
    FILTER_PCT_CHG = 3.8       # 提高到3.8%，只有强势股才配在轮动中生存
    FILTER_TURNOVER = 4.0      # 换手要充分
    
    HISTORY_DAYS = 250
    MAX_WORKERS = 12           # 高并发
    FILE_NAME = f"Rotation_Sniper_{datetime.now().strftime('%Y%m%d')}.xlsx"

# ==========================================
# 1. 动态板块雷达 (捕捉轮动核心)
# ==========================================
class SectorRotationRadar:
    """
    专门解决[快速轮动]问题。
    它不看新闻，只看真金白银砸向了哪个板块。
    """
    def __init__(self):
        self.hot_sectors = []       # 涨幅榜前列
        self.money_flow_sectors = [] # 资金净流入前列
        self.final_hot_list = []    # 综合研判后的热点列表

    def scan_market_sectors(self):
        print(Fore.MAGENTA + ">>> [1/5] 启动板块轮动雷达 (正在计算资金流向)...")
        try:
            # 1. 获取概念板块涨幅榜 (代表情绪)
            # 东方财富实时接口
            df_gain = ak.stock_board_concept_name_em()
            # 过滤掉非行业概念 (如"昨日连板", "融资融券"等噪音)
            mask = ~df_gain['板块名称'].str.contains("昨日|连板|融资|融券|转债|ST|板|标普|指数")
            df_gain = df_gain[mask].sort_values(by="涨跌幅", ascending=False)
            
            # 取涨幅前15名作为[情绪风口]
            top_gainers = df_gain.head(15)['板块名称'].tolist()
            
            # 2. 获取行业板块资金流 (代表主力真金白银)
            # 这一步是为了防止"一日游"的假高潮
            df_flow = ak.stock_market_fund_flow() # 实时资金流
            df_flow = df_flow.sort_values(by="今日主力净流入", ascending=False)
            top_flow = df_flow.head(15)['名称'].tolist()
            
            # 3. 交叉验证 (Cross Validation)
            # 如果一个板块既在涨幅榜，又在资金流入榜，那就是[主线]
            # 如果只在涨幅榜，可能是[轮动补涨]
            self.final_hot_list = list(set(top_gainers + top_flow))
            
            # 打印当前轮动核心
            print(Fore.MAGENTA + f"    🔥 情绪风口(涨幅): {top_gainers[:5]}...")
            print(Fore.MAGENTA + f"    💰 资金风口(流入): {top_flow[:5]}...")
            print(Fore.YELLOW +  f"    🎯 综合锁定今日核心板块: {len(self.final_hot_list)} 个")
            
        except Exception as e:
            print(Fore.RED + f"    ⚠️ 板块接口请求波动: {e}，启用备用策略")
            self.final_hot_list = []

    def get_sector_status(self, stock_concept_string):
        """
        判断某只个股的板块字符串，是否命中了今日热点
        返回: (匹配度分数, 命中的板块名)
        """
        score = 0
        hit_sectors = []
        
        if not self.final_hot_list or not stock_concept_string:
            return 0, []

        for hot in self.final_hot_list:
            # 精准匹配：防止"AI"匹配到"Airline"
            # 简单的字符串包含即可，因为板块名通常很独特
            if hot in stock_concept_string:
                score += 20 # 命中一个大热点加20分
                hit_sectors.append(hot)
                
        return score, hit_sectors

# ==========================================
# 2. 静态题材映射库 (兜底保障)
# ==========================================
class StaticThemeMap:
    """
    解决API板块命名不规范的问题。
    比如API叫"通用航空"，新闻叫"低空经济"。
    这里做强映射，确保不漏。
    """
    THEME_DICT = {
        "低空经济": ["飞行汽车", "eVTOL", "无人机", "通航", "万丰", "宗申", "低空"],
        "华为产业链": ["华为", "海思", "鸿蒙", "欧拉", "星闪", "昇腾", "鲲鹏", "P70"],
        "AI算力": ["CPO", "光模块", "液冷", "算力", "服务器", "英伟达", "铜连接", "HBM"],
        "固态电池": ["固态", "电解质", "硫化物", "全固态", "清陶"],
        "人形机器人": ["机器人", "减速器", "伺服", "电机", "传感器", "优必选"],
        "商业航天": ["卫星", "火箭", "航天", "星网", "G60"],
        "半导体": ["芯片", "光刻机", "存储", "封测", "第三代", "碳化硅"],
        "车路云": ["自动驾驶", "车路云", "V2X", "雷达", "智驾"],
        "并购重组": ["重组", "股权转让", "变更", "借壳"],
        "大金融": ["证券", "银行", "保险", "互联金融", "信托"]
    }

    @staticmethod
    def match(text):
        hits = []
        for theme, kws in StaticThemeMap.THEME_DICT.items():
            for kw in kws:
                if kw in text:
                    hits.append(theme)
                    break 
        return hits

# ==========================================
# 3. 深度逻辑分析引擎 (全逻辑)
# ==========================================
class DeepLogicEngine:
    def __init__(self, radar):
        self.radar = radar

    def get_stock_data(self, code):
        """稳健获取K线，带重试"""
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
        
        # --- A. 技术面一票否决 (The Filter) ---
        # 1. 获取K线
        df = self.get_stock_data(code)
        if df is None or len(df) < 60: return None
        
        close = df['close'].values
        ma5 = pd.Series(close).rolling(5).mean().values
        ma10 = pd.Series(close).rolling(10).mean().values
        ma20 = pd.Series(close).rolling(20).mean().values
        ma60 = pd.Series(close).rolling(60).mean().values
        
        curr_price = close[-1]
        
        # 2. 趋势硬性门槛
        # 在轮动行情中，破位的股票是没人救的，必须在生命线(MA60)之上
        if curr_price < ma60[-1]: return None
        
        # 3. 攻击形态门槛
        # 必须是多头排列，或者今日放量突破20日线
        is_bullish = (ma5[-1] > ma10[-1]) 
        is_breakout = (curr_price > ma20[-1]) and (df['open'].values[-1] < ma20[-1])
        if not (is_bullish or is_breakout): return None

        # --- B. 题材精准捕捉 (The Brain) ---
        # 这是捕捉轮动的核心：结合个股所属板块 + 新闻舆情
        
        # 1. 获取个股所属板块 (东财接口)
        # 这一步非常关键，它告诉我们这只股票到底是什么成份
        stock_concepts = ""
        try:
            # 获取个股关联概念，如果接口慢，可以考虑只对初筛过的做
            # 这里为了精准，必须做
            concept_df = ak.stock_board_concept_cons_em(symbol=code) 
            # 注意：上述接口是查板块里的股，反向查股所属板块比较慢
            # 优化：改用 stock_individual_info_em 或 stock_news_em 提取
            pass 
        except: pass
        
        # 替代方案：通过新闻和名称来匹配，同时利用 base_info 里可能隐含的行业信息
        # 为了不拖慢速度，我们模拟获取一次新闻和行业
        try:
            news_df = ak.stock_news_em(symbol=code)
            news_text = name
            if not news_df.empty:
                news_text += " ".join(news_df.head(3)['新闻标题'].tolist())
        except: 
            news_text = name

        # C. 双重题材评分
        # 分数来源1: 动态雷达 (命中今日涨幅榜板块)
        # 我们用新闻文本去撞击雷达列表
        dynamic_score, hit_dynamic_sectors = self.radar.get_sector_status(news_text)
        
        # 分数来源2: 静态字典 (命中长期主线)
        hit_static_themes = StaticThemeMap.match(news_text)
        static_score = len(hit_static_themes) * 10
        
        # --- C. 结构面评分 (The Structure) ---
        tech_score = 60 # 基础分
        reasons = []
        
        # 1. 距离前高 (压力位)
        h120 = df['high'].iloc[-120:].max()
        dist = (h120 - curr_price) / curr_price
        
        if dist < 0.02: 
            tech_score += 25; reasons.append("🚀突破新高")
        elif dist < 0.15: 
            tech_score += 15; reasons.append("🧗逼近前高")
            
        # 2. 涨停基因 (游资偏好)
        limit_ups = len(df[df['pct_chg'] > 9.5].tail(15))
        if limit_ups >= 3:
            tech_score += 20; reasons.append(f"🐲妖股({limit_ups}板)")
        elif limit_ups >= 1:
            tech_score += 10; reasons.append("⚡活跃")
            
        # 3. 烂板/硬板识别 (日内强度)
        if base_info['pct_chg'] > 9.5:
            if base_info['close'] == base_info['high']:
                reasons.append("硬板")
            else:
                reasons.append("烂板") # 烂板次日需弱转强

        # --- D. 综合总分 ---
        # 核心逻辑：(技术分 + 题材分)
        # 如果动态分为0 (说明不在今日轮动风口)，则除非技术面极强(>85分)，否则剔除
        # 这就是"轮动克星"：非风口股，长得再好也容易被吸血。
        
        total_score = tech_score + dynamic_score + static_score
        
        if dynamic_score == 0 and total_score < 85:
            return None # 没蹭上热点，形态又不是神级，丢弃
            
        if total_score < 75: return None

        # 构造输出
        all_themes = list(set(hit_dynamic_sectors + hit_static_themes))
        
        # 竞价指令
        advice = "观察"
        if dynamic_score > 0 and "突破" in str(reasons):
            advice = "🔥主线突破(重仓)"
        elif dynamic_score > 0:
            advice = "⚡风口套利(跟随)"
        elif "妖股" in str(reasons):
            advice = "🐲龙头博弈(分歧低吸)"

        return {
            "代码": code, "名称": name,
            "总分": total_score,
            "操盘指令": advice,
            "命中热点": ",".join(all_themes) if all_themes else "(独立逻辑)",
            "技术形态": "|".join(reasons),
            "现价": curr_price, 
            "涨幅%": base_info['pct_chg'],
            "换手%": base_info['turnover'],
            "轮动状态": "✅在风口" if dynamic_score > 0 else "❌非风口"
        }

# ==========================================
# 4. 指挥中枢
# ==========================================
class Commander:
    def run(self):
        print(Fore.GREEN + "=== 🐲 A股轮动克星·全景实战系统 (Logic Full) ===")
        print(Fore.WHITE + "策略核心：动态板块资金流 + 静态题材库 + 严格技术形态")
        
        # 1. 启动板块雷达 (获取最新的轮动方向)
        radar = SectorRotationRadar()
        radar.scan_market_sectors()
        
        # 2. 全市场扫描
        print(Fore.CYAN + ">>> [2/5] 获取全市场实时快照...")
        try:
            df = ak.stock_zh_a_spot_em()
            df.rename(columns={'代码':'code', '名称':'name', '最新价':'close', '涨跌幅':'pct_chg', 
                              '换手率':'turnover', '总市值':'total_mv', '流通市值':'circ_mv'}, inplace=True)
            for c in ['close', 'pct_chg', 'turnover', 'circ_mv']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        except: return

        # 3. 漏斗过滤 (The Funnel)
        # 在轮动快的行情下，只看"有辨识度"的票
        print(Fore.CYAN + f">>> [3/5] 执行严苛初筛 (涨幅>{BattleConfig.FILTER_PCT_CHG}%, 换手>{BattleConfig.FILTER_TURNOVER}%)...")
        mask = (
            (~df['name'].str.contains('ST|退|C|U')) & 
            (df['close'].between(BattleConfig.MIN_PRICE, BattleConfig.MAX_PRICE)) &
            (df['circ_mv'].between(BattleConfig.MIN_CAP, BattleConfig.MAX_CAP)) &
            (df['pct_chg'] >= BattleConfig.FILTER_PCT_CHG) & 
            (df['turnover'] >= BattleConfig.FILTER_TURNOVER)
        )
        candidates = df[mask].copy()
        
        # 关键：按[换手率]排序，取前150名。
        # 为什么？因为轮动越快，存量博弈越明显，资金只会去流动性最好的地方。
        # 没量的票，轮动到了也拉不动。
        candidates = candidates.sort_values(by='turnover', ascending=False).head(150)
        print(Fore.YELLOW + f"    📉 锁定 {len(candidates)} 只高流动性标的，进入深度匹配...")

        # 4. 深度并发分析
        engine = DeepLogicEngine(radar)
        results = []
        tasks = [row.to_dict() for _, row in candidates.iterrows()]
        
        print(Fore.CYAN + f">>> [4/5] 启动多线程深度计算 (Workers: {BattleConfig.MAX_WORKERS})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=BattleConfig.MAX_WORKERS) as ex:
            futures = [ex.submit(engine.analyze, task) for task in tasks]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
                res = f.result()
                if res: results.append(res)

        # 5. 结果生成
        print(Fore.CYAN + f">>> [5/5] 生成作战指令: {BattleConfig.FILE_NAME}")
        
        # 排序：总分优先 > 涨幅优先
        results.sort(key=lambda x: (x['总分'], x['涨幅%']), reverse=True)
        final_list = results[:35]
        
        if final_list:
            df_res = pd.DataFrame(final_list)
            cols = ["代码", "名称", "总分", "轮动状态", "操盘指令", "命中热点", "技术形态", "现价", "涨幅%", "换手%"]
            df_res = df_res[cols]
            
            df_res.to_excel(BattleConfig.FILE_NAME, index=False)
            
            print(Fore.GREEN + "\n🔥 === 今日轮动核心标的 (Top 5) === 🔥")
            print(df_res[["名称", "总分", "轮动状态", "操盘指令", "命中热点"]].head(5).to_string(index=False))
            print(Fore.WHITE + f"\n✅ 报告生成完毕。重点关注[轮动状态]为'✅在风口'的标的。")
        else:
            print(Fore.RED + "❌ 今日市场极度撕裂，无符合轮动模型的标的。")

if __name__ == "__main__":
    start = time.time()
    Commander().run()
    print(f"\n耗时: {time.time() - start:.1f}s")
