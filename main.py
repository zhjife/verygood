import akshare as ak
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import logging
import concurrent.futures
from datetime import datetime, timedelta
from tqdm import tqdm
from colorama import init, Fore, Style
import warnings

# ==========================================
# 0. 全局配置与初始化
# ==========================================
init(autoreset=True)
warnings.filterwarnings('ignore')

class Config:
    # --- 基础门槛 (硬性过滤: 游资审美) ---
    MIN_CAP = 12 * 10**8      # 12亿
    MAX_CAP = 400 * 10**8     # 400亿
    MIN_PRICE = 2.5           # 最低价
    MAX_PRICE = 90.0          # 最高价
    
    # --- 核心交易参数 ---
    MIN_TURNOVER = 5.0        # 最小换手
    TARGET_TURNOVER = (5.0, 25.0) 
    LIMIT_THRESHOLD = 9.5     
    HISTORY_DAYS = 120        
    
    # --- 知名席位词库 ---
    FAMOUS_SEATS = [
        "机构专用", "深股通", "沪股通", 
        "中信证券西安朱雀", "国泰君安上海江苏路", "财通证券杭州上塘路", 
        "华鑫证券上海分公司", "中国银河北京中关村", "东吴证券苏州西北街"
    ]
    
    # --- 系统参数 ---
    MAX_WORKERS = 12          
    TIMEOUT = 5               
    FILE_NAME = f"实战指令单_{datetime.now().strftime('%Y%m%d')}.xlsx"

# ==========================================
# 通用工具：带重试的数据拉取 (放在 Config 类外面，顶格写)
# ==========================================
def fetch_data_with_retry(func, max_retries=10, delay=5, *args, **kwargs):
    """
    通用重试函数：解决 GitHub 网络不稳定问题
    """
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"    [网络波动] 第 {i+1}/{max_retries} 次尝试失败，{delay}秒后重试... 错误: {e}")
            time.sleep(delay)
    print("    [严重错误] 重试多次仍失败，放弃。")
    return pd.DataFrame() # 返回空表

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Warlord")
    

# ==========================================
# 1. 大盘风控雷达 (Market Risk Radar)
# ==========================================
class MarketRadar:
    """
    负责判断大盘情绪：冰点/混沌/高潮
    只有环境安全，才允许开仓。
    """
    def __init__(self):
        self.sentiment = "中性"
        self.is_safe = True
        
    def scan(self):
        print(Fore.CYAN + ">>> [1/5] 正在测算全市场温度 (风控扫描)...")
        try:
            # 获取实时快照
            df = ak.stock_zh_a_spot_em()
            
            # 统计核心数据
            up_count = len(df[df['涨跌幅'] > 0])
            down_count = len(df[df['涨跌幅'] < 0])
            limit_up = len(df[df['涨跌幅'] >= 9.0])
            limit_down = len(df[df['涨跌幅'] <= -9.0])
            
            # 风控模型 logic
            # 1. 冰点熔断：跌停 > 20 且 跌停 > 涨停
            if limit_down > 20 and limit_down > limit_up:
                self.sentiment = "❄️ 冰点退潮 (空仓)"
                self.is_safe = False
                print(Fore.RED + f"    ⚠️ 严重警告：跌停家数({limit_down})激增，亏钱效应显著！系统触发熔断。")
            # 2. 情绪高潮
            elif limit_up > 60:
                self.sentiment = "🔥 情绪高潮 (积极)"
                self.is_safe = True
            # 3. 普跌迷茫
            elif up_count < 1200:
                self.sentiment = "☁️ 普跌迷茫 (防守)"
                self.is_safe = False # 普跌日尽量不做首板，只做抱团核心
            # 4. 正常轮动
            else:
                self.sentiment = "🌤️ 震荡轮动 (试错)"
                self.is_safe = True
                
            print(f"    市场状态: {self.sentiment} | 涨停: {limit_up} | 跌停: {limit_down} | 上涨家数: {up_count}")
            return self.is_safe
            
        except Exception as e:
            print(Fore.YELLOW + f"    风控数据获取异常: {e}，默认谨慎放行。")
            return True

# ==========================================
# 2. 情报与题材局 (Intelligence Bureau)
# ==========================================
class IntelligenceBureau:
    """
    负责搜集全网舆情，建立 [关键词 -> 题材] 的映射
    用于判断题材是否破圈、是否是主流。
    """
    def __init__(self):
        self.hot_buzz_words = [] # 百度热搜词
        self.market_mainline = [] # 股市领涨题材
        
        # 核心题材映射表 (需定期维护更新)
        # 这是游资联想力的核心
        self.theme_map = {
            "低空经济": ["飞行汽车", "eVTOL", "无人机", "通航", "低空", "万丰", "宗申"],
            "AI算力": ["CPO", "光模块", "液冷", "英伟达", "算力", "服务器", "铜连接", "中际"],
            "华为产业链": ["鸿蒙", "P70", "华为", "海思", "欧拉", "星闪", "昇腾", "Mate"],
            "固态电池": ["锂电", "固态", "电池", "电解质", "三祥", "清陶"],
            "有色资源": ["黄金", "铜", "铝", "有色", "紫金", "洛阳"],
            "设备更新": ["机床", "机器人", "工业母机", "农机", "电梯"],
            "合成生物": ["生物", "发酵", "合成", "川宁", "蔚蓝"],
            "出海逻辑": ["出海", "跨境", "海运", "家电", "工程机械"]
        }

    def fetch_intelligence(self):
        print(Fore.CYAN + ">>> [2/5] 扫描全网热搜与主线题材...")
        
        # 1. 爬取百度热搜 (社会舆情佐证)
        try:
            url = "https://top.baidu.com/board?tab=realtime"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=Config.TIMEOUT)
            soup = BeautifulSoup(resp.text, 'html.parser')
            self.hot_buzz_words = [item.get_text().strip() for item in soup.find_all('div', class_='c-single-text-ellipsis')[:40]]
            print(Fore.YELLOW + f"    全网热搜捕获: {len(self.hot_buzz_words)} 条")
        except: pass

        # 2. 扫描股市涨幅榜 (资金投票佐证)
        try:
            concept_df = ak.stock_board_concept_name_em()
            self.market_mainline = concept_df.sort_values(by="涨跌幅", ascending=False).head(15)['板块名称'].tolist()
            print(Fore.YELLOW + f"    今日资金主攻方向: {self.market_mainline[:6]}...")
        except: pass

    def analyze_text_for_themes(self, text):
        """
        分析文本，返回：(命中的题材列表, 是否涉及破圈热搜)
        """
        hits = []
        is_viral = False
        
        # A. 匹配预设题材库
        for theme, keywords in self.theme_map.items():
            for kw in keywords:
                if kw in text:
                    hits.append(theme)
                    # 检查是否破圈 (该题材的关键词同时也出现在百度热搜中)
                    for buzz in self.hot_buzz_words:
                        if kw in buzz or theme in buzz:
                            is_viral = True
                    break
        
        # B. 匹配股市主线
        for main in self.market_mainline:
            if main in text:
                hits.append(f"{main}")
                
        return list(set(hits)), is_viral

# ==========================================
# 3. 深度分析引擎 (Deep Analysis Engine)
# ==========================================
class AnalysisEngine:
    """
    负责单只股票的深度政审：筹码、资金、心理、形态
    """
    def __init__(self, intel):
        self.intel = intel

    def check_pressure_and_structure(self, code, current_price):
        """
        [筹码结构分析]
        计算: 距120日新高的距离。判断上方是否有套牢盘。
        """
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=Config.HISTORY_DAYS)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            
            if df.empty or len(df) < 60: return "数据不足", 0
            
            max_high = df['最高'].max()
            dist_to_high = (max_high - current_price) / current_price
            
            # 结构判定 logic
            if dist_to_high < 0.03:
                return "🌌突破新高(无阻力)", 25, dist_to_high
            elif dist_to_high < 0.15:
                return "🧗接近前高(需换手)", 10, dist_to_high
            elif dist_to_high > 0.40:
                return f"🌊深水套牢(距前高{dist_to_high:.0%})", -20, dist_to_high
            else:
                return "⚖️震荡区间", 0, dist_to_high
        except:
            return "分析失败", 0, 0

    def check_smart_money(self, code):
        """
        [龙虎榜分析]
        查询最近上榜记录，寻找主力痕迹
        """
        try:
            target_date = datetime.now().strftime("%Y%m%d")
            # 查今日，若无查昨日 (增加容错)
            lhb = ak.stock_lhb_detail_daily_sina(date=target_date, symbol=code)
            if lhb is None or lhb.empty:
                target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
                lhb = ak.stock_lhb_detail_daily_sina(date=target_date, symbol=code)
            
            if lhb is None or lhb.empty: return "无榜", 0
            
            # 分析买入席位
            buy_seats = " ".join(lhb.head(5)['买方名称'].astype(str).tolist())
            
            tags = []
            score = 5 # 上榜本身就有关注度
            
            if "机构专用" in buy_seats:
                tags.append("🔥机构大买")
                score += 20
            
            if "深股通" in buy_seats or "沪股通" in buy_seats:
                tags.append("💰北向加仓")
                score += 15
                
            for seat in Config.FAMOUS_SEATS:
                if seat in buy_seats and "机构" not in seat:
                    tags.append("🐉顶级游资")
                    score += 15
                    break
            
            return "|".join(tags) if tags else "普通榜", score
        except:
            return "查询异常", 0

    def profile_psychology(self, row, dist_to_high, money_status, is_viral, is_high_risk):
        """
        [心理画像引擎]
        将数据翻译为游资的情绪状态，辅助人工决策
        """
        psy_tags = []
        
        # 1. 风险画像
        if is_high_risk:
            return "⚠️雷区(主力出逃)"
        
        # 2. 空间心理
        if dist_to_high < 0.03:
            psy_tags.append("🚀破顶博弈")
        elif dist_to_high > 0.3:
            psy_tags.append("😰深水压力")
            
        # 3. 接力心理 (换手率与封板)
        if row['pct_chg'] > 9.5:
            if 8 <= row['turnover'] <= 20:
                psy_tags.append("🤝分歧转一致")
            elif row['turnover'] < 4:
                psy_tags.append("🔒缩量加速")
            elif row['turnover'] > 25:
                psy_tags.append("⚡高位大分歧")
        
        # 4. 信仰心理
        if "机构" in money_status:
            psy_tags.append("🏦机构背书")
        elif "游资" in money_status:
            psy_tags.append("🗡️游资合力")
            
        # 5. 舆情心理
        if is_viral:
            psy_tags.append("🔥全网共识")
        
        if not psy_tags:
            psy_tags.append("😐情绪一般")
            
        return " | ".join(psy_tags)

    def analyze_one_stock(self, row):
        """
        单只股票全流程分析
        """
        code, name = row['code'], row['name']
        score = 60 # 基础分
        reasons = []
        risks = []
        
        try:
            # --- 1. 硬性排雷 (新闻NLP) ---
            news_df = pd.DataFrame()
            try: news_df = ak.stock_news_em(symbol=code)
            except: pass
            
            latest_news = ""
            has_risk = False
            
            if not news_df.empty:
                full_text = " ".join(news_df.head(8)['新闻标题'].tolist())
                latest_news = news_df.iloc[0]['新闻标题']
                
                # 致命关键词 (排雷)
                risk_kws = ["立案", "调查", "警示", "违规", "减持", "退市", "ST"]
                for kw in risk_kws:
                    if kw in full_text:
                        # 有雷直接返回，不浪费算力
                        return None 
                
                # --- 2. 题材共振 ---
                themes, is_viral = self.intel.analyze_text_for_themes(full_text)
                if themes:
                    t_str = ",".join(themes)
                    score += 20
                    if is_viral:
                        score += 10
                        reasons.append(f"🔥破圈:{t_str}")
                    else:
                        reasons.append(f"题材:{t_str}")
            
            # --- 3. 筹码结构 ---
            struct_status, struct_score, dist_val = self.check_pressure_and_structure(code, row['close'])
            score += struct_score
            reasons.append(struct_status)
            
            # --- 4. 资金痕迹 ---
            money_status, money_score = self.check_smart_money(code)
            score += money_score
            
            # --- 5. 心理画像生成 ---
            # 整合以上数据，生成可读标签
            psy_profile = self.profile_psychology(row, dist_val, money_status, is_viral, has_risk)
            
            # --- 6. 技术形态与量能 ---
            # 涨停加分
            if row['pct_chg'] > Config.LIMIT_THRESHOLD:
                score += 15
                if row['close'] == row['high']:
                    reasons.append("硬板")
                else:
                    score -= 5
                    reasons.append("烂板")
            
            # 换手率加分
            if Config.TARGET_TURNOVER[0] <= row['turnover'] <= Config.TARGET_TURNOVER[1]:
                score += 10
                reasons.append("黄金换手")
            
            # --- 7. 实战指令生成 ---
            # 分数门槛：低于75分的杂毛股不显示
            if score < 75: return None
            
            # 建议仓位
            pos_pct = "10% (轻仓)"
            if score >= 90: pos_pct = "40% (重仓)"
            elif score >= 85: pos_pct = "20% (中仓)"
            
            # 竞价达标价 (预期高开2%为强势，低于此价不买)
            target_price = row['close'] * 1.02
            
            # 杂毛标记
            role_tag = "🐲核心龙" if score >= 90 else "🐕跟风/杂毛"
            
            return {
                "代码": code, "名称": name, 
                "总评分": score,
                "角色定位": role_tag,
                "心理画像": psy_profile,  # <--- 核心分析结果
                "建议仓位": pos_pct,
                "竞价开枪价": f"> {target_price:.2f}",
                "现价": row['close'], "涨幅%": row['pct_chg'], "换手%": row['turnover'],
                "市值(亿)": round(row['circ_mv']/10**8, 2),
                "主力痕迹": money_status,
                "最新资讯": latest_news
            }

        except Exception as e:
            return None

# ==========================================
# 4. 指挥官系统 (The Warlord)
# ==========================================
class DragonWarlord:
    def __init__(self):
        self.radar = MarketRadar()
        self.intel = IntelligenceBureau()
        self.engine = AnalysisEngine(self.intel)

    def execute(self):
        print(Fore.GREEN + "=============================================")
        print(Fore.GREEN + "   🐉 游资实战终极系统 (DragonWarlord Ult)   ")
        print(Fore.GREEN + "=============================================")
        
        # 1. 风控扫描
        if not self.radar.scan():
            # 熔断：如果是冰点，停止计算
            return

        # 2. 获取情报
        self.intel.fetch_intelligence()
      # ... (上接代码)

        # 3. 市场初筛 (Funnel Level 1)
        print(Fore.CYAN + ">>> [3/5] 拉取全市场数据并执行硬过滤 (增强重试版)...")
        
        # === 修改开始：使用重试机制 ===
        df = fetch_data_with_retry(ak.stock_zh_a_spot_em, max_retries=10, delay=5)
        # === 修改结束 ===

        if df.empty:
            print(Fore.RED + "    严重：无法获取市场数据，任务终止。可能是IP被封或接口维护。")
            return

        try:
            # 清洗
            df = df.rename(columns={"代码":"code", "名称":"name", "最新价":"close", "涨跌幅":"pct_chg", "换手率":"turnover", "流通市值":"circ_mv", "最高":"high"})
            for c in ['close', 'pct_chg', 'turnover', 'circ_mv', 'high']: df[c] = pd.to_numeric(df[c], errors='coerce')
            
            # 硬性门槛过滤
            mask = (
                (~df['name'].str.contains('ST|退')) &
                (df['close'].between(Config.MIN_PRICE, Config.MAX_PRICE)) &
                (df['circ_mv'].between(Config.MIN_CAP, Config.MAX_CAP)) &
                (df['turnover'].between(Config.MIN_TURNOVER, 40)) & 
                (df['pct_chg'] > 5.0) 
            )
            candidates = df[mask]
            print(f"    初筛入围: {len(candidates)} 只 (强势且有流动性)")
            
        except Exception as e:
            print(Fore.RED + f"数据处理失败: {e}")
            return

        # ... (下接深度分析代码)

        # 4. 深度并发分析 (Funnel Level 2)
        print(Fore.CYAN + f">>> [4/5] 启动深度政审 (并发数: {Config.MAX_WORKERS})...")
        results = []
        tasks = [row for _, row in candidates.iterrows()]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            data_iter = tqdm(executor.map(self.engine.analyze_one_stock, tasks), total=len(tasks))
            results = [x for x in data_iter if x is not None]
        
        # 排序
        results.sort(key=lambda x: x['总评分'], reverse=True)
        
        # 5. 生成作战指令
        self.export(results)

    def export(self, data):
        print(Fore.CYAN + f">>> [5/5] 生成作战指令: {Config.FILE_NAME}")
        if not data:
            print(Fore.RED + "    今日无符合严选标准的标的。")
            return
            
        df = pd.DataFrame(data)
        
        with pd.ExcelWriter(Config.FILE_NAME, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='核心战部', index=False)
            wb = writer.book
            ws = writer.sheets['核心战部']
            
            # 样式定义
            f_header = wb.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
            f_red = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'bold': True}) # 极好
            f_cmd = wb.add_format({'bg_color': '#FFFFCC', 'border': 1, 'bold': True}) # 指令
            f_psy = wb.add_format({'italic': True, 'font_color': '#0000FF'}) # 心理画像
            
            # 格式应用
            ws.set_row(0, 20, f_header)
            ws.set_column('B:B', 12) # 名称
            ws.set_column('C:C', 8)  # 分数
            ws.set_column('E:E', 35) # 心理画像
            ws.set_column('G:G', 15) # 竞价指令
            ws.set_column('L:L', 35) # 资讯
            
            # 视觉高亮
            ws.conditional_format('C2:C200', {'type': 'cell', 'criteria': '>=', 'value': 90, 'format': f_red}) # 高分
            ws.conditional_format('E2:E200', {'type': 'text', 'criteria': 'containing', 'value': '破顶', 'format': f_red}) # 破顶
            ws.set_column('G:G', 15, f_cmd) # 指令列
            ws.set_column('E:E', 35, f_psy) # 心理列
            
        print(Fore.GREEN + f"✅ 作战指令已下达！请打开 Excel 查看。")

if __name__ == "__main__":
    start = time.time()
    warlord = DragonWarlord()
    warlord.execute()
    print(f"Total Time: {time.time() - start:.1f}s")
