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
import random

# ==========================================
# 0. 全局配置 (System Config)
# ==========================================
init(autoreset=True)
warnings.filterwarnings('ignore')

class Config:
    # --- 1. 基础门槛 (游资审美) ---
    MIN_CAP = 12 * 10**8      # 12亿 (壳资源/微盘股风险大)
    MAX_CAP = 400 * 10**8     # 400亿 (除非是大中军，否则游资拉不动)
    MIN_PRICE = 2.5           # 剔除绝对垃圾股
    MAX_PRICE = 120.0         # 剔除散户接不动的高价股
    
    # --- 2. 交易参数 ---
    # [逻辑保留] 代码B的3%下限更好，能捕捉首板前的潜伏
    TARGET_TURNOVER = (3.0, 25.0) 
    MIN_TURNOVER = 3.0
    LIMIT_THRESHOLD = 9.5         
    HISTORY_DAYS = 400        # [逻辑保留] 代码B的400天看长做短逻辑
    
    # --- 3. 知名席位词库 ---
    FAMOUS_SEATS = [
        "机构专用", "深股通", "沪股通", 
        "中信证券西安朱雀", "国泰君安上海江苏路", "财通证券杭州上塘路", 
        "华鑫证券上海分公司", "中国银河北京中关村", "东吴证券苏州西北街",
        "国盛证券宁波桑田路", "招商证券交易单元", "东方财富拉萨"
    ]
    
    # --- 4. 系统运行参数 ---
    # [重要] 降低并发数，因为我们要计算复杂的K线指标，请求量大，容易被封
    MAX_WORKERS = 8           
    TIMEOUT = 5               
    FILE_NAME = f"实战指令单_{datetime.now().strftime('%Y%m%d')}.xlsx"

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ==========================================
# 1. 大盘风控雷达 (Market Risk Radar)
# ==========================================
class MarketRadar:
    def __init__(self):
        self.sentiment = "中性"
        self.is_safe = True
        
    def scan(self):
        print(Fore.CYAN + ">>> [1/5] 正在测算全市场温度 (风控扫描)...")
        try:
            df = ak.stock_zh_a_spot_em()
            # 兼容性清洗
            df.rename(columns={'涨跌幅': 'pct_chg'}, inplace=True)
            df['pct_chg'] = pd.to_numeric(df['pct_chg'], errors='coerce')
            
            up_count = len(df[df['pct_chg'] > 0])
            limit_down = len(df[df['pct_chg'] <= -9.0])
            limit_up = len(df[df['pct_chg'] >= 9.0])
            
            # 风控模型
            if limit_down > 20 and limit_down > limit_up:
                self.sentiment = "❄️ 冰点退潮 (空仓)"
                self.is_safe = False
                print(Fore.RED + f"    ⚠️ 警告：跌停({limit_down}) > 涨停({limit_up})，触以熔断！")
            elif limit_up > 60:
                self.sentiment = "🔥 情绪高潮 (积极)"
                self.is_safe = True
            elif up_count < 1200:
                self.sentiment = "☁️ 普跌迷茫 (防守)"
                self.is_safe = False
            else:
                self.sentiment = "🌤️ 震荡轮动 (试错)"
                self.is_safe = True
                
            print(f"    市场状态: {self.sentiment} | 涨停: {limit_up} | 跌停: {limit_down} | 上涨: {up_count}")
            return self.is_safe
        except Exception as e:
            print(Fore.YELLOW + f"    风控数据获取异常: {e}，默认放行。")
            return True

# ==========================================
# 2. 情报与题材局 (Intelligence Bureau)
# ==========================================
class IntelligenceBureau:
    def __init__(self):
        self.hot_buzz_words = [] 
        self.market_mainline = []
        
        # 扩展题材库
        self.theme_map = {
            "低空经济": ["飞行汽车", "eVTOL", "无人机", "通航", "低空", "万丰", "宗申"],
            "AI算力": ["CPO", "光模块", "液冷", "英伟达", "算力", "服务器", "铜连接", "中际"],
            "华为产业链": ["鸿蒙", "P70", "华为", "海思", "欧拉", "星闪", "昇腾", "Mate"],
            "固态电池": ["锂电", "固态", "电池", "电解质", "三祥", "清陶"],
            "有色资源": ["黄金", "铜", "铝", "有色", "紫金", "洛阳"],
            "设备更新": ["机床", "机器人", "工业母机", "农机", "电梯"],
            "商业航天": ["航天", "卫星", "火箭", "西昌"],
            "车路云": ["车路云", "自动驾驶", "智慧交通", "路侧"],
            "金融科技": ["互联网金融", "信创", "数字货币"]
        }

    def fetch_intelligence(self):
        print(Fore.CYAN + ">>> [2/5] 扫描全网热搜与主线题材...")
        
        # [优化保留] 必须加Headers，否则百度会返回403
        try:
            url = "https://top.baidu.com/board?tab=realtime"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
            }
            resp = requests.get(url, headers=headers, timeout=Config.TIMEOUT)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                self.hot_buzz_words = [item.get_text().strip() for item in soup.find_all('div', class_='c-single-text-ellipsis')[:40]]
                print(Fore.YELLOW + f"    全网热搜捕获: {len(self.hot_buzz_words)} 条")
            else:
                print(Fore.RED + "    百度热搜拒绝访问，使用本地兜底词库。")
                self.hot_buzz_words = ["华为", "算力", "低空", "无人驾驶"]
        except Exception as e: 
            print(Fore.RED + f"    热搜获取失败: {e}")

        try:
            concept_df = ak.stock_board_concept_name_em()
            self.market_mainline = concept_df.sort_values(by="涨跌幅", ascending=False).head(15)['板块名称'].tolist()
            print(Fore.YELLOW + f"    今日资金主攻: {self.market_mainline[:6]}")
        except: pass

    def analyze_text_for_themes(self, text):
        hits = []
        is_viral = False
        for theme, keywords in self.theme_map.items():
            for kw in keywords:
                if kw in text:
                    hits.append(theme)
                    for buzz in self.hot_buzz_words:
                        if kw in buzz or theme in buzz:
                            is_viral = True
                    break
        for main in self.market_mainline:
            if main in text: hits.append(f"{main}(主线)")
        return list(set(hits)), is_viral

# ==========================================
# 3. [核心] K线形态识别引擎 (来自 Code B)
# ==========================================
class KLineStrictLib:
    """
    负责识别具体的K线组合形态，这是判断主力意图的关键。
    """
    @staticmethod
    def detect(df):
        if len(df) < 30: return 0, []
        
        c = df['close']; o = df['open']; h = df['high']; l = df['low']; v = df['volume']
        # 确保有MA数据
        if 'ma5' not in df.columns: return 0, []
        ma5, ma10, ma20 = df['ma5'], df['ma10'], df['ma20']
        
        body = np.abs(c - o)
        avg_body = body.rolling(10).mean()
        
        # 辅助函数：安全获取iloc
        def get(s, i): return s.iloc[i] if len(s) > abs(i) else 0
        
        buy_pats = []
        score = 0
        
        # 1. 旭日东升 (大阴线后接大阳线反包，且高开)
        if (get(c,-2)<get(o,-2)) and (get(body,-2)>get(avg_body,-2)*1.2) and (get(o,-1)>get(c,-2)) and (get(c,-1)>get(o,-2)):
            buy_pats.append("旭日东升"); score += 20
            
        # 2. 红三兵 (连续三根阳线，重心上移)
        if (get(c,-3)>get(o,-3)) and (get(c,-2)>get(o,-2)) and (get(c,-1)>get(o,-1)) and (get(c,-1)>get(c,-2)>get(c,-3)):
            buy_pats.append("红三兵"); score += 15
            
        # 3. 一阳穿三线 (强力突破)
        if (get(c,-1)>max(get(ma5,-1),get(ma10,-1),get(ma20,-1))) and (get(o,-1)<min(get(ma5,-1),get(ma10,-1),get(ma20,-1))):
            buy_pats.append("一阳穿三线"); score += 25
            
        # 4. 倍量过左峰 (有量有价)
        # 寻找过去20天的高点（不含今天）
        past_high = h.iloc[-21:-1].max()
        if (get(v,-1)>get(v,-2)*1.9) and (get(c,-1) >= past_high):
            buy_pats.append("倍量过左峰"); score += 20
            
        # 5. 蜻蜓点水 (回踩生命线)
        if (get(l,-1) <= get(ma20,-1)) and (min(get(o,-1), get(c,-1)) > get(ma20,-1)) and (get(c,-1)>get(o,-1)):
            buy_pats.append("蜻蜓点水"); score += 15

        return score, buy_pats

# ==========================================
# 4. [核心] 高级指标计算引擎 (来自 Code B)
# ==========================================
class IndicatorEngine:
    """
    负责计算 MACD, KDJ, RSI, 量比等技术指标。
    """
    @staticmethod
    def calculate(df):
        if len(df) < 60: return None
        c = df['close']; h = df['high']; l = df['low']; v = df['volume']
        
        # 均线
        ma5=c.rolling(5).mean(); ma10=c.rolling(10).mean(); ma20=c.rolling(20).mean()
        df['ma5'], df['ma10'], df['ma20'] = ma5, ma10, ma20
        
        # 量比 (简化版：今日量/5日均量)
        vol_ma5 = v.rolling(5).mean()
        vol_ratio = v / vol_ma5.replace(0, 1) # 避免除零
        
        # MACD
        exp12 = c.ewm(span=12, adjust=False).mean()
        exp26 = c.ewm(span=26, adjust=False).mean()
        dif = exp12 - exp26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_bar = 2 * (dif - dea)
        
        # KDJ
        low_min = l.rolling(9).min()
        high_max = h.rolling(9).max()
        rsv = (c - low_min) / (high_max - low_min) * 100
        K = rsv.ewm(com=2, adjust=False).mean()
        D = K.ewm(com=2, adjust=False).mean()
        
        # RSI
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain/loss))

        # 返回最新一天的指标
        return {
            'ma20': ma20.iloc[-1],
            'vol_ratio': vol_ratio.iloc[-1],
            'rsi': rsi.iloc[-1],
            # MACD数据
            'dif': dif.iloc[-1], 'dea': dea.iloc[-1], 
            'bar': macd_bar.iloc[-1], 'prev_bar': macd_bar.iloc[-2],
            # KDJ数据
            'k': K.iloc[-1], 'd': D.iloc[-1]
        }

# ==========================================
# 5. 深度分析引擎 (Integration)
# ==========================================
class AnalysisEngine:
    def __init__(self, intel):
        self.intel = intel

    def check_pressure_and_structure(self, code, current_price):
        """
        融合逻辑：同时计算筹码结构、指标、形态
        """
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=Config.HISTORY_DAYS)).strftime("%Y%m%d")
            
            # [重要] 增加重试机制，防止akshare超时
            df = None
            for _ in range(2):
                try:
                    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                    if df is not None and not df.empty: break
                except: 
                    time.sleep(0.5)
            
            if df is None or len(df) < 60: return "数据不足", 0, 0, None, None
            
            # 统一列名
            df.rename(columns={'日期':'date', '开盘':'open', '收盘':'close', '最高':'high', '最低':'low', '成交量':'volume'}, inplace=True)
            
            # 1. 计算高级指标 (Code B)
            indicators = IndicatorEngine.calculate(df)
            if not indicators: return "指标计算失败", 0, 0, None, None
            
            # 2. 识别K线形态 (Code B)
            k_score, k_patterns = KLineStrictLib.detect(df)
            
            # 3. 结构判定 (Code A)
            max_high = df['high'].max()
            dist_to_high = (max_high - current_price) / current_price
            
            struct_status = "⚖️震荡"
            struct_score = 0
            
            # 只有在震荡或突破时才适合介入，深水套牢股不碰
            if dist_to_high < 0.03: 
                struct_status = "🌌突破新高"
                struct_score = 25
            elif dist_to_high < 0.15: 
                struct_status = "🧗接近前高"
                struct_score = 10
            elif dist_to_high > 0.40: 
                struct_status = f"🌊深水套牢({dist_to_high:.0%})"
                struct_score = -20
            
            # 合并分数
            struct_score += k_score
            if k_patterns: struct_status += f" | {' '.join(k_patterns)}"
            
            return struct_status, struct_score, dist_to_high, indicators, k_patterns
        except Exception as e:
            return f"分析异常", 0, 0, None, None

    def check_smart_money(self, code):
        """
        [龙虎榜分析] 融合重试逻辑
        """
        try:
            target_date = datetime.now().strftime("%Y%m%d")
            lhb = None
            
            # 尝试今日
            try: lhb = ak.stock_lhb_detail_daily_sina(date=target_date, symbol=code)
            except: pass
            
            # 若无，尝试昨日
            if lhb is None or lhb.empty:
                target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
                try: lhb = ak.stock_lhb_detail_daily_sina(date=target_date, symbol=code)
                except: pass
            
            if lhb is None or lhb.empty: return "无榜/数据未更新", 0
            
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

    def profile_psychology(self, row, dist_to_high, money_status, is_viral, is_high_risk, indicators):
        """
        [心理画像引擎] 全维度融合
        """
        psy_tags = []
        if is_high_risk: return "⚠️雷区(主力出逃)"
        
        # 1. 空间心理
        if dist_to_high < 0.03: psy_tags.append("🚀破顶博弈")
        elif dist_to_high > 0.3: psy_tags.append("😰深水压力")
        
        # 2. 接力心理
        if row['pct_chg'] > 9.5:
            if 8 <= row['turnover'] <= 20: psy_tags.append("🤝分歧转一致")
            elif row['turnover'] < 4: psy_tags.append("🔒缩量加速")
            elif row['turnover'] > 25: psy_tags.append("⚡高位大分歧")
        
        # 3. 指标状态 (MACD)
        if indicators:
            if indicators['dif'] > indicators['dea'] and indicators['bar'] > indicators['prev_bar']:
                psy_tags.append("📈MACD加速")
            elif indicators['rsi'] > 80:
                psy_tags.append("⚠️RSI超买")
                
        # 4. 信仰心理
        if "机构" in money_status: psy_tags.append("🏦机构背书")
        elif "游资" in money_status: psy_tags.append("🗡️游资合力")
        
        # 5. 舆情
        if is_viral: psy_tags.append("🔥全网共识")
        
        if not psy_tags: psy_tags.append("😐观察")
        return " | ".join(psy_tags)

    def analyze_one_stock(self, row):
        # [关键] 随机延迟，防止IP被封，这是保证程序能跑完几百只股票的关键
        time.sleep(random.uniform(0.2, 0.6))
        
        code, name = row['code'], row['name']
        score = 60
        reasons = []
        
        try:
            # 1. 基础过滤 (PE < 0 剔除亏损股，来自 Code B)
            if row['pe'] < 0: return None 
            
            # 2. 新闻与题材 (Code A)
            news_df = pd.DataFrame()
            try: news_df = ak.stock_news_em(symbol=code)
            except: pass
            
            latest_news = ""
            is_viral = False
            
            if not news_df.empty:
                full_text = " ".join(news_df.head(6)['新闻标题'].tolist())
                latest_news = news_df.iloc[0]['新闻标题']
                # 排雷
                if any(w in full_text for w in ["立案", "调查", "警示", "违规", "退市", "ST"]):
                    return None 
                
                themes, is_viral = self.intel.analyze_text_for_themes(full_text)
                if themes:
                    t_str = ",".join(themes)
                    score += 20
                    reasons.append(f"🔥破圈:{t_str}" if is_viral else f"题材:{t_str}")

            # 3. 深度技术分析 (Code B的核心)
            struct_status, struct_score, dist_val, indicators, k_patterns = self.check_pressure_and_structure(code, row['close'])
            score += struct_score
            reasons.append(struct_status)
            
            # 4. 资金痕迹 (龙虎榜)
            money_status, money_score = self.check_smart_money(code)
            score += money_score
            
            # 5. 心理画像
            psy_profile = self.profile_psychology(row, dist_val, money_status, is_viral, False, indicators)
            
            # 6. 量价共振加分 (Code B)
            if row['pct_chg'] > Config.LIMIT_THRESHOLD:
                score += 15
                if row['close'] == row['high']: reasons.append("封板")
                else: score -= 5; reasons.append("烂板")
            
            if indicators and indicators['dif'] > indicators['dea'] and indicators['vol_ratio'] > 1.5:
                score += 10
                reasons.append("量价共振")

            # 7. 黄金换手
            if 5 <= row['turnover'] <= 15:
                score += 10; reasons.append("黄金换手")
            
            # --- 最终判定 ---
            if score < 75: return None
            
            pos_pct = "40% (重仓)" if score >= 90 else ("20% (中仓)" if score >= 85 else "10% (轻仓)")
            target_price = row['close'] * 1.02
            role_tag = "🐲核心龙" if score >= 90 else "🐕跟风"
            
            return {
                "代码": code, "名称": name, 
                "总评分": score,
                "角色定位": role_tag,
                "心理画像": psy_profile,
                "建议仓位": pos_pct,
                "竞价开枪价": f"> {target_price:.2f}",
                "现价": row['close'], "涨幅%": row['pct_chg'], "换手%": row['turnover'],
                "市值(亿)": round(row['circ_mv']/10**8, 2),
                "市盈率": row['pe'],
                "主力痕迹": money_status,
                "最新资讯": latest_news
            }
        except Exception as e:
            return None

# ==========================================
# 6. 指挥官系统
# ==========================================
class DragonWarlord:
    def __init__(self):
        self.radar = MarketRadar()
        self.intel = IntelligenceBureau()
        self.engine = AnalysisEngine(self.intel)

    def execute(self):
        print(Fore.GREEN + "=== 🐉 游资实战终极融合版 (DragonWarlord Ultimate) ===")
        
        # 1. 风控
        if not self.radar.scan(): return

        # 2. 情报
        self.intel.fetch_intelligence()
        
        # 3. 市场初筛
        print(Fore.CYAN + ">>> [3/5] 拉取全市场数据...")
        try:
            df = ak.stock_zh_a_spot_em()
            
            # 清洗与格式化
            cols_map = {'代码': 'code', '名称': 'name', '最新价': 'close', '涨跌幅': 'pct_chg', 
                        '换手率': 'turnover', '总市值': 'circ_mv', '最高': 'high', '市盈率-动态': 'pe'}
            df.rename(columns=cols_map, inplace=True)
            
            numeric_cols = ['close', 'pct_chg', 'turnover', 'circ_mv', 'high', 'pe']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # [筛选策略融合]
            # 1. 剔除ST、退市
            # 2. 价格、市值门槛 (代码A)
            # 3. PE > 0 (代码B，剔除垃圾股)
            # 4. 放开300/688限制 (代码A，机会更多)
            mask = (
                (~df['name'].str.contains('ST|退|C')) &
                (df['close'].between(Config.MIN_PRICE, Config.MAX_PRICE)) &
                (df['circ_mv'].between(Config.MIN_CAP, Config.MAX_CAP)) &
                (df['turnover'] > Config.MIN_TURNOVER) & 
                (df['pct_chg'] > 4.0) & # 保持强势股筛选
                (df['pe'] > 0) # 业绩避雷
            )
            candidates = df[mask]
            print(f"    初筛入围: {len(candidates)} 只 (强势 + 业绩正 + 流动性充足)")
            
        except Exception as e:
            print(Fore.RED + f"数据拉取失败: {e}")
            return

        # 4. 深度分析
        print(Fore.CYAN + f">>> [4/5] 启动深度政审 (并发数: {Config.MAX_WORKERS})...")
        results = []
        tasks = [row for _, row in candidates.iterrows()]
        
        # 使用tqdm进度条
        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            data_iter = tqdm(executor.map(self.engine.analyze_one_stock, tasks), total=len(tasks))
            results = [x for x in data_iter if x is not None]
        
        results.sort(key=lambda x: x['总评分'], reverse=True)
        
        # 5. 导出
        self.export(results)

    def export(self, data):
        print(Fore.CYAN + f">>> [5/5] 生成作战指令: {Config.FILE_NAME}")
        
        if not data:
            print(Fore.YELLOW + "    [提示] 严选结果为空，生成空表。")
            data = [{"代码": "000000", "名称": "空仓", "总评分": 0, "心理画像": "全市场无符合标的"}]
            
        df = pd.DataFrame(data)
        
        try:
            with pd.ExcelWriter(Config.FILE_NAME, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='核心战部', index=False)
                wb = writer.book
                ws = writer.sheets['核心战部']
                
                f_header = wb.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
                f_red = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'bold': True})
                f_cmd = wb.add_format({'bg_color': '#FFFFCC', 'border': 1, 'bold': True})
                f_info = wb.add_format({'text_wrap': True})
                
                ws.set_row(0, 20, f_header)
                ws.set_column('B:B', 12) # 名称
                ws.set_column('C:C', 6)  # 评分
                ws.set_column('E:E', 35) # 画像
                ws.set_column('G:G', 15) # 指令
                ws.set_column('L:L', 35, f_info) # 资讯
                
                if len(data) > 0 and data[0]['代码'] != "000000":
                    ws.conditional_format('C2:C200', {'type': 'cell', 'criteria': '>=', 'value': 90, 'format': f_red})
                    ws.set_column('G:G', 15, f_cmd)
                
            print(Fore.GREEN + f"✅ 任务完成！文件已生成。")
        except Exception as e:
            print(Fore.RED + f"Excel 生成失败: {e}")

if __name__ == "__main__":
    start = time.time()
    DragonWarlord().execute()
    print(f"Total Time: {time.time() - start:.1f}s")
