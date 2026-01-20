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
    # --- 1. 基础门槛 (保持宽泛，确保能扫到更多票) ---
    MIN_CAP = 10 * 10**8      # 10亿起，不做太严苛限制
    MAX_CAP = 500 * 10**8     
    MIN_PRICE = 2.0           
    MAX_PRICE = 150.0         
    
    # --- 2. 交易参数 ---
    # [优化] 换手率放宽至 2.0%，模仿代码B的宽口径
    TARGET_TURNOVER = (2.0, 35.0) 
    MIN_TURNOVER = 2.0
    LIMIT_THRESHOLD = 9.5         
    HISTORY_DAYS = 400        # 代码B的长周期回溯
    
    # --- 3. 知名席位词库 ---
    FAMOUS_SEATS = [
        "机构专用", "深股通", "沪股通", 
        "中信证券西安朱雀", "国泰君安上海江苏路", "财通证券杭州上塘路", 
        "华鑫证券上海分公司", "中国银河北京中关村", "东吴证券苏州西北街",
        "国盛证券宁波桑田路", "招商证券交易单元", "东方财富拉萨"
    ]
    
    # --- 4. 系统运行参数 ---
    # [恢复代码B的高并发]
    MAX_WORKERS = 16          
    TIMEOUT = 8  # 稍微延长超时时间适应高并发             
    FILE_NAME = f"实战指令单_{datetime.now().strftime('%Y%m%d')}.xlsx"

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ==========================================
# 1. 大盘风控雷达
# ==========================================
class MarketRadar:
    def __init__(self):
        self.sentiment = "中性"
        self.is_safe = True
        
    def scan(self):
        print(Fore.CYAN + ">>> [1/5] 正在测算全市场温度 (风控扫描)...")
        try:
            df = ak.stock_zh_a_spot_em()
            # 兼容性重命名
            rename_map = {'涨跌幅': 'pct_chg', '最新价': 'close'}
            df.rename(columns=rename_map, inplace=True)
            df['pct_chg'] = pd.to_numeric(df['pct_chg'], errors='coerce')
            
            up_count = len(df[df['pct_chg'] > 0])
            limit_down = len(df[df['pct_chg'] <= -9.0])
            limit_up = len(df[df['pct_chg'] >= 9.0])
            
            # 仅做提示，不熔断，保证数据产出
            if limit_down > 20 and limit_down > limit_up:
                self.sentiment = "❄️ 冰点退潮"
                print(Fore.RED + f"    ⚠️ 风险提示：跌停({limit_down}) > 涨停({limit_up})，请谨慎出手。")
            elif limit_up > 60:
                self.sentiment = "🔥 情绪高潮"
            else:
                self.sentiment = "🌤️ 震荡轮动"
                
            print(f"    市场状态: {self.sentiment} | 涨停: {limit_up} | 跌停: {limit_down} | 上涨: {up_count}")
            return True
        except Exception as e:
            print(Fore.YELLOW + f"    风控接口异常: {e}，默认放行。")
            return True

# ==========================================
# 2. 情报与题材局
# ==========================================
class IntelligenceBureau:
    def __init__(self):
        self.hot_buzz_words = [] 
        self.market_mainline = []
        
        self.theme_map = {
            "低空经济": ["飞行汽车", "eVTOL", "无人机", "通航", "万丰", "宗申"],
            "AI算力": ["CPO", "光模块", "液冷", "英伟达", "算力", "服务器", "铜连接"],
            "华为产业链": ["鸿蒙", "P70", "华为", "海思", "欧拉", "星闪", "昇腾", "Mate"],
            "固态电池": ["锂电", "固态", "电池", "电解质", "三祥", "清陶"],
            "有色资源": ["黄金", "铜", "铝", "有色", "紫金", "洛阳"],
            "商业航天": ["航天", "卫星", "火箭", "西昌", "星网"],
            "车路云": ["车路云", "自动驾驶", "智慧交通", "路侧", "V2X"],
            "半导体": ["芯片", "光刻机", "存储", "封测"],
            "并购重组": ["重组", "股权转让", "收购"]
        }

    def fetch_intelligence(self):
        print(Fore.CYAN + ">>> [2/5] 扫描全网热搜与主线题材...")
        
        # 1. 百度热搜 (带Headers伪装)
        try:
            url = "https://top.baidu.com/board?tab=realtime"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                self.hot_buzz_words = [item.get_text().strip() for item in soup.find_all('div', class_='c-single-text-ellipsis')[:40]]
                print(Fore.YELLOW + f"    全网热搜: {len(self.hot_buzz_words)} 条")
        except: 
            self.hot_buzz_words = ["华为", "算力", "低空", "电池"] # 兜底

        # 2. 资金主线
        try:
            concept_df = ak.stock_board_concept_name_em()
            self.market_mainline = concept_df.sort_values(by="涨跌幅", ascending=False).head(15)['板块名称'].tolist()
            print(Fore.YELLOW + f"    资金主攻: {self.market_mainline[:6]}")
        except: pass

    def analyze_text_for_themes(self, text):
        hits = []
        is_viral = False
        for theme, keywords in self.theme_map.items():
            for kw in keywords:
                if kw in text:
                    hits.append(theme)
                    for buzz in self.hot_buzz_words:
                        if kw in buzz or theme in buzz: is_viral = True
                    break
        for main in self.market_mainline:
            if main in text: hits.append(f"{main}(主线)")
        return list(set(hits)), is_viral

# ==========================================
# 3. K线与指标引擎 (保留Alpha Galaxy逻辑)
# ==========================================
class IndicatorEngine:
    @staticmethod
    def calculate(df):
        if len(df) < 60: return None
        c, h, l, v = df['close'], df['high'], df['low'], df['volume']
        
        ma5=c.rolling(5).mean(); ma10=c.rolling(10).mean(); ma20=c.rolling(20).mean()
        
        # 量比
        vol_ma5 = v.rolling(5).mean()
        vol_ratio = v / vol_ma5.replace(0, 1)
        
        # MACD
        exp12 = c.ewm(span=12, adjust=False).mean()
        exp26 = c.ewm(span=26, adjust=False).mean()
        dif = exp12 - exp26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_bar = 2 * (dif - dea)
        
        # RSI
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain/loss))

        return {
            'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
            'vol_ratio': vol_ratio.iloc[-1],
            'rsi': rsi.iloc[-1],
            'dif': dif.iloc[-1], 'dea': dea.iloc[-1], 
            'bar': macd_bar.iloc[-1], 'prev_bar': macd_bar.iloc[-2]
        }

class KLineStrictLib:
    @staticmethod
    def detect(df, inds):
        if inds is None: return 0, []
        c, o, v = df['close'], df['open'], df['volume']
        ma5, ma20 = inds['ma5'], inds['ma20']
        
        def get(s, i): return s.iloc[i] if len(s) > abs(i) else 0
        
        buy_pats = []
        score = 0
        
        # 旭日东升
        body = np.abs(c - o)
        avg_body = body.rolling(10).mean()
        if (get(c,-2)<get(o,-2)) and (get(body,-2)>get(avg_body,-2)*1.2) and (get(o,-1)>get(c,-2)) and (get(c,-1)>get(o,-2)):
            buy_pats.append("旭日东升"); score += 20
        
        # 红三兵
        if (get(c,-3)>get(o,-3)) and (get(c,-2)>get(o,-2)) and (get(c,-1)>get(o,-1)) and (get(c,-1)>get(c,-2)>get(c,-3)):
            buy_pats.append("红三兵"); score += 15

        # 一阳穿三线
        if (get(c,-1)>max(get(ma5,-1),get(ma20,-1))) and (get(o,-1)<min(get(ma5,-1),get(ma20,-1))):
            buy_pats.append("一阳穿三线"); score += 25
            
        return score, buy_pats

# ==========================================
# 4. 深度分析引擎 (核心)
# ==========================================
class AnalysisEngine:
    def __init__(self, intel):
        self.intel = intel

    def analyze_one_stock(self, row):
        # [优化] 使用更短的随机延迟，因为我们要处理更多数据
        # 依靠重试机制来保证数据获取，而不是单纯的等待
        time.sleep(random.uniform(0.1, 0.3))
        
        code, name = row['code'], row['name']
        score = 60
        reasons = []
        
        try:
            # 1. PE 过滤 (来自代码B)
            if row['pe'] < 0: return None
            
            # 2. 获取K线 (关键：增强重试机制)
            df = None
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=Config.HISTORY_DAYS)).strftime("%Y%m%d")
            
            for _ in range(3): # 失败重试3次
                try:
                    # 使用qfq (前复权)
                    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                    if df is not None and not df.empty: break
                except: 
                    time.sleep(0.5) # 失败后稍微休息
            
            if df is None or len(df) < 60: return None 
            
            # 统一列名
            df.rename(columns={'日期':'date', '开盘':'open', '收盘':'close', '最高':'high', '最低':'low', '成交量':'volume'}, inplace=True)
            
            # 3. 计算指标 & 形态
            inds = IndicatorEngine.calculate(df)
            k_score, k_patterns = KLineStrictLib.detect(df, inds)
            score += k_score
            if k_patterns: reasons.append(" | ".join(k_patterns))
            
            # 4. 结构与压力
            current_price = row['close']
            max_high = df['high'].max()
            dist_to_high = (max_high - current_price) / current_price
            
            if dist_to_high < 0.03: score += 25; reasons.append("🚀突破新高")
            elif dist_to_high < 0.15: score += 10; reasons.append("🧗接近前高")
            elif dist_to_high > 0.40: score -= 20; reasons.append("🌊深水套牢")
            
            # 5. 题材挖掘
            news_df = pd.DataFrame()
            try: news_df = ak.stock_news_em(symbol=code)
            except: pass
            
            latest_news = "无"
            if not news_df.empty:
                full_text = " ".join(news_df.head(5)['新闻标题'].tolist())
                latest_news = news_df.iloc[0]['新闻标题']
                # 排雷
                if any(x in full_text for x in ["立案", "调查", "退市", "ST"]): return None
                
                themes, is_viral = self.intel.analyze_text_for_themes(full_text)
                if themes:
                    t_str = ",".join(themes)
                    score += 15
                    reasons.append(f"🔥{t_str}" if is_viral else f"题材:{t_str}")
            
            # 6. 量价与盘口
            if row['pct_chg'] > Config.LIMIT_THRESHOLD:
                if row['close'] == row['high']: score += 15; reasons.append("硬板")
                else: reasons.append("烂板")
                
            if inds and inds['dif'] > inds['dea'] and inds['vol_ratio'] > 1.5:
                score += 10; reasons.append("量价共振")
                
            # 7. 资金查询 (只查高分股)
            money_status = "-"
            if score >= 85:
                try:
                    lhb = ak.stock_lhb_detail_daily_sina(date=end_date, symbol=code)
                    if lhb is not None and not lhb.empty:
                        txt = str(lhb['买方名称'])
                        if "机构" in txt: score += 15; money_status = "机构"
                        elif "桑田路" in txt or "拉萨" in txt: score += 10; money_status = "游资"
                except: pass

            # 最终门槛
            if score < 75: return None
            
            # =========== 动态竞价策略 (已集成) ============
            target_price = 0
            action = "观察"
            if score >= 90:
                action = "低吸"
                target_price = current_price * 0.98
            elif "突破新高" in reasons:
                action = "博弈"
                target_price = current_price * 1.01
            elif "烂板" in reasons:
                action = "弱转强"
                target_price = current_price * 1.03
            else:
                action = "确认"
                target_price = current_price * 1.02
                
            bid_str = f"{action} > {target_price:.2f}"

            return {
                "代码": code, "名称": name, 
                "总评分": score,
                "角色": "🐲龙头" if score>=90 else "🐕跟风",
                "画像": " | ".join(reasons),
                "竞价指令": bid_str,
                "现价": current_price, "涨幅%": row['pct_chg'],
                "市值": round(row['circ_mv']/10**8, 1),
                "PE": round(row['pe'], 1),
                "主力": money_status,
                "资讯": latest_news
            }

        except Exception as e:
            return None

# ==========================================
# 5. 主程序 (使用代码B的数据获取方式)
# ==========================================
class DragonWarlord:
    def execute(self):
        print(Fore.GREEN + "=== 🐉 游资实战终极版 (Max Data Mode) ===")
        
        radar = MarketRadar()
        radar.scan()
        
        intel = IntelligenceBureau()
        intel.fetch_intelligence()
        
        print(Fore.CYAN + ">>> [3/5] 拉取全市场数据 (Code B Mode)...")
        try:
            # 1. 全量获取 (这里就是代码B获取2000+数据的关键)
            df = ak.stock_zh_a_spot_em()
            
            # 2. 立即重命名与清洗
            rename = {'代码':'code', '名称':'name', '最新价':'close', '涨跌幅':'pct_chg', 
                      '换手率':'turnover', '总市值':'circ_mv', '最高':'high', '市盈率-动态':'pe'}
            df.rename(columns=rename, inplace=True)
            
            for c in ['close', 'pct_chg', 'turnover', 'circ_mv', 'high', 'pe']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            
            # 3. 宽泛过滤 (不使用 head 限制，依靠逻辑过滤)
            # 只要涨幅大于 2% 且 换手大于 2% 的票都纳入分析范围
            # 这样在行情好时可能有 500+ 只，行情差时也有 100+ 只
            mask = (
                (~df['name'].str.contains('ST|退|C')) &
                (df['close'].between(Config.MIN_PRICE, Config.MAX_PRICE)) &
                (df['circ_mv'].between(Config.MIN_CAP, Config.MAX_CAP)) &
                (df['turnover'] > Config.MIN_TURNOVER) & 
                (df['pct_chg'] > 2.0) &  # [关键] 放宽至2%，确保扫描面够广
                (df['pe'] > 0)
            )
            candidates = df[mask]
            
            print(f"    初筛入围: {len(candidates)} 只 (将全部进行深度扫描，请耐心等待...)")
            
        except Exception as e:
            print(Fore.RED + f"数据拉取失败: {e}")
            return

        print(Fore.CYAN + f">>> [4/5] 启动深度并发分析 (并发数: {Config.MAX_WORKERS})...")
        # 实例化引擎
        engine = AnalysisEngine(intel)
        results = []
        tasks = [row for _, row in candidates.iterrows()]
        
        # 这里的 tqdm 会显示真实的进度，如果入围 500 个，就会跑 500 个
        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            data_iter = tqdm(executor.map(engine.analyze_one_stock, tasks), total=len(tasks))
            results = [x for x in data_iter if x is not None]
        
        results.sort(key=lambda x: x['总评分'], reverse=True)
        
        self.export(results)

    def export(self, data):
        print(Fore.CYAN + f">>> [5/5] 导出Excel: {Config.FILE_NAME}")
        if not data:
            print(Fore.YELLOW + "今日无符合严选标准的标的。")
            return

        df = pd.DataFrame(data)
        try:
            with pd.ExcelWriter(Config.FILE_NAME, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='核心战部', index=False)
                wb = writer.book
                ws = writer.sheets['核心战部']
                
                f_red = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'bold': True})
                f_cmd = wb.add_format({'bg_color': '#FFFFCC', 'border': 1, 'bold': True})
                
                ws.set_column('B:B', 12)
                ws.set_column('E:E', 35) # 画像列宽
                ws.set_column('L:L', 30) # 资讯列宽
                ws.conditional_format('C2:C200', {'type': 'cell', 'criteria': '>=', 'value': 90, 'format': f_red})
                ws.set_column('F:F', 18, f_cmd) # 指令列
                
            print(Fore.GREEN + f"✅ 任务完成！已生成 {len(data)} 条实战指令。")
        except Exception as e:
            print(Fore.RED + f"Excel保存失败: {e}")

if __name__ == "__main__":
    start = time.time()
    DragonWarlord().execute()
    print(f"Total Time: {time.time() - start:.1f}s")
