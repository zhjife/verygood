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
    # --- 1. 硬性门槛 (本地过滤用) ---
    MIN_CAP = 10 * 10**8       # 10亿
    MAX_CAP = 500 * 10**8      # 500亿 (大票难拉)
    MIN_PRICE = 2.0            
    MAX_PRICE = 120.0          
    
    # [关键] 本地漏斗过滤标准
    # 游资通常只看涨幅 > 3% 且换手活跃的票，这样能把请求数控制在安全范围
    FILTER_PCT_CHG = 3.0       
    FILTER_TURNOVER = 2.5      
    
    HISTORY_DAYS = 400         # 回溯400天看年线和长期结构
    
    # --- 2. 知名席位词库 (Smart Money) ---
    FAMOUS_SEATS = [
        "机构专用", "深股通", "沪股通", 
        "中信证券西安朱雀", "国泰君安上海江苏路", "财通证券杭州上塘路", 
        "华鑫证券上海分公司", "中国银河北京中关村", "东吴证券苏州西北街",
        "国盛证券宁波桑田路", "招商证券交易单元", "东方财富拉萨"
    ]
    
    # --- 3. 运行参数 ---
    MAX_WORKERS = 8            # 适中并发，兼顾速度与防封
    TIMEOUT = 5
    # 建议改为 Report_日期.xlsx
    FILE_NAME = f"Strategy_Report_{datetime.now().strftime('%Y%m%d')}.xlsx"

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ==========================================
# 1. 市场雷达 (基于快照数据)
# ==========================================
class MarketRadar:
    def scan(self, df_snapshot):
        """利用快照数据进行风控"""
        print(Fore.CYAN + ">>> [1/5] 市场温度扫描 (基于快照)...")
        try:
            # 统计数据
            up = len(df_snapshot[df_snapshot['pct_chg'] > 0])
            limit_up = len(df_snapshot[df_snapshot['pct_chg'] >= 9.0])
            limit_down = len(df_snapshot[df_snapshot['pct_chg'] <= -9.0])
            
            sentiment = "🌤️ 震荡轮动"
            is_safe = True
            
            if limit_down > 20 and limit_down > limit_up:
                sentiment = "❄️ 冰点退潮"
                print(Fore.RED + f"    ⚠️ 风险提示：跌停({limit_down}) > 涨停({limit_up})，亏钱效应显著！")
                is_safe = False
            elif limit_up > 60:
                sentiment = "🔥 情绪高潮"
            elif up < 1500:
                sentiment = "☁️ 普跌迷茫"
                
            print(f"    状态: {sentiment} | 涨停: {limit_up} | 跌停: {limit_down} | 上涨: {up}")
            return is_safe
        except:
            return True

# ==========================================
# 2. 情报局 (热点与题材)
# ==========================================
class IntelligenceBureau:
    def __init__(self):
        self.hot_words = []
        self.mainline = []
        
        # 完整的题材映射
        self.theme_map = {
            "低空经济": ["飞行汽车", "eVTOL", "无人机", "通航", "万丰", "宗申", "设计"],
            "AI算力": ["CPO", "光模块", "液冷", "英伟达", "算力", "服务器", "铜连接"],
            "华为产业链": ["鸿蒙", "P70", "华为", "海思", "欧拉", "星闪", "昇腾", "Mate"],
            "固态电池": ["锂电", "固态", "电池", "电解质", "三祥", "清陶", "赣锋"],
            "有色资源": ["黄金", "铜", "铝", "有色", "紫金", "洛阳"],
            "商业航天": ["航天", "卫星", "火箭", "西昌", "星网"],
            "车路云": ["车路云", "自动驾驶", "智慧交通", "路侧", "V2X"],
            "半导体": ["芯片", "光刻机", "存储", "封测", "海光"],
            "并购重组": ["重组", "股权转让", "收购", "壳"]
        }

    def fetch(self):
        print(Fore.CYAN + ">>> [2/5] 获取热点题材...")
        # 百度热搜 (带伪装)
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            resp = requests.get("https://top.baidu.com/board?tab=realtime", headers=headers, timeout=5)
            soup = BeautifulSoup(resp.text, 'html.parser')
            self.hot_words = [x.get_text().strip() for x in soup.find_all('div', class_='c-single-text-ellipsis')[:40]]
            print(Fore.YELLOW + f"    捕获热搜: {len(self.hot_words)} 条")
        except: 
            self.hot_words = ["华为", "低空", "算力", "电池"] # 兜底

        # 资金主线
        try:
            cdf = ak.stock_board_concept_name_em()
            self.mainline = cdf.sort_values(by="涨跌幅", ascending=False).head(15)['板块名称'].tolist()
            print(Fore.YELLOW + f"    主线: {self.mainline[:6]}")
        except: pass

    def match(self, text):
        hits = []
        viral = False
        if not text: return [], False
        
        for t, kws in self.theme_map.items():
            for kw in kws:
                if kw in text:
                    hits.append(t)
                    for buzz in self.hot_words:
                        if kw in buzz or t in buzz: viral = True
                    break
        for m in self.mainline:
            if m in text: hits.append(f"{m}(主线)")
        return list(set(hits)), viral

# ==========================================
# 3. 高级指标与形态引擎 (完全恢复逻辑)
# ==========================================
class IndicatorEngine:
    @staticmethod
    def calculate(df):
        if len(df) < 60: return None
        c, h, l, v = df['close'], df['high'], df['low'], df['volume']
        
        # 均线系统
        ma5=c.rolling(5).mean(); ma10=c.rolling(10).mean(); ma20=c.rolling(20).mean()
        
        # 量比 (近似计算)
        vol_ma5 = v.rolling(5).mean()
        vol_ratio = v / vol_ma5.replace(0, 1)
        
        # MACD
        exp12 = c.ewm(span=12, adjust=False).mean()
        exp26 = c.ewm(span=26, adjust=False).mean()
        dif = exp12 - exp26
        dea = dif.ewm(span=9, adjust=False).mean()
        bar = 2 * (dif - dea)
        
        # RSI
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain/loss))

        return {
            'ma5': ma5, 'ma10': ma10, 'ma20': ma20, # 序列，供形态识别用
            'vol_ratio': vol_ratio.iloc[-1],
            'rsi': rsi.iloc[-1],
            'dif': dif.iloc[-1], 'dea': dea.iloc[-1], 
            'bar': bar.iloc[-1], 'prev_bar': bar.iloc[-2]
        }

class KLineStrictLib:
    @staticmethod
    def detect(df, inds):
        if inds is None: return 0, []
        c, o, v, h, l = df['close'], df['open'], df['volume'], df['high'], df['low']
        ma5, ma10, ma20 = inds['ma5'], inds['ma10'], inds['ma20']
        
        def get(s, i): return s.iloc[i] if len(s) > abs(i) else 0
        
        buy_pats = []
        score = 0
        
        # 1. 旭日东升 (大阳反包)
        body = np.abs(c - o)
        avg_body = body.rolling(10).mean()
        if (get(c,-2)<get(o,-2)) and (get(body,-2)>get(avg_body,-2)*1.2) and (get(o,-1)>get(c,-2)) and (get(c,-1)>get(o,-2)):
            buy_pats.append("旭日东升"); score += 20
            
        # 2. 红三兵 (多头排列)
        if (get(c,-3)>get(o,-3)) and (get(c,-2)>get(o,-2)) and (get(c,-1)>get(o,-1)) and (get(c,-1)>get(c,-2)>get(c,-3)):
            buy_pats.append("红三兵"); score += 15
            
        # 3. 一阳穿三线 (强力突破)
        if (get(c,-1)>max(get(ma5,-1),get(ma10,-1),get(ma20,-1))) and (get(o,-1)<min(get(ma5,-1),get(ma10,-1),get(ma20,-1))):
            buy_pats.append("一阳穿三线"); score += 25
            
        # 4. 倍量过左峰
        past_high = h.iloc[-21:-1].max()
        if (get(v,-1)>get(v,-2)*1.8) and (get(c,-1) >= past_high):
            buy_pats.append("倍量过峰"); score += 20
            
        # 5. 蜻蜓点水 (回踩生命线)
        if (get(l,-1) <= get(ma20,-1)) and (min(get(o,-1), get(c,-1)) > get(ma20,-1)) and (get(c,-1)>get(o,-1)):
            buy_pats.append("蜻蜓点水"); score += 15

        return score, buy_pats

# ==========================================
# 4. 深度分析引擎 (整合所有逻辑)
# ==========================================
class AnalysisEngine:
    def __init__(self, intel):
        self.intel = intel

    def get_kline_safe(self, code):
        """带重试与随机延迟的K线获取"""
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=Config.HISTORY_DAYS)).strftime("%Y%m%d")
        
        for _ in range(3): # 重试3次
            try:
                # 随机延迟防止封IP
                time.sleep(random.uniform(0.2, 0.5))
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
                if df is not None and not df.empty: return df
            except: time.sleep(0.5)
        return None

    def check_smart_money(self, code):
        """恢复龙虎榜查询"""
        try:
            date = datetime.now().strftime("%Y%m%d")
            # 查今日，失败查昨日
            lhb = None
            try: lhb = ak.stock_lhb_detail_daily_sina(date=date, symbol=code)
            except: pass
            
            if lhb is None or lhb.empty:
                date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
                try: lhb = ak.stock_lhb_detail_daily_sina(date=date, symbol=code)
                except: pass
            
            if lhb is None or lhb.empty: return "无榜", 0
            
            buy_seats = str(lhb['买方名称'].tolist())
            tags = []
            score = 5
            
            if "机构专用" in buy_seats: tags.append("🔥机构"); score += 20
            if "深股通" in buy_seats or "沪股通" in buy_seats: tags.append("💰北向"); score += 15
            
            for seat in Config.FAMOUS_SEATS:
                if seat in buy_seats and "机构" not in seat:
                    tags.append("🐉游资"); score += 15; break
            
            return "|".join(tags) if tags else "普通榜", score
        except: return "查询失败", 0

    def profile_psychology(self, row, dist, money_status, is_viral, inds):
        """恢复详细的心理画像"""
        tags = []
        # 空间
        if dist < 0.03: tags.append("🚀破顶博弈")
        elif dist > 0.40: tags.append("🌊深水压力")
        
        # 接力
        if row['pct_chg'] > 9.5:
            if 8 <= row['turnover'] <= 20: tags.append("🤝分歧转一致")
            elif row['turnover'] < 4: tags.append("🔒缩量加速")
            elif row['turnover'] > 25: tags.append("⚡高位大分歧")
            
        # 指标状态
        if inds:
            if inds['dif'] > inds['dea'] and inds['bar'] > inds['prev_bar']: tags.append("📈MACD加速")
            if inds['rsi'] > 80: tags.append("⚠️RSI超买")
            
        # 信仰
        if "机构" in money_status: tags.append("🏦机构背书")
        if is_viral: tags.append("🔥全网共识")
        
        return " | ".join(tags) if tags else "😐观察"

    def analyze(self, row):
        code, name = row['code'], row['name']
        score = 60
        reasons = []
        
        # --- 1. 获取K线 ---
        df = self.get_kline_safe(code)
        
        # 即使K线失败，也尽量保留基本信息，而不是直接丢弃
        k_valid = False
        dist_to_high = 0
        inds = None
        
        if df is not None and len(df) > 30:
            k_valid = True
            df.rename(columns={'日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'volume'}, inplace=True)
            
            # (A) 计算指标与形态
            inds = IndicatorEngine.calculate(df)
            k_score, k_patterns = KLineStrictLib.detect(df, inds)
            score += k_score
            if k_patterns: reasons.append(" | ".join(k_patterns))
            
            # (B) 结构分析
            max_high = df['high'].max()
            current_price = row['close']
            dist_to_high = (max_high - current_price) / current_price
            
            if dist_to_high < 0.03: score += 20; reasons.append("🚀新高")
            elif dist_to_high < 0.15: score += 10; reasons.append("🧗近高")
            elif dist_to_high > 0.40: score -= 20; reasons.append("🌊深水")
            
            # (C) 量价共振
            if inds and inds['dif'] > inds['dea'] and inds['vol_ratio'] > 1.5:
                score += 10; reasons.append("量价共振")
        else:
            reasons.append("⚠️K线缺失")

        # --- 2. 题材与舆情 ---
        try:
            news = ak.stock_news_em(symbol=code)
            if not news.empty:
                full_text = " ".join(news.head(5)['新闻标题'].tolist())
                # 排雷
                if any(x in full_text for x in ["立案", "调查", "退市", "ST"]): return None
                
                tags, is_viral = self.intel.match(full_text)
                if tags:
                    t_str = ",".join(tags)
                    score += 15
                    reasons.append(f"🔥{t_str}" if is_viral else f"题材:{t_str}")
        except: is_viral = False

        # --- 3. 资金查询 (只查高分股) ---
        money_status = "-"
        if score >= 80:
            money_status, m_score = self.check_smart_money(code)
            score += m_score

        # --- 4. 封板属性 ---
        if row['pct_chg'] > 9.0:
            if row['close'] == row['high']: score += 10; reasons.append("硬板")
            else: reasons.append("烂板")

        # --- 5. 心理画像与竞价 ---
        psy_profile = self.profile_psychology(row, dist_to_high, money_status, is_viral, inds)
        
        # 动态竞价计算
        target = row['close'] * 1.02
        action = "确认"
        role = "🐕跟风"
        
        if score >= 90: 
            action = "低吸"; target = row['close'] * 0.98; role = "🐲龙头"
        elif "新高" in reasons: 
            action = "博弈"; target = row['close'] * 1.01; role = "🔥先锋"
        elif "烂板" in reasons: 
            action = "弱转强"; target = row['close'] * 1.03
            
        bid_instruction = f"{action} > {target:.2f}"
        
        # 最终门槛
        if score < 75: return None

        return {
            "代码": code, "名称": name, 
            "总评分": score,
            "角色": role,
            "心理画像": psy_profile,
            "形态/题材": " | ".join(reasons),
            "竞价指令": bid_instruction,
            "现价": row['close'], "涨幅%": row['pct_chg'],
            "换手%": row['turnover'], "市值(亿)": round(row['circ_mv']/10**8, 1),
            "PE": row['pe'],
            "主力": money_status
        }

# ==========================================
# 5. 主流程 (漏斗筛选模式)
# ==========================================
class DragonWarlord:
    def run(self):
        print(Fore.GREEN + "=== 🐉 游资实战系统 (逻辑无损·漏斗加速版) ===")
        
        # Step 1: 快照 (一次请求 5000+)
        print(Fore.CYAN + ">>> [1/4] 获取全市场实时快照...")
        try:
            df = ak.stock_zh_a_spot_em()
            # 立即清洗
            rename = {'代码':'code', '名称':'name', '最新价':'close', '涨跌幅':'pct_chg', 
                      '换手率':'turnover', '总市值':'circ_mv', '最高':'high', '市盈率-动态':'pe'}
            df.rename(columns=rename, inplace=True)
            for c in ['close', 'pct_chg', 'turnover', 'circ_mv', 'pe']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                
            print(f"    成功获取 {len(df)} 只股票。")
        except Exception as e:
            print(Fore.RED + f"❌ 快照失败: {e}"); return

        # Step 2: 风控与情报
        radar = MarketRadar()
        radar.scan(df)
        
        intel = IntelligenceBureau()
        intel.fetch()
        
        # Step 3: 本地漏斗 (关键步骤)
        print(Fore.CYAN + ">>> [2/4] 执行游资审美标准初筛 (本地内存)...")
        # 这里的标准必须足够严，才能保证后续请求 K 线时不崩
        mask = (
            (~df['name'].str.contains('ST|退|C')) &
            (df['close'].between(Config.MIN_PRICE, Config.MAX_PRICE)) &
            (df['circ_mv'].between(Config.MIN_CAP, Config.MAX_CAP)) &
            (df['pe'] > 0) & 
            # 核心过滤：只有涨幅和换手达标的才值得深入分析
            (df['pct_chg'] >= Config.FILTER_PCT_CHG) & 
            (df['turnover'] >= Config.FILTER_TURNOVER)
        )
        candidates = df[mask].copy()
        
        # 如果数量太多，强制取前300强，防止API封号
        if len(candidates) > 300:
            print(Fore.YELLOW + f"    ⚠️ 候选股过多({len(candidates)})，截取前300只强势股。")
            candidates = candidates.sort_values(by='pct_chg', ascending=False).head(300)
            
        print(Fore.YELLOW + f"    📉 最终入围深度分析: {len(candidates)} 只")

        # Step 4: 深度并发 (全逻辑回补)
        print(Fore.CYAN + f">>> [3/4] 启动全量深度分析 (并发: {Config.MAX_WORKERS})...")
        engine = AnalysisEngine(intel)
        tasks = [row for _, row in candidates.iterrows()]
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as ex:
            # tqdm 显示真实进度
            data_iter = tqdm(ex.map(engine.analyze, tasks), total=len(tasks))
            results = [r for r in data_iter if r is not None]
            
        results.sort(key=lambda x: x['总评分'], reverse=True)
        
        # Step 5: 导出
        print(Fore.CYAN + f">>> [4/4] 导出结果: {Config.FILE_NAME}")
        if results:
            df_res = pd.DataFrame(results)
            # 调整列顺序
            cols = ["代码", "名称", "总评分", "角色", "竞价指令", "心理画像", "形态/题材", "主力", "现价", "涨幅%", "换手%", "市值(亿)", "PE"]
            # 防止列缺失报错
            final_cols = [c for c in cols if c in df_res.columns]
            df_res = df_res[final_cols]
            
            df_res.to_excel(Config.FILE_NAME, index=False)
            print(Fore.GREEN + f"✅ 成功生成 {len(results)} 条指令！包含完整逻辑分析。")
        else:
            print(Fore.RED + "❌ 今日无符合条件的股票。")

if __name__ == "__main__":
    start = time.time()
    DragonWarlord().run()
    print(f"Total Time: {time.time() - start:.1f}s")
