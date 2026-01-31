# -*- coding: utf-8 -*-
"""
A股游资·天眼系统 (Ultimate Full-Armor Stable / 最终全装甲·Playwright版)
版本: v2.2 Dual-Source
优化内容: 
1. 数据源：优先雪球(自动翻页) -> 备用东方财富(Playwright)
2. 宽进严出：数据获取阶段不做严格过滤，保留全市场数据
3. 稳定性：全链路异常熔断
"""

import pandas as pd
import numpy as np
import time
import concurrent.futures
from datetime import datetime, timedelta
from tqdm import tqdm
from colorama import init, Fore, Style, Back
import warnings
import random
import sys
import http.client
import requests
import functools
import json
import re
import akshare as ak  # 仅用于非快照数据的获取(如K线、新闻)

# === 引入 Playwright ===
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print(Fore.RED + "❌ 缺少 playwright 库，请先运行: pip install playwright && playwright install chromium")
    sys.exit(1)

# 初始化
init(autoreset=True)
warnings.filterwarnings('ignore')

# ==========================================
# 0. 全局作战配置 (Battle Configuration)
# ==========================================
class BattleConfig:
    # --- 基础筛选 (Funnel) ---
    MIN_CAP = 15 * 10**8       # 最小流通市值 15亿
    MAX_CAP = 400 * 10**8      # 最大流通市值 400亿 (容纳中军)
    MIN_PRICE = 3.0            # 最低价
    MAX_PRICE = 90.0           # 最高价
    
    # --- 活跃度门槛 ---
    FILTER_PCT_CHG = 2.0       # 涨幅 > 2% (捕捉起爆点，不过滤太多)
    FILTER_TURNOVER = 4.5      # 换手 > 4.5% (游资票必须活跃)
    
    # --- 系统参数 ---
    HISTORY_DAYS = 60          # K线回溯天数
    MAX_WORKERS = 8            # 分析引擎并发线程数
    FILE_NAME = f"Dragon_FullArmor_{datetime.now().strftime('%Y%m%d')}.xlsx"

# ==========================================
# 0.1 核心工具链 (Core Toolchain)
# ==========================================
def retry_robust(max_retries=3, base_delay=1.0, backoff_factor=2.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries:
                        sleep_time = delay * (1 + random.random() * 0.5)
                        time.sleep(sleep_time)
                        delay *= backoff_factor
            return None
        return wrapper
    return decorator

# ==========================================
# 1. 舆情风控哨兵 (News Sentry)
# ==========================================
class NewsSentry:
    NEGATIVE_KEYWORDS = [
        "立案", "调查", "违规", "警示", "减持", "亏损", "大幅下降", 
        "无法表示意见", "ST", "退市", "诉讼", "冻结", "留置", "黑天鹅"
    ]
    _cache = {} 

    @staticmethod
    @retry_robust(max_retries=2, base_delay=0.5)
    def check_news(code):
        if code in NewsSentry._cache:
            return NewsSentry._cache[code]
        try:
            df = ak.stock_news_em(symbol=code)
            if df is None or df.empty:
                return False, "无近期资讯"
            recent_titles = df.head(10)['新闻标题'].astype(str).tolist()
            combined_text = " ".join(recent_titles)
            risk_msgs = []
            for kw in NewsSentry.NEGATIVE_KEYWORDS:
                if kw in combined_text:
                    risk_msgs.append(kw)
            if risk_msgs:
                unique_risks = sorted(list(set(risk_msgs)))
                result = (True, f"⚠️利空含:{','.join(unique_risks)}")
            else:
                result = (False, "舆情平稳")
            NewsSentry._cache[code] = result
            return result
        except:
            return False, "资讯接口跳过"

# ==========================================
# 2. 龙虎榜基因雷达 (Dragon-Tiger Radar)
# ==========================================
class DragonTigerRadar:
    def __init__(self):
        self.lhb_stocks = set()

    def scan(self):
        print(Fore.MAGENTA + ">>> [3/8] 扫描游资龙虎榜基因...")
        try:
            for i in range(3): 
                d = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
                self._fetch_daily_lhb(d)
            print(Fore.GREEN + f"    ✅ 基因库构建完毕，收录 {len(self.lhb_stocks)} 只游资票")
        except Exception as e:
            print(Fore.YELLOW + f"    ⚠️ 龙虎榜接口波动(非致命): {e}")

    @retry_robust(max_retries=2, base_delay=0.5)
    def _fetch_daily_lhb(self, date_str):
        try:
            df = ak.stock_lhb_detail_daily_sina(date=date_str)
            if df is not None and not df.empty:
                codes = df['代码'].astype(str).tolist()
                self.lhb_stocks.update(codes)
        except:
            raise ValueError("LHB fetch failed")

    def has_gene(self, code):
        return code in self.lhb_stocks

# ==========================================
# 3. 热点与龙头锚定雷达 (Hot Concept & Leader)
# ==========================================
class HotConceptRadar:
    def __init__(self):
        self.stock_concept_map = {} 
        self.concept_leader_map = {} 

    def scan(self):
        print(Fore.MAGENTA + ">>> [4/8] 扫描顶级热点 & 锁定板块龙头...")
        try:
            df_board = ak.stock_board_concept_name_em()
            noise = ["昨日", "连板", "首板", "涨停", "融资", "融券", "转债", "ST", "板块", "指数", "深股通", "沪股通"]
            mask = ~df_board['板块名称'].str.contains("|".join(noise))
            df_top = df_board[mask].sort_values(by="涨跌幅", ascending=False).head(10)
            hot_list = df_top['板块名称'].tolist()
            
            print(Fore.MAGENTA + f"    🔥 顶级风口: {hot_list[:6]}...")
            
            print(Fore.CYAN + "    ⚡ 正在精密扫描热点 (已开启限流保护模式)...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                futures = [ex.submit(self._fetch_constituents_safe, t) for t in hot_list]
                for f in concurrent.futures.as_completed(futures):
                    c_name, codes, l_info = f.result()
                    self.concept_leader_map[c_name] = l_info
                    for code in codes:
                        if code not in self.stock_concept_map: 
                            self.stock_concept_map[code] = []
                        self.stock_concept_map[code].append(c_name)
                        
            print(Fore.GREEN + f"    ✅ 龙头锚定完毕 (示例: {list(self.concept_leader_map.items())[0] if self.concept_leader_map else '无'})")
            
        except Exception as e:
            print(Fore.RED + f"    ⚠️ 热点雷达波动: {e}")

    @retry_robust(max_retries=2, base_delay=1.0)
    def _fetch_constituents_safe(self, name):
        try:
            df = ak.stock_board_concept_cons_em(symbol=name)
            if df is not None and not df.empty:
                leader_info = "未知"
                if '涨跌幅' in df.columns:
                    df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
                    df.sort_values(by='涨跌幅', ascending=False, inplace=True)
                    top_stock = df.iloc[0]
                    leader_info = f"{top_stock['名称']}({top_stock['涨跌幅']}%)"
                return name, df['代码'].tolist(), leader_info
            return name, [], "-"
        except Exception:
            raise ValueError("Concept fetch failed")

    def get_info(self, code):
        concepts = self.stock_concept_map.get(code, [])
        if not concepts: return False, "-", "-"
        main_concept = concepts[0]
        leader_info = self.concept_leader_map.get(main_concept, "-")
        return True, main_concept, leader_info

# ==========================================
# 4. 市场哨兵 (Market Sentry)
# ==========================================
class MarketSentry:
    @staticmethod
    @retry_robust(max_retries=2, base_delay=0.5)
    def check_market():
        print(Fore.MAGENTA + ">>> [2/8] 侦测大盘环境...")
        try:
            df = ak.stock_zh_index_daily(symbol="sh000001")
            if df is None or df.empty: raise ValueError("Index data missing")
            today = df.iloc[-1]
            pct = (today['close'] - today['open']) / today['open'] * 100
            if pct < -1.5:
                print(Fore.RED + f"    ⚠️ 警告：大盘暴跌 ({round(pct,2)}%)，已启动【防御模式】(只看硬板)。")
                BattleConfig.FILTER_PCT_CHG = 5.0
            else:
                print(Fore.GREEN + f"    ✅ 大盘环境正常 ({round(pct,2)}%)。")
        except:
            print(Fore.YELLOW + "    ⚠️ 大盘数据获取失败，默认正常模式。")

# ==========================================
# 5. 核心分析引擎 (Identity Engine)
# ==========================================
class IdentityEngine:
    def __init__(self, concept_radar, lhb_radar):
        self.concept_radar = concept_radar
        self.lhb_radar = lhb_radar

    @retry_robust(max_retries=3, base_delay=0.3)
    def get_kline(self, code):
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=BattleConfig.HISTORY_DAYS + 10)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
        if df is not None and not df.empty:
            df.rename(columns={'日期':'date','开盘':'open','收盘':'close','最高':'high',
                               '最低':'low','成交量':'volume','成交额':'amount','涨跌幅':'pct_chg'}, inplace=True)
            return df
        raise ValueError("Empty K-line")

    def calculate_cmf(self, df):
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df['volume']
            range_hl = (high - low)
            range_hl = range_hl.replace(0, 0.01)
            mf_vol = (((close - low) - (high - close)) / range_hl) * volume
            cmf_val = mf_vol.rolling(20).sum() / volume.rolling(20).sum()
            val = cmf_val.iloc[-1]
            return 0.0 if (np.isnan(val) or np.isinf(val)) else val
        except: 
            return 0.0

    def check_overheat(self, df, turnover):
        try:
            close = df['close']; pct_chg = df['pct_chg']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(6).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
            loss = loss.replace(0, 0.01)
            rsi = 100 - (100 / (1 + gain / loss))
            if rsi.iloc[-1] > 90: return True, "RSI超买"
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
        
        is_risk = False
        risk_msg = []
        score = 60
        features = []
        
        if high >= prev['close'] * 1.095 and (high - close) / close > 0.03:
            is_risk = True; risk_msg.append("炸板/烂板")
        ma5 = df['close'].rolling(5).mean().iloc[-1]
        if ma5 > 0 and (close - ma5) / ma5 > 0.18:
            is_risk = True; risk_msg.append("乖离率大")
        vwap = amount / volume if volume > 0 else close
        if close < vwap * 0.985 and pct_chg < 9.8:
            is_risk = True; risk_msg.append("均价压制")
        is_oh, oh_msg = self.check_overheat(df, turnover)
        if is_oh: is_risk = True; risk_msg.append(oh_msg)

        if vol_ratio > 8.0: score += 15; features.append(f"竞价抢筹(量比{vol_ratio})")
        open_pct = (open_p - prev['close']) / prev['close'] * 100
        if prev['pct_chg'] < 3.0 and 2.0 < open_pct < 6.0:
            score += 20; features.append("🔥弱转强")
        limit_ups = len(df[df['pct_chg'] > 9.5].tail(20))
        if limit_ups > 0: score += 10; features.append(f"妖股({limit_ups}板)")
        if self.lhb_radar.has_gene(code): score += 20; features.append("🐉龙虎榜")
        if cmf_val > 0.15: score += 15; features.append("主力锁仓")
        elif cmf_val < -0.1: score -= 15; features.append("资金流出")
        
        is_hot, concept_name, leader_info = self.concept_radar.get_info(code)
        if is_hot:
            score += 25
            if name in leader_info:
                features.append(f"🔥板块龙头:{concept_name}")
                leader_display = "★本机★"
            else:
                features.append(f"热点:{concept_name}")
                leader_display = leader_info
        else:
            leader_display = "-"

        news_msg = "平稳"
        if score > 80 and not is_risk:
            has_bad_news, n_msg = NewsSentry.check_news(code)
            if has_bad_news:
                is_risk = True
                risk_msg.append(n_msg)
                score -= 100
            news_msg = n_msg

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
# 6. 指挥官 (Commander - 重点修改数据获取)
# ==========================================
class Commander:
    def _fetch_xueqiu_playwright(self, page):
        """
        [主源] 雪球：自动翻页 + 宽进严出
        """
        print(Fore.CYAN + "    ⚡ 正在从 [雪球] 拉取数据 (自动翻页中)...")
        data_list = []
        
        try:
            # 1. 获取 Cookie
            page.goto("https://xueqiu.com", timeout=20000, wait_until='domcontentloaded')
            time.sleep(2) 
            
            # 2. 循环翻页
            current_page = 1
            max_page = 60 # 60 * 90 = 5400，足够覆盖全市场
            page_size = 90
            
            pbar = tqdm(total=max_page, desc="    ❄️ 雪球抓取", unit="页", leave=False)
            
            while current_page <= max_page:
                xq_url = f"https://xueqiu.com/service/v5/stock/screener/quote/list?page={current_page}&size={page_size}&order=desc&order_by=percent&exchange=CN&market=CN&type=sha,shb,sza,szb"
                
                try:
                    response = page.goto(xq_url, timeout=8000, wait_until='domcontentloaded')
                    if response.status != 200: break
                    
                    json_data = response.json()
                    if 'data' not in json_data or 'list' not in json_data['data']: break
                    raw_list = json_data['data']['list']
                    if not raw_list: break
                    
                    for item in raw_list:
                        try:
                            # 字段映射
                            # 雪球 symbol: SH600xxx -> 600xxx
                            raw_code = str(item.get('symbol', ''))
                            code = re.sub(r'^[A-Za-z]+', '', raw_code)
                            name = str(item.get('name', ''))
                            
                            # 数值字段 (处理 None)
                            price = float(item.get('current') or 0)
                            turnover = float(item.get('turnover_rate') or 0)
                            volume_ratio = float(item.get('volume_ratio') or 1.0)
                            
                            # 雪球市值单位通常是元
                            total_cap = float(item.get('market_capital') or 0)
                            float_cap = float(item.get('float_market_capital') or 0)
                            
                            # --- 宽进严出逻辑 ---
                            # 只剔除北交所(8/4/92开头)和退市股，保留所有主板/创业/科创以便后续筛选
                            if code.startswith(('8', '4', '92')): continue
                            if '退' in name: continue
                            
                            data_list.append({
                                'code': code, 'name': name, 
                                'close': price, 'pct_chg': float(item.get('percent') or 0),
                                'turnover': turnover, 'circ_mv': float_cap, 
                                '量比': volume_ratio
                            })
                        except: continue
                    
                    current_page += 1
                    pbar.update(1)
                    time.sleep(0.3) # 防封间隔
                except: break
            
            pbar.close()
            print(Fore.GREEN + f"    ✅ 雪球获取结束: 共 {len(data_list)} 条")
            return pd.DataFrame(data_list)
            
        except Exception as e:
            print(Fore.RED + f"    ❌ 雪球获取失败: {e}")
            return pd.DataFrame()

    def _fetch_eastmoney_playwright(self, page):
        """
        [备用] 东方财富：单节点自动翻页
        """
        print(Fore.YELLOW + "    ⚠️ 雪球异常，切换至 [东方财富] 备用源...")
        data_list = []
        # 使用单一稳定节点
        target_domain = "push2.eastmoney.com"
        # 增加字段 f10(量比-待定, 常用f10为量比/金额, 这里用f23 PB占位, 后续算) 和 f21(流通市值)
        base_url = "https://{DOMAIN}/api/qt/clist/get?pn={PAGE}&pz=100&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048&fields=f12,f14,f2,f3,f8,f21"
        
        page_num = 1
        pbar = tqdm(desc="    💰 东财抓取", unit="页", leave=False)
        
        while True:
            url = base_url.format(DOMAIN=target_domain, PAGE=page_num)
            try:
                response = page.goto(url, timeout=8000, wait_until='domcontentloaded')
                if response.status != 200: break
                
                txt = response.text()
                if not txt: break
                js = json.loads(txt)
                
                if 'data' in js and 'diff' in js['data']:
                    rows = js['data']['diff']
                    if not rows: break
                    
                    for item in rows:
                        try:
                            code = str(item.get('f12'))
                            name = str(item.get('f14'))
                            if code.startswith(('8','4','92')) or '退' in name: continue
                            
                            data_list.append({
                                'code': code, 'name': name,
                                'close': float(item.get('f2') or 0),
                                'pct_chg': float(item.get('f3') or 0),
                                'turnover': float(item.get('f8') or 0),
                                'circ_mv': float(item.get('f21') or 0),
                                '量比': 1.0 # 东财列表接口很难直接拿到量比，暂设默认
                            })
                        except: continue
                    
                    page_num += 1
                    pbar.update(1)
                    time.sleep(0.5)
                else: break
            except: break
            
        pbar.close()
        print(Fore.GREEN + f"    ✅ 东方财富获取结束: 共 {len(data_list)} 条")
        return pd.DataFrame(data_list)

    def get_snapshot_robust(self):
        """
        Playwright 驱动的双源获取逻辑
        """
        print(Fore.CYAN + f">>> [1/8] 启动 Playwright 获取全市场快照...")
        
        df_result = pd.DataFrame()
        
        try:
            with sync_playwright() as p:
                # 启动浏览器 (无头模式 + 抗反爬参数)
                browser = p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-blink-features=AutomationControlled']
                )
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                    viewport={'width': 1920, 'height': 1080}
                )
                page = context.new_page()
                
                # 1. 优先雪球
                df_result = self._fetch_xueqiu_playwright(page)
                
                # 2. 失败则切东财
                if df_result.empty:
                    df_result = self._fetch_eastmoney_playwright(page)
                
                browser.close()
                
        except Exception as e:
            print(Fore.RED + f"❌ Playwright 核心进程崩溃: {e}")
            return None

        if df_result.empty:
            print(Fore.RED + "❌ 所有数据源均未返回有效数据！")
            return None
            
        return df_result

    def generate_excel(self, df_res):
        try:
            with pd.ExcelWriter(BattleConfig.FILE_NAME, engine='xlsxwriter') as writer:
                df_res.to_excel(writer, sheet_name='真龙榜', index=False)
                
                manual_data = {
                    '关键列名': ['身份', '板块龙头', '舆情风控', '量比', 'CMF', '特征-弱转强'],
                    '实战含义': [
                        '【真龙T0】: 确定性最高；【陷阱】: 坚决不买。',
                        '锚定效应。如果龙头涨停，你的跟风票才安全。',
                        '一票否决。含“立案、调查”等字眼，回避。',
                        '竞价抢筹指标。> 5.0 表示主力急不可耐。',
                        '主力意图指标。> 0.15 表示主力锁仓。',
                        '最强游资信号。昨日弱势，今日高开爆量。'
                    ]
                }
                pd.DataFrame(manual_data).to_excel(writer, sheet_name='实战说明书', index=False)
                
                wb = writer.book
                ws = writer.sheets['真龙榜']
                fmt_bad = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
                ws.conditional_format('C2:C150', {'type': 'text', 'criteria': 'containing', 'value': '陷阱', 'format': fmt_bad})
                ws.conditional_format('F2:F150', {'type': 'text', 'criteria': 'containing', 'value': '利空', 'format': fmt_bad})
                fmt_good = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
                ws.conditional_format('C2:C150', {'type': 'text', 'criteria': 'containing', 'value': '真龙', 'format': fmt_good})
        except Exception as e:
            print(Fore.RED + f"Excel生成出错: {e}")

    def run(self):
        print(Fore.GREEN + f"=== 🐲 A股游资·天眼系统 (Playwright Core / v2.2) ===")
        print(Fore.YELLOW + f"🕒 当前时间: {datetime.now().strftime('%H:%M:%S')}")

        # STEP 1: 获取快照
        df = self.get_snapshot_robust()
        if df is None: return

        # STEP 2: 战术冷却
        print(Fore.YELLOW + "\n>>> ❄️ 核心数据获取完毕，战术冷却 3 秒...")
        time.sleep(3)

        # STEP 3 & 4: 启动雷达
        MarketSentry.check_market()
        lhb = DragonTigerRadar()
        lhb.scan()
        concept = HotConceptRadar()
        concept.scan()

        # STEP 5: 漏斗筛选 (在这里做严格筛选)
        print(Fore.CYAN + ">>> [5/8] 漏斗筛选 (资金/市值/价格)...")
        # 确保列类型正确
        cols = ['close', 'circ_mv', 'pct_chg', 'turnover', '量比']
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        mask = (
            (df['close'].between(BattleConfig.MIN_PRICE, BattleConfig.MAX_PRICE)) &
            (df['circ_mv'].between(BattleConfig.MIN_CAP, BattleConfig.MAX_CAP)) &
            (df['pct_chg'] >= BattleConfig.FILTER_PCT_CHG) &
            (df['turnover'] >= BattleConfig.FILTER_TURNOVER)
        )
        candidates = df[mask].copy()
        print(Fore.YELLOW + f"    📉 初始池: {len(df)} -> 入围: {len(candidates)} 只")

        if candidates.empty:
            print(Fore.RED + "❌ 没有股票符合筛选条件，流程结束。")
            return

        # STEP 6: 深度运算
        print(Fore.CYAN + ">>> [6/8] 深度运算 (资金+风控+舆情+龙头锚定)...")
        engine = IdentityEngine(concept, lhb)
        results = []
        
        # 优先分析量比高的
        target_rows = candidates.sort_values(by='量比', ascending=False).head(200)
        tasks = [row.to_dict() for _, row in target_rows.iterrows()]
        
        pbar = tqdm(total=len(tasks), desc="    ⚡ 分析进度", unit="股", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=BattleConfig.MAX_WORKERS) as ex:
            futures = {ex.submit(engine.analyze, task): task for task in tasks}
            for f in concurrent.futures.as_completed(futures):
                try:
                    res = f.result(timeout=30)
                    if res: results.append(res)
                except: pass
                finally: pbar.update(1)
        pbar.close()

        # STEP 7: 导出
        print(Fore.CYAN + f">>> [7/8] 生成战报: {BattleConfig.FILE_NAME}")
        if results:
            df_res = pd.DataFrame(results)
            df_res.sort_values(by='总分', ascending=False, inplace=True)
            cols = ['代码','名称','身份','建议','板块龙头','舆情风控','总分','涨幅%','换手%','量比','CMF','特征']
            final_cols = [c for c in cols if c in df_res.columns]
            df_res = df_res[final_cols]
            self.generate_excel(df_res)
            print(Fore.GREEN + f"✅ 成功! 请打开 Excel 查看【真龙榜】")
            print(df_res[['名称','身份','板块龙头','特征']].head(5).to_string(index=False))
        else:
            print(Fore.RED + "❌ 无有效标的。")

if __name__ == "__main__":
    commander = Commander()
    commander.run()
