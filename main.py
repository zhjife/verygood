# -*- coding: utf-8 -*-
"""
Alpha Galaxy Omni Pro Max - 机构全维量化系统 (v2.8 THS视觉抓取版)
Features: 
1. [Radar Fix] 热点雷达：使用 Playwright 直接抓取同花顺(10jqka)网页，解决无涨跌幅问题。
2. [Data Source] 选股数据：雪球 (Playwright自动翻页) + 宽进严出。
3. [Stability] 全链路浏览器模拟，抗反爬能力最强。
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
import functools
import json
import re
import akshare as ak

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
# 0. 全局作战配置
# ==========================================
class BattleConfig:
    MIN_CAP = 15 * 10**8       
    MAX_CAP = 400 * 10**8      
    MIN_PRICE = 3.0            
    MAX_PRICE = 90.0           
    FILTER_PCT_CHG = 2.0       
    FILTER_TURNOVER = 4.5      
    HISTORY_DAYS = 60          
    MAX_WORKERS = 8            
    FILE_NAME = f"Dragon_FullArmor_{datetime.now().strftime('%Y%m%d')}.xlsx"

# ==========================================
# 0.1 核心工具链
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
# 1. 舆情风控哨兵
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
# 2. 龙虎榜基因雷达
# ==========================================
class DragonTigerRadar:
    def __init__(self):
        self.lhb_stocks = set()

    def scan(self):
        print(Fore.MAGENTA + ">>> [3/8] 扫描游资龙虎榜基因 (东方财富源)...")
        try:
            found_days = 0
            for i in range(5): 
                if found_days >= 3: break 
                d = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
                count = self._fetch_daily_lhb(d)
                if count > 0:
                    found_days += 1
            print(Fore.GREEN + f"    ✅ 基因库构建完毕，收录 {len(self.lhb_stocks)} 只游资票")
        except Exception as e:
            print(Fore.YELLOW + f"    ⚠️ 龙虎榜接口波动: {e}")

    def _fetch_daily_lhb(self, date_str):
        try:
            df = ak.stock_lhb_detail_daily_em(date=date_str)
            if df is not None and not df.empty:
                codes = df['代码'].astype(str).tolist()
                self.lhb_stocks.update(codes)
                return len(codes)
            return 0
        except:
            return 0

    def has_gene(self, code):
        return code in self.lhb_stocks

# ==========================================
# ==========================================
# 3. 热点与龙头锚定雷达 (修复版：智能视觉识别)
# ==========================================
class HotConceptRadar:
    """
    [Fix] 修复同花顺网页抓取为空的问题。
    采用"特征搜索"而非"固定索引"，自动在表格中寻找包含概念链接的单元格。
    """
    def __init__(self):
        self.stock_concept_map = {}   
        self.concept_leader_map = {}  

    def scan(self):
        print(Fore.MAGENTA + ">>> [4/8] 扫描顶级热点 & 锁定板块龙头 (THS智能视觉)...")
        
        try:
            with sync_playwright() as p:
                # 启动浏览器
                browser = p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-blink-features=AutomationControlled']
                )
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                    viewport={'width': 1920, 'height': 1080} # 大分辨率防止布局坍塌
                )
                page = context.new_page()

                # --- 1. 访问同花顺概念板块涨幅榜 ---
                target_url = "http://q.10jqka.com.cn/gn/index/field/199112/order/desc/page/1/"
                page.goto(target_url, timeout=25000, wait_until='domcontentloaded')
                
                # 显式等待表格出现
                try:
                    page.wait_for_selector('.m-table tbody tr', timeout=10000)
                    time.sleep(1.5) # 等待JS渲染文字
                except:
                    print(Fore.YELLOW + "    ⚠️ 网页加载超时，无法定位表格")
                    browser.close()
                    return

                # --- 2. 智能提取热门板块 ---
                rows = page.query_selector_all('.m-table tbody tr')
                top_boards = []
                
                for row in rows:
                    if len(top_boards) >= 6: break 
                    
                    try:
                        # 获取该行所有链接
                        links = row.query_selector_all('a')
                        
                        name = ""
                        href = ""
                        pct_txt = ""
                        
                        # 智能寻找：链接地址包含 'gn/detail' 的就是概念名
                        for link in links:
                            url = link.get_attribute('href')
                            txt = link.inner_text().strip()
                            if url and 'gn/detail' in url and txt:
                                name = txt
                                href = url
                                break
                        
                        # 如果没找到名字，跳过此行
                        if not name: continue

                        # 尝试获取涨跌幅 (通常在最后几列，带颜色)
                        cols = row.query_selector_all('td')
                        for col in cols:
                            txt = col.inner_text().strip()
                            if txt.endswith('%'):
                                pct_txt = txt
                                break
                        
                        # 过滤杂音
                        if any(x in name for x in ["ST", "昨日", "连板", "融资", "新股", "同花顺"]): continue
                        
                        top_boards.append({'name': name, 'url': href, 'pct': pct_txt})
                        
                    except Exception:
                        continue

                # 打印结果进行验证
                names_found = [b['name'] for b in top_boards]
                if not names_found or names_found == ['']:
                    print(Fore.RED + "    ❌ 未能提取到有效板块名，请检查网页结构。")
                else:
                    print(Fore.MAGENTA + f"    🔥 视觉识别风口: {names_found}...")

                # --- 3. 逐个点击进去获取成分股 ---
                pbar = tqdm(top_boards, desc="    ⚡ 视觉锚定", unit="板块")
                
                for board in pbar:
                    b_name = board['name']
                    b_url = board['url']
                    
                    try:
                        page.goto(b_url, timeout=15000, wait_until='domcontentloaded')
                        
                        # 详情页可能有弹窗或加载慢，增加容错
                        try:
                            page.wait_for_selector('.m-table tbody tr', timeout=8000)
                        except:
                            # 详情页加载失败，跳过
                            continue

                        stock_rows = page.query_selector_all('.m-table tbody tr')
                        
                        leader_name = "未知"
                        max_pct = -100.0
                        board_stocks = []
                        
                        for sr in stock_rows:
                            try:
                                # 获取该行所有文本，按列分割
                                # 这种方式比 cols[index] 更抗结构变化
                                row_text = sr.inner_text().split('\t')
                                if len(row_text) < 3: 
                                    # 如果 split 失败，回退到 query_selector
                                    scols = sr.query_selector_all('td')
                                    if len(scols) < 4: continue
                                    s_code = scols[1].inner_text().strip()
                                    s_name = scols[2].inner_text().strip()
                                    s_pct_str = scols[3].inner_text().strip()
                                else:
                                    # 假如 inner_text 是 "1\n600519\n茅台\n..."
                                    lines = sr.inner_text().split('\n')
                                    if len(lines) < 4: continue
                                    s_code = lines[1].strip()
                                    s_name = lines[2].strip()
                                    s_pct_str = lines[3].strip()

                                s_pct_str = s_pct_str.replace('%', '')
                                try:
                                    s_pct = float(s_pct_str)
                                except: s_pct = 0.0
                                
                                board_stocks.append(s_code)
                                
                                if s_pct > max_pct:
                                    max_pct = s_pct
                                    leader_name = s_name
                            except:
                                continue
                        
                        # 只有找到真正的涨跌幅才记录
                        if max_pct > -100:
                            self.concept_leader_map[b_name] = f"{leader_name}({max_pct}%)"
                        
                        for c in board_stocks:
                            if c not in self.stock_concept_map:
                                self.stock_concept_map[c] = []
                            self.stock_concept_map[c].append(b_name)
                            
                        time.sleep(1.0)
                        
                    except Exception:
                        continue
                
                pbar.close()
                browser.close()
                
            if self.stock_concept_map:
                print(Fore.GREEN + f"    ✅ 热点雷达构建完毕 (包含 {len(self.stock_concept_map)} 只关联个股)")
            else:
                print(Fore.YELLOW + "    ⚠️ 热点雷达未匹配到任何个股")

        except Exception as e:
            print(Fore.RED + f"    ❌ 热点雷达运行时异常: {e}")
            pass

    def get_info(self, code):
        concepts = self.stock_concept_map.get(code, [])
        if not concepts: return False, "-", "-"
        main_concept = concepts[0]
        leader_info = self.concept_leader_map.get(main_concept, "-")
        return True, main_concept, leader_info
# 4. 市场哨兵
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
# 5. 核心分析引擎
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
# 6. 指挥官 (Commander)
# ==========================================
class Commander:
    def _fetch_xueqiu_playwright(self, page):
        """[主源] 雪球：自动翻页 + 宽进严出"""
        print(Fore.CYAN + "    ⚡ 正在从 [雪球] 拉取数据 (自动翻页中)...")
        data_list = []
        try:
            page.goto("https://xueqiu.com", timeout=20000, wait_until='domcontentloaded')
            time.sleep(2) 
            current_page = 1
            max_page = 60 
            page_size = 90 
            
            pbar = tqdm(total=max_page, desc="    ❄️ 雪球抓取", unit="页", leave=False)
            
            while current_page <= max_page:
                xq_url = f"https://xueqiu.com/service/v5/stock/screener/quote/list?page={current_page}&size={page_size}&order=desc&order_by=percent&exchange=CN&market=CN&type=sha,shb,sza,szb"
                try:
                    response = page.goto(xq_url, timeout=10000, wait_until='domcontentloaded')
                    if response.status != 200: break
                    json_data = response.json()
                    if 'data' not in json_data or 'list' not in json_data['data']: break
                    raw_list = json_data['data']['list']
                    if not raw_list: break
                    
                    for item in raw_list:
                        try:
                            raw_code = str(item.get('symbol', ''))
                            code = re.sub(r'^[A-Za-z]+', '', raw_code)
                            name = str(item.get('name', ''))
                            price = float(item.get('current') or 0)
                            turnover = float(item.get('turnover_rate') or 0)
                            volume_ratio = float(item.get('volume_ratio') or 1.0)
                            float_cap = float(item.get('float_market_capital') or 0)
                            
                            # 宽进严出：只剔除北交所/退市
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
                    time.sleep(0.3)
                except: break
            
            pbar.close()
            print(Fore.GREEN + f"    ✅ 雪球获取结束: 共 {len(data_list)} 条")
            return pd.DataFrame(data_list)
        except Exception as e:
            print(Fore.RED + f"    ❌ 雪球获取失败: {e}")
            return pd.DataFrame()

    def _fetch_eastmoney_akshare(self):
        """备用源: 东财(Akshare)"""
        print(Fore.YELLOW + "    ⚠️ 雪球异常，切换至 [东方财富] 备用源(Akshare)...")
        try:
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty: return pd.DataFrame()
            
            data_list = []
            numeric_cols = ['最新价', '涨跌幅', '换手率', '流通市值', '量比']
            for c in numeric_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
            for _, row in df.iterrows():
                try:
                    code = str(row['代码'])
                    name = str(row['名称'])
                    if code.startswith(('8','4','92')) or '退' in name: continue
                    
                    data_list.append({
                        'code': code, 'name': name,
                        'close': row['最新价'],
                        'pct_chg': row['涨跌幅'],
                        'turnover': row['换手率'],
                        'circ_mv': row['流通市值'],
                        '量比': row['量比'] if '量比' in row else 1.0
                    })
                except: continue
            print(Fore.GREEN + f"    ✅ 东方财富获取结束: 共 {len(data_list)} 条")
            return pd.DataFrame(data_list)
        except Exception as e:
            print(Fore.RED + f"    ❌ 东方财富获取失败: {e}")
            return pd.DataFrame()

    def get_snapshot_robust(self):
        print(Fore.CYAN + f">>> [1/8] 启动全市场快照获取...")
        df_result = pd.DataFrame()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-blink-features=AutomationControlled']
                )
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                    viewport={'width': 1920, 'height': 1080}
                )
                page = context.new_page()
                df_result = self._fetch_xueqiu_playwright(page)
                browser.close()
        except Exception as e:
            print(Fore.RED + f"❌ Playwright 异常: {e}")

        if df_result.empty:
            df_result = self._fetch_eastmoney_akshare()

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
        print(Fore.GREEN + f"=== 🐲 A股游资·天眼系统 (THS Visual / v2.8) ===")
        print(Fore.YELLOW + f"🕒 当前时间: {datetime.now().strftime('%H:%M:%S')}")

        # STEP 1
        df = self.get_snapshot_robust()
        if df is None: return

        # STEP 2
        print(Fore.YELLOW + "\n>>> ❄️ 核心数据获取完毕，战术冷却 3 秒...")
        time.sleep(3)

        # STEP 3 & 4
        MarketSentry.check_market()
        lhb = DragonTigerRadar()
        lhb.scan()
        concept = HotConceptRadar()
        concept.scan()

        # STEP 5
        print(Fore.CYAN + ">>> [5/8] 漏斗筛选 (资金/市值/价格)...")
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

        # STEP 6
        print(Fore.CYAN + ">>> [6/8] 深度运算 (资金+风控+舆情+龙头锚定)...")
        engine = IdentityEngine(concept, lhb)
        results = []
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

        # STEP 7
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
