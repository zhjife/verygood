import akshare as ak
import pandas as pd
import time

print("🔍 开始诊断东方财富快照接口...")
print(f"当前 Akshare 版本: {ak.__version__}")

def diagnose_spot():
    try:
        # 1. 尝试拉取数据
        print(">>> 正在请求 stock_zh_a_spot_em ...")
        start_time = time.time()
        
        # 这里的 adjust 参数有时会影响返回，通常留空
        df = ak.stock_zh_a_spot_em()
        
        end_time = time.time()
        print(f"✅ 请求成功！耗时: {end_time - start_time:.2f} 秒")
        
        # 2. 检查返回数据
        if df is None or df.empty:
            print("❌ 错误：返回了 空 DataFrame。可能是非交易时间或接口维护。")
            return

        # 3. 打印实际列名 (这是最关键的一步)
        print("\n📋 接口实际返回的列名如下 (请对比你的 rename 映射):")
        print(df.columns.tolist())
        
        # 4. 打印前3行数据
        print("\n📊 数据预览:")
        print(df.head(3))
        
        # 5. 模拟你的重命名逻辑进行测试
        expected_cols = ['代码', '名称', '最新价', '涨跌幅', '换手率', '流通市值']
        missing_cols = [c for c in expected_cols if c not in df.columns]
        
        if missing_cols:
            print(f"\n❌ 严重警告：以下关键列在返回数据中找不到: {missing_cols}")
            print("这会导致你的 rename 或调用失败！请根据上方'实际列名'修改代码。")
        else:
            print("\n✅ 关键列检查通过。")

    except Exception as e:
        print(f"\n❌ 接口调用发生异常 (Python 报错):")
        print(f"类型: {type(e)}")
        print(f"详情: {e}")
        
        # 建议
        if "timeout" in str(e).lower():
            print("💡 建议：网络超时。请检查网络稳定性，或稍后重试。")
        elif "json" in str(e).lower():
            print("💡 建议：解析失败。通常是因为 IP 虽然没封，但被重定向到了验证码页面，或者 akshare 版本太旧。")
        elif "connection" in str(e).lower():
            print("💡 建议：连接被拒绝。服务器可能暂时无响应。")

if __name__ == "__main__":
    diagnose_spot()
