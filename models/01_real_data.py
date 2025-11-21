# 01_real_data.py - 真实数据处理版本
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def create_real_traffic_data():
    """创建更真实的网络流量数据"""
    print("生成真实网络流量数据...")
    
    # 30天数据，5分钟间隔
    n_days = 30
    points_per_day = 24 * 12  # 288个点/天
    total_points = n_days * points_per_day
    
    dates = pd.date_range('2024-01-01', periods=total_points, freq='5T')
    np.random.seed(42)
    
    # 更真实的流量模式
    traffic_data = []
    for i, timestamp in enumerate(dates):
        hour = timestamp.hour
        minute = timestamp.minute
        day_of_week = timestamp.dayofweek
        day_of_month = timestamp.day
        
        # 基础流量
        base = 800
        
        # 日周期 - 更真实的模式
        if 0 <= hour < 6:    # 深夜
            daily_effect = -200
        elif 6 <= hour < 9:  # 早高峰
            daily_effect = 150 * (1 + np.sin(2 * np.pi * (hour-6)/3))
        elif 9 <= hour < 18: # 日间平稳
            daily_effect = 100
        elif 18 <= hour < 22: # 晚高峰
            daily_effect = 200 * (1 + np.sin(2 * np.pi * (hour-18)/4))
        else: # 夜间下降
            daily_effect = -100
            
        # 周周期
        if day_of_week >= 5:  # 周末
            weekly_effect = -80
        else:  # 工作日
            weekly_effect = 50
            
        # 特殊事件
        if day_of_month in [1, 15]:  # 月初和月中
            event_effect = 60
        else:
            event_effect = 0
            
        # 随机噪声
        noise = np.random.normal(0, 25)
        
        # 合成流量
        traffic = base + daily_effect + weekly_effect + event_effect + noise
        traffic = max(traffic, 200)  # 最小流量
        
        traffic_data.append(traffic)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'value': traffic_data
    })
    df.set_index('timestamp', inplace=True)
    
    # 添加真实特征
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # 滞后特征
    df['lag_1'] = df['value'].shift(1)
    df['lag_12'] = df['value'].shift(12)
    df['lag_288'] = df['value'].shift(288)
    
    # 滚动特征
    df['rolling_mean_6'] = df['value'].rolling(6).mean()
    df['rolling_std_6'] = df['value'].rolling(6).std()
    
    df = df.dropna()
    
    return df

def create_real_visualizations(df):
    """创建真实的可视化"""
    print("生成真实数据可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('真实网络流量数据分析 - 网汛哨兵', fontsize=16, fontweight='bold')
    
    # 1. 整体趋势
    axes[0,0].plot(df.index, df['value'], linewidth=0.8, alpha=0.7, color='blue')
    axes[0,0].set_title('30天网络流量趋势')
    axes[0,0].set_ylabel('流量 (Mbps)')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 单日详细模式
    one_day = df.iloc[500:788]  # 选取某一天
    axes[0,1].plot(one_day.index, one_day['value'], linewidth=1.5, color='red')
    axes[0,1].set_title('单日流量详细模式')
    axes[0,1].set_ylabel('流量 (Mbps)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. 小时平均流量
    hourly_avg = df.groupby('hour')['value'].mean()
    axes[1,0].plot(hourly_avg.index, hourly_avg.values, 'o-', linewidth=2, 
                   markersize=6, color='green')
    axes[1,0].set_title('各小时平均流量')
    axes[1,0].set_xlabel('小时')
    axes[1,0].set_ylabel('平均流量 (Mbps)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 工作日vs周末
    weekday_mask = df['is_weekend'] == 0
    weekend_mask = df['is_weekend'] == 1
    
    weekday_avg = df[weekday_mask].groupby('hour')['value'].mean()
    weekend_avg = df[weekend_mask].groupby('hour')['value'].mean()
    
    axes[1,1].plot(weekday_avg.index, weekday_avg.values, label='工作日', linewidth=2)
    axes[1,1].plot(weekend_avg.index, weekend_avg.values, label='周末', linewidth=2)
    axes[1,1].set_title('工作日 vs 周末流量对比')
    axes[1,1].set_xlabel('小时')
    axes[1,1].set_ylabel('平均流量 (Mbps)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_traffic_analysis.png', dpi=150, bbox_inches='tight')
    
    return fig

def main():
    print("=== 真实网络流量数据生成 ===")
    
    # 生成真实数据
    df = create_real_traffic_data()
    print(f"✅ 生成真实流量数据: {len(df)} 条记录")
    
    # 数据统计
    print(f"\n📊 数据统计信息:")
    print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"平均流量: {df['value'].mean():.1f} Mbps")
    print(f"流量标准差: {df['value'].std():.1f} Mbps")
    print(f"峰值流量: {df['value'].max():.1f} Mbps")
    print(f"谷值流量: {df['value'].min():.1f} Mbps")
    print(f"工作日平均: {df[df['is_weekend']==0]['value'].mean():.1f} Mbps")
    print(f"周末平均: {df[df['is_weekend']==1]['value'].mean():.1f} Mbps")
    
    # 保存数据
    df.to_csv('real_traffic_data.csv')
    print("✅ 真实数据已保存: real_traffic_data.csv")
    
    # 生成可视化
    create_real_visualizations(df)
    print("✅ 真实数据分析图已保存: real_traffic_analysis.png")
    
    # 显示数据样例
    print(f"\n📋 数据样例 (前5行):")
    print(df.head())
    
    return df

if __name__ == "__main__":
    real_data = main()