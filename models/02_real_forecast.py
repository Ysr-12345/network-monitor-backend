import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def calculate_smape(actual, forecast):
    """计算对称平均绝对百分比误差"""
    return 100/len(actual) * np.sum(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))

def create_real_features(df):
    """创建真实特征工程"""
    print("进行真实特征工程...")
    
    features_df = df.copy()
    
    # 时间周期特征
    features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
    features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
    features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
    features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
    
    # 更多滞后特征
    for lag in [1, 2, 3, 6, 12, 24, 36]:
        features_df[f'lag_{lag}'] = features_df['value'].shift(lag)
    
    # 更多滚动特征
    for window in [6, 12, 24, 36]:
        features_df[f'rolling_mean_{window}'] = features_df['value'].rolling(window).mean()
        features_df[f'rolling_std_{window}'] = features_df['value'].rolling(window).std()
    
    # 差分特征
    features_df['diff_1'] = features_df['value'].diff(1)
    features_df['diff_12'] = features_df['value'].diff(12)
    
    features_df = features_df.dropna()
    
    return features_df

def train_linear_model(X_train, y_train, X_test, y_test):
    """训练线性回归模型"""
    print("训练线性回归模型...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    smape = calculate_smape(y_test, y_pred)
    
    return model, y_pred, {'RMSE': rmse, 'MAE': mae, 'sMAPE': smape}

def train_random_forest(X_train, y_train, X_test, y_test):
    """训练随机森林模型"""
    print("训练随机森林模型...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    smape = calculate_smape(y_test, y_pred)
    
    return model, y_pred, {'RMSE': rmse, 'MAE': mae, 'sMAPE': smape}

def create_real_forecasts(df, features_df):
    """创建真实预测"""
    print("开始真实模型预测...")
    
    # 选择特征列 - 确保不包含目标变量
    feature_columns = [col for col in features_df.columns if col not in ['value', 'timestamp']]
    
    # 准备数据
    X = features_df[feature_columns].values
    y = features_df['value'].values
    
    # 分割数据 - 确保有足够的数据
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 训练多个模型
    models = {}
    predictions = {}
    metrics = {}
    
    # 线性回归
    lr_model, lr_pred, lr_metrics = train_linear_model(X_train, y_train, X_test, y_test)
    models['Linear'] = lr_model
    predictions['Linear'] = lr_pred
    metrics['Linear'] = lr_metrics
    
    # 随机森林
    rf_model, rf_pred, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    models['RandomForest'] = rf_model
    predictions['RandomForest'] = rf_pred
    metrics['RandomForest'] = rf_metrics
    
    # 简单基准模型 (历史平均)
    historical_mean = np.mean(y_train)
    baseline_pred = np.full_like(y_test, historical_mean)
    metrics['Baseline'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, baseline_pred)),
        'MAE': mean_absolute_error(y_test, baseline_pred),
        'sMAPE': calculate_smape(y_test, baseline_pred)
    }
    
    return models, predictions, metrics, y_test, X_test.shape[0]

def plot_real_results(df, features_df, predictions, metrics, y_test, test_size):
    """绘制真实预测结果"""
    print("生成真实预测图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('真实模型预测结果 - 网汛哨兵', fontsize=16, fontweight='bold')
    
    # 1. 预测 vs 实际对比
    # 获取测试集对应的时间戳
    split_idx = int(0.8 * len(features_df))
    test_dates = features_df.index[split_idx:split_idx + len(y_test)]
    
    axes[0,0].plot(test_dates, y_test, label='实际流量', color='blue', linewidth=2)
    colors = ['red', 'green']
    model_names = ['Linear', 'RandomForest']
    
    for i, model_name in enumerate(model_names):
        if model_name in predictions:
            pred = predictions[model_name]
            # 确保预测值和实际值长度一致
            min_len = min(len(y_test), len(pred))
            axes[0,0].plot(test_dates[:min_len], pred[:min_len], 
                          label=f'{model_name}预测', color=colors[i], linewidth=1.5, alpha=0.8)
    
    axes[0,0].set_title('模型预测 vs 实际流量')
    axes[0,0].set_ylabel('流量 (Mbps)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. 误差分布
    errors_data = []
    labels = []
    
    for model_name in model_names:
        if model_name in predictions:
            pred = predictions[model_name]
            min_len = min(len(y_test), len(pred))
            errors = y_test[:min_len] - pred[:min_len]
            errors_data.append(errors)
            labels.append(model_name)
    
    if errors_data:
        axes[0,1].boxplot(errors_data, labels=labels)
        axes[0,1].set_title('预测误差分布')
        axes[0,1].set_ylabel('预测误差 (Mbps)')
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. 性能指标对比
    model_names = list(metrics.keys())
    rmse_values = [metrics[name]['RMSE'] for name in model_names]
    mae_values = [metrics[name]['MAE'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[1,0].bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8, color='lightblue')
    axes[1,0].bar(x + width/2, mae_values, width, label='MAE', alpha=0.8, color='lightcoral')
    axes[1,0].set_title('模型性能指标对比')
    axes[1,0].set_xlabel('模型')
    axes[1,0].set_ylabel('误差值')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(model_names)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. sMAPE对比
    smape_values = [metrics[name]['sMAPE'] for name in model_names]
    bars = axes[1,1].bar(model_names, smape_values, alpha=0.8, color=['red', 'green', 'blue'])
    axes[1,1].set_title('sMAPE误差对比 (%)')
    axes[1,1].set_ylabel('sMAPE (%)')
    axes[1,1].grid(True, alpha=0.3)
    
    # 在柱状图上显示数值
    for bar, value in zip(bars, smape_values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('real_forecast_results.png', dpi=150, bbox_inches='tight')
    
    return fig

def main():
    print("=== 真实模型预测分析 ===")
    
    # 加载数据
    try:
        df = pd.read_csv('real_traffic_data.csv', index_col='timestamp', parse_dates=True)
        print(f"✅ 加载真实数据: {len(df)} 条记录")
    except:
        print("❌ 请先运行 01_real_data.py")
        return
    
    # 特征工程
    features_df = create_real_features(df)
    print(f"✅ 特征工程完成: {features_df.shape[1]} 个特征")
    
    # 模型训练和预测
    models, predictions, metrics, y_test, test_size = create_real_forecasts(df, features_df)
    
    # 显示结果
    print(f"\n📈 模型性能对比:")
    for model_name, model_metrics in metrics.items():
        print(f"{model_name:12} | RMSE: {model_metrics['RMSE']:6.1f} | "
              f"MAE: {model_metrics['MAE']:6.1f} | sMAPE: {model_metrics['sMAPE']:5.1f}%")
    
    # 选择最佳模型
    best_model = min(metrics.items(), key=lambda x: x[1]['sMAPE'])
    print(f"\n🎯 最佳模型: {best_model[0]} (sMAPE: {best_model[1]['sMAPE']:.1f}%)")
    
    # 生成图表
    plot_real_results(df, features_df, predictions, metrics, y_test, test_size)
    print("✅ 真实预测图表已保存: real_forecast_results.png")
    
    # 保存预测结果
    split_idx = int(0.8 * len(features_df))
    test_dates = features_df.index[split_idx:split_idx + len(y_test)]
    
    results_df = pd.DataFrame({
        'timestamp': test_dates,
        'actual': y_test
    })
    
    # 添加各模型预测结果
    for model_name in ['Linear', 'RandomForest']:
        if model_name in predictions:
            pred = predictions[model_name]
            min_len = min(len(y_test), len(pred))
            results_df[f'{model_name.lower()}_pred'] = np.nan
            results_df.iloc[:min_len, results_df.columns.get_loc(f'{model_name.lower()}_pred')] = pred[:min_len]
    
    results_df.to_csv('real_predictions.csv', index=False)
    print("✅ 预测结果已保存: real_predictions.csv")
    
    return models, metrics

if __name__ == "__main__":
    trained_models, model_metrics = main()