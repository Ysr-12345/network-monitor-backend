import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict


class ForecastService:
    """
    模拟预测服务 - 在实际项目中这里会集成真正的机器学习模型
    """

    def __init__(self):
        self.models = ["ARIMA", "LSTM", "Prophet"]

    def generate_forecast(self, device_name: str, port_name: str, horizon: str = "1h") -> Dict:
        """生成模拟预测数据"""

        # 模拟基础流量值（根据设备类型）
        base_traffic = self._get_base_traffic(device_name, port_name)

        # 根据预测时长设置数据点数量
        if horizon == "1h":
            points = 12  # 5分钟间隔，1小时12个点
        else:  # 24h
            points = 288  # 5分钟间隔，24小时288个点

        forecasts = []
        current_time = datetime.utcnow()

        for i in range(points):
            forecast_time = current_time + timedelta(minutes=5 * (i + 1))

            # 模拟流量值（加入时间周期性和随机性）
            hour = forecast_time.hour
            traffic_value = base_traffic * self._get_time_factor(hour)
            traffic_value *= (1 + 0.1 * np.sin(i * 0.1))  # 加入周期性

            # 添加一些随机波动
            traffic_value *= np.random.uniform(0.95, 1.05)

            # 计算置信区间
            confidence_range = traffic_value * 0.1  # 10% 置信区间

            forecasts.append({
                "timestamp": forecast_time.isoformat(),
                "predicted_value": round(traffic_value, 2),
                "confidence_upper": round(traffic_value + confidence_range, 2),
                "confidence_lower": round(max(0, traffic_value - confidence_range), 2)
            })

        return {
            "device_name": device_name,
            "port_name": port_name,
            "forecasts": forecasts,
            "model_used": np.random.choice(self.models),
            "confidence_interval": 0.95
        }

    def _get_base_traffic(self, device_name: str, port_name: str) -> float:
        """根据设备名获取基础流量值"""
        if "core" in device_name.lower():
            return 8000000000  # 8 Gbps
        elif "access" in device_name.lower():
            return 1000000000  # 1 Gbps
        else:
            return 500000000  # 500 Mbps

    def _get_time_factor(self, hour: int) -> float:
        """根据小时获取时间因子（模拟潮汐效应）"""
        # 模拟网络使用模式：晚上高，凌晨低
        if 8 <= hour <= 11:  # 上午工作时间
            return 0.7
        elif 14 <= hour <= 17:  # 下午工作时间
            return 0.9
        elif 19 <= hour <= 23:  # 晚上高峰
            return 1.2
        elif 0 <= hour <= 6:  # 凌晨低峰
            return 0.3
        else:  # 其他时间
            return 0.5


# 创建全局实例
forecast_service = ForecastService()