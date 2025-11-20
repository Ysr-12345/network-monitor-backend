from datetime import datetime
from typing import List, Dict
import smtplib
from email.mime.text import MIMEText
import requests
import json


class AlertService:
    """
    告警服务 - 处理告警生成和推送
    """

    def __init__(self):
        self.alert_history = []

    def check_congestion_alert(self, device_name: str, port_name: str,
                               current_traffic: float, bandwidth: float,
                               forecast_traffic: float = None) -> Dict:
        """检查拥塞告警"""

        utilization = (current_traffic / bandwidth) * 100

        if utilization >= 120:
            level = "critical"
            message = f"设备 {device_name} 端口 {port_name} 流量严重超限！当前利用率: {utilization:.1f}%"
        elif utilization >= 100:
            level = "error"
            message = f"设备 {device_name} 端口 {port_name} 流量超限！当前利用率: {utilization:.1f}%"
        elif utilization >= 80:
            level = "warning"
            message = f"设备 {device_name} 端口 {port_name} 流量接近上限！当前利用率: {utilization:.1f}%"
        else:
            return None

        alert_data = {
            "device_name": device_name,
            "port_name": port_name,
            "alert_type": "congestion",
            "alert_level": level,
            "message": message,
            "timestamp": datetime.utcnow(),
            "current_utilization": utilization,
            "forecast_traffic": forecast_traffic
        }

        # 记录告警历史
        self.alert_history.append(alert_data)

        return alert_data

    def check_anomaly_alert(self, device_name: str, port_name: str,
                            anomaly_score: float) -> Dict:
        """检查异常告警"""

        if anomaly_score > 0.7:
            level = "critical"
            message = f"设备 {device_name} 端口 {port_name} 检测到严重流量异常！异常分数: {anomaly_score:.2f}"
        elif anomaly_score > 0.5:
            level = "error"
            message = f"设备 {device_name} 端口 {port_name} 检测到流量异常！异常分数: {anomaly_score:.2f}"
        elif anomaly_score > 0.3:
            level = "warning"
            message = f"设备 {device_name} 端口 {port_name} 检测到轻微流量异常！异常分数: {anomaly_score:.2f}"
        else:
            return None

        alert_data = {
            "device_name": device_name,
            "port_name": port_name,
            "alert_type": "anomaly",
            "alert_level": level,
            "message": message,
            "timestamp": datetime.utcnow(),
            "anomaly_score": anomaly_score
        }

        # 记录告警历史
        self.alert_history.append(alert_data)

        return alert_data

    def send_dingtalk_alert(self, alert_data: Dict, webhook_url: str = None):
        """发送钉钉告警（模拟）"""
        print(f"📢 发送钉钉告警: {alert_data['message']}")
        # 实际实现会调用钉钉webhook API

    def send_wechat_alert(self, alert_data: Dict):
        """发送微信告警（模拟）"""
        print(f"📱 发送微信告警: {alert_data['message']}")
        # 实际实现会调用企业微信API

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """获取最近告警"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alert_history
                if alert['timestamp'] > cutoff_time]


# 创建全局实例
alert_service = AlertService()