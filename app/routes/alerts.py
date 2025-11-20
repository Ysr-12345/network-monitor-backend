from fastapi import APIRouter
from app.services.alert_service import alert_service
from app.models import AlertResponse
from datetime import datetime, timedelta

router = APIRouter(prefix="/alerts", tags=["告警管理"])


@router.get("/", response_model=list[AlertResponse])
async def get_alerts(hours: int = 24, resolved: bool = None):
    """获取告警列表"""
    alerts = alert_service.get_recent_alerts(hours)

    if resolved is not None:
        alerts = [alert for alert in alerts if alert.get('is_resolved', False) == resolved]

    return alerts


@router.post("/test-congestion")
async def test_congestion_alert(device_name: str = "core-switch-1",
                                port_name: str = "GigabitEthernet0/1"):
    """测试拥塞告警"""
    # 模拟高流量触发告警
    alert = alert_service.check_congestion_alert(
        device_name, port_name,
        current_traffic=900000000,  # 900 Mbps
        bandwidth=1000000000  # 1 Gbps
    )

    if alert:
        # 发送告警通知
        alert_service.send_dingtalk_alert(alert)
        alert_service.send_wechat_alert(alert)

        return {
            "success": True,
            "alert": alert,
            "message": "拥塞告警测试成功"
        }
    else:
        return {
            "success": False,
            "message": "未触发告警条件"
        }


@router.post("/test-anomaly")
async def test_anomaly_alert(device_name: str = "core-switch-1",
                             port_name: str = "GigabitEthernet0/1"):
    """测试异常告警"""
    # 模拟高异常分数触发告警
    alert = alert_service.check_anomaly_alert(
        device_name, port_name, anomaly_score=0.85
    )

    if alert:
        alert_service.send_dingtalk_alert(alert)
        alert_service.send_wechat_alert(alert)

        return {
            "success": True,
            "alert": alert,
            "message": "异常告警测试成功"
        }
    else:
        return {
            "success": False,
            "message": "未触发告警条件"
        }