from fastapi import APIRouter
import random
from datetime import datetime

router = APIRouter(prefix="/anomaly", tags=["异常检测"])


@router.get("/{device_name}/{port_name}")
async def check_anomaly(device_name: str, port_name: str, last: str = "1h"):
    """
    检查流量异常

    - **device_name**: 设备名称
    - **port_name**: 端口名称
    - **last**: 检查时长 (1h, 6h, 24h)
    """
    # 模拟异常检测结果
    anomaly_score = round(random.uniform(0, 1), 2)

    # 模拟异常时间段
    anomalies = []
    if anomaly_score > 0.3:
        current_time = datetime.utcnow()
        anomalies = [
            {
                "start_time": (current_time - timedelta(minutes=30)).isoformat(),
                "end_time": (current_time - timedelta(minutes=25)).isoformat(),
                "severity": "high" if anomaly_score > 0.7 else "medium",
                "anomaly_type": "burst" if random.random() > 0.5 else "drop"
            }
        ]

    return {
        "device_name": device_name,
        "port_name": port_name,
        "anomaly_score": anomaly_score,
        "is_anomaly": anomaly_score > 0.5,
        "anomalies": anomalies,
        "check_period": last,
        "timestamp": datetime.utcnow().isoformat()
    }