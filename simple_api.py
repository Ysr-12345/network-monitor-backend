from fastapi import FastAPI
import random
from datetime import datetime, timedelta

app = FastAPI(
    title="网络流量智能监控系统",
    description="基于AI的网络流量预测与异常检测系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/")
async def root():
    return {
        "message": "网络流量监控系统 API",
        "status": "运行中",
        "endpoints": {
            "预测接口": "/api/v1/forecast/{device}/{port}",
            "异常检测": "/api/v1/anomaly/{device}/{port}",
            "告警管理": "/api/v1/alerts/"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/v1/forecast/{device_name}/{port_name}")
async def get_forecast(device_name: str, port_name: str, horizon: str = "1h"):
    """模拟流量预测"""
    forecasts = []
    current_time = datetime.utcnow()

    points = 12 if horizon == "1h" else 288

    for i in range(points):
        forecast_time = current_time + timedelta(minutes=5 * (i + 1))

        # 模拟不同设备的基准流量
        if "core" in device_name.lower():
            base_traffic = 8000000000  # 8 Gbps
        else:
            base_traffic = 1000000000  # 1 Gbps

        # 加入时间周期性
        hour = forecast_time.hour
        time_factor = 0.3 if 0 <= hour <= 6 else 0.7 if hour <= 17 else 1.2

        traffic_value = base_traffic * time_factor * random.uniform(0.95, 1.05)

        forecasts.append({
            "timestamp": forecast_time.isoformat(),
            "predicted_value": round(traffic_value, 2),
            "confidence_upper": round(traffic_value * 1.1, 2),
            "confidence_lower": round(traffic_value * 0.9, 2)
        })

    return {
        "device_name": device_name,
        "port_name": port_name,
        "forecasts": forecasts,
        "model_used": random.choice(["ARIMA", "LSTM", "Prophet"]),
        "confidence_interval": 0.95
    }


@app.get("/api/v1/anomaly/{device_name}/{port_name}")
async def check_anomaly(device_name: str, port_name: str, last: str = "1h"):
    """模拟异常检测"""
    anomaly_score = round(random.uniform(0, 1), 2)

    # 模拟偶尔出现高异常分数
    if random.random() < 0.3:  # 30% 几率出现异常
        anomaly_score = round(random.uniform(0.6, 0.95), 2)

    anomalies = []
    if anomaly_score > 0.5:
        current_time = datetime.utcnow()
        anomalies = [
            {
                "start_time": (current_time - timedelta(minutes=10)).isoformat(),
                "end_time": (current_time - timedelta(minutes=5)).isoformat(),
                "severity": "high" if anomaly_score > 0.7 else "medium",
                "anomaly_type": random.choice(["burst", "drop", "scan"])
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


@app.get("/api/v1/alerts/")
async def get_alerts(hours: int = 24):
    """模拟告警列表"""
    alerts = [
        {
            "id": 1,
            "device_name": "core-switch-1",
            "port_name": "GigabitEthernet0/1",
            "alert_type": "congestion",
            "alert_level": "warning",
            "message": "流量接近上限！当前利用率: 85%",
            "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            "is_resolved": False
        },
        {
            "id": 2,
            "device_name": "access-switch-1",
            "port_name": "GigabitEthernet0/24",
            "alert_type": "anomaly",
            "alert_level": "critical",
            "message": "检测到严重流量异常！异常分数: 0.89",
            "timestamp": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            "is_resolved": True
        }
    ]
    return alerts


@app.post("/api/v1/alerts/test-congestion")
async def test_congestion_alert():
    """测试拥塞告警"""
    return {
        "success": True,
        "alert": {
            "device_name": "core-switch-1",
            "port_name": "GigabitEthernet0/1",
            "alert_type": "congestion",
            "alert_level": "warning",
            "message": "测试告警：流量接近上限！",
            "timestamp": datetime.utcnow().isoformat()
        },
        "message": "拥塞告警测试成功"
    }


@app.post("/api/v1/alerts/test-anomaly")
async def test_anomaly_alert():
    """测试异常告警"""
    return {
        "success": True,
        "alert": {
            "device_name": "core-switch-1",
            "port_name": "GigabitEthernet0/1",
            "alert_type": "anomaly",
            "alert_level": "critical",
            "message": "测试告警：检测到严重流量异常！",
            "timestamp": datetime.utcnow().isoformat()
        },
        "message": "异常告警测试成功"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("simple_api:app", host="0.0.0.0", port=8000, reload=True)