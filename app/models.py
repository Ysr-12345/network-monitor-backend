from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class NetworkTraffic(Base):
    __tablename__ = "network_traffic"

    id = Column(Integer, primary_key=True, index=True)
    device_name = Column(String(100), nullable=False)
    port_name = Column(String(100), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    traffic_in = Column(Float, nullable=False)  # 入向流量 bps
    traffic_out = Column(Float, nullable=False)  # 出向流量 bps
    bandwidth = Column(Float, nullable=False)  # 端口带宽 bps


class TrafficForecast(Base):
    __tablename__ = "traffic_forecast"

    id = Column(Integer, primary_key=True, index=True)
    device_name = Column(String(100), nullable=False)
    port_name = Column(String(100), nullable=False)
    forecast_timestamp = Column(DateTime, nullable=False)
    predicted_value = Column(Float, nullable=False)
    confidence_upper = Column(Float)  # 置信区间上限
    confidence_lower = Column(Float)  # 置信区间下限
    model_type = Column(String(50))  # ARIMA, LSTM, Prophet
    created_at = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    device_name = Column(String(100), nullable=False)
    port_name = Column(String(100), nullable=False)
    alert_type = Column(String(50), nullable=False)  # congestion, anomaly
    alert_level = Column(String(20), nullable=False)  # warning, error, critical
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

# Pydantic 模型用于请求和响应数据验证

class TrafficData(BaseModel):
    device_name: str
    port_name: str
    traffic_in: float
    traffic_out: float
    bandwidth: float

class ForecastRequest(BaseModel):
    device_name: str
    port_name: str
    horizon: str = "1h"  # 1h 或 24h

class ForecastResponse(BaseModel):
    device_name: str
    port_name: str
    forecasts: List[dict]
    model_used: str
    confidence_interval: float

class AlertResponse(BaseModel):
    id: int
    device_name: str
    port_name: str
    alert_type: str
    alert_level: str
    message: str
    timestamp: datetime
    is_resolved: bool

    class Config:
        from_attributes = True