from fastapi import APIRouter, HTTPException
from app.services.forecast_service import forecast_service

# 移除: from app.models import ForecastRequest, ForecastResponse

router = APIRouter(prefix="/forecast", tags=["流量预测"])


@router.post("/{device_name}/{port_name}")
async def get_forecast(device_name: str, port_name: str, horizon: str = "1h"):
    """
    获取流量预测
    """
    try:
        if horizon not in ["1h", "24h"]:
            raise HTTPException(status_code=400, detail="horizon 必须是 '1h' 或 '24h'")

        forecast_data = forecast_service.generate_forecast(device_name, port_name, horizon)
        return forecast_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测生成失败: {str(e)}")


@router.get("/models")
async def get_available_models():
    """获取可用的预测模型"""
    return {
        "available_models": forecast_service.models,
        "description": "系统支持 ARIMA、LSTM、Prophet 三种预测模型"
    }