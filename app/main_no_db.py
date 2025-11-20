from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import forecast, anomaly, alerts

# 创建 FastAPI 应用实例
app = FastAPI(
    title="网络流量智能监控系统",
    description="基于AI的网络流量预测与异常检测系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由（跳过数据库相关部分）
app.include_router(forecast.router, prefix="/api/v1")
app.include_router(anomaly.router, prefix="/api/v1")
app.include_router(alerts.router, prefix="/api/v1")

# 根路径
@app.get("/")
async def root():
    return {
        "message": "网络流量智能监控系统 API (无数据库模式)",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "运行中 - 数据库功能已禁用"
    }

# 健康检查
@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "disabled"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)