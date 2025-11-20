




from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import create_tables
from app.routes import forecast, anomaly, alerts

# 创建 FastAPI 应用实例
app = FastAPI(
    title="网络流量智能监控系统",
    description="基于AI的网络流量预测与异常检测系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置 CORS（跨域资源共享）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建数据库表
@app.on_event("startup")
async def startup_event():
    create_tables()
    print("✅ 数据库表创建完成")

# 注册路由
app.include_router(forecast.router, prefix="/api/v1")
app.include_router(anomaly.router, prefix="/api/v1")
app.include_router(alerts.router, prefix="/api/v1")

# 根路径
@app.get("/")
async def root():
    return {
        "message": "网络流量智能监控系统 API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "预测接口": "/api/v1/forecast",
            "异常检测": "/api/v1/anomaly",
            "告警管理": "/api/v1/alerts"
        }
    }

# 健康检查
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)