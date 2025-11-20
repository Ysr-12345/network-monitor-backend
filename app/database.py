from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base

# 使用 SQLite 作为临时数据库
SQLALCHEMY_DATABASE_URL = "sqlite:///./network_monitor.db"

# 创建数据库引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建数据库表
def create_tables():
    Base.metadata.create_all(bind=engine)

# 依赖注入，用于获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()