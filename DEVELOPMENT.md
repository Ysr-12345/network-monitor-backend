# 开发指南

## 分支策略
- `main` - 稳定版本（生产就绪代码）
- `develop` - 开发分支（功能集成）
- `feature/*` - 功能分支（新功能开发）
- `hotfix/*` - 热修复分支（紧急bug修复）

## 开发流程
1. 从 `develop` 分支创建功能分支：`git checkout -b feature/功能名称`
2. 在功能分支上开发并提交代码
3. 完成开发后，创建 Pull Request 到 `develop` 分支
4. 代码审查通过后合并到 `develop`
5. 测试稳定后合并到 `main` 分支

## API 接口

### 流量预测
```http
POST /api/v1/forecast/{device_name}/{port_name}?horizon=1h

