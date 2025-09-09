# Trajectory Tree Viewer

一个现代化的前后端架构程序，用于交互式可视化轨迹树，支持点击节点查看截图和详细信息。

## 🎯 功能特点

- **交互式轨迹树可视化**: 基于 vis-network 的现代化图形界面
- **点击节点查看截图**: 支持点击任何节点查看对应的页面截图
- **详细信息展示**: 显示节点的思考过程、动作含义、URL等详细信息
- **响应式设计**: 支持桌面和移动设备
- **实时数据**: 自动检测和加载新的轨迹文件
- **全屏模式**: 支持全屏查看轨迹图

## 🏗️ 架构

### 后端 (FastAPI)
- **API服务**: 提供轨迹数据、截图文件等RESTful API
- **静态文件服务**: 直接提供截图文件访问
- **CORS支持**: 支持跨域请求
- **自动发现**: 自动扫描并列出所有可用的轨迹文件

### 前端 (React + TypeScript)
- **现代化UI**: 基于React 18和TypeScript
- **图形可视化**: 使用vis-network进行轨迹树渲染
- **状态管理**: React Hooks进行状态管理
- **类型安全**: 完整的TypeScript类型定义

## 🚀 快速开始

### 1. 启动后端服务

```bash
cd VAB-WebArena-Lite/trajectory_viewer
./start_backend.sh
```

后端服务将在 http://localhost:8000 启动

### 2. 启动前端服务

```bash
# 在新的终端窗口中
cd VAB-WebArena-Lite/trajectory_viewer
./start_frontend.sh
```

前端应用将在 http://localhost:3000 启动

### 3. 访问应用

打开浏览器访问 http://localhost:3000

## 📁 目录结构

```
trajectory_viewer/
├── backend/
│   ├── main.py              # FastAPI主服务
│   └── requirements.txt     # Python依赖
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/      # React组件
│   │   ├── types.ts         # TypeScript类型定义
│   │   ├── api.ts           # API客户端
│   │   └── App.tsx          # 主应用组件
│   └── package.json         # Node.js依赖
├── start_backend.sh         # 后端启动脚本
├── start_frontend.sh        # 前端启动脚本
└── README.md               # 本文档
```

## 🔧 API接口

### 轨迹相关
- `GET /api/trajectories` - 获取所有轨迹列表
- `GET /api/trajectories/{run_id}` - 获取特定轨迹详情
- `GET /api/trajectories/{run_id}/graphviz` - 获取Graphviz DOT源码

### 文件服务
- `GET /screenshots/{filename}` - 获取截图文件
- `GET /api/health` - 健康检查

## 🎨 界面说明

### 轨迹列表
- 显示所有可用的轨迹文件
- 显示轨迹的基本信息：run_id、意图、节点数量等
- 点击轨迹项可以加载并可视化

### 轨迹可视化
- **节点类型**:
  - 🔵 蓝色圆形: 根节点（任务开始）
  - 🟢 绿色方形: 状态节点（执行步骤）
  - 🟡 黄色方形: 候选节点（未选择）
  - 🟢 绿色方形: 选中节点（已执行）
- **边类型**:
  - 绿色实线: 已执行的动作路径
  - 灰色虚线: 候选动作（未选择）

### 交互功能
- **点击节点**: 查看节点详细信息
- **截图查看**: 如果节点有截图，可以点击查看
- **全屏模式**: 支持全屏查看轨迹图
- **视图重置**: 重置图形视图到最佳位置

## 🔄 与现有系统集成

这个查看器会自动读取 `outputs/trajectory/` 目录下的 `*_final.json` 文件，这些文件由 `run_reward_guided.py` 生成。

### 数据流程
1. `run_reward_guided.py` 运行任务并生成轨迹JSON文件
2. 后端API自动发现新的轨迹文件
3. 前端界面显示可用的轨迹列表
4. 用户选择轨迹进行交互式可视化

## 🛠️ 开发

### 后端开发
```bash
cd trajectory_viewer/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### 前端开发
```bash
cd trajectory_viewer/frontend
npm install
npm start
```

## 📱 移动端支持

界面采用响应式设计，在移动设备上也能正常使用，支持触摸操作。

## 🐛 故障排除

### 1. 后端无法启动
- 检查Python环境是否正确
- 确认依赖是否安装完整
- 检查端口8000是否被占用

### 2. 前端无法启动
- 检查Node.js版本（需要16+）
- 确认依赖是否安装完整
- 检查端口3000是否被占用

### 3. 截图不显示
- 检查截图文件是否存在
- 确认文件路径正确
- 检查文件权限

### 4. 轨迹数据不显示
- 确认 `outputs/trajectory/` 目录存在
- 检查JSON文件格式是否正确
- 查看浏览器控制台错误信息

## 🔮 未来计划

- [ ] 支持轨迹对比功能
- [ ] 添加搜索和过滤功能
- [ ] 支持轨迹导出功能
- [ ] 添加更多可视化选项
- [ ] 支持实时轨迹更新
