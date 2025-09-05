 # 交互式轨迹可视化指南

## 🎯 功能特点

- **点击节点显示截图**: 点击轨迹图中的任何节点，可以查看该步骤的截图
- **详细信息展示**: 显示步骤信息、URL、候选动作、思考过程等
- **候选动作对比**: 绿色表示执行的动作，红色表示被放弃的候选动作
- **实时交互**: 基于Web的交互式界面，支持缩放、拖拽等操作

## 🚀 快速开始

### 1. 生成交互式可视化
```bash
# 生成最新的交互式轨迹图
python visualize_trajectory.py --interactive --open

# 生成指定文件的交互式可视化
python visualize_trajectory.py --interactive --file trajectory_xxx_final.dot --open
```

### 2. 在代码中使用
```python
from agent.reward_guided_agent import RewardGuidedAgent

# 创建agent
agent = RewardGuidedAgent(...)

# 运行任务...

# 保存最终轨迹（自动生成交互式HTML）
agent.save_final_trajectory()
```

## 📁 生成的文件

- `trajectory_xxx_final.json`: 完整轨迹数据（包含候选动作）
- `trajectory_xxx_final.dot`: Graphviz DOT格式
- `trajectory_xxx.svg`: 静态SVG图
- `trajectory_xxx.png`: 静态PNG图
- `trajectory_xxx_interactive.html`: **交互式HTML文件** ⭐

> **注意**: 系统只保存完整的final文件，不再生成步骤文件（`_step_X.json`）

## 🖼️ 截图支持

### 自动截图保存
系统会自动保存每个步骤的截图到轨迹树中：

```python
# 在RuntimeManager中，截图路径会自动保存到节点
node.screenshot_path = "/path/to/screenshot.png"
```

### 手动添加截图
```python
# 为特定节点添加截图
node = tree.get_node("node_id")
node.screenshot_path = "path/to/screenshot.png"
```

## 🎨 交互式界面说明

### 节点类型
- **蓝色节点**: 任务开始节点
- **绿色节点**: 执行步骤节点

### 边类型
- **绿色实线**: 已执行的动作（主路径）
- **红色虚线**: 候选动作（被放弃的备选方案）

### 点击节点显示信息
- **步骤信息**: 步骤号、URL
- **截图**: 该步骤的页面截图
- **候选动作**: 所有候选动作及其思考过程
- **Checkpoint**: 完整的检查点信息

## 🔧 高级功能

### 1. 自定义截图路径
```python
# 在TrajNode中设置截图路径
node.screenshot_path = "/custom/path/screenshot.png"
```

### 2. 批量生成可视化
```python
# 为多个轨迹文件生成交互式可视化
import glob
for dot_file in glob.glob("outputs/trajectory/*_final.dot"):
    subprocess.run([
        "python", "visualize_trajectory.py", 
        "--interactive", 
        "--file", dot_file
    ])
```

### 3. 自定义HTML模板
可以修改`TrajectoryTree.to_interactive_html()`方法来自定义HTML模板。

## 🌐 浏览器兼容性

- Chrome/Chromium: ✅ 完全支持
- Firefox: ✅ 完全支持
- Safari: ✅ 完全支持
- Edge: ✅ 完全支持

## 📱 移动端支持

交互式HTML在移动设备上也能正常工作，支持触摸操作。

## 🐛 故障排除

### 1. 截图不显示
- 检查截图文件是否存在
- 确认文件路径正确
- 检查文件权限

### 2. 交互式页面无法打开
- 确保浏览器支持JavaScript
- 检查网络连接（需要加载vis-network库）
- 尝试使用不同的浏览器

### 3. 轨迹图显示异常
- 检查JSON文件是否完整
- 确认DOT文件格式正确
- 查看浏览器控制台错误信息

## 💡 使用技巧

1. **最佳实践**: 使用交互式HTML进行轨迹分析和调试
2. **分享**: 可以将HTML文件分享给团队成员
3. **存档**: 保存JSON文件用于后续分析
4. **演示**: 使用交互式界面进行演示和汇报

## 🔄 更新日志

- v1.0: 基础交互式可视化
- v1.1: 添加截图支持
- v1.2: 优化界面和用户体验
