# Reward Guided Search Module

一个智能的奖励引导搜索模块，帮助代理在Web自动化任务中找到最优的行动路径。

## 🚀 主要功能

- **智能搜索**: 使用多种搜索策略找到最佳解决方案
- **奖励引导**: 基于奖励函数指导搜索方向
- **自适应学习**: 根据任务表现自动调整策略
- **性能优化**: 内置缓存和性能监控

## 📋 快速开始

```python
from agent.reward_guided_search import RewardGuidedSearchAgent, SearchConfig

# 创建代理
config = SearchConfig(
    search_type="beam_search",
    beam_width=5,
    max_depth=8
)
agent = RewardGuidedSearchAgent(config)

# 使用代理
action = agent.next_action(trajectory, intent, meta_data)
```

## 🔧 配置选项

- `search_type`: 搜索策略 (beam_search, monte_carlo, a_star)
- `beam_width`: 搜索宽度
- `max_depth`: 最大搜索深度
- `max_iterations`: 最大迭代次数

## 📊 监控性能

```python
# 获取性能统计
stats = agent.get_search_statistics()
print(f"成功率: {stats['success_rate']:.2%}")
```

## 🎯 适用场景

- Web自动化任务规划
- 用户界面导航优化
- 任务完成路径搜索
- 智能代理行为优化

## 📁 文件说明

- `example_usage.py` - 使用示例
- `README.md` - 本文档

## 🤝 贡献

欢迎提交Issue和Pull Request！
