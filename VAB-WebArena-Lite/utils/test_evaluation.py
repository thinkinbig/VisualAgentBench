"""
测试轨迹评估脚本
"""

import json
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))
from evaluate_trajectories import TrajectoryEvaluator


def test_single_trajectory():
    """测试单个轨迹评估"""
    
    # 设置API密钥（请替换为你的API密钥）
    api_key = "your-openai-api-key-here"
    
    if api_key == "your-openai-api-key-here":
        print("请先设置你的OpenAI API密钥")
        return
    
    evaluator = TrajectoryEvaluator(api_key=api_key)
    
    # 测试任务ID 0
    task_id = 0
    
    # 加载任务配置
    config_path = "VAB-WebArena-Lite/config_files/wa/test_webarena_lite.raw.json"
    task_configs = evaluator.load_task_config(config_path)
    
    if task_id not in task_configs:
        print(f"任务 {task_id} 配置不存在")
        return
    
    task_config = task_configs[task_id]
    print(f"任务意图: {task_config.get('intent', '')}")
    
    # 加载轨迹
    trajectory_path = f"WADataset/data/learn_by_interact/Webarena_trajectory_lite/{task_id}.json"
    trajectory = evaluator.load_trajectory(trajectory_path)
    
    if trajectory is None:
        print(f"轨迹文件 {trajectory_path} 不存在")
        return
    
    print(f"轨迹文件大小: {len(json.dumps(trajectory))} 字符")
    
    # 评估轨迹
    print("开始评估...")
    result = evaluator.evaluate_trajectory(task_config, trajectory)
    
    print("\n评估结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def show_trajectory_structure():
    """显示轨迹文件结构"""
    trajectory_path = "WADataset/data/learn_by_interact/Webarena_trajectory_lite/0.json"
    
    if not Path(trajectory_path).exists():
        print(f"轨迹文件不存在: {trajectory_path}")
        return
    
    with open(trajectory_path, 'r', encoding='utf-8') as f:
        trajectory = json.load(f)
    
    print("轨迹文件结构:")
    print(json.dumps(trajectory, indent=2, ensure_ascii=False)[:2000] + "...")


def show_task_config():
    """显示任务配置结构"""
    config_path = "VAB-WebArena-Lite/config_files/wa/test_webarena_lite.raw.json"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    # 显示第一个任务的配置
    if tasks:
        print("任务配置结构 (第一个任务):")
        print(json.dumps(tasks[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试轨迹评估")
    parser.add_argument("--show-config", action="store_true", help="显示任务配置结构")
    parser.add_argument("--show-trajectory", action="store_true", help="显示轨迹文件结构")
    parser.add_argument("--test", action="store_true", help="测试单个轨迹评估")
    
    args = parser.parse_args()
    
    if args.show_config:
        show_task_config()
    elif args.show_trajectory:
        show_trajectory_structure()
    elif args.test:
        test_single_trajectory()
    else:
        print("请指定操作: --show-config, --show-trajectory, 或 --test")
