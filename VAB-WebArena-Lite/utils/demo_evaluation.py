#!/usr/bin/env python3
"""
轨迹评估工具演示脚本
"""

import json
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))
from evaluate_trajectories import TrajectoryEvaluator


def demo_without_api():
    """演示评估流程（不需要API调用）"""
    
    print("WebArena Lite 轨迹评估工具演示")
    print("=" * 50)
    
    # 创建评估器（使用假API密钥）
    evaluator = TrajectoryEvaluator(api_key="demo-key", model="gpt-4o")
    
    # 加载任务配置
    config_path = "VAB-WebArena-Lite/config_files/wa/test_webarena_lite.raw.json"
    print(f"加载任务配置: {config_path}")
    task_configs = evaluator.load_task_config(config_path)
    print(f"加载了 {len(task_configs)} 个任务配置")
    
    # 显示几个任务配置示例
    print("\n任务配置示例:")
    for i, (task_id, config) in enumerate(list(task_configs.items())[:3]):
        print(f"\n任务 {task_id}:")
        print(f"  意图: {config.get('intent', '')}")
        print(f"  网站: {', '.join(config.get('sites', []))}")
        print(f"  评估类型: {', '.join(config.get('eval', {}).get('eval_types', []))}")
    
    # 加载轨迹文件
    trajectory_dir = Path("WADataset/data/learn_by_interact/Webarena_trajectory_lite")
    print(f"\n轨迹文件夹: {trajectory_dir}")
    
    if trajectory_dir.exists():
        trajectory_files = list(trajectory_dir.glob("*.json"))
        print(f"找到 {len(trajectory_files)} 个轨迹文件")
        
        # 显示几个轨迹文件示例
        print("\n轨迹文件示例:")
        for i, traj_file in enumerate(trajectory_files[:3]):
            task_id = int(traj_file.stem)
            trajectory = evaluator.load_trajectory(str(traj_file))
            
            if trajectory:
                print(f"\n任务 {task_id} 轨迹:")
                print(f"  文件大小: {traj_file.stat().st_size} 字节")
                print(f"  步骤数: {len(trajectory)}")
                
                # 显示轨迹摘要
                summary = evaluator._extract_trajectory_summary(trajectory)
                print(f"  轨迹摘要: {summary[:200]}...")
    else:
        print("轨迹文件夹不存在")
    
    print("\n" + "=" * 50)
    print("演示完成!")
    print("\n要运行实际评估，请:")
    print("1. 设置 OPENAI_API_KEY 环境变量")
    print("2. 运行: python run_evaluation.py")


def show_evaluation_prompt():
    """显示评估提示词示例"""
    
    print("评估提示词示例:")
    print("=" * 50)
    
    # 创建评估器
    evaluator = TrajectoryEvaluator(api_key="demo-key", model="gpt-4o")
    
    # 加载示例数据
    config_path = "VAB-WebArena-Lite/config_files/wa/test_webarena_lite.raw.json"
    task_configs = evaluator.load_task_config(config_path)
    
    trajectory_dir = Path("WADataset/data/learn_by_interact/Webarena_trajectory_lite")
    if trajectory_dir.exists():
        trajectory_files = list(trajectory_dir.glob("*.json"))
        if trajectory_files:
            # 使用第一个轨迹文件
            task_id = int(trajectory_files[0].stem)
            task_config = task_configs.get(task_id)
            trajectory = evaluator.load_trajectory(str(trajectory_files[0]))
            
            if task_config and trajectory:
                prompt = evaluator.create_evaluation_prompt(task_config, trajectory)
                print(prompt)
            else:
                print("无法加载示例数据")
        else:
            print("没有找到轨迹文件")
    else:
        print("轨迹文件夹不存在")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="轨迹评估工具演示")
    parser.add_argument("--show-prompt", action="store_true", help="显示评估提示词示例")
    
    args = parser.parse_args()
    
    if args.show_prompt:
        show_evaluation_prompt()
    else:
        demo_without_api()


if __name__ == "__main__":
    main()
