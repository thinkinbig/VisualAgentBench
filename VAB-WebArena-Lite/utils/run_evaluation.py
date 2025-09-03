#!/usr/bin/env python3
"""
运行轨迹评估的示例脚本
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))
from evaluate_trajectories import TrajectoryEvaluator


def main():
    """主函数"""
    
    # 检查API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误: 请设置环境变量 OPENAI_API_KEY")
        print("例如: export OPENAI_API_KEY='your-api-key-here'")
        return 1
    
    # 创建评估器
    evaluator = TrajectoryEvaluator(api_key=api_key, model="gpt-4o")
    
    # 配置路径
    config_path = "VAB-WebArena-Lite/config_files/wa/test_webarena_lite.raw.json"
    trajectory_dir = "WADataset/data/learn_by_interact/Webarena_trajectory_lite"
    output_path = "trajectory_evaluation_results.json"
    
    print("WebArena Lite 轨迹评估工具")
    print("=" * 50)
    print(f"配置文件: {config_path}")
    print(f"轨迹文件夹: {trajectory_dir}")
    print(f"输出文件: {output_path}")
    print(f"使用模型: gpt-4o")
    print()
    
    # 询问用户是否继续
    response = input("是否开始评估? (y/N): ").strip().lower()
    if response != 'y':
        print("评估已取消")
        return 0
    
    try:
        # 开始评估（限制前10个任务用于测试）
        print("开始评估前10个任务...")
        stats = evaluator.evaluate_all_trajectories(
            config_path=config_path,
            trajectory_dir=trajectory_dir,
            output_path=output_path,
            max_tasks=10
        )
        
        print("\n" + "="*50)
        print("评估完成!")
        print(f"总任务数: {stats['total']}")
        print(f"成功: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
        print(f"失败: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
        print(f"API错误: {stats['api_errors']}")
        print(f"文件错误: {stats['file_errors']}")
        print(f"结果已保存到: {output_path}")
        
        # 询问是否评估所有任务
        if stats['total'] > 0:
            response = input("\n是否评估所有165个任务? (y/N): ").strip().lower()
            if response == 'y':
                print("开始评估所有任务...")
                stats = evaluator.evaluate_all_trajectories(
                    config_path=config_path,
                    trajectory_dir=trajectory_dir,
                    output_path=output_path,
                    max_tasks=None
                )
                
                print("\n" + "="*50)
                print("全部评估完成!")
                print(f"总任务数: {stats['total']}")
                print(f"成功: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
                print(f"失败: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
                print(f"API错误: {stats['api_errors']}")
                print(f"文件错误: {stats['file_errors']}")
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
