"""
将WebArena轨迹文件映射到WebArena Lite
"""

import os
import shutil
from pathlib import Path
from typing import Optional
import sys

# 添加当前目录到Python路径，以便导入task_id_mapping
sys.path.append(str(Path(__file__).parent))
from task_id_mapping import arena_to_lite_id


def map_webarena_trajectories_to_lite(
    source_dir: str,
    target_dir: str,
    dry_run: bool = False
) -> None:
    """
    将WebArena轨迹文件映射到WebArena Lite
    
    Args:
        source_dir: WebArena轨迹文件夹路径
        target_dir: 目标文件夹路径（WebArena Lite轨迹）
        dry_run: 如果为True，只显示将要复制的文件，不实际复制
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"源文件夹不存在: {source_path}")
    
    # 创建目标文件夹
    if not dry_run:
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"创建目标文件夹: {target_path}")
    
    # 统计信息
    total_files = 0
    mapped_files = 0
    skipped_files = 0
    
    print(f"开始处理轨迹文件...")
    print(f"源文件夹: {source_path}")
    print(f"目标文件夹: {target_path}")
    print(f"模式: {'预览模式' if dry_run else '复制模式'}")
    print("-" * 50)
    
    # 遍历所有JSON文件
    for json_file in source_path.glob("*.json"):
        total_files += 1
        
        # 从文件名提取arena_task_id
        try:
            arena_task_id = int(json_file.stem)
        except ValueError:
            print(f"跳过非数字文件名: {json_file.name}")
            skipped_files += 1
            continue
        
        # 检查是否有对应的lite_task_id
        lite_task_id = arena_to_lite_id(arena_task_id)
        
        if lite_task_id is not None:
            # 有映射，复制文件并重命名
            target_file = target_path / f"{lite_task_id}.json"
            
            if dry_run:
                print(f"将复制: {json_file.name} -> {target_file.name} (Arena {arena_task_id} -> Lite {lite_task_id})")
            else:
                shutil.copy2(json_file, target_file)
                print(f"已复制: {json_file.name} -> {target_file.name} (Arena {arena_task_id} -> Lite {lite_task_id})")
            
            mapped_files += 1
        else:
            # 没有映射，跳过
            if dry_run:
                print(f"跳过: {json_file.name} (Arena {arena_task_id} 无对应Lite映射)")
            skipped_files += 1
    
    print("-" * 50)
    print(f"处理完成!")
    print(f"总文件数: {total_files}")
    print(f"已映射: {mapped_files}")
    print(f"已跳过: {skipped_files}")
    print(f"映射率: {mapped_files/total_files*100:.1f}%" if total_files > 0 else "映射率: 0%")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="将WebArena轨迹文件映射到WebArena Lite")
    parser.add_argument(
        "--source", 
        default="WADataset/data/learn_by_interact/Webarena_trajectory-20250828T023914Z-1-001/Webarena_trajectory",
        help="WebArena轨迹文件夹路径"
    )
    parser.add_argument(
        "--target",
        default="WADataset/data/learn_by_interact/Webarena_trajectory_lite",
        help="目标文件夹路径"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，不实际复制文件"
    )
    
    args = parser.parse_args()
    
    try:
        map_webarena_trajectories_to_lite(
            source_dir=args.source,
            target_dir=args.target,
            dry_run=args.dry_run
        )
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
