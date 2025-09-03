"""
任务标签管理工具
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional


def add_tag_to_tasks(task_ids: List[int], tag: str, config_path: str) -> int:
    """
    为指定的任务ID添加标签
    
    Args:
        task_ids: 要添加标签的任务ID列表
        tag: 要添加的标签
        config_path: 配置文件路径
        
    Returns:
        实际添加标签的任务数量
    """
    # 加载配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    updated_count = 0
    task_ids_set = set(task_ids)
    
    for task in config_data:
        task_id = task.get('task_id')
        if task_id in task_ids_set:
            # 检查是否已经有tags字段
            if 'tags' not in task:
                task['tags'] = []
            
            # 添加新标签（避免重复）
            if tag not in task['tags']:
                task['tags'].append(tag)
                updated_count += 1
    
    # 保存结果
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)
    
    return updated_count


def add_tag_from_evaluation_results(eval_results_path: str, tag: str, config_path: str) -> int:
    """
    从评估结果中提取成功的任务ID并添加标签
    
    Args:
        eval_results_path: 评估结果文件路径
        tag: 要添加的标签
        config_path: 配置文件路径
        
    Returns:
        实际添加标签的任务数量
    """
    # 加载评估结果
    with open(eval_results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 提取成功的任务ID
    success_ids = []
    for result in results:
        if result.get('success', False):
            task_id = result.get('task_id')
            if task_id is not None:
                success_ids.append(task_id)
    
    print(f"从评估结果中找到 {len(success_ids)} 个成功的任务")
    print(f"成功的任务ID: {sorted(success_ids)[:20]}{'...' if len(success_ids) > 20 else ''}")
    
    # 添加标签
    updated_count = add_tag_to_tasks(success_ids, tag, config_path)
    
    print(f"为 {updated_count} 个任务添加了标签 '{tag}'")
    return updated_count


def add_tag_from_webpilot_success(webpilot_success_path: str, tag: str, config_path: str) -> int:
    """
    从webpilot成功数据中提取任务ID并添加标签
    
    Args:
        webpilot_success_path: webpilot成功数据文件路径
        tag: 要添加的标签
        config_path: 配置文件路径
        
    Returns:
        实际添加标签的任务数量
    """
    # 加载webpilot成功数据
    with open(webpilot_success_path, 'r', encoding='utf-8') as f:
        webpilot_data = json.load(f)
    
    # 提取任务ID
    task_ids = []
    for item in webpilot_data:
        task_id = item.get('task_id')
        if task_id is not None:
            task_ids.append(task_id)
    
    print(f"从webpilot成功数据中找到 {len(task_ids)} 个任务")
    print(f"任务ID: {sorted(task_ids)[:20]}{'...' if len(task_ids) > 20 else ''}")
    
    # 添加标签
    updated_count = add_tag_to_tasks(task_ids, tag, config_path)
    
    print(f"为 {updated_count} 个任务添加了标签 '{tag}'")
    return updated_count


def remove_tag_from_tasks(task_ids: List[int], tag: str, config_path: str) -> int:
    """
    从指定的任务ID中移除标签
    
    Args:
        task_ids: 要移除标签的任务ID列表
        tag: 要移除的标签
        config_path: 配置文件路径
        
    Returns:
        实际移除标签的任务数量
    """
    # 加载配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    updated_count = 0
    task_ids_set = set(task_ids)
    
    for task in config_data:
        task_id = task.get('task_id')
        if task_id in task_ids_set and 'tags' in task:
            if tag in task['tags']:
                task['tags'].remove(tag)
                updated_count += 1
                # 如果tags列表为空，删除该字段
                if not task['tags']:
                    del task['tags']
    
    # 保存结果
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)
    
    return updated_count


def get_tasks_by_tag(tag: str, config_path: str) -> List[int]:
    """
    获取具有指定标签的所有任务ID
    
    Args:
        tag: 标签名称
        config_path: 配置文件路径
        
    Returns:
        具有该标签的任务ID列表
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    tagged_tasks = []
    for task in config_data:
        if 'tags' in task and tag in task['tags']:
            tagged_tasks.append(task['task_id'])
    
    return sorted(tagged_tasks)


def list_all_tags(config_path: str) -> Dict[str, int]:
    """
    列出所有标签及其对应的任务数量
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        标签名称到任务数量的映射
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    tag_counts = {}
    for task in config_data:
        if 'tags' in task:
            for tag in task['tags']:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    return tag_counts


def main():
    """主函数 - 提供命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="任务标签管理工具")
    parser.add_argument(
        "--config",
        default="VAB-WebArena-Lite/config_files/wa/test_webarena_lite.raw.json",
        help="配置文件路径"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 添加标签命令
    add_parser = subparsers.add_parser('add', help='添加标签')
    add_parser.add_argument('--tag', required=True, help='要添加的标签')
    add_parser.add_argument('--task-ids', nargs='+', type=int, help='任务ID列表')
    add_parser.add_argument('--from-eval', help='从评估结果文件添加标签')
    add_parser.add_argument('--from-webpilot', help='从webpilot成功数据文件添加标签')
    
    # 移除标签命令
    remove_parser = subparsers.add_parser('remove', help='移除标签')
    remove_parser.add_argument('--tag', required=True, help='要移除的标签')
    remove_parser.add_argument('--task-ids', nargs='+', type=int, required=True, help='任务ID列表')
    
    # 列出标签命令
    list_parser = subparsers.add_parser('list', help='列出所有标签')
    list_parser.add_argument('--tag', help='列出具有指定标签的任务ID')
    
    args = parser.parse_args()
    
    if args.command == 'add':
        if args.from_eval:
            add_tag_from_evaluation_results(args.from_eval, args.tag, args.config)
        elif args.from_webpilot:
            add_tag_from_webpilot_success(args.from_webpilot, args.tag, args.config)
        elif args.task_ids:
            updated_count = add_tag_to_tasks(args.task_ids, args.tag, args.config)
            print(f"为 {updated_count} 个任务添加了标签 '{args.tag}'")
        else:
            print("错误: 必须指定 --task-ids、--from-eval 或 --from-webpilot")
    
    elif args.command == 'remove':
        updated_count = remove_tag_from_tasks(args.task_ids, args.tag, args.config)
        print(f"从 {updated_count} 个任务中移除了标签 '{args.tag}'")
    
    elif args.command == 'list':
        if args.tag:
            task_ids = get_tasks_by_tag(args.tag, args.config)
            print(f"具有标签 '{args.tag}' 的任务ID: {task_ids}")
            print(f"共 {len(task_ids)} 个任务")
        else:
            tag_counts = list_all_tags(args.config)
            print("所有标签统计:")
            for tag, count in sorted(tag_counts.items()):
                print(f"  {tag}: {count} 个任务")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
