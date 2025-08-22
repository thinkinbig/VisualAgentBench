#!/usr/bin/env python3
"""
分析WebPRMCollection_preference_pair数据集中不寻常的action
统计哪些action不在标准的action space中，以及它们出现在哪些题目中
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from collections import defaultdict, Counter
import json
import os
import time
import logging
from typing import Any, Dict, List, Optional, Set
from tqdm import tqdm

def load_dataset_and_analyze():
    """加载数据集并分析不寻常的action"""
    print("正在下载WebPRMCollection_preference_pair数据集...")
    
    try:
        dataset = load_dataset("WebShepherd/WebPRMCollection_preference_pair")
        print(f"数据集加载成功！包含 {len(dataset['test'])} 个样本")
        
        # 转换为pandas DataFrame
        df = dataset['test'].to_pandas()
        
        print(f"数据已加载，共{len(df)}行")
        print(f"包含{df['task_id'].nunique()}个不同的task_id")
        
        return df
        
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return None

def get_standard_action_space():
    """定义标准的action space，基于WebArena的actions.py"""
    # 基于WebArena prompt文档中定义的action space
    action_space = [
        # Page Operation Actions
        "click", "type", "hover", "press", "scroll",
        
        # Tab Management Actions  
        "new_tab", "tab_focus", "close_tab",
        
        # URL Navigation Actions
        "goto", "go_back", "go_forward",
        
        # Completion Action
        "stop",
        
        # Progress Reporting Action
        "send_msg_to_user"
    ]
    
    return set(action_space)

def extract_actions_from_action_history(action_history):
    """从action_history中提取所有action"""
    actions = set()
    
    if not isinstance(action_history, list):
        return actions
    
    for item in action_history:
        if isinstance(item, str):
            # 提取base action (动作名称)
            if '(' in item or '[' in item:
                # 处理复合action，如 "click('button')" 或 "click [123]"
                base_action = item.split('(')[0].split('[')[0].strip()
                actions.add(base_action)
            else:
                # 简单action
                actions.add(item.strip())
        elif isinstance(item, dict):
            # 如果是字典，提取action字段
            if 'action' in item:
                action = item['action']
                if isinstance(action, str):
                    # 处理复合action
                    if '(' in action or '[' in action:
                        base_action = action.split('(')[0].split('[')[0].strip()
                        actions.add(base_action)
                    else:
                        actions.add(action.strip())
    
    return actions

def analyze_unusual_actions(df, standard_action_space):
    """分析不寻常的action"""
    print("\n=== 分析不寻常的action ===")
    
    unusual_actions = defaultdict(lambda: {
        'count': 0,
        'task_ids': set(),
        'examples': []
    })
    
    total_actions = 0
    unusual_count = 0
    
    # 添加调试信息
    debug_samples = 0
    max_debug_samples = 5
    
    # 遍历所有样本
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="分析action", unit="行"):
        task_id = row['task_id']
        action_history = row.get('action_history', [])
        
        if action_history is None or len(action_history) == 0:
            continue
        
        # 调试信息：显示前几个样本的action_history内容
        if debug_samples < max_debug_samples:
            print(f"\n调试样本 {debug_samples + 1}:")
            print(f"  Task ID: {task_id}")
            print(f"  Action History: {action_history}")
            print(f"  Type: {type(action_history)}")
            print(f"  Length: {len(action_history)}")
            if len(action_history) > 0:
                print(f"  First item: {action_history[0]} (type: {type(action_history[0])})")
            debug_samples += 1
        
        # 提取当前样本的所有action
        sample_actions = extract_actions_from_action_history(action_history.tolist() if hasattr(action_history, 'tolist') else action_history)
        total_actions += len(sample_actions)
        
        # 检查每个action是否在标准action space中
        for action in sample_actions:
            if action not in standard_action_space:
                unusual_count += 1
                unusual_actions[action]['count'] += 1
                unusual_actions[action]['task_ids'].add(task_id)
                
                # 记录示例信息
                unusual_actions[action]['examples'].append({
                    'task_id': task_id,
                    'step_id': row.get('step_id', ''),
                    'row_idx': idx,
                    'action_history': action_history.tolist() if hasattr(action_history, 'tolist') else action_history,
                    'website_name': row.get('website_name', ''),
                    'intent': row.get('intent', '')[:100] + '...' if len(str(row.get('intent', ''))) > 100 else row.get('intent', '')
                })
    
    print(f"\n总共检查了 {total_actions} 个action")
    print(f"发现 {unusual_count} 个不寻常的action")
    print(f"涉及 {len(unusual_actions)} 种不同的不寻常action类型")
    
    return unusual_actions

def print_unusual_actions_analysis(unusual_actions, standard_action_space):
    """打印不寻常action的分析结果"""
    print(f"\n=== 不寻常action详细分析 ===")
    
    # 按出现次数排序
    sorted_actions = sorted(unusual_actions.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print(f"\n发现的不寻常action (按出现次数排序):")
    print("-" * 80)
    
    for action, info in sorted_actions:
        print(f"\nAction: '{action}'")
        print(f"  出现次数: {info['count']}")
        print(f"  涉及题目数: {len(info['task_ids'])}")
        print(f"  题目ID列表: {sorted(list(info['task_ids']))[:10]}{'...' if len(info['task_ids']) > 10 else ''}")
        
        # 显示前几个示例
        print(f"  示例:")
        for i, example in enumerate(info['examples'][:3]):
            print(f"    {i+1}. Task {example['task_id']}-{example['step_id']} ({example['website_name']})")
            print(f"       Intent: {example['intent']}")
            print(f"       Action History: {example['action_history'][:3]}{'...' if len(example['action_history']) > 3 else ''}")
    
    # 统计信息
    print(f"\n=== 统计摘要 ===")
    print(f"标准action space包含 {len(standard_action_space)} 种action")
    print(f"发现 {len(unusual_actions)} 种不寻常的action")
    
    # 按出现次数分组
    frequency_groups = {
        '高频(>=10次)': 0,
        '中频(5-9次)': 0,
        '低频(2-4次)': 0,
        '单次': 0
    }
    
    for action, info in unusual_actions.items():
        count = info['count']
        if count >= 10:
            frequency_groups['高频(>=10次)'] += 1
        elif count >= 5:
            frequency_groups['中频(5-9次)'] += 1
        elif count >= 2:
            frequency_groups['低频(2-4次)'] += 1
        else:
            frequency_groups['单次'] += 1
    
    print(f"\n按出现频率分组:")
    for group, count in frequency_groups.items():
        print(f"  {group}: {count} 种")
    
    # 分析可能的原因
    print(f"\n=== 可能的原因分析 ===")
    
    # 检查是否是标准action的变体
    potential_variants = defaultdict(list)
    for unusual_action in unusual_actions.keys():
        for standard_action in standard_action_space:
            if (unusual_action.lower() in standard_action.lower() or 
                standard_action.lower() in unusual_action.lower() or
                unusual_action.replace('_', '').lower() == standard_action.replace('_', '').lower()):
                potential_variants[unusual_action].append(standard_action)
    
    if potential_variants:
        print(f"\n可能是标准action变体的action:")
        for unusual_action, variants in potential_variants.items():
            print(f"  '{unusual_action}' -> 可能的变体: {variants}")
    
    # 检查是否是复合action
    compound_actions = [action for action in unusual_actions.keys() if '(' in action or ' ' in action]
    if compound_actions:
        print(f"\n可能是复合action的action:")
        for action in compound_actions[:10]:  # 只显示前10个
            print(f"  '{action}'")
    
    return sorted_actions

def save_analysis_results(unusual_actions, output_dir="analysis/scripts/results"):
    """保存分析结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/unusual_actions_analysis_{timestamp}.json"
    
    # 转换set为list以便JSON序列化
    serializable_actions = {}
    for action, info in unusual_actions.items():
        serializable_actions[action] = {
            'count': info['count'],
            'task_ids': sorted(list(info['task_ids'])),
            'examples': info['examples']
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_actions, f, ensure_ascii=False, indent=2)
    
    print(f"\n分析结果已保存到: {output_file}")
    return output_file

def main():
    """主函数"""
    print("开始分析WebPRMCollection_preference_pair数据集中的不寻常action...")
    
    # 加载数据集
    df = load_dataset_and_analyze()
    if df is None:
        return
    
    # 获取标准action space
    standard_action_space = get_standard_action_space()
    print(f"标准action space包含 {len(standard_action_space)} 种action")
    
    # 分析不寻常的action
    unusual_actions = analyze_unusual_actions(df, standard_action_space)
    
    if not unusual_actions:
        print("未发现不寻常的action")
        return
    
    # 打印分析结果
    sorted_actions = print_unusual_actions_analysis(unusual_actions, standard_action_space)
    
    # 保存结果
    output_file = save_analysis_results(unusual_actions)
    
    print(f"\n分析完成！")
    print(f"发现 {len(unusual_actions)} 种不寻常的action")
    print(f"详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
