#!/usr/bin/env python3
"""
探索WebRewardBench数据集的真实结构，找到go_back和send_msg_to_user动作
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import json
from collections import defaultdict

def explore_dataset():
    """探索数据集结构"""
    print("正在下载WebRewardBench数据集...")
    
    try:
        dataset = load_dataset("WebShepherd/WebRewardBench")
        print(f"数据集加载成功！包含 {len(dataset['test'])} 个样本")
        
        # 转换为pandas DataFrame
        df = dataset['test'].to_pandas()
        
        # 查看前几个样本的详细结构
        print("\n=== 查看前3个样本的详细结构 ===")
        for i in range(min(3, len(df))):
            print(f"\n--- 样本 {i+1} ---")
            row = df.iloc[i]
            print(f"Source: {row['source_name']}")
            print(f"Intent: {row['intent'][:100]}...")
            print(f"Task ID: {row['task_id']}")
            print(f"Step ID: {row['step_id']}")
            
            # 查看action_history
            print(f"Action History (类型: {type(row['action_history'])})")
            if isinstance(row['action_history'], np.ndarray):
                for j, action in enumerate(row['action_history']):
                    print(f"  {j+1}. {action}")
            elif isinstance(row['action_history'], list):
                for j, action in enumerate(row['action_history']):
                    print(f"  {j+1}. {action}")
            else:
                print(f"  {row['action_history']}")
            
            # 查看chosen
            print(f"Chosen (类型: {type(row['chosen'])})")
            if isinstance(row['chosen'], np.ndarray) and len(row['chosen']) > 0:
                chosen = row['chosen'][0]
                if isinstance(chosen, dict):
                    for key, value in chosen.items():
                        print(f"    {key}: {value}")
            elif isinstance(row['chosen'], list) and len(row['chosen']) > 0:
                chosen = row['chosen'][0]
                if isinstance(chosen, dict):
                    for key, value in chosen.items():
                        print(f"    {key}: {value}")
            else:
                print(f"  {row['chosen']}")
            
            # 查看rejected
            print(f"Rejected (类型: {type(row['rejected'])})")
            if isinstance(row['rejected'], np.ndarray) and len(row['rejected']) > 0:
                for j, rejected in enumerate(row['rejected']):
                    print(f"  {j+1}. {rejected}")
            elif isinstance(row['rejected'], list) and len(row['rejected']) > 0:
                for j, rejected in enumerate(row['rejected']):
                    print(f"  {j+1}. {rejected}")
            else:
                print(f"  {row['rejected']}")
        
        # 搜索所有可能的动作类型
        print("\n=== 搜索所有可能的动作类型 ===")
        all_actions = set()
        action_patterns = defaultdict(int)
        
        for idx, row in df.iterrows():
            # 搜索action_history
            if isinstance(row['action_history'], np.ndarray):
                for action in row['action_history']:
                    if isinstance(action, str):
                        all_actions.add(action)
                        # 提取动作类型（第一个括号前的部分）
                        if '(' in action:
                            action_type = action.split('(')[0].strip()
                            action_patterns[action_type] += 1
                        else:
                            action_patterns[action] += 1
            elif isinstance(row['action_history'], list):
                for action in row['action_history']:
                    if isinstance(action, str):
                        all_actions.add(action)
                        if '(' in action:
                            action_type = action.split('(')[0].strip()
                            action_patterns[action_type] += 1
                        else:
                            action_patterns[action] += 1
            
            # 搜索chosen中的action
            if isinstance(row['chosen'], np.ndarray) and len(row['chosen']) > 0:
                chosen = row['chosen'][0]
                if isinstance(chosen, dict) and 'action' in chosen:
                    action = chosen['action']
                    if isinstance(action, str):
                        all_actions.add(action)
                        if '(' in action:
                            action_type = action.split('(')[0].strip()
                            action_patterns[action_type] += 1
                        else:
                            action_patterns[action] += 1
            elif isinstance(row['chosen'], list) and len(row['chosen']) > 0:
                chosen = row['chosen'][0]
                if isinstance(chosen, dict) and 'action' in chosen:
                    action = chosen['action']
                    if isinstance(action, str):
                        all_actions.add(action)
                        if '(' in action:
                            action_type = action.split('(')[0].strip()
                            action_patterns[action_type] += 1
                        else:
                            action_patterns[action] += 1
            
            # 搜索rejected中的action
            if isinstance(row['rejected'], np.ndarray) and len(row['rejected']) > 0:
                for rejected in row['rejected']:
                    if isinstance(rejected, dict) and 'action' in rejected:
                        action = rejected['action']
                        if isinstance(action, str):
                            all_actions.add(action)
                            if '(' in action:
                                action_type = action.split('(')[0].strip()
                                action_patterns[action_type] += 1
                            else:
                                action_patterns[action] += 1
            elif isinstance(row['rejected'], list):
                for rejected in row['rejected']:
                    if isinstance(rejected, dict) and 'action' in rejected:
                        action = rejected['action']
                        if isinstance(action, str):
                            all_actions.add(action)
                            if '(' in action:
                                action_type = action.split('(')[0].strip()
                                action_patterns[action_type] += 1
                            else:
                                action_patterns[action] += 1
        
        print(f"\n找到 {len(all_actions)} 个不同的动作")
        print("\n动作类型统计 (前20个):")
        sorted_patterns = sorted(action_patterns.items(), key=lambda x: x[1], reverse=True)
        for action_type, count in sorted_patterns[:20]:
            print(f"  {action_type}: {count}")
        
        # 搜索包含特定关键词的动作
        print("\n=== 搜索包含特定关键词的动作 ===")
        keywords = ['go', 'back', 'send', 'msg', 'message', 'user', 'stop']
        
        for keyword in keywords:
            matching_actions = []
            for action in all_actions:
                if keyword.lower() in action.lower():
                    matching_actions.append(action)
            
            if matching_actions:
                print(f"\n包含 '{keyword}' 的动作:")
                for action in matching_actions[:10]:  # 只显示前10个
                    print(f"  {action}")
        
        # 按source_name分析动作分布
        print("\n=== 按source_name分析动作分布 ===")
        source_actions = defaultdict(lambda: defaultdict(int))
        
        for idx, row in df.iterrows():
            source_name = row['source_name']
            
            # 统计action_history中的动作
            if isinstance(row['action_history'], np.ndarray):
                for action in row['action_history']:
                    if isinstance(action, str):
                        if '(' in action:
                            action_type = action.split('(')[0].strip()
                        else:
                            action_type = action
                        source_actions[source_name][action_type] += 1
            elif isinstance(row['action_history'], list):
                for action in row['action_history']:
                    if isinstance(action, str):
                        if '(' in action:
                            action_type = action.split('(')[0].strip()
                        else:
                            action_type = action
                        source_actions[source_name][action_type] += 1
        
        for source_name in source_actions:
            print(f"\n{source_name} 的动作分布 (前10个):")
            sorted_actions = sorted(source_actions[source_name].items(), key=lambda x: x[1], reverse=True)
            for action_type, count in sorted_actions[:10]:
                print(f"  {action_type}: {count}")
        
        return df
        
    except Exception as e:
        print(f"探索数据集时出错: {e}")
        return None

def analyze_send_msg_to_user_positions(df):
    """分析send_msg_to_user动作在action_history中的位置"""
    print("\n=== 分析send_msg_to_user动作位置 ===")
    
    send_msg_samples = []
    total_send_msg = 0
    
    for idx, row in df.iterrows():
        source_name = row['source_name']
        task_id = row['task_id']
        step_id = row['step_id']
        intent = row['intent'][:100] + '...' if len(row['intent']) > 100 else row['intent']
        
        # 检查action_history中是否有send_msg_to_user
        if isinstance(row['action_history'], (np.ndarray, list)):
            action_history = row['action_history'] if isinstance(row['action_history'], list) else row['action_history'].tolist()
            
            for action_idx, action in enumerate(action_history):
                if isinstance(action, str) and 'send_msg_to_user' in action.lower():
                    total_send_msg += 1
                    
                    # 记录详细信息
                    sample_info = {
                        'source_name': source_name,
                        'task_id': task_id,
                        'step_id': step_id,
                        'intent': intent,
                        'action': action,
                        'action_position': action_idx + 1,  # 从1开始计数
                        'total_actions': len(action_history),
                        'is_last': action_idx == len(action_history) - 1,
                        'is_first': action_idx == 0,
                        'row_idx': idx
                    }
                    send_msg_samples.append(sample_info)
    
    print(f"总共找到 {total_send_msg} 个send_msg_to_user动作")
    print(f"分布在 {len(send_msg_samples)} 个样本中")
    
    if send_msg_samples:
        # 按位置统计
        position_stats = defaultdict(int)
        last_position_count = 0
        first_position_count = 0
        middle_position_count = 0
        
        for sample in send_msg_samples:
            position = sample['action_position']
            position_stats[position] += 1
            
            if sample['is_last']:
                last_position_count += 1
            elif sample['is_first']:
                first_position_count += 1
            else:
                middle_position_count += 1
        
        print(f"\n位置统计:")
        print(f"  在最后位置: {last_position_count} 个 ({last_position_count/total_send_msg*100:.1f}%)")
        print(f"  在第一个位置: {first_position_count} 个 ({first_position_count/total_send_msg*100:.1f}%)")
        print(f"  在中间位置: {middle_position_count} 个 ({middle_position_count/total_send_msg*100:.1f}%)")
        
        print(f"\n具体位置分布:")
        sorted_positions = sorted(position_stats.items())
        for pos, count in sorted_positions:
            print(f"  第{pos}位: {count} 个")
        
        # 显示一些具体示例
        print(f"\n=== 具体示例 ===")
        
        # 显示在最后位置的示例
        last_position_samples = [s for s in send_msg_samples if s['is_last']]
        if last_position_samples:
            print(f"\n在最后位置的示例 (前3个):")
            for i, sample in enumerate(last_position_samples[:3]):
                print(f"  {i+1}. {sample['source_name']} - Task {sample['task_id']}-{sample['step_id']}")
                print(f"     Intent: {sample['intent']}")
                print(f"     Action: {sample['action']}")
                print(f"     位置: 第{sample['action_position']}位 (共{sample['total_actions']}个动作)")
        
        # 显示不在最后位置的示例
        not_last_samples = [s for s in send_msg_samples if not s['is_last']]
        if not_last_samples:
            print(f"\n不在最后位置的示例 (前3个):")
            for i, sample in enumerate(not_last_samples[:3]):
                print(f"  {i+1}. {sample['source_name']} - Task {sample['task_id']}-{sample['step_id']}")
                print(f"     Intent: {sample['intent']}")
                print(f"     Action: {sample['action']}")
                print(f"     位置: 第{sample['action_position']}位 (共{sample['total_actions']}个动作)")
        
        # 按source_name统计
        print(f"\n=== 按source_name统计 ===")
        source_stats = defaultdict(lambda: {'total': 0, 'last': 0, 'not_last': 0})
        
        for sample in send_msg_samples:
            source = sample['source_name']
            source_stats[source]['total'] += 1
            if sample['is_last']:
                source_stats[source]['last'] += 1
            else:
                source_stats[source]['not_last'] += 1
        
        for source_name, stats in source_stats.items():
            print(f"\n{source_name}:")
            print(f"  总数: {stats['total']}")
            print(f"  在最后: {stats['last']} ({stats['last']/stats['total']*100:.1f}%)")
            print(f"  不在最后: {stats['not_last']} ({stats['not_last']/stats['total']*100:.1f}%)")
    
    return send_msg_samples

def main():
    """主函数"""
    print("开始探索WebRewardBench数据集结构...")
    
    df = explore_dataset()
    
    if df is not None:
        # 分析send_msg_to_user的位置
        send_msg_samples = analyze_send_msg_to_user_positions(df)
        print("\n探索完成！")
    else:
        print("数据集探索失败")

if __name__ == "__main__":
    main()
