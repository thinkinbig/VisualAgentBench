#!/usr/bin/env python3
"""
分析WebPRMCollection_preference_pair数据集中go_back动作的正确性
判断每一步go_back是否都是正确的步骤选择，而不是action操作错误进入了错误页面
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from collections import defaultdict
import json
import re
import os
import time
import logging
from typing import Any, Dict, List, Optional
from tqdm import tqdm

try:
    # OpenAI SDK (requires OPENAI_API_KEY)
    from openai import OpenAI  # type: ignore
    _openai_available = True
except Exception:
    _openai_available = False

def load_and_sort_dataset():
    """加载数据集并按task_id和step_id排序"""
    print("正在下载WebPRMCollection_preference_pair数据集...")
    
    try:
        dataset = load_dataset("WebShepherd/WebPRMCollection_preference_pair")
        print(f"数据集加载成功！包含 {len(dataset['test'])} 个样本")
        
        # 转换为pandas DataFrame
        df = dataset['test'].to_pandas()
        
        # 按task_id和step_id排序
        df = df.sort_values(by=['task_id', 'step_id']).reset_index(drop=True)
        
        # 添加has_prev_step标记
        df['has_prev_step'] = df['task_id'].eq(df['task_id'].shift(1))
        
        print(f"数据已按task_id, step_id排序，共{len(df)}行")
        print(f"包含{df['task_id'].nunique()}个不同的task_id")
        
        return df
        
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return None

def analyze_goback_correctness(df):
    """分析go_back动作的正确性"""
    print("\n=== 分析go_back动作的正确性 ===")
    
    go_back_samples = []
    
    # 只分析chosen中的go_back动作
    print("只分析chosen中的go_back动作...")
    for idx, row in df.iterrows():
        source_name = row.get('website_name', 'unknown')
        task_id = row['task_id']
        step_id = row['step_id']
        intent = row['intent']
        has_prev_step = bool(row.get('has_prev_step', False))
        
        # 只检查chosen中的go_back
        if isinstance(row.get('chosen'), dict) and 'action' in row['chosen']:
            action = row['chosen']['action']
            if isinstance(action, str) and 'go_back' in action.lower():
                go_back_samples.append({
                    'source_name': source_name,
                    'task_id': task_id,
                    'step_id': step_id,
                    'intent': intent[:200] + '...' if len(intent) > 200 else intent,
                    'action': action,
                    'row_idx': idx,
                    'context': 'chosen',
                    'has_prev_step': has_prev_step,
                    'current_url': row.get('current_url', ''),
                    'start_url': row.get('start_url', ''),
                    'thought_history': row.get('thought_history', []),
                    'checklist': row.get('checklist', ''),
                    'checklist_target_list': row.get('checklist_target_list', []),
                    'chosen_thought': row['chosen'].get('thought', '')
                })
    
    print(f"找到 {len(go_back_samples)} 个go_back动作")
    
    # 按context分组统计
    context_stats = defaultdict(int)
    for sample in go_back_samples:
        context_stats[sample['context']] += 1
    
    print(f"\ngo_back动作按context分布:")
    for context, count in context_stats.items():
        print(f"  {context}: {count}")
    
    return go_back_samples

def analyze_goback_context(df, go_back_samples):
    """分析go_back动作的上下文，判断是否正确"""
    print(f"\n=== 分析go_back动作的上下文正确性 ===")
    
    go_back_analysis = []
    
    # 使用tqdm显示进度条
    for sample in tqdm(go_back_samples, desc="分析go_back动作", unit="个"):
        row_idx = sample['row_idx']
        task_id = sample['task_id']
        step_id = sample['step_id']
        
        # 获取当前步骤的上下文信息
        current_row = df.iloc[row_idx]
        
        # 获取前一步骤的信息（如果存在）
        prev_step_info = None
        if sample['has_prev_step'] and row_idx > 0:
            prev_row = df.iloc[row_idx - 1]
            if prev_row['task_id'] == task_id:
                prev_step_info = {
                    'step_id': prev_row['step_id'],
                    'action_history': prev_row.get('action_history', []),
                    'current_url': prev_row.get('current_url', ''),
                    'chosen_action': prev_row.get('chosen', {}).get('action', '') if isinstance(prev_row.get('chosen'), dict) else ''
                }
        
        # 分析go_back的合理性
        analysis = {
            'source_name': sample['source_name'],
            'task_id': task_id,
            'step_id': step_id,
            'context': sample['context'],
            'action': sample['action'],
            'intent': sample['intent'],
            'has_prev_step': sample['has_prev_step'],
            'current_url': sample['current_url'],
            'start_url': sample['start_url'],
            'prev_step_info': prev_step_info,
            'thought_history': sample.get('thought_history', []),
            'checklist': sample.get('checklist', ''),
            'chosen_thought': sample.get('chosen_thought', ''),
            'rejected_thought': sample.get('rejected_thought', '')
        }
        
        # 不再需要reasonableness字段

        # 可选：使用LLM作为裁判进行判断
        use_llm = os.getenv("USE_LLM_AS_JUDGE", "1") == "1"
        api_key_present = bool(os.getenv("OPENAI_API_KEY"))
        if use_llm and _openai_available and api_key_present:
            try:
                task_ctx = build_task_context(df, task_id)
                llm_result = judge_goback_with_llm(
                    task_context=task_ctx,
                    focus_step_id=step_id,
                    proposed_action=analysis['action'],
                )
                analysis['llm_judgement'] = llm_result
            except Exception as e:
                analysis['llm_judgement'] = {
                    'error': f'LLM judging failed: {e}'
                }
        else:
            if not _openai_available:
                analysis['llm_judgement'] = {'skipped': 'openai sdk not available'}
            elif not api_key_present:
                analysis['llm_judgement'] = {'skipped': 'OPENAI_API_KEY not set'}
            else:
                analysis['llm_judgement'] = {'skipped': 'USE_LLM_AS_JUDGE disabled'}
        
        go_back_analysis.append(analysis)
    
    return go_back_analysis

def _truncate_text(text: str, max_len: int = 400) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."

def _safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return []

def build_task_context(df: pd.DataFrame, task_id: str) -> Dict[str, Any]:
    """构造给LLM的完整任务上下文（按该task_id的所有步骤）。"""
    task_df = df[df['task_id'] == task_id].sort_values(by=['step_id']).reset_index(drop=True)

    steps: List[Dict[str, Any]] = []
    for _, r in task_df.iterrows():
        chosen = r.get('chosen', {}) if isinstance(r.get('chosen'), dict) else {}
        rejected = _safe_list(r.get('rejected'))
        rejected_actions = []
        for rej in rejected:
            if isinstance(rej, dict) and 'action' in rej:
                rejected_actions.append({'action': rej.get('action', ''), 'thought': _truncate_text(rej.get('thought', ''))})

        steps.append({
            'step_id': r['step_id'],
            'current_url': r.get('current_url', ''),
            'start_url': r.get('start_url', ''),
            'thought_history': [_truncate_text(t) for t in _safe_list(r.get('thought_history'))],
            'action_history': _safe_list(r.get('action_history')),
            'chosen': {
                'action': chosen.get('action', ''),
                'thought': _truncate_text(chosen.get('thought', '')),
            },
            'rejected': rejected_actions,
            'checklist': r.get('checklist', ''),
            'checklist_target_list': r.get('checklist_target_list', []),
            'text_observation': _truncate_text(r.get('text_observation', ''), max_len=300),
        })

    task_context = {
        'task_id': task_id,
        'intent': task_df.iloc[0]['intent'] if len(task_df) > 0 else '',
        'start_url': task_df.iloc[0].get('start_url', '') if len(task_df) > 0 else '',
        'website_name': task_df.iloc[0].get('website_name', '') if len(task_df) > 0 else '',
        'steps': steps,
    }
    return task_context

def judge_goback_with_llm(task_context: Dict[str, Any], focus_step_id: Any, proposed_action: str, model: str = None) -> Dict[str, Any]:
    """调用OpenAI作为裁判，基于完整任务流程判断go_back是否合理。

    返回字段:
      - is_reasonable: bool - go_back是否合理
      - category: str - 分类 (strategic_return | error_recovery | unnecessary | wrong_path)
      - confidence: float [0,1] - 置信度
      - reason: str - 详细理由
      - flow_analysis: str - 流程分析
    """
    if model is None:
        model = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")

    if not _openai_available:
        raise RuntimeError("OpenAI SDK not available")

    client = OpenAI()

    system_prompt = (
        "You are an expert web navigation analyst. Analyze the COMPLETE task flow to judge whether "
        "a 'go_back' action at the focus step is reasonable. Consider:\n"
        "1. TASK CONTEXT: What is the user's goal and current progress?\n"
        "2. FLOW ANALYSIS: What led to the current page? Was it a strategic choice or navigation error?\n"
        "3. GO_BACK PURPOSE: Is it to return to a better path, recover from an error, or unnecessary?\n"
        "4. TASK COMPLETION: Does go_back help or hinder task completion?\n\n"
        "Return JSON with: is_reasonable (bool), category (strategic_return|error_recovery|unnecessary|wrong_path), "
        "confidence (0-1), reason (string), flow_analysis (string)"
    )

    user_payload = {
        'task': {
            'task_id': task_context.get('task_id', ''),
            'intent': task_context.get('intent', ''),
            'website_name': task_context.get('website_name', ''),
            'start_url': task_context.get('start_url', ''),
        },
        'focus_step_id': focus_step_id,
        'proposed_action': proposed_action,
        'steps': task_context.get('steps', []),
        'instruction': (
            "Analyze the COMPLETE task flow. At focus_step_id, the user wants to execute 'go_back'. "
            "Look at ALL previous steps, actions, and thoughts to understand:\n"
            "1. Why are they on the current page?\n"
            "2. Is go_back reasonable in this context?\n"
            "3. What would be the consequences?\n"
            "Provide detailed flow analysis and reasoning."
        ),
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content if resp and resp.choices else "{}"
    try:
        data = json.loads(content)
        # 归一化输出
        return {
            'is_reasonable': bool(data.get('is_reasonable', False)),
            'category': data.get('category', 'unknown'),
            'confidence': float(data.get('confidence', 0.0)),
            'reason': data.get('reason', ''),
            'flow_analysis': data.get('flow_analysis', ''),
            'model': model,
        }
    except Exception:
        return {
            'raw': content,
            'parse_error': True,
            'model': model,
        }

# 删除整个analyze_single_goback函数，不再需要

def print_analysis_results(go_back_analysis):
    """打印分析结果"""
    logging.info("=== go_back动作合理性分析结果 ===")
    
    # 按LLM判断结果分组
    llm_reasonable = [a for a in go_back_analysis if a.get('llm_judgement', {}).get('is_reasonable', False)]
    llm_unreasonable = [a for a in go_back_analysis if a.get('llm_judgement', {}).get('is_reasonable', False) == False]
    llm_no_judgement = [a for a in go_back_analysis if 'llm_judgement' not in a or 'error' in a.get('llm_judgement', {})]
    
    logging.info(f"LLM认为合理: {len(llm_reasonable)}个")
    logging.info(f"LLM认为不合理: {len(llm_unreasonable)}个")
    logging.info(f"LLM未判断: {len(llm_no_judgement)}个")
    
    # 记录详细示例到日志
    logging.info("=== LLM认为合理的go_back示例 ===")
    for i, analysis in enumerate(llm_reasonable[:3]):
        logging.info(f"{i+1}. Source: {analysis['source_name']}")
        logging.info(f"   Task {analysis['task_id']}-{analysis['step_id']}")
        logging.info(f"   Intent: {analysis['intent'][:100]}...")
        logging.info(f"   Action: {analysis['action']}")
        # 显示LLM判断结果
        if 'llm_judgement' in analysis and 'error' not in analysis['llm_judgement']:
            llm = analysis['llm_judgement']
            logging.info(f"   LLM判断: {llm.get('is_reasonable', 'N/A')}")
            logging.info(f"   LLM分类: {llm.get('category', 'N/A')}")
            logging.info(f"   LLM置信度: {llm.get('confidence', 'N/A')}")
            logging.info(f"   LLM理由: {llm.get('reason', 'N/A')}")
            logging.info(f"   流程分析: {llm.get('flow_analysis', 'N/A')}")
    
    logging.info("=== LLM认为不合理的go_back示例 ===")
    for i, analysis in enumerate(llm_unreasonable[:3]):
        logging.info(f"{i+1}. Source: {analysis['source_name']}")
        logging.info(f"   Task {analysis['task_id']}-{analysis['step_id']}")
        logging.info(f"   Intent: {analysis['intent'][:100]}...")
        logging.info(f"   Action: {analysis['action']}")
        # 显示LLM判断结果
        if 'llm_judgement' in analysis and 'error' not in analysis['llm_judgement']:
            llm = analysis['llm_judgement']
            logging.info(f"   LLM判断: {llm.get('is_reasonable', 'N/A')}")
            logging.info(f"   LLM分类: {llm.get('category', 'N/A')}")
            logging.info(f"   LLM置信度: {llm.get('confidence', 'N/A')}")
            logging.info(f"   LLM理由: {llm.get('reason', 'N/A')}")
            logging.info(f"   流程分析: {llm.get('flow_analysis', 'N/A')}")
    
    # 按source_name统计LLM判断结果
    source_stats = defaultdict(lambda: {'total': 0, 'reasonable': 0, 'unreasonable': 0, 'no_judgement': 0})
    for analysis in go_back_analysis:
        source = analysis['source_name']
        source_stats[source]['total'] += 1
        
        if 'llm_judgement' in analysis and 'error' not in analysis['llm_judgement']:
            if analysis['llm_judgement'].get('is_reasonable', False):
                source_stats[source]['reasonable'] += 1
            else:
                source_stats[source]['unreasonable'] += 1
        else:
            source_stats[source]['no_judgement'] += 1
    
    # 新增：显示LLM判断的详细统计
    logging.info("=== LLM判断结果统计 ===")
    llm_stats = {
        'total_judged': 0,
        'reasonable': 0,
        'unreasonable': 0,
        'categories': defaultdict(int),
        'avg_confidence': 0.0
    }
    
    confidence_scores = []
    low_confidence_samples = []  # 置信度低于0.8的样本
    
    for analysis in go_back_analysis:
        if 'llm_judgement' in analysis and 'error' not in analysis['llm_judgement']:
            llm = analysis['llm_judgement']
            llm_stats['total_judged'] += 1
            
            if llm.get('is_reasonable', False):
                llm_stats['reasonable'] += 1
            else:
                llm_stats['unreasonable'] += 1
            
            category = llm.get('category', 'unknown')
            llm_stats['categories'][category] += 1
            
            confidence = llm.get('confidence', 0.0)
            if confidence > 0:
                confidence_scores.append(confidence)
                # 收集置信度低于0.8的样本
                if confidence < 0.8:
                    low_confidence_samples.append({
                        'analysis': analysis,
                        'confidence': confidence
                    })
    
    if confidence_scores:
        llm_stats['avg_confidence'] = sum(confidence_scores) / len(confidence_scores)
    
    logging.info(f"LLM判断总数: {llm_stats['total_judged']}")
    logging.info(f"LLM认为合理: {llm_stats['reasonable']}")
    logging.info(f"LLM认为不合理: {llm_stats['unreasonable']}")
    logging.info(f"平均置信度: {llm_stats['avg_confidence']:.2f}")
    logging.info(f"低置信度样本(<0.8): {len(low_confidence_samples)}个")
    
    logging.info("LLM分类统计:")
    for category, count in llm_stats['categories'].items():
        logging.info(f"  {category}: {count}")
    
    # 特别列出置信度低于0.8的样本供详细分析
    logging.info("=== 置信度低于0.8的样本详细分析 ===")
    if low_confidence_samples:
        # 按置信度排序，最低的在前
        low_confidence_samples.sort(key=lambda x: x['confidence'])
        
        for i, sample_data in enumerate(low_confidence_samples):
            analysis = sample_data['analysis']
            confidence = sample_data['confidence']
            llm = analysis['llm_judgement']
            
            logging.info(f"\n【低置信度样本 {i+1}】置信度: {confidence}")
            logging.info(f"Source: {analysis['source_name']}")
            logging.info(f"Task: {analysis['task_id']}-{analysis['step_id']}")
            logging.info(f"Intent: {analysis['intent']}")
            logging.info(f"Action: {analysis['action']}")
            logging.info(f"Current URL: {analysis.get('current_url', 'N/A')}")
            logging.info(f"Start URL: {analysis.get('start_url', 'N/A')}")
            logging.info(f"Has Prev Step: {analysis.get('has_prev_step', 'N/A')}")
            logging.info(f"LLM判断: {'合理' if llm.get('is_reasonable') else '不合理'}")
            logging.info(f"LLM分类: {llm.get('category', 'N/A')}")
            logging.info(f"LLM理由: {llm.get('reason', 'N/A')}")
            logging.info(f"流程分析: {llm.get('flow_analysis', 'N/A')}")
            
            # 显示思考历史（如果有的话）
            thought_history = analysis.get('thought_history', [])
            if thought_history:
                logging.info("思考历史:")
                for j, thought in enumerate(thought_history[-3:]):  # 只显示最后3个
                    logging.info(f"  {j+1}. {thought[:200]}...")
            
            logging.info("=" * 80)
    else:
        logging.info("所有样本的置信度都>=0.8")
    
    # 控制台显示摘要信息
    print(f"\n=== 分析结果摘要 ===")
    print(f"LLM判断总数: {llm_stats['total_judged']}")
    print(f"LLM认为合理: {llm_stats['reasonable']}")
    print(f"LLM认为不合理: {llm_stats['unreasonable']}")
    print(f"平均置信度: {llm_stats['avg_confidence']:.2f}")
    print(f"低置信度样本(<0.8): {len(low_confidence_samples)}个")
    
    if low_confidence_samples:
        print(f"\n置信度最低的3个样本:")
        for i, sample_data in enumerate(low_confidence_samples[:3]):
            analysis = sample_data['analysis']
            confidence = sample_data['confidence']
            llm = analysis['llm_judgement']
            print(f"  {i+1}. Task {analysis['task_id']}-{analysis['step_id']} ({analysis['source_name']})")
            print(f"     置信度: {confidence}, 判断: {'合理' if llm.get('is_reasonable') else '不合理'}")
            print(f"     理由: {llm.get('reason', 'N/A')[:100]}...")
    
    print(f"\n详细分析结果已保存到日志文件中")

def setup_logging():
    """设置logging配置，同时输出到控制台和文件"""
    # 创建logs目录
    os.makedirs('analysis/scripts/logs', exist_ok=True)
    
    # 生成日志文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f'analysis/scripts/logs/goback_analysis_{timestamp}.log'
    
    # 配置logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 输出到控制台
        ]
    )
    
    logging.info(f"日志文件: {log_filename}")
    return log_filename

def main():
    """主函数"""
    # 设置logging
    log_filename = setup_logging()
    
    logging.info("开始分析WebPRMCollection_preference_pair数据集中的go_back动作正确性...")
    
    # 加载并排序数据集
    df = load_and_sort_dataset()
    if df is None:
        return
    
    # 分析go_back动作
    go_back_samples = analyze_goback_correctness(df)
    
    if not go_back_samples:
        logging.info("未找到go_back动作")
        return
    
    # 分析go_back的上下文正确性
    go_back_analysis = analyze_goback_context(df, go_back_samples)
    
    # 打印分析结果
    print_analysis_results(go_back_analysis)
    
    logging.info(f"分析完成！总共分析了 {len(go_back_samples)} 个go_back动作")
    logging.info(f"详细日志已保存到: {log_filename}")

if __name__ == "__main__":
    main()
