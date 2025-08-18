#!/usr/bin/env python3
"""
使用DuckDB进行真正的SQL查询分析WebPRM数据集中的task 397
特别关注thoughts和action的关系，以及chosen列中的决策过程
显示完整的thoughts内容
"""

import duckdb
from datasets import load_dataset
import re
import os
import json
from datetime import datetime
import pandas as pd

def main():
    # 连接到DuckDB数据库
    con = duckdb.connect(':memory:')
    
    # 从Hugging Face加载真实的WebPRM数据集
    print("✅ 正在从Hugging Face加载WebPRM数据集...")
    
    try:
        # 加载数据集（优先尝试 WebShepherd/，回退到 LangAGI-Lab/）
        try:
            dataset = load_dataset("WebShepherd/WebPRMCollection_preference_pair")
        except Exception:
            dataset = load_dataset("LangAGI-Lab/WebPRMCollection_preference_pair")
        print(f"✅ 成功加载数据集！包含 {len(dataset['test'])} 条记录")
        
        # 将数据集转换为DuckDB表
        df = dataset['test'].to_pandas()
        con.register('webprm', df)
        
        print(f"✅ 数据集已导入DuckDB，开始分析...")
        
    except Exception as e:
        print(f"❌ 加载数据集时出错: {e}")
        print("请确保已安装datasets库: pip install datasets")
        return
    
    # 显示数据集结构
    print("\n1. 数据集结构:")
    schema = con.execute("DESCRIBE webprm").fetchall()
    for field_name, field_type, null, key, default, extra in schema:
        print(f"   {field_name}: {field_type}")
    
    # 检查chosen列的结构
    print(f"\n2. Chosen列的结构分析:")
    chosen_sample = con.execute("""
        SELECT chosen, typeof(chosen) as chosen_type
        FROM webprm 
        LIMIT 1
    """).fetchall()
    
    for chosen, chosen_type in chosen_sample:
        print(f"   Chosen类型: {chosen_type}")
        print(f"   Chosen内容: {chosen}")
        print()
    
    # 检查task 397是否存在
    task_exists = con.execute("""
        SELECT COUNT(*) as count 
        FROM webprm 
        WHERE task_id = '397'
    """).fetchone()[0]
    
    if task_exists == 0:
        print(f"\n❌ 数据集中没有找到task 397")
        print("可用的task_id列表:")
        available_tasks = con.execute("""
            SELECT DISTINCT task_id, COUNT(*) as step_count
            FROM webprm 
            GROUP BY task_id 
            ORDER BY step_count DESC
            LIMIT 10
        """).fetchall()
        
        for task_id, step_count in available_tasks:
            print(f"   Task {task_id}: {step_count} 个步骤")
        return
    
    # 核心分析：任务397的Thoughts和Action关系分析
    print(f"\n3. 任务397的Thoughts和Action关系分析:")
    
    # 基础信息
    task_info = con.execute("""
        SELECT COUNT(DISTINCT step_id) as step_count, 
               ANY_VALUE(intent) as intent, 
               ANY_VALUE(website_name) as website_name
        FROM webprm 
        WHERE task_id = '397'
        GROUP BY task_id
        LIMIT 5
    """).fetchall()
    
    if task_info:
        step_count, intent, website = task_info[0]
        print(f"   任务397: {website} - {step_count} 个步骤")
        print(f"   目标: {intent}")
    
    # 分析每个step的thoughts和action关系
    print(f"\n4. 每个Step的完整Thoughts → Action → Chosen决策链:")
    
    # 获取每个step的详细信息
    step_details = con.execute("""
        SELECT 
            step_id,
            intent,
            website_name,
            current_url,
            thought_history,
            action_history,
            chosen,
            rejected
        FROM webprm 
        WHERE task_id = '397'
        ORDER BY CAST(step_id AS INTEGER)
    """).fetchall()
    
    for step_id, intent, website, current_url, thought_history, action_history, chosen, rejected in step_details:
        print(f"\n{'='*100}")
        print(f"Step {step_id}: {website} - {current_url}")
        print(f"{'='*100}")
        print(f"目标: {intent}")
        print()
        
        # 分析thoughts - 显示完整内容
        if thought_history and len(thought_history) > 0:
            print(f"💭 Thoughts ({len(thought_history)} 条):")
            print("-" * 50)
            for i, thought in enumerate(thought_history):
                print(f"Thought {i+1}:")
                print(f"{thought}")
                print()
        else:
            print("💭 Thoughts: 无")
            print()
        
        # 分析actions - 显示完整内容
        if action_history and len(action_history) > 0:
            print(f"🎯 Actions ({len(action_history)} 条):")
            print("-" * 50)
            for i, action in enumerate(action_history):
                print(f"Action {i+1}:")
                print(f"{action}")
                print()
        else:
            print("🎯 Actions: 无")
            print()
        
        # 分析chosen决策 - 显示完整内容
        if chosen:
            print(f"✅ Chosen决策:")
            print("-" * 50)
            if isinstance(chosen, dict):
                # 如果chosen是字典，显示其结构
                for key, value in chosen.items():
                    print(f"{key}:")
                    print(f"{value}")
                    print()
            else:
                print(f"{chosen}")
                print()
        else:
            print("✅ Chosen决策: 无")
            print()
        
        # 分析rejected决策 - 显示完整内容
        if rejected and len(rejected) > 0:
            print(f"❌ Rejected决策 ({len(rejected)} 条):")
            print("-" * 50)
            for i, reject in enumerate(rejected):
                print(f"Rejected {i+1}:")
                print(f"{reject}")
                print()
        else:
            print("❌ Rejected决策: 无")
            print()
    
    # 深入分析：Thoughts和Actions的对应关系
    print(f"\n5. Thoughts和Actions的对应关系分析:")
    print("=" * 100)
    
    for step_id, intent, website, current_url, thought_history, action_history, chosen, rejected in step_details:
        print(f"\nStep {step_id} 的决策过程:")
        print("-" * 50)
        
        if thought_history and action_history and len(thought_history) == len(action_history):
            print(f"💡 每个Thought对应一个Action:")
            for i in range(len(thought_history)):
                thought = thought_history[i]
                action = action_history[i]
                print(f"\nThought {i+1}:")
                print(f"{thought}")
                print(f"\nAction {i+1}:")
                print(f"{action}")
                print("-" * 30)
        elif thought_history and action_history:
            print(f"⚠️  Thoughts数量({len(thought_history)})与Actions数量({len(action_history)})不匹配")
        else:
            print(f"ℹ️  缺少Thoughts或Actions数据")
    
    # 分析chosen列中的决策模式
    print(f"\n6. Chosen列中的决策模式分析:")
    print("=" * 100)
    
    # 统计chosen列的类型分布
    chosen_types = con.execute("""
        SELECT 
            typeof(chosen) as chosen_type,
            COUNT(*) as count
        FROM webprm 
        WHERE task_id = '397'
        GROUP BY typeof(chosen)
    """).fetchall()
    
    print(f"Chosen列类型分布:")
    for chosen_type, count in chosen_types:
        print(f"  {chosen_type}: {count} 条记录")
    
    # 分析chosen列的内容模式
    print(f"\nChosen列内容模式:")
    for step_id, intent, website, current_url, thought_history, action_history, chosen, rejected in step_details:
        if chosen:
            print(f"\nStep {step_id}:")
            print("-" * 30)
            if isinstance(chosen, dict):
                print(f"决策类型: 结构化决策")
                for key, value in chosen.items():
                    print(f"{key}:")
                    print(f"{value}")
                    print()
            elif isinstance(chosen, str):
                print(f"决策类型: 文本决策")
                print(f"内容:")
                print(f"{chosen}")
                print()
            else:
                print(f"决策类型: {type(chosen)}")
                print(f"内容:")
                print(f"{chosen}")
                print()
    
    # ========== 新增：跨数据集提取 go_back 与 send_msg_to_user/send_msg 的出现位置与上下文 ==========
    print("\n7. 跨数据集扫描 go_back 与 send_msg_to_user/send_msg 的出现位置与上下文")
    print("=" * 100)

    def to_str(obj):
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)

    def find_occurrences_in_record(record: dict, row_index: int, dataset_name: str, split_name: str):
        occurrences = []
        action_regexes = {
            "go_back": [
                r"\bgo_back\b", r"\bgo\s*back\b", r"\bgo_backward\b", r"\bgo\s*backward\b"
            ],
            "send_msg": [
                r"send_msg_to_user\s*\(", r"send_msg\s*\("
            ],
        }

        def match_any(text: str, patterns: list[str]) -> bool:
            return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

        # 基本标识
        task_id = record.get("task_id")
        step_id = record.get("step_id")
        intent = record.get("intent") or record.get("instruction")
        current_url = record.get("current_url") or record.get("url")
        start_url = record.get("start_url")
        text_observation = record.get("text_observation")

        # 1) action_history（若存在）
        if isinstance(record.get("action_history"), list):
            for idx, act in enumerate(record["action_history"]):
                act_text = to_str(act)
                for action_key, patterns in action_regexes.items():
                    if match_any(act_text, patterns):
                        occurrences.append({
                            "dataset": dataset_name,
                            "split": split_name,
                            "row_index": row_index,
                            "task_id": task_id,
                            "step_id": step_id,
                            "occurrence_type": "action_history",
                            "position": idx,
                            "action_group": action_key,
                            "raw": act_text,
                            "intent": intent,
                            "current_url": current_url,
                            "start_url": start_url,
                            "text_observation_snippet": (text_observation[:400] if isinstance(text_observation, str) else None),
                        })

        # 2) chosen
        chosen = record.get("chosen")
        if chosen is not None:
            chosen_text = to_str(chosen)
            for action_key, patterns in action_regexes.items():
                if match_any(chosen_text, patterns):
                    occurrences.append({
                        "dataset": dataset_name,
                        "split": split_name,
                        "row_index": row_index,
                        "task_id": task_id,
                        "step_id": step_id,
                        "occurrence_type": "chosen",
                        "position": None,
                        "action_group": action_key,
                        "raw": chosen_text,
                        "intent": intent,
                        "current_url": current_url,
                        "start_url": start_url,
                        "text_observation_snippet": (text_observation[:400] if isinstance(text_observation, str) else None),
                    })

        # 3) rejected（列表）
        rejected = record.get("rejected")
        if isinstance(rejected, list):
            for ridx, rej in enumerate(rejected):
                rej_text = to_str(rej)
                for action_key, patterns in action_regexes.items():
                    if match_any(rej_text, patterns):
                        occurrences.append({
                            "dataset": dataset_name,
                            "split": split_name,
                            "row_index": row_index,
                            "task_id": task_id,
                            "step_id": step_id,
                            "occurrence_type": "rejected",
                            "position": ridx,
                            "action_group": action_key,
                            "raw": rej_text,
                            "intent": intent,
                            "current_url": current_url,
                            "start_url": start_url,
                            "text_observation_snippet": (text_observation[:400] if isinstance(text_observation, str) else None),
                        })

        return occurrences

    def extract_from_hf_dataset(hf_ds, dataset_name: str, split_name: str):
        results = []
        for i, rec in enumerate(hf_ds[split_name]):
            if isinstance(rec, dict):
                record = rec
            else:
                try:
                    record = rec.to_dict()
                except Exception:
                    record = dict(rec)
            results.extend(find_occurrences_in_record(record, i, dataset_name, split_name))
        return results

    # 7.1 WebPRMCollection_preference_pair（test split）
    ws_prm_results = extract_from_hf_dataset(dataset, "WebPRMCollection_preference_pair", "test")
    print(f"WebPRMCollection_preference_pair（test）共检出: {len(ws_prm_results)} 条相关出现")

    # 7.2 WebRewardBench（test split）
    reward_bench_results = []
    try:
        wrb = load_dataset("WebShepherd/WebRewardBench")
        reward_bench_results = extract_from_hf_dataset(wrb, "WebRewardBench", "test")
        print(f"WebRewardBench（test）共检出: {len(reward_bench_results)} 条相关出现")
    except Exception as e:
        print(f"⚠️ 加载 WebRewardBench 失败: {e}")

    # 合并并保存结果（便于后续监测对比）
    all_results = ws_prm_results + reward_bench_results
    out_dir = os.path.join("VAB-WebArena-Lite", "log_files")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"go_back_send_msg_occurrences_{timestamp}.json")
    csv_path = os.path.join(out_dir, f"go_back_send_msg_occurrences_{timestamp}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    if all_results:
        pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"已保存识别到的ID与上下文到:\n- {json_path}\n- {csv_path}")


    con.close()

if __name__ == "__main__":
    main()
