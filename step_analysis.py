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
    
    # ========== 新增：跨数据集提取 chosen 的 thought 与 action ==========
    print("\n7. 跨数据集提取 chosen 的 thought 与 action")
    print("=" * 100)

    def to_str(obj):
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)

    def find_occurrences_in_record(record: dict, row_index: int, dataset_name: str, split_name: str):
        occurrences = []

        def split_thought_action_from_text(text: str):
            if not isinstance(text, str):
                return None, None
            thought_pat = r"(?:^|\n)\s*(?:Thoughts?|Reasoning|思考|想法)\s*[:：]\s*(.*?)(?=\n\s*(?:Action|行动|动作)\s*[:：]|\Z)"
            action_pat = r"(?:^|\n)\s*(?:Action|行动|动作)\s*[:：]\s*(.*)"
            thought_match = re.search(thought_pat, text, flags=re.IGNORECASE | re.DOTALL)
            action_match = re.search(action_pat, text, flags=re.IGNORECASE | re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else None
            action = action_match.group(1).strip() if action_match else None
            # 如果没匹配到，尝试把文本当作 JSON 解析
            if (thought is None or action is None) and isinstance(text, str):
                try:
                    if text.strip().startswith(('{', '[')):
                        obj = json.loads(text)
                        if isinstance(obj, dict):
                            if thought is None:
                                thought = obj.get('thought') or obj.get('thoughts') or obj.get('reasoning')
                            if action is None:
                                a = obj.get('action') or obj.get('tool_action')
                                if a is not None and not isinstance(a, str):
                                    action = json.dumps(a, ensure_ascii=False)
                                elif isinstance(a, str):
                                    action = a
                except Exception:
                    pass
            return thought, action

        def extract_chosen_parts(chosen_obj):
            chosen_full_text = to_str(chosen_obj) if chosen_obj is not None else None
            chosen_thought_val = None
            chosen_action_val = None
            if isinstance(chosen_obj, dict):
                chosen_thought_val = chosen_obj.get('thought') or chosen_obj.get('thoughts') or chosen_obj.get('reasoning')
                a = chosen_obj.get('action') or chosen_obj.get('tool_action')
                if a is not None and not isinstance(a, str):
                    chosen_action_val = json.dumps(a, ensure_ascii=False)
                elif isinstance(a, str):
                    chosen_action_val = a
            elif isinstance(chosen_obj, list):
                # 取最后一个包含 action 的元素；否则取列表最后一个元素进行解析
                candidate = None
                for item in reversed(chosen_obj):
                    if isinstance(item, dict) and (item.get('action') or item.get('tool_action')):
                        candidate = item
                        break
                if candidate is None and len(chosen_obj) > 0:
                    candidate = chosen_obj[-1]
                if isinstance(candidate, dict):
                    chosen_thought_val = candidate.get('thought') or candidate.get('thoughts') or candidate.get('reasoning')
                    a = candidate.get('action') or candidate.get('tool_action')
                    if a is not None and not isinstance(a, str):
                        chosen_action_val = json.dumps(a, ensure_ascii=False)
                    elif isinstance(a, str):
                        chosen_action_val = a
                elif isinstance(candidate, str):
                    t, a = split_thought_action_from_text(candidate)
                    chosen_thought_val, chosen_action_val = t, a
            elif isinstance(chosen_obj, str):
                t, a = split_thought_action_from_text(chosen_obj)
                chosen_thought_val, chosen_action_val = t, a
            return chosen_full_text, chosen_thought_val, chosen_action_val

        def classify_action(action_text: str):
            if not action_text or not isinstance(action_text, str):
                return None
            text = action_text.strip()
            # 统一小写便于匹配
            low = text.lower()
            # send_msg 系列
            if re.search(r"\bsend_msg_to_user\s*\(", low) or re.search(r"\bsend_msg\s*\(", low):
                return "send_msg"
            # go back 系列
            if re.search(r"\bgo_back\b", low) or re.search(r"\bgo\s*back\b", low) or re.search(r"\bgo_backward\b", low) or re.search(r"\bgo\s*backward\b", low):
                return "go_back"
            # JSON 风格字符串里可能包含 id/name 字段
            try:
                if (low.startswith('{') and low.endswith('}')) or (low.startswith('[') and low.endswith(']')):
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        cand = (
                            obj.get('id') or obj.get('name') or obj.get('action') or obj.get('tool') or obj.get('tool_action')
                        )
                        if isinstance(cand, str):
                            c = cand.lower()
                            if 'send_msg' in c or 'send-msg' in c or 'send message' in c:
                                return 'send_msg'
                            if 'go_back' in c or 'go back' in c or 'backward' in c:
                                return 'go_back'
            except Exception:
                pass
            return None

        # 基本标识
        task_id = record.get("task_id")
        step_id = record.get("step_id")
        intent = record.get("intent") or record.get("instruction")
        current_url = record.get("current_url") or record.get("url")
        start_url = record.get("start_url")
        text_observation = record.get("text_observation")

        # 仅基于 chosen 生成一条汇总记录（按 step 粒度），并拆分 thought/action
        chosen_obj = record.get("chosen")
        if chosen_obj is not None:
            chosen_full_text, chosen_thought_val, chosen_action_val = extract_chosen_parts(chosen_obj)
            action_type = classify_action(chosen_action_val or chosen_full_text)
            # 仅当 chosen 的动作属于 send_msg 或 go_back 时记录
            if action_type in ("send_msg", "go_back"):
                    occurrences.append({
                        "dataset": dataset_name,
                        "split": split_name,
                        "row_index": row_index,
                        "task_id": task_id,
                        "step_id": step_id,
                        "occurrence_type": "chosen",
                    "chosen_full": chosen_full_text,
                    "chosen_thought": chosen_thought_val,
                    "chosen_action": chosen_action_val,
                    "chosen_action_type": action_type,
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

    # 基于 task_id + step_id 分组（按数据集与切分进一步区分），仅保留 chosen 动作为 send_msg 或 go_back 的样本
    groups = {}
    for r in all_results:
        key = (
            r.get("dataset"),
            r.get("split"),
            r.get("task_id"),
            r.get("step_id"),
        )
        if key not in groups:
            groups[key] = {
                "dataset": r.get("dataset"),
                "split": r.get("split"),
                "task_id": r.get("task_id"),
                "step_id": r.get("step_id"),
                "intent": r.get("intent"),
                "current_url": r.get("current_url"),
                "start_url": r.get("start_url"),
                "num_occurrences": 0,
                "chosen_full": None,
                "chosen_thought": None,
                "chosen_action": None,
                "chosen_action_type": None,
            }
        g = groups[key]
        g["num_occurrences"] += 1
        # 以 chosen 相关字段为主进行聚合，优先保留有值的字段
        cf = r.get("chosen_full")
        ct = r.get("chosen_thought")
        ca = r.get("chosen_action")
        cat = r.get("chosen_action_type")
        if cf and not g.get("chosen_full"):
            g["chosen_full"] = cf
        if ct and not g.get("chosen_thought"):
            g["chosen_thought"] = ct
        if ca and not g.get("chosen_action"):
            g["chosen_action"] = ca
        if cat and not g.get("chosen_action_type"):
            g["chosen_action_type"] = cat

    # 只保留确认为 send_msg / go_back 的分组
    grouped_list = [g for g in groups.values() if g.get("chosen_action_type") in ("send_msg", "go_back")]

    # 保存分组后的去冗余结果
    grouped_json_path = os.path.join(out_dir, f"go_back_send_msg_occurrences_grouped_{timestamp}.json")
    grouped_csv_path = os.path.join(out_dir, f"go_back_send_msg_occurrences_grouped_{timestamp}.csv")
    with open(grouped_json_path, "w", encoding="utf-8") as f:
        json.dump(grouped_list, f, ensure_ascii=False, indent=2)
    if grouped_list:
        pd.DataFrame(grouped_list).to_csv(grouped_csv_path, index=False)

    print(
        "已保存识别到的ID与上下文到:\n- {}\n- {}\n并保存按 task_id+step_id 分组后的去冗余结果到:\n- {}\n- {}".format(
            json_path, csv_path, grouped_json_path, grouped_csv_path
        )
    )

    # 8. 校验：每个 task 的最后一步是否为 send_msg
    print("\n8. 校验：每个 task 的最后一步是否为 send_msg")
    print("=" * 100)

    def try_int(v):
        try:
            return int(v)
        except Exception:
            return None

    def split_thought_action_from_text_check(text: str):
        if not isinstance(text, str):
            return None, None
        thought_pat = r"(?:^|\n)\s*(?:Thoughts?|Reasoning|思考|想法)\s*[:：]\s*(.*?)(?=\n\s*(?:Action|行动|动作)\s*[:：]|\Z)"
        action_pat = r"(?:^|\n)\s*(?:Action|行动|动作)\s*[:：]\s*(.*)"
        thought_match = re.search(thought_pat, text, flags=re.IGNORECASE | re.DOTALL)
        action_match = re.search(action_pat, text, flags=re.IGNORECASE | re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        if (thought is None or action is None) and isinstance(text, str):
            try:
                if text.strip().startswith(('{', '[')):
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        if thought is None:
                            thought = obj.get('thought') or obj.get('thoughts') or obj.get('reasoning')
                        if action is None:
                            a = obj.get('action') or obj.get('tool_action')
                            if a is not None and not isinstance(a, str):
                                action = json.dumps(a, ensure_ascii=False)
                            elif isinstance(a, str):
                                action = a
            except Exception:
                pass
        return thought, action

    def extract_thought_action_from_chosen_check(chosen_obj):
        thought_val = None
        action_val = None
        if chosen_obj is None:
            return thought_val, action_val
        if isinstance(chosen_obj, dict):
            thought_val = chosen_obj.get('thought') or chosen_obj.get('thoughts') or chosen_obj.get('reasoning')
            a = chosen_obj.get('action') or chosen_obj.get('tool_action')
            if a is not None and not isinstance(a, str):
                action_val = json.dumps(a, ensure_ascii=False)
            elif isinstance(a, str):
                action_val = a
        elif isinstance(chosen_obj, list):
            candidate = None
            for item in reversed(chosen_obj):
                if isinstance(item, dict) and (item.get('action') or item.get('tool_action')):
                    candidate = item
                    break
            if candidate is None and len(chosen_obj) > 0:
                candidate = chosen_obj[-1]
            if isinstance(candidate, dict):
                thought_val = candidate.get('thought') or candidate.get('thoughts') or candidate.get('reasoning')
                a = candidate.get('action') or candidate.get('tool_action')
                if a is not None and not isinstance(a, str):
                    action_val = json.dumps(a, ensure_ascii=False)
                elif isinstance(a, str):
                    action_val = a
            elif isinstance(candidate, str):
                t, a = split_thought_action_from_text_check(candidate)
                thought_val, action_val = t, a
        elif isinstance(chosen_obj, str):
            t, a = split_thought_action_from_text_check(chosen_obj)
            thought_val, action_val = t, a
        return thought_val, action_val

    def classify_action_for_check(action_text: str):
        if not action_text or not isinstance(action_text, str):
            return None
        text = action_text.strip()
        low = text.lower()
        if re.search(r"\bsend_msg_to_user\s*\(", low) or re.search(r"\bsend_msg\s*\(", low):
            return "send_msg"
        if re.search(r"\bgo_back\b", low) or re.search(r"\bgo\s*back\b", low) or re.search(r"\bgo_backward\b", low) or re.search(r"\bgo\s*backward\b", low):
            return "go_back"
        try:
            if (low.startswith('{') and low.endswith('}')) or (low.startswith('[') and low.endswith(']')):
                obj = json.loads(text)
                if isinstance(obj, dict):
                    cand = (
                        obj.get('id') or obj.get('name') or obj.get('action') or obj.get('tool') or obj.get('tool_action')
                    )
                    if isinstance(cand, str):
                        c = cand.lower()
                        if 'send_msg' in c or 'send-msg' in c or 'send message' in c:
                            return 'send_msg'
                        if 'go_back' in c or 'go back' in c or 'backward' in c:
                            return 'go_back'
        except Exception:
            pass
        return None

    def check_last_step_send_msg(hf_ds, dataset_name: str, split_name: str):
        if hf_ds is None or split_name not in hf_ds:
            print(f"跳过 {dataset_name}：split {split_name} 不可用")
            return
        # 先找每个 task 的最大 step_id
        task_to_max_step = {}
        for rec in hf_ds[split_name]:
            r = rec if isinstance(rec, dict) else rec.to_dict()
            tid = r.get('task_id')
            sid = try_int(r.get('step_id'))
            if tid is None or sid is None:
                continue
            if (tid not in task_to_max_step) or (sid > task_to_max_step[tid]):
                task_to_max_step[tid] = sid
        # 检查最后一步的 chosen
        ok = 0
        not_ok = []
        total = len(task_to_max_step)
        for rec in hf_ds[split_name]:
            r = rec if isinstance(rec, dict) else rec.to_dict()
            tid = r.get('task_id')
            sid = try_int(r.get('step_id'))
            if tid is None or sid is None:
                continue
            if task_to_max_step.get(tid) != sid:
                continue
            thought_text, act_text = extract_thought_action_from_chosen_check(r.get('chosen'))
            act_type = classify_action_for_check(act_text or r.get('chosen'))
            if act_type == 'send_msg':
                ok += 1
            else:
                not_ok.append({
                    'task_id': tid,
                    'last_step_id': sid,
                    'classified_action_type': act_type,
                    'action_text': act_text,
                    'thought_text': thought_text,
                })
        print(f"[{dataset_name}:{split_name}] 任务总数: {total}，最后一步为 send_msg 的任务: {ok}，不是的: {len(not_ok)}")
        if not_ok:
            # 仅打印前若干条，避免刷屏
            limit = 20
            print(f"示例（前{limit}项）非 send_msg 的最后一步任务，含 action_type 与 thought：")
            for item in not_ok[:limit]:
                print(f"  task {item['task_id']} last_step {item['last_step_id']} -> type: {item['classified_action_type']}")
                if item.get('thought_text'):
                    snippet = item['thought_text']
                    if isinstance(snippet, str) and len(snippet) > 400:
                        snippet = snippet[:400] + ' ...'
                    print("    thought:")
                    print(f"    {snippet}")
                if item.get('action_text'):
                    a_snippet = item['action_text']
                    if isinstance(a_snippet, str) and len(a_snippet) > 300:
                        a_snippet = a_snippet[:300] + ' ...'
                    print("    action_text:")
                    print(f"    {a_snippet}")
        return total, ok, not_ok

    # 对已加载的数据集进行校验
    try:
        check_last_step_send_msg(dataset, 'WebPRMCollection_preference_pair', 'test')
    except Exception as e:
        print(f"校验 WebPRMCollection_preference_pair 失败: {e}")
    try:
        # wrb 可能在前面加载失败
        wrb_available = 'wrb' in locals()
        if wrb_available:
            check_last_step_send_msg(wrb, 'WebRewardBench', 'test')
    except Exception as e:
        print(f"校验 WebRewardBench 失败: {e}")

    # 9. 打印 task_id = '1' 的最后一步 chosen 中的 thought 与 action
    print("\n9. 打印 task_id = '1' 最后一步的 thought 与 action")
    print("=" * 100)

    def try_get_dict(rec):
        return rec if isinstance(rec, dict) else rec.to_dict()

    def split_thought_action_from_text_for_print(text: str):
        if not isinstance(text, str):
            return None, None
        thought_pat = r"(?:^|\n)\s*(?:Thoughts?|Reasoning|思考|想法)\s*[:：]\s*(.*?)(?=\n\s*(?:Action|行动|动作)\s*[:：]|\Z)"
        action_pat = r"(?:^|\n)\s*(?:Action|行动|动作)\s*[:：]\s*(.*)"
        thought_match = re.search(thought_pat, text, flags=re.IGNORECASE | re.DOTALL)
        action_match = re.search(action_pat, text, flags=re.IGNORECASE | re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        if (thought is None or action is None) and isinstance(text, str):
            try:
                if text.strip().startswith(('{', '[')):
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        if thought is None:
                            thought = obj.get('thought') or obj.get('thoughts') or obj.get('reasoning')
                        if action is None:
                            a = obj.get('action') or obj.get('tool_action')
                            if a is not None and not isinstance(a, str):
                                action = json.dumps(a, ensure_ascii=False)
                            elif isinstance(a, str):
                                action = a
            except Exception:
                pass
        return thought, action

    def extract_thought_action_from_chosen_for_print(chosen_obj):
        chosen_full_text = None
        chosen_thought_val = None
        chosen_action_val = None
        if chosen_obj is None:
            return chosen_full_text, chosen_thought_val, chosen_action_val
        chosen_full_text = json.dumps(chosen_obj, ensure_ascii=False) if not isinstance(chosen_obj, str) else chosen_obj
        if isinstance(chosen_obj, dict):
            chosen_thought_val = chosen_obj.get('thought') or chosen_obj.get('thoughts') or chosen_obj.get('reasoning')
            a = chosen_obj.get('action') or chosen_obj.get('tool_action')
            if a is not None and not isinstance(a, str):
                chosen_action_val = json.dumps(a, ensure_ascii=False)
            elif isinstance(a, str):
                chosen_action_val = a
        elif isinstance(chosen_obj, str):
            t, a = split_thought_action_from_text_for_print(chosen_obj)
            chosen_thought_val, chosen_action_val = t, a
        return chosen_full_text, chosen_thought_val, chosen_action_val


    con.close()

if __name__ == "__main__":
    main()
