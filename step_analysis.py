#!/usr/bin/env python3
"""
使用DuckDB进行真正的SQL查询分析WebPRM数据集中的task 397
"""

import duckdb
from datasets import load_dataset

def main():
    # 连接到DuckDB数据库
    con = duckdb.connect(':memory:')
    
    # 从Hugging Face加载真实的WebPRM数据集
    print("✅ 正在从Hugging Face加载WebPRM数据集...")
    
    try:
        # 加载数据集
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
    
    # 打印task 397的action history示例
    print(f"\n2. Task 397的Action History示例:")
    action_history_sample = con.execute("""
        SELECT task_id, step_id, action_history
        FROM webprm 
        WHERE task_id = '397'
        LIMIT 1
    """).fetchall()
    
    for task_id, step_id, action_history in action_history_sample:
        print(f"   Task {task_id}, Step {step_id}:")
        print(f"     Action History: {action_history}")
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
    
    # 核心分析：任务397的Step差异分析
    print(f"\n3. 任务397的Step差异分析:")
    
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
        print(f"   目标: {intent[:100] if intent else 'N/A'}...")
    
    # 核心差异分析：显示每个step之间的实际变化
    print(f"\n4. Step之间的实际变化:")
    step_changes = con.execute("""
        WITH step_data AS (
            SELECT 
                step_id,
                intent,
                website_name,
                start_url,
                current_url,
                text_observation,
                checklist,
                checklist_target_list,
                array_length(thought_history) as thought_count,
                array_length(action_history) as action_count,
                array_length(rejected) as rejected_count,
                rejected,
                LAG(step_id) OVER (ORDER BY CAST(step_id AS INTEGER)) as prev_step_id,
                LAG(intent) OVER (ORDER BY CAST(step_id AS INTEGER)) as prev_intent,
                LAG(website_name) OVER (ORDER BY CAST(step_id AS INTEGER)) as prev_website_name,
                LAG(start_url) OVER (ORDER BY CAST(step_id AS INTEGER)) as prev_start_url,
                LAG(current_url) OVER (ORDER BY CAST(step_id AS INTEGER)) as prev_current_url,
                LAG(text_observation) OVER (ORDER BY CAST(step_id AS INTEGER)) as prev_text_observation,
                LAG(checklist) OVER (ORDER BY CAST(step_id AS INTEGER)) as prev_checklist,
                LAG(checklist_target_list) OVER (ORDER BY CAST(step_id AS INTEGER)) as prev_checklist_target_list,
                LAG(array_length(thought_history)) OVER (ORDER BY CAST(step_id AS INTEGER)) as prev_thought_count,
                LAG(array_length(action_history)) OVER (ORDER BY CAST(step_id AS INTEGER)) as prev_action_count,
                LAG(array_length(rejected)) OVER (ORDER BY CAST(step_id AS INTEGER)) as prev_rejected_count,
                LAG(rejected) OVER (ORDER BY CAST(step_id AS INTEGER)) as prev_rejected
            FROM webprm 
            WHERE task_id = '397'
        )
        SELECT 
            step_id,
            prev_step_id,
            -- 只显示发生变化的字段
            CASE 
                WHEN intent != prev_intent THEN 
                    '意图: ' || COALESCE(prev_intent, 'N/A') || ' → ' || COALESCE(intent, 'N/A')
                ELSE NULL
            END as intent_change,
            CASE 
                WHEN website_name != prev_website_name THEN 
                    '网站: ' || COALESCE(prev_website_name, 'N/A') || ' → ' || COALESCE(website_name, 'N/A')
                ELSE NULL
            END as website_change,
            CASE 
                WHEN start_url != prev_start_url THEN 
                    '起始URL: ' || COALESCE(prev_start_url, 'N/A') || ' → ' || COALESCE(start_url, 'N/A')
                ELSE NULL
            END as start_url_change,
            CASE 
                WHEN current_url != prev_current_url THEN 
                    '当前URL: ' || COALESCE(prev_current_url, 'N/A') || ' → ' || COALESCE(current_url, 'N/A')
                ELSE NULL
            END as current_url_change,
            CASE 
                WHEN text_observation != prev_text_observation THEN 
                    '文本观察: 已变化 (长度: ' || COALESCE(length(prev_text_observation), 0) || ' → ' || COALESCE(length(text_observation), 0) || ')'
                ELSE NULL
            END as text_obs_change,
            -- 添加文本观察的具体内容
            CASE 
                WHEN text_observation != prev_text_observation THEN 
                    '文本内容变化: ' || COALESCE(LEFT(prev_text_observation, 100), 'N/A') || ' → ' || COALESCE(LEFT(text_observation, 100), 'N/A')
                ELSE NULL
            END as text_content_change,
            CASE 
                WHEN checklist != prev_checklist THEN 
                    '检查清单: ' || COALESCE(prev_checklist, 'N/A') || ' → ' || COALESCE(checklist, 'N/A')
                ELSE NULL
            END as checklist_change,
            CASE 
                WHEN checklist_target_list != prev_checklist_target_list THEN 
                    '检查目标: ' || COALESCE(prev_checklist_target_list::VARCHAR, 'N/A') || ' → ' || COALESCE(checklist_target_list::VARCHAR, 'N/A')
                ELSE NULL
            END as checklist_target_change,
            CASE 
                WHEN thought_count != prev_thought_count THEN 
                    '思考历史: ' || COALESCE(prev_thought_count, 0) || ' → ' || COALESCE(thought_count, 0) || ' 条'
                ELSE NULL
            END as thought_count_change,
            CASE 
                WHEN action_count != prev_action_count THEN 
                    '动作历史: ' || COALESCE(prev_action_count, 0) || ' → ' || COALESCE(action_count, 0) || ' 条'
                ELSE NULL
            END as action_count_change,
            CASE 
                WHEN rejected_count != prev_rejected_count THEN 
                    '拒绝记录: ' || COALESCE(prev_rejected_count, 0) || ' → ' || COALESCE(rejected_count, 0) || ' 条'
                ELSE NULL
            END as rejected_count_change,
            -- 添加拒绝内容的具体变化
            CASE 
                WHEN rejected_count != prev_rejected_count THEN 
                    '拒绝内容变化: ' || COALESCE(LEFT(prev_rejected::VARCHAR, 100), 'N/A') || ' → ' || COALESCE(LEFT(rejected::VARCHAR, 100), 'N/A')
                ELSE NULL
            END as rejected_content_change
        FROM step_data
        WHERE prev_intent IS NOT NULL
        ORDER BY CAST(step_id AS INTEGER)
    """).fetchall()
    
    print(f"   每个Step的变化详情:")
    for row in step_changes:
        step_id, prev_step_id, intent_change, website_change, start_url_change, current_url_change, \
        text_obs_change, text_content_change, checklist_change, checklist_target_change, thought_count_change, \
        action_count_change, rejected_count_change, rejected_content_change = row
        
        # 只显示发生变化的字段
        changes = [change for change in [intent_change, website_change, start_url_change, current_url_change,
                                       text_obs_change, text_content_change, checklist_change, checklist_target_change,
                                       thought_count_change, action_count_change, rejected_count_change, rejected_content_change] 
                  if change is not None]
        
        if changes:
            print(f"     Step {prev_step_id} → Step {step_id}:")
            for change in changes:
                print(f"       {change}")
            print()
        else:
            print(f"     Step {prev_step_id} → Step {step_id}: 所有字段都相同")
            print()
    
    con.close()

if __name__ == "__main__":
    main()
