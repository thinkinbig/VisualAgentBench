#!/usr/bin/env python3
"""
使用DuckDB进行真正的SQL查询分析WebPRM数据集中的task 397
特别关注thoughts和action的关系，以及chosen列中的决策过程
显示完整的thoughts内容
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
    
    con.close()

if __name__ == "__main__":
    main()
