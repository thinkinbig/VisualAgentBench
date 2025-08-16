import duckdb
from datasets import load_dataset

def main():
    # 连接到DuckDB数据库
    con = duckdb.connect(':memory:')
    
    # 从Hugging Face加载真实的WebRewardBench数据集
    print("✅ 正在从Hugging Face加载WebRewardBench数据集...")
    
    try:
        # 加载数据集
        dataset = load_dataset("LangAGI-Lab/WebPRMCollection_preference_pair")
        print(f"✅ 成功加载数据集！包含 {len(dataset['test'])} 条记录")
        
        # 将数据集转换为DuckDB表
        df = dataset['test'].to_pandas()
        con.register('webprm', df)
        
        print(f"✅ 数据集已导入DuckDB，开始分析...")
        
        # 显示数据集结构
        print(f"\n1. 数据集结构:")
        schema = con.execute("DESCRIBE webprm").fetchall()
        for field_name, field_type, null, key, default, extra in schema:
            print(f"   {field_name}: {field_type}")
        
        # 分析包含多个send_msg_to_user的任务
        print(f"\n2. 分析包含多个send_msg_to_user的任务:")
        
        # 统计每个任务中send_msg_to_user的总数
        send_msg_stats = con.execute("""
            SELECT 
                task_id,
                COUNT(*) as total_steps,
                SUM(
                    (SELECT COUNT(*) 
                     FROM unnest(action_history) AS action 
                     WHERE action::VARCHAR LIKE '%send_msg_to_user%')
                ) as total_send_msg_count,
                MAX(
                    (SELECT COUNT(*) 
                     FROM unnest(action_history) AS action 
                     WHERE action::VARCHAR LIKE '%send_msg_to_user%')
                ) as max_send_msg_per_step
            FROM webprm 
            GROUP BY task_id
            HAVING SUM(
                (SELECT COUNT(*) 
                 FROM unnest(action_history) AS action 
                 WHERE action::VARCHAR LIKE '%send_msg_to_user%')
            ) > 0
            ORDER BY total_send_msg_count DESC
            LIMIT 20
        """).fetchall()
        
        print(f"   包含send_msg_to_user的任务统计 (前20个):")
        print(f"   任务ID | 总步骤数 | 总send_msg数量 | 单步最大send_msg数量")
        print(f"   -------|----------|----------------|-------------------")
        for task_id, total_steps, total_send_msg, max_per_step in send_msg_stats:
            print(f"   {task_id:>6} | {total_steps:>8} | {total_send_msg:>14} | {max_per_step:>17}")
        
        # 详细分析包含多个send_msg_to_user的任务
        print(f"\n3. 详细分析包含多个send_msg_to_user的任务:")
        
        multiple_send_msg_tasks = con.execute("""
            SELECT 
                task_id,
                intent,
                start_url,
                action_history,
                (SELECT COUNT(*) 
                 FROM unnest(action_history) AS action 
                 WHERE action::VARCHAR LIKE '%send_msg_to_user%') as send_msg_count
            FROM webprm 
            WHERE (SELECT COUNT(*) 
                   FROM unnest(action_history) AS action 
                   WHERE action::VARCHAR LIKE '%send_msg_to_user%') > 0
            ORDER BY task_id
            LIMIT 15
        """).fetchall()
        
        current_task = None
        for task_id, intent, start_url, action_history, send_msg_count in multiple_send_msg_tasks:
            if current_task != task_id:
                current_task = task_id
                print(f"\n   === 任务 {task_id} ===")
                print(f"   意图: {intent[:100]}...")
                print(f"   起始URL: {start_url}")
            
            print(f"     send_msg_to_user数量: {send_msg_count}")
            print(f"       动作历史: {action_history}")
        
        # 统计总结
        print(f"\n4. 统计总结:")
        
        # 总任务数
        total_tasks = con.execute("SELECT COUNT(DISTINCT task_id) FROM webprm").fetchone()[0]
        
        # 包含send_msg_to_user的任务数
        tasks_with_send_msg = con.execute("""
            SELECT COUNT(DISTINCT task_id) 
            FROM webprm 
            WHERE (SELECT COUNT(*) 
                   FROM unnest(action_history) AS action 
                   WHERE action::VARCHAR LIKE '%send_msg_to_user%') > 0
        """).fetchone()[0]
        
        # 包含多个send_msg_to_user的任务数
        tasks_with_multiple_send_msg = con.execute("""
            SELECT COUNT(DISTINCT task_id) 
            FROM webprm 
            GROUP BY task_id
            HAVING SUM(
                (SELECT COUNT(*) 
                 FROM unnest(action_history) AS action 
                 WHERE action::VARCHAR LIKE '%send_msg_to_user%')
            ) > 1
        """).fetchone()[0]
        
        print(f"   📊 总体统计:")
        print(f"     总任务数: {total_tasks}")
        print(f"     包含send_msg_to_user的任务数: {tasks_with_send_msg}")
        print(f"     包含多个send_msg_to_user的任务数: {tasks_with_multiple_send_msg}")
        print(f"     占比: {tasks_with_send_msg}/{total_tasks} = {tasks_with_send_msg/total_tasks*100:.1f}%")
        
        # 分析send_msg_to_user的分布模式
        print(f"\n5. send_msg_to_user分布模式分析:")
        
        send_msg_patterns = con.execute("""
            SELECT 
                CASE 
                    WHEN total_send_msg = 1 THEN '1个'
                    WHEN total_send_msg = 2 THEN '2个'
                    WHEN total_send_msg = 3 THEN '3个'
                    WHEN total_send_msg = 4 THEN '4个'
                    WHEN total_send_msg = 5 THEN '5个'
                    ELSE '6个+'
                END as send_msg_category,
                COUNT(*) as task_count
            FROM (
                SELECT 
                    task_id,
                    SUM(
                        (SELECT COUNT(*) 
                         FROM unnest(action_history) AS action 
                         WHERE action::VARCHAR LIKE '%send_msg_to_user%')
                    ) as total_send_msg
                FROM webprm 
                GROUP BY task_id
                HAVING SUM(
                    (SELECT COUNT(*) 
                     FROM unnest(action_history) AS action 
                     WHERE action::VARCHAR LIKE '%send_msg_to_user%')
                ) > 0
            ) t
            GROUP BY send_msg_category
            ORDER BY MIN(total_send_msg)
        """).fetchall()
        
        print(f"   send_msg_to_user数量分布:")
        for category, count in send_msg_patterns:
            print(f"     {category}: {count} 个任务")
        
        # 分析网站分布
        print(f"\n6. 包含send_msg_to_user的网站分布:")
        
        website_distribution = con.execute("""
            SELECT 
                start_url,
                COUNT(DISTINCT task_id) as task_count
            FROM webprm 
            WHERE (SELECT COUNT(*) 
                   FROM unnest(action_history) AS action 
                   WHERE action::VARCHAR LIKE '%send_msg_to_user%') > 0
            GROUP BY start_url
            ORDER BY task_count DESC
            LIMIT 10
        """).fetchall()
        
        print(f"   网站分布 (前10个):")
        for start_url, task_count in website_distribution:
            print(f"     {start_url}: {task_count} 个任务")
        
    except Exception as e:
        print(f"❌ 加载数据集时出错: {e}")
        print("请确保已安装datasets库: pip install datasets")
        return
    
    con.close()

if __name__ == "__main__":
    main()
