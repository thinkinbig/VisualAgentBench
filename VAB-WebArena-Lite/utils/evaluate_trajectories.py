"""
使用GPT-4o评估WebArena Lite轨迹的执行成功性
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import openai
from openai import OpenAI
import time
import argparse

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))
from task_id_mapping import lite_to_arena_id


class TrajectoryEvaluator:
    """轨迹评估器"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        初始化评估器
        
        Args:
            api_key: OpenAI API密钥
            model: 使用的模型名称
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def load_task_config(self, config_path: str) -> Dict[int, Dict]:
        """
        加载任务配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Dict[task_id, task_config]
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        
        task_configs = {}
        for task in tasks:
            task_id = task.get('task_id')
            if task_id is not None:
                task_configs[task_id] = task
                
        return task_configs
    
    def load_trajectory(self, trajectory_path: str) -> Optional[Dict]:
        """
        加载轨迹文件
        
        Args:
            trajectory_path: 轨迹文件路径
            
        Returns:
            轨迹数据或None
        """
        if not os.path.exists(trajectory_path):
            return None
            
        try:
            with open(trajectory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载轨迹文件失败 {trajectory_path}: {e}")
            return None
    
    def create_evaluation_prompt(self, task_config: Dict, trajectory: Dict) -> str:
        """
        创建评估提示词
        
        Args:
            task_config: 任务配置
            trajectory: 轨迹数据
            
        Returns:
            评估提示词
        """
        intent = task_config.get('intent', '')
        eval_config = task_config.get('eval', {})
        sites = task_config.get('sites', [])
        
        # 提取轨迹中的关键信息
        trajectory_summary = self._extract_trajectory_summary(trajectory)
        
        # 提取标准答案信息
        reference_info = self._extract_reference_answers(eval_config)
        
        prompt = f"""
你是一个专业的Web自动化任务评估专家。请评估以下WebArena Lite任务的执行轨迹是否成功完成了任务目标。

## 任务信息
**任务意图**: {intent}
**涉及网站**: {', '.join(sites)}
**评估类型**: {', '.join(eval_config.get('eval_types', []))}

## 标准答案参考
{reference_info}

## 执行轨迹摘要
{trajectory_summary}

## 评估要求
请从以下角度评估轨迹的执行情况：

1. **任务完成度**: 是否成功完成了任务意图？
2. **操作正确性**: 执行的操作是否符合任务要求？
3. **结果准确性**: 最终结果是否满足标准答案要求？
4. **执行效率**: 操作路径是否合理高效？

**重要**: 请特别关注标准答案中的要求，确保评估结果与预期答案一致。

## 输出格式
请按以下JSON格式输出评估结果：

```json
{{
    "success": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "详细的评估理由，包括与标准答案的对比",
    "key_actions": ["关键操作1", "关键操作2", ...],
    "issues": ["问题1", "问题2", ...],
    "suggestions": ["改进建议1", "改进建议2", ...],
    "reference_comparison": "与标准答案的详细对比分析"
}}
```

请仔细分析轨迹，给出客观、准确的评估结果。
"""
        return prompt
    
    def _extract_trajectory_summary(self, trajectory: List) -> str:
        """
        提取轨迹摘要
        
        Args:
            trajectory: 轨迹数据（列表格式）
            
        Returns:
            轨迹摘要字符串
        """
        if not trajectory:
            return "轨迹数据为空"
        
        summary_parts = []
        
        # 基本信息
        summary_parts.append(f"执行步骤数: {len(trajectory)}")
        
        # 提取关键步骤
        key_steps = []
        for i, step in enumerate(trajectory[:15]):  # 只显示前15步
            if isinstance(step, dict):
                # 提取动作信息
                if 'action' in step:
                    action = step['action']
                    if isinstance(action, dict):
                        action_type = action.get('action_type', 'unknown')
                        if 'element_id' in action:
                            key_steps.append(f"步骤{i+1}: {action_type} (元素ID: {action['element_id']})")
                        else:
                            key_steps.append(f"步骤{i+1}: {action_type}")
                    else:
                        key_steps.append(f"步骤{i+1}: {action}")
                
                # 提取观察信息
                elif 'observation' in step:
                    obs = step['observation']
                    # 截取观察信息的前100个字符
                    obs_short = obs[:100] + "..." if len(obs) > 100 else obs
                    key_steps.append(f"步骤{i+1}: 观察 - {obs_short}")
        
        if key_steps:
            summary_parts.append("关键步骤:\n" + "\n".join(key_steps))
        
        # 提取最终状态
        if trajectory:
            last_step = trajectory[-1]
            if isinstance(last_step, dict) and 'observation' in last_step:
                final_obs = last_step['observation']
                # 提取URL信息
                if 'url=' in final_obs:
                    url_start = final_obs.find("url='") + 5
                    url_end = final_obs.find("'", url_start)
                    if url_start > 4 and url_end > url_start:
                        final_url = final_obs[url_start:url_end]
                        summary_parts.append(f"最终URL: {final_url}")
        
        return "\n".join(summary_parts) if summary_parts else "无法提取轨迹信息"
    
    def _extract_reference_answers(self, eval_config: Dict) -> str:
        """
        提取标准答案信息
        
        Args:
            eval_config: 评估配置
            
        Returns:
            标准答案信息字符串
        """
        reference_parts = []
        
        # 提取reference_answers
        reference_answers = eval_config.get('reference_answers')
        if reference_answers:
            reference_parts.append("**期望答案**:")
            if isinstance(reference_answers, dict):
                for key, value in reference_answers.items():
                    if key == 'exact_match':
                        reference_parts.append(f"- 精确匹配: {value}")
                    elif key == 'must_include':
                        if isinstance(value, list):
                            reference_parts.append(f"- 必须包含: {', '.join(value)}")
                        else:
                            reference_parts.append(f"- 必须包含: {value}")
                    elif key == 'fuzzy_match':
                        if isinstance(value, list):
                            reference_parts.append(f"- 模糊匹配: {', '.join(value)}")
                        else:
                            reference_parts.append(f"- 模糊匹配: {value}")
            else:
                reference_parts.append(f"- 标准答案: {reference_answers}")
        
        # 提取reference_url
        reference_url = eval_config.get('reference_url')
        if reference_url:
            reference_parts.append(f"**期望URL**: {reference_url}")
        
        # 提取program_html检查规则
        program_html = eval_config.get('program_html', [])
        if program_html:
            reference_parts.append("**程序化检查规则**:")
            for i, rule in enumerate(program_html):
                if isinstance(rule, dict):
                    url = rule.get('url', '')
                    locator = rule.get('locator', '')
                    required_contents = rule.get('required_contents', {})
                    reference_parts.append(f"- 规则{i+1}: 检查URL '{url}' 中的元素 '{locator}'")
                    if required_contents:
                        for content_type, content_value in required_contents.items():
                            reference_parts.append(f"  要求: {content_type} = {content_value}")
        
        # 提取原始注释
        raw_annotation = eval_config.get('reference_answer_raw_annotation')
        if raw_annotation:
            reference_parts.append(f"**原始注释**: {raw_annotation}")
        
        # 提取其他评估说明
        string_note = eval_config.get('string_note')
        if string_note:
            reference_parts.append(f"**字符串匹配说明**: {string_note}")
        
        url_note = eval_config.get('url_note')
        if url_note:
            reference_parts.append(f"**URL匹配说明**: {url_note}")
        
        if not reference_parts:
            return "**无标准答案参考**"
        
        return "\n".join(reference_parts)
    
    def evaluate_trajectory(self, task_config: Dict, trajectory: Dict, use_auto_validation: bool = True) -> Dict:
        """
        评估单个轨迹
        
        Args:
            task_config: 任务配置
            trajectory: 轨迹数据
            use_auto_validation: 是否使用自动验证（基于标准答案）
            
        Returns:
            评估结果
        """
        prompt = self.create_evaluation_prompt(task_config, trajectory)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的Web自动化任务评估专家，擅长分析用户行为轨迹和任务完成情况。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # 尝试解析JSON结果
            try:
                # 提取JSON部分
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = content[json_start:json_end]
                    result = json.loads(json_str)
                    
                    # 如果启用自动验证，进行额外的验证
                    if use_auto_validation:
                        auto_validation = self._auto_validate_with_reference(task_config, trajectory, result)
                        result.update(auto_validation)
                    
                    return result
                else:
                    return {
                        "success": False,
                        "confidence": 0.0,
                        "reasoning": "无法解析GPT响应",
                        "key_actions": [],
                        "issues": ["响应格式错误"],
                        "suggestions": []
                    }
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "confidence": 0.0,
                    "reasoning": f"JSON解析失败: {content}",
                    "key_actions": [],
                    "issues": ["响应格式错误"],
                    "suggestions": []
                }
                
        except Exception as e:
            return {
                "success": False,
                "confidence": 0.0,
                "reasoning": f"API调用失败: {str(e)}",
                "key_actions": [],
                "issues": ["API错误"],
                "suggestions": []
            }
    
    def _auto_validate_with_reference(self, task_config: Dict, trajectory: Dict, llm_result: Dict) -> Dict:
        """
        基于标准答案进行自动验证
        
        Args:
            task_config: 任务配置
            trajectory: 轨迹数据
            llm_result: LLM评估结果
            
        Returns:
            自动验证结果
        """
        eval_config = task_config.get('eval', {})
        validation_result = {
            "auto_validation": {
                "enabled": True,
                "checks": [],
                "final_verdict": None,
                "confidence_adjustment": 0.0
            }
        }
        
        # 检查URL匹配
        reference_url = eval_config.get('reference_url')
        if reference_url:
            final_url = self._extract_final_url(trajectory)
            if final_url:
                url_match = self._check_url_match(final_url, reference_url, eval_config.get('url_note', 'EXACT'))
                validation_result["auto_validation"]["checks"].append({
                    "type": "url_match",
                    "expected": reference_url,
                    "actual": final_url,
                    "match": url_match
                })
                if not url_match:
                    validation_result["auto_validation"]["confidence_adjustment"] -= 0.3
        
        # 检查字符串匹配
        reference_answers = eval_config.get('reference_answers')
        if reference_answers:
            trajectory_text = self._extract_trajectory_text(trajectory)
            string_match = self._check_string_match(trajectory_text, reference_answers)
            validation_result["auto_validation"]["checks"].append({
                "type": "string_match",
                "expected": reference_answers,
                "actual": trajectory_text[:500] + "..." if len(trajectory_text) > 500 else trajectory_text,
                "match": string_match
            })
            if not string_match:
                validation_result["auto_validation"]["confidence_adjustment"] -= 0.4
        
        # 综合判断
        all_checks_passed = all(check["match"] for check in validation_result["auto_validation"]["checks"])
        if validation_result["auto_validation"]["checks"]:
            if all_checks_passed:
                validation_result["auto_validation"]["final_verdict"] = "PASS"
                validation_result["auto_validation"]["confidence_adjustment"] += 0.2
            else:
                validation_result["auto_validation"]["final_verdict"] = "FAIL"
        
        return validation_result
    
    def _extract_final_url(self, trajectory: List) -> Optional[str]:
        """提取轨迹中的最终URL"""
        if not trajectory:
            return None
        
        last_step = trajectory[-1]
        if isinstance(last_step, dict) and 'observation' in last_step:
            obs = last_step['observation']
            if 'url=' in obs:
                url_start = obs.find("url='") + 5
                url_end = obs.find("'", url_start)
                if url_start > 4 and url_end > url_start:
                    return obs[url_start:url_end]
        return None
    
    def _check_url_match(self, actual_url: str, expected_url: str, matching_rule: str) -> bool:
        """检查URL匹配"""
        if matching_rule == "EXACT":
            return actual_url == expected_url
        elif matching_rule == "GOLD in PRED":
            return expected_url in actual_url
        else:
            return actual_url == expected_url
    
    def _extract_trajectory_text(self, trajectory: List) -> str:
        """提取轨迹中的文本内容"""
        text_parts = []
        for step in trajectory:
            if isinstance(step, dict):
                if 'observation' in step:
                    text_parts.append(step['observation'])
                if 'action' in step and isinstance(step['action'], dict):
                    if 'text' in step['action']:
                        text_parts.append(step['action']['text'])
        return " ".join(text_parts)
    
    def _check_string_match(self, actual_text: str, reference_answers: Dict) -> bool:
        """检查字符串匹配"""
        if not isinstance(reference_answers, dict):
            return False
        
        actual_text_lower = actual_text.lower()
        
        # 检查exact_match
        if 'exact_match' in reference_answers:
            expected = str(reference_answers['exact_match']).lower()
            return expected in actual_text_lower
        
        # 检查must_include
        if 'must_include' in reference_answers:
            must_include = reference_answers['must_include']
            if isinstance(must_include, list):
                return all(str(item).lower() in actual_text_lower for item in must_include)
            else:
                return str(must_include).lower() in actual_text_lower
        
        # 检查fuzzy_match (简化版本)
        if 'fuzzy_match' in reference_answers:
            fuzzy_match = reference_answers['fuzzy_match']
            if isinstance(fuzzy_match, list):
                # 至少匹配一个
                return any(str(item).lower() in actual_text_lower for item in fuzzy_match)
            else:
                return str(fuzzy_match).lower() in actual_text_lower
        
        return False
    
    def evaluate_all_trajectories(self, 
                                config_path: str, 
                                trajectory_dir: str,
                                output_path: str,
                                max_tasks: Optional[int] = None,
                                use_auto_validation: bool = True) -> Dict:
        """
        评估所有轨迹
        
        Args:
            config_path: 任务配置文件路径
            trajectory_dir: 轨迹文件夹路径
            output_path: 输出结果文件路径
            max_tasks: 最大评估任务数（用于测试）
            
        Returns:
            评估结果统计
        """
        print("加载任务配置...")
        task_configs = self.load_task_config(config_path)
        print(f"加载了 {len(task_configs)} 个任务配置")
        
        trajectory_path = Path(trajectory_dir)
        if not trajectory_path.exists():
            raise FileNotFoundError(f"轨迹文件夹不存在: {trajectory_path}")
        
        # 获取所有轨迹文件
        trajectory_files = list(trajectory_path.glob("*.json"))
        print(f"找到 {len(trajectory_files)} 个轨迹文件")
        
        if max_tasks:
            trajectory_files = trajectory_files[:max_tasks]
            print(f"限制评估前 {max_tasks} 个任务")
        
        results = []
        stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "api_errors": 0,
            "file_errors": 0
        }
        
        for i, traj_file in enumerate(trajectory_files):
            task_id = int(traj_file.stem)
            print(f"\n评估任务 {i+1}/{len(trajectory_files)}: Task ID {task_id}")
            
            # 检查任务配置是否存在
            if task_id not in task_configs:
                print(f"  跳过: 任务配置不存在")
                stats["file_errors"] += 1
                continue
            
            # 加载轨迹
            trajectory = self.load_trajectory(str(traj_file))
            if trajectory is None:
                print(f"  跳过: 轨迹文件加载失败")
                stats["file_errors"] += 1
                continue
            
            # 评估轨迹
            task_config = task_configs[task_id]
            result = self.evaluate_trajectory(task_config, trajectory, use_auto_validation)
            
            # 记录结果
            result["task_id"] = task_id
            result["intent"] = task_config.get("intent", "")
            results.append(result)
            
            # 更新统计
            stats["total"] += 1
            if result.get("success", False):
                stats["success"] += 1
                print(f"  ✅ 成功 (置信度: {result.get('confidence', 0):.2f})")
            else:
                stats["failed"] += 1
                print(f"  ❌ 失败 (置信度: {result.get('confidence', 0):.2f})")
            
            if "API错误" in result.get("issues", []):
                stats["api_errors"] += 1
            
            # 保存中间结果
            if (i + 1) % 10 == 0:
                self._save_results(results, output_path)
                print(f"  已保存中间结果 ({i+1} 个任务)")
            
            # 避免API限制
            time.sleep(1)
        
        # 保存最终结果
        self._save_results(results, output_path)
        
        return stats
    
    def _save_results(self, results: List[Dict], output_path: str):
        """保存结果到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用GPT-4o评估WebArena Lite轨迹")
    parser.add_argument(
        "--api-key",
        required=True,
        help="OpenAI API密钥"
    )
    parser.add_argument(
        "--config",
        default="VAB-WebArena-Lite/config_files/wa/test_webarena_lite.raw.json",
        help="任务配置文件路径"
    )
    parser.add_argument(
        "--trajectory-dir",
        default="WADataset/data/learn_by_interact/Webarena_trajectory_lite",
        help="轨迹文件夹路径"
    )
    parser.add_argument(
        "--output",
        default="trajectory_evaluation_results.json",
        help="输出结果文件路径"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="最大评估任务数（用于测试）"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="使用的模型名称"
    )
    parser.add_argument(
        "--disable-auto-validation",
        action="store_true",
        help="禁用基于标准答案的自动验证"
    )
    
    args = parser.parse_args()
    
    try:
        evaluator = TrajectoryEvaluator(api_key=args.api_key, model=args.model)
        
        print("开始评估WebArena Lite轨迹...")
        print(f"配置文件: {args.config}")
        print(f"轨迹文件夹: {args.trajectory_dir}")
        print(f"输出文件: {args.output}")
        print(f"使用模型: {args.model}")
        
        use_auto_validation = not args.disable_auto_validation
        stats = evaluator.evaluate_all_trajectories(
            config_path=args.config,
            trajectory_dir=args.trajectory_dir,
            output_path=args.output,
            max_tasks=args.max_tasks,
            use_auto_validation=use_auto_validation
        )
        
        print("\n" + "="*50)
        print("评估完成!")
        print(f"总任务数: {stats['total']}")
        print(f"成功: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
        print(f"失败: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
        print(f"API错误: {stats['api_errors']}")
        print(f"文件错误: {stats['file_errors']}")
        print(f"结果已保存到: {args.output}")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
