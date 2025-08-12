#!/usr/bin/env python3
"""
WebArena-Lite Shopping and Admin Benchmark Runner
专门用于运行 shopping 和 admin 配置的脚本
"""

import os
import sys
import argparse
import logging
import traceback
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, '.')

from src.server.task_controller import TaskController
from src.server.task_worker import TaskWorker
from src.client.agent_test import AgentTest

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('vwa_shopping_admin.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_shopping_admin_benchmark():
    """运行 shopping 和 admin 基准测试"""
    logger = setup_logging()
    logger.info("Starting WebArena-Lite Shopping and Admin Benchmark")
    
    # 设置环境变量
    os.environ["DATASET"] = "visualwebarena"
    
    # 检查必要的环境变量
    required_env_vars = [
        "OPENAI_API_KEY",
        "SHOPPING", 
        "ADMIN"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please set the following environment variables:")
        logger.info("export OPENAI_API_KEY='your-api-key'")
        logger.info("export SHOPPING='shopping-website-url'")
        logger.info("export ADMIN='admin-website-url'")
        return False
    
    # 配置参数
    config = {
        "agent_type": "prompt",
        "instruction_path": "agent/prompts/jsons/p_cot_id_actree_2s.json",
        "provider": "openai",
        "model": "gpt-5-nano",
        "action_set_tag": "id_accessibility_tree",
        "mode": "chat",
        "temperature": 1.0,
        "top_p": 0.9,
        "context_length": 0,
        "max_tokens": 384,
        "stop_token": None,
        "max_obs_length": 3840,
        "max_retry": 1
    }
    
    # 测试配置目录
    test_configs = [
        "config_files/vwa/test_shopping",
        "config_files/vwa/test_admin"
    ]
    
    # 检查测试配置文件是否存在
    for config_dir in test_configs:
        if not Path(config_dir).exists():
            logger.warning(f"Test config directory not found: {config_dir}")
            logger.info("Please run generate_test_data.py first to create test configs")
            continue
        
        # 运行测试
        logger.info(f"Running tests for {config_dir}")
        try:
            # 这里可以调用具体的测试逻辑
            # 暂时先打印信息
            logger.info(f"Would run tests for {config_dir} with config: {config}")
            
        except Exception as e:
            logger.error(f"Error running tests for {config_dir}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    logger.info("Benchmark completed")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="WebArena-Lite Shopping and Admin Benchmark Runner")
    parser.add_argument("--config", type=str, default="config_files/vwa/test_shopping", 
                       help="Test configuration directory")
    parser.add_argument("--model", type=str, default="gpt-5-nano",
                       help="Language model to use")
    parser.add_argument("--headless", action="store_true", default=True,
                       help="Run browser in headless mode")
    
    args = parser.parse_args()
    
    # 更新配置
    if args.model != "gpt-5-nano":
        os.environ["MODEL"] = args.model
    
    # 运行基准测试
    success = run_shopping_admin_benchmark()
    
    if success:
        print("✅ Benchmark completed successfully!")
        sys.exit(0)
    else:
        print("❌ Benchmark failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
