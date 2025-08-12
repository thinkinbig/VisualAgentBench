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

from browser_env import ScriptBrowserEnv
from agent.agent import construct_agent

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
    
    # 加载环境变量配置文件
    try:
        from dotenv import load_dotenv
        if load_dotenv('.env'):
            logger.info("Environment variables loaded from .env")
        else:
            logger.warning(".env not found, using default values")
    except ImportError:
        logger.warning("python-dotenv not installed, using default values")
    
    # 设置默认环境变量
    if not os.environ.get("DATASET"):
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
    
    # 根据 USE_CHEAP_MODEL 选择模型
    use_cheap_model = os.environ.get("USE_CHEAP_MODEL", "false").lower() == "true"
    if use_cheap_model:
        default_model = "gpt-5-nano"
    else:
        default_model = "gpt-5"  # 或者其他功能完整的模型
    
    # 测试配置目录
    test_configs = [
        "config_files/vwa/test_shopping",
        "config_files/vwa/test_admin"
    ]
    
    # 检查 admin 认证状态
    admin_auth_file = "./.auth/shopping_admin_state.json"
    if not Path(admin_auth_file).exists():
        logger.warning(f"Admin authentication file not found: {admin_auth_file}")
        logger.info("Please run prepare.sh to set up admin authentication")
        # 从测试配置中移除 admin，只测试 shopping
        test_configs = ["config_files/vwa/test_shopping"]
    
    # 检查测试配置文件是否存在
    for config_dir in test_configs:
        if not Path(config_dir).exists():
            logger.warning(f"Test config directory not found: {config_dir}")
            logger.info("Please run generate_test_data.py first to create test configs")
            continue
        
        # 运行测试
        logger.info(f"Running tests for {config_dir}")
        try:
            # 获取配置文件列表
            config_files = list(Path(config_dir).glob("*.json"))
            if not config_files:
                logger.warning(f"No test config files found in {config_dir}")
                continue
            
            # 选择第一个配置文件进行测试
            test_config = config_files[0]
            logger.info(f"Testing with config: {test_config}")
            
            # 创建浏览器环境
            env = ScriptBrowserEnv(
                headless=True,
                observation_type='accessibility_tree',
                current_viewport_only=True,
                viewport_size={"width": 1280, "height": 720}
            )
            
            # 创建 agent
            import argparse
            args = argparse.Namespace()
            args.agent_type = "prompt"
            args.instruction_path = "agent/prompts/jsons/p_cot_id_actree_2s.json"
            args.provider = "openai"
            args.model = os.environ.get("MODEL", default_model)
            args.action_set_tag = "id_accessibility_tree"
            args.planner_ip = None
            args.mode = "chat"
            args.temperature = float(os.environ.get("TEMPERATURE", "1.0"))
            args.top_p = float(os.environ.get("TOP_P", "0.9"))
            args.context_length = int(os.environ.get("CONTEXT_LENGTH", "0"))
            args.max_tokens = int(os.environ.get("MAX_TOKENS", "384"))
            args.stop_token = os.environ.get("STOP_TOKEN")
            args.max_obs_length = int(os.environ.get("MAX_OBS_LENGTH", "3840"))
            args.max_retry = int(os.environ.get("MAX_RETRY", "1"))
            
            agent = construct_agent(args)
            logger.info(f"Agent created successfully: {type(agent)}")
            
            # 重置环境
            obs, info = env.reset(options={"config_file": str(test_config)})
            logger.info(f"Environment reset successful, observation type: {type(obs)}")
            
            # 重置 agent
            agent.reset(str(test_config))
            logger.info("Agent reset successful")
            
            # 创建轨迹
            trajectory = []
            state_info = {"observation": obs, "info": info}
            trajectory.append(state_info)
            
            # 尝试获取下一个动作
            try:
                action = agent.next_action(
                    trajectory=trajectory,
                    intent="test",
                    meta_data={"action_history": ["None"]}
                )
                logger.info(f"Successfully got action: {action}")
            except Exception as e:
                logger.error(f"Error getting action: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # 清理
            env.close()
            logger.info(f"Test completed for {test_config}")
            
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
