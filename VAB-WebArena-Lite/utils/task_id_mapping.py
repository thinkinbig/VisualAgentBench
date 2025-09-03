"""
WebArena Lite 到 WebArena 的 Task ID 映射函数
"""

import json
from typing import Dict, List, Optional
from pathlib import Path

# 全局缓存映射关系
_lite_to_arena: Optional[Dict[int, int]] = None
_arena_to_lite: Optional[Dict[int, int]] = None


def _init_mappings(mapping_file_path: Optional[str] = None) -> None:
    """初始化映射关系，只加载一次"""
    global _lite_to_arena, _arena_to_lite
    
    if _lite_to_arena is not None and _arena_to_lite is not None:
        return  # 已经初始化过了
    
    if mapping_file_path is None:
        current_dir = Path(__file__).parent
        mapping_file_path = current_dir.parent / "config_files" / "wa" / "test_webarena_lite.raw.json"
    
    with open(mapping_file_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    _lite_to_arena = {}
    _arena_to_lite = {}
    
    for item in mapping_data:
        lite_task_id = item.get('task_id')
        old_task_id = item.get('old_task_id')
        
        if lite_task_id is not None and old_task_id is not None:
            _lite_to_arena[lite_task_id] = old_task_id
            _arena_to_lite[old_task_id] = lite_task_id


def load_task_mapping(mapping_file_path: Optional[str] = None) -> Dict[int, int]:
    """
    加载WebArena Lite到WebArena的task ID映射
    
    Args:
        mapping_file_path: 映射文件路径，默认为test_webarena_lite.raw.json
        
    Returns:
        Dict[lite_task_id, arena_old_task_id]
    """
    _init_mappings(mapping_file_path)
    return _lite_to_arena


def lite_to_arena_id(lite_task_id: int, mapping_file_path: Optional[str] = None) -> int:
    """
    将WebArena Lite的task_id映射到WebArena的old_task_id
    
    Args:
        lite_task_id: WebArena Lite的task_id
        mapping_file_path: 映射文件路径
        
    Returns:
        WebArena的old_task_id
        
    Raises:
        KeyError: 如果lite_task_id不存在映射
    """
    _init_mappings(mapping_file_path)
    if lite_task_id not in _lite_to_arena:
        raise KeyError(f"WebArena Lite task_id {lite_task_id} 没有对应的WebArena映射")
    return _lite_to_arena[lite_task_id]


def arena_to_lite_id(arena_task_id: int, mapping_file_path: Optional[str] = None) -> Optional[int]:
    """
    将WebArena的old_task_id映射到WebArena Lite的task_id
    
    Args:
        arena_task_id: WebArena的old_task_id
        mapping_file_path: 映射文件路径
        
    Returns:
        WebArena Lite的task_id，如果不存在映射则返回None
    """
    _init_mappings(mapping_file_path)
    return _arena_to_lite.get(arena_task_id)

