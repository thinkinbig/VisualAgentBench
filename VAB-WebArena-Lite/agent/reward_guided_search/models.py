"""
Data models for Reward Guided Search Module
Using Pydantic for type safety and validation
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

# ============================================================================
# STATE MODELS - 网页状态相关模型
# ============================================================================

class ElementType(str, Enum):
    """网页元素类型"""
    BUTTON = "button"
    INPUT = "input"
    LINK = "link"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    IMAGE = "image"
    TEXT = "text"
    DIV = "div"
    SPAN = "span"
    FORM = "form"
    TABLE = "table"
    LIST = "list"
    OTHER = "other"

class WebElement(BaseModel):
    """网页元素模型"""
    bid: str = Field(..., description="Unique element identifier")
    type: ElementType = Field(..., description="Element type")
    text: Optional[str] = Field(None, description="Element text content")
    coordinates: Optional[List[int]] = Field(None, description="Element coordinates [x, y]")
    attributes: Dict[str, str] = Field(default_factory=dict, description="HTML attributes")
    visible: bool = Field(True, description="Whether element is visible")
    enabled: bool = Field(True, description="Whether element is interactive")
    
    @validator('coordinates')
    def validate_coordinates(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError("Coordinates must be [x, y]")
        return v

class PageState(BaseModel):
    """网页状态模型"""
    url: str = Field(..., description="Current page URL")
    title: str = Field(..., description="Page title")
    elements: List[WebElement] = Field(default_factory=list, description="Interactive elements")
    text_observation: Optional[str] = Field(None, description="AXTree text observation")
    page_content: Optional[str] = Field(None, description="Page text content")
    current_focus: Optional[str] = Field(None, description="Currently focused element")
    scroll_position: Optional[Dict[str, int]] = Field(None, description="Scroll position {x, y}")
    window_size: Optional[Dict[str, int]] = Field(None, description="Window size {width, height}")
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v
    
    def get_elements_by_type(self, element_type: ElementType) -> List[WebElement]:
        """根据类型获取元素"""
        return [e for e in self.elements if e.type == element_type]
    
    def get_clickable_elements(self) -> List[WebElement]:
        """获取可点击元素"""
        return [e for e in self.elements if e.type in [ElementType.BUTTON, ElementType.LINK] and e.enabled]
    
    def get_input_elements(self) -> List[WebElement]:
        """获取输入元素"""
        return [e for e in self.elements if e.type in [ElementType.INPUT, ElementType.TEXTAREA, ElementType.SELECT]]

# ============================================================================
# ACTION MODELS - 动作相关模型
# ============================================================================

class ActionType(str, Enum):
    """动作类型"""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    NAVIGATE = "navigate"
    HOVER = "hover"
    SELECT = "select"
    SUBMIT = "submit"

class Action(BaseModel):
    """动作模型"""
    action_type: ActionType = Field(..., description="Type of action to perform")
    target: Optional[str] = Field(None, description="Target element bid or selector")
    coordinates: Optional[List[int]] = Field(None, description="Target coordinates [x, y]")
    text: Optional[str] = Field(None, description="Text to type or select")
    value: Optional[str] = Field(None, description="Value to set")
    duration: Optional[float] = Field(None, description="Duration for wait actions")
    thought: Optional[str] = Field(None, description="Reasoning behind this action")
    
    @validator('coordinates')
    def validate_coordinates(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError("Coordinates must be [x, y]")
        return v
    
    @validator('duration')
    def validate_duration(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Duration must be positive")
        return v

# ============================================================================
# CONTEXT MODELS - 上下文相关模型
# ============================================================================

class TaskContext(BaseModel):
    """任务上下文模型"""
    intent: str = Field(..., description="User's intent/goal")
    start_url: Optional[str] = Field(None, description="Starting URL for the task")
    meta_data: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    constraints: List[str] = Field(default_factory=list, description="Task constraints")
    priority: int = Field(default=1, description="Task priority (1-5)")
    
    @validator('priority')
    def validate_priority(cls, v):
        if not 1 <= v <= 5:
            raise ValueError("Priority must be between 1 and 5")
        return v

class Feedback(BaseModel):
    """反馈模型"""
    thought: Optional[str] = Field(None, description="Thought process feedback")
    checklist: List[str] = Field(default_factory=list, description="Checklist items")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    score: Optional[float] = Field(None, description="Feedback score")
    reasoning: Optional[str] = Field(None, description="Reasoning for feedback")

# ============================================================================
# PROMPT MODELS - 提示相关模型
# ============================================================================

class PromptRequest(BaseModel):
    """提示请求模型"""
    prompt_type: str = Field(..., description="Type of prompt to generate")
    role: str = Field(..., description="Agent role for the prompt")
    state: PageState = Field(..., description="Current page state")
    context: TaskContext = Field(..., description="Task context")
    action: Optional[Action] = Field(None, description="Action to evaluate/refine")
    feedback: Optional[Feedback] = Field(None, description="Feedback for refinement")
    trajectory: Optional[str] = Field(None, description="Action trajectory history")

class PromptResponse(BaseModel):
    """提示响应模型"""
    role: str = Field(..., description="Message role (system/user)")
    content: str = Field(..., description="Prompt content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

# ============================================================================
# UTILITY FUNCTIONS - 工具函数
# ============================================================================

def create_page_state_from_dict(data: Dict[str, Any]) -> PageState:
    """从字典创建PageState"""
    return PageState(**data)

def create_action_from_dict(data: Dict[str, Any]) -> Action:
    """从字典创建Action"""
    return Action(**data)

def validate_state_data(data: Dict[str, Any]) -> bool:
    """验证状态数据"""
    try:
        PageState(**data)
        return True
    except Exception:
        return False
