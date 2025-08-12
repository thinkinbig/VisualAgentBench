"""
Test Pydantic models integration with prompts system
"""

import pytest
from .models import PageState, WebElement, ElementType, Action, ActionType, TaskContext
from .prompts import build_action_generation_prompt, build_webspherd_evaluation_prompt

def test_page_state_creation():
    """Test creating PageState with valid data"""
    
    # Create valid page state
    state = PageState(
        url="https://example.com",
        title="Example Page",
        elements=[
            WebElement(
                bid="btn1",
                type=ElementType.BUTTON,
                text="Submit",
                coordinates=[100, 200],
                visible=True,
                enabled=True
            ),
            WebElement(
                bid="input1", 
                type=ElementType.INPUT,
                text="Username",
                coordinates=[50, 100],
                visible=True,
                enabled=True
            )
        ]
    )
    
    assert state.url == "https://example.com"
    assert state.title == "Example Page"
    assert len(state.elements) == 2
    assert state.elements[0].type == ElementType.BUTTON
    assert state.elements[1].type == ElementType.INPUT

def test_page_state_validation():
    """Test PageState validation"""
    
    # Test invalid URL
    with pytest.raises(ValueError, match="URL must start with http:// or https://"):
        PageState(
            url="invalid-url",
            title="Test",
            elements=[]
        )
    
    # Test valid URL
    state = PageState(
        url="https://example.com",
        title="Test",
        elements=[]
    )
    assert state.url == "https://example.com"

def test_web_element_validation():
    """Test WebElement validation"""
    
    # Test invalid coordinates
    with pytest.raises(ValueError, match="Coordinates must be \\[x, y\\]"):
        WebElement(
            bid="test",
            type=ElementType.BUTTON,
            coordinates=[100]  # Should be [x, y]
        )
    
    # Test valid coordinates
    element = WebElement(
        bid="test",
        type=ElementType.BUTTON,
        coordinates=[100, 200]
    )
    assert element.coordinates == [100, 200]

def test_action_creation():
    """Test Action model creation"""
    
    action = Action(
        action_type=ActionType.CLICK,
        target="btn1",
        coordinates=[100, 200],
        thought="Click the submit button"
    )
    
    assert action.action_type == ActionType.CLICK
    assert action.target == "btn1"
    assert action.coordinates == [100, 200]
    assert action.thought == "Click the submit button"

def test_action_validation():
    """Test Action validation"""
    
    # Test invalid coordinates
    with pytest.raises(ValueError, match="Coordinates must be \\[x, y\\]"):
        Action(
            action_type=ActionType.CLICK,
            coordinates=[100]  # Should be [x, y]
        )
    
    # Test invalid duration
    with pytest.raises(ValueError, match="Duration must be positive"):
        Action(
            action_type=ActionType.WAIT,
            duration=-1.0  # Should be positive
        )

def test_task_context_validation():
    """Test TaskContext validation"""
    
    # Test invalid priority
    with pytest.raises(ValueError, match="Priority must be between 1 and 5"):
        TaskContext(
            intent="Test task",
            priority=6  # Should be 1-5
        )
    
    # Test valid priority
    context = TaskContext(
        intent="Test task",
        priority=3
    )
    assert context.priority == 3

def test_prompts_with_pydantic_models():
    """Test prompts work with Pydantic models"""
    
    # Create PageState
    state = PageState(
        url="https://example.com",
        title="Login Page",
        elements=[
            WebElement(
                bid="btn1",
                type=ElementType.BUTTON,
                text="Login",
                coordinates=[100, 200]
            )
        ]
    )
    
    # Test action generation prompt
    prompt = build_action_generation_prompt(
        state=state,
        intent="Login to the system",
        meta_data={}
    )
    
    assert "Login Page" in prompt
    assert "https://example.com" in prompt
    assert "1 button(s)" in prompt
    
    # Test with dict fallback
    dict_state = {
        "url": "https://example.com",
        "title": "Login Page",
        "elements": []
    }
    
    prompt2 = build_action_generation_prompt(
        state=dict_state,
        intent="Login to the system", 
        meta_data={}
    )
    
    assert "Login Page" in prompt2
    assert "https://example.com" in prompt2

def test_page_state_utility_methods():
    """Test PageState utility methods"""
    
    state = PageState(
        url="https://example.com",
        title="Test Page",
        elements=[
            WebElement(
                bid="btn1",
                type=ElementType.BUTTON,
                text="Submit",
                enabled=True
            ),
            WebElement(
                bid="input1",
                type=ElementType.INPUT,
                text="Username"
            ),
            WebElement(
                bid="link1",
                type=ElementType.LINK,
                text="Help",
                enabled=True
            )
        ]
    )
    
    # Test get_elements_by_type
    buttons = state.get_elements_by_type(ElementType.BUTTON)
    assert len(buttons) == 1
    assert buttons[0].text == "Submit"
    
    # Test get_clickable_elements
    clickable = state.get_clickable_elements()
    assert len(clickable) == 2  # Button and Link
    assert all(e.enabled for e in clickable)
    
    # Test get_input_elements
    inputs = state.get_input_elements()
    assert len(inputs) == 1
    assert inputs[0].type == ElementType.INPUT

if __name__ == "__main__":
    # Run basic tests
    print("Testing Pydantic models...")
    
    try:
        test_page_state_creation()
        test_action_creation()
        test_prompts_with_pydantic_models()
        test_page_state_utility_methods()
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
