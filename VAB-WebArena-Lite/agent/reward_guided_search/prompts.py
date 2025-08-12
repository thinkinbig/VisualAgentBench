"""
Prompts Management System for Reward Guided Search Module
Clean, organized, and maintainable prompt engineering system with enums
"""

from typing import Dict, Any, List, Union
from enum import Enum
from .models import PageState, Action, TaskContext, Feedback

# ============================================================================
# ENUMS - Define prompt types and roles
# ============================================================================

class PromptType(Enum, str):
    """Enumeration of all prompt types"""
    ACTION_GENERATION = "action_generation"
    ACTION_REFINEMENT = "action_refinement"
    WEBSPHERD_EVALUATION = "webspherd_evaluation"
    LIKERT_SCALE_REWARD_EVALUATION = "likert_scale_reward_evaluation"
    WEB_SHEPHERD_REWARD_EVALUATION = "web_shepherd_reward_evaluation"
    WEB_SHEPHERD_CHECKLIST_GENERATION = "web_shepherd_checklist_generation"
    CHECKLIST_BASED_ACTION_REFINEMENT = "checklist_based_action_refinement"
    CONTEXT_ANALYSIS = "context_analysis"
    ERROR_RECOVERY = "error_recovery"

class AgentRole(Enum, str):
    """Enumeration of agent roles"""
    ACTION_GENERATOR = "action_generator"
    ACTION_REFINER = "action_refiner"
    WEBSPHERD_EVALUATOR = "webspherd_evaluator"
    LIKERT_SCALE_REWARD_EVALUATOR = "likert_scale_reward_evaluator"
    WEB_SHEPHERD_REWARD_EVALUATOR = "web_shepherd_reward_evaluator"
    WEB_SHEPHERD_CHECKLIST_GENERATOR = "web_shepherd_checklist_generator"
    CHECKLIST_BASED_ACTION_REFINER = "checklist_based_action_refiner"
    CONTEXT_ANALYZER = "context_analyzer"
    ERROR_RECOVERY_SPECIALIST = "error_recovery_specialist"

# ============================================================================
# SYSTEM PROMPTS - Define system prompts for different roles
# ============================================================================

SYSTEM_PROMPTS = {
    AgentRole.ACTION_GENERATOR.value: {
        "role": "system",
        "content": """You are an expert web automation agent specialized in completing web tasks efficiently and accurately.

Your capabilities:
- Analyze web page states and user intents
- Generate specific, actionable web automation commands
- Understand UI elements and their interactions
- Plan sequential actions to achieve goals
- Adapt to different website layouts and structures

Always provide actions in valid JSON format with the following structure:
{
    "action_type": "click|type|scroll|wait|navigate",
    "coordinates": [x, y],
    "text": "text content if applicable",
    "element_id": "element identifier if available",
    "reasoning": "brief explanation of why this action"
}"""
    },
    
    AgentRole.ACTION_REFINER.value: {
        "role": "system",
        "content": """You are an expert web automation agent focused on refining and improving actions based on feedback.

Your task is to:
- Analyze feedback from previous action evaluations
- Identify areas for improvement in action planning
- Generate refined actions that address specific issues
- Maintain consistency with the overall task goal
- Provide better alternatives when possible

Generate refined actions in valid JSON format with the same structure as the original action."""
    },
    
    AgentRole.WEBSPHERD_EVALUATOR.value: {
        "role": "system",
        "content": """You are an expert evaluator of web agent.
Your task is to assess how helpful a given agent's THOUGHT and ACTION is in making
progress toward the user's goal, based on the current state of the webpage.

You will evaluate actions on a scale of 1-5:
- 5: Very Helpful - directly and effectively moves toward the goal
- 4: Helpful - contributes meaningfully to progress
- 3: Somewhat Helpful - partially relevant or preparatory step
- 2: Slightly Helpful - weakly related or indirectly helpful
- 1: Not Helpful - unrelated, redundant, or distracting

Always provide your response in the specified format with REASON, SCORE, and ADDITIONAL_FEEDBACK."""
    },
    
    AgentRole.WEB_SHEPHERD_REWARD_EVALUATOR.value: {
        "role": "system",
        "content": """You are an expert evaluator of web agent.
Your task is to assess how helpful a given agent's THOUGHT and ACTION is in making
progress toward the user's goal, based on the current state of the webpage.

Your task is to evaluate how well the agent's THOUGHT and ACTION satisfy each item in the checklist
using the task instruction, trajectory (including previously completed steps), current webpage
state, and the agent's latest response.

Start by writing a concise paragraph summarizing the agent's overall performance.
Refer to the reasoning provided in the trajectory, and discuss whether the THOUGHT is
appropriate and the ACTION moves the task forward.

Then, assess each checklist item individually using the following labels:
- Yes: The item is fully and clearly satisfied, either in the current response or previously completed.
- In Progress: There is meaningful partial progress toward completing the item.
- No: The item is not satisfied due to ambiguity, insufficient evidence, or lack of progress.

Always provide your response in the specified format with REASON, SCORE, and ADDITIONAL_FEEDBACK."""
    },
    
    AgentRole.WEB_SHEPHERD_CHECKLIST_GENERATOR.value: {
        "role": "system",
        "content": """You are an AI assistant tasked with generating structured checklists that highlight key
subgoals necessary to complete a task.

Your task is to:
1. Identify essential high-level subgoals that represent significant steps
2. Consolidate related user actions into single subgoals
3. Prioritize only critical interactions for meaningful progression
4. Provide concise subgoal analysis before the checklist
5. Ensure the checklist contains only essential steps (max 5 items)

Focus on:
- User interactions that lead to noticeable page transitions
- Meaningful changes in system state
- Critical steps necessary for task completion
- Logical sequence of subgoals

Avoid including:
- Minor or unnecessary steps (scroll, hover)
- Separate items for closely related actions
- More than 5 critical subgoals"""
    },
    
    AgentRole.CONTEXT_ANALYZER.value: {
        "role": "system",
        "content": """You are an expert web page context analyzer specialized in understanding page structure and purpose.

Your task is to:
- Analyze the current page context and purpose
- Identify available actions and navigation options
- Assess task relevance and potential obstacles
- Provide strategic planning guidance
- Understand page structure and element relationships

Focus on:
- Page purpose and functionality
- Available interactive elements
- Navigation and action possibilities
- Task completion strategies"""
    },
    
    AgentRole.ERROR_RECOVERY_SPECIALIST.value: {
        "role": "system",
        "content": """You are an expert error recovery specialist for web automation tasks.

Your task is to:
- Analyze error situations and their causes
- Generate recovery strategies and alternative approaches
- Maintain task progress despite failures
- Prevent error recurrence
- Provide graceful error handling

Focus on:
- Understanding what went wrong
- Finding alternative solutions
- Maintaining forward progress
- Learning from failures"""
    }
}

# ============================================================================
# USER PROMPT BUILDERS - Functions to build user prompts
# ============================================================================

def build_action_generation_prompt(state: Union[PageState, Dict[str, Any]], intent: str, meta_data: Dict[str, Any]) -> str:
    """Build action generation prompt"""
    
    # Convert dict to PageState if needed
    if isinstance(state, dict):
        try:
            state = PageState(**state)
        except Exception as e:
            # Fallback to dict if conversion fails
            pass
    
    # Extract data with type safety
    if isinstance(state, PageState):
        url = state.url
        title = state.title
        elements = state.elements
        
        # Count element types using Pydantic model
        element_counts = {}
        for element in elements:
            element_type = element.type.value if hasattr(element.type, 'value') else str(element.type)
            element_counts[element_type] = element_counts.get(element_type, 0) + 1
    else:
        # Fallback for dict
        url = state.get('url', 'Unknown')
        title = state.get('title', 'Unknown')
        elements = state.get('elements', [])
        
        # Count element types
        element_counts = {}
        for element in elements:
            element_type = element.get('type', 'unknown')
            element_counts[element_type] = element_counts.get(element_type, 0) + 1
    
    element_summary = [f"{count} {elem_type}(s)" for elem_type, count in element_counts.items()]
    
    return f"""Task: {intent}

Current Web Page State:
- URL: {url}
- Page Title: {title}
- Interactive Elements: {', '.join(element_summary) if element_summary else 'None detected'}

Available Action Types:
1. **click** - Click on an interactive element
   - Use when: Clicking buttons, links, checkboxes, etc.
   - Requires: coordinates or element_id
   
2. **type** - Enter text into input fields
   - Use when: Filling forms, search boxes, etc.
   - Requires: coordinates/element_id + text content
   
3. **scroll** - Navigate page content
   - Use when: Need to see more content or elements
   - Requires: scroll direction and amount
   
4. **wait** - Pause execution
   - Use when: Waiting for page load, animations, etc.
   - Requires: duration in seconds
   
5. **navigate** - Go to a specific URL
   - Use when: Direct navigation is needed
   - Requires: target URL

Task Analysis:
- Current goal: {intent}
- Page context: {title}
- Available elements: {len(elements)} interactive elements

Generate the next action that will best progress towards completing this task. Consider:
- What is the most logical next step?
- Which element should be interacted with?
- What information needs to be provided?
- How to ensure the action is safe and effective?

Response Format: Provide ONLY a valid JSON object with the action details."""

def build_action_refinement_prompt(state: Union[PageState, Dict[str, Any]], 
                                intent: str, 
                                meta_data: Dict[str, Any], 
                                feedback: Union[Feedback, Dict[str, Any]], 
                                refinement_step: int) -> str:
    """Build action refinement prompt"""
    
    url = state.get('url', 'Unknown')
    title = state.get('title', 'Unknown')
    
    thought = feedback.get('thought', 'No thought process provided')
    checklist = feedback.get('checklist', [])
    suggestions = feedback.get('suggestions', [])
    
    checklist_text = "\n".join([f"- {item}" for item in checklist]) if checklist else "No specific criteria provided"
    suggestions_text = "\n".join([f"- {suggestion}" for suggestion in suggestions]) if suggestions else "No specific suggestions provided"
    
    return f"""Task: {intent}

Current Web Page State:
- URL: {url}
- Page Title: {title}

Previous Action Feedback (Refinement Step {refinement_step}):

Thought Process:
{thought}

Evaluation Checklist:
{checklist_text}

Improvement Suggestions:
{suggestions_text}

Refinement Instructions:
Based on the feedback above, generate a refined action that addresses the identified issues. The refined action should:

1. **Address Specific Problems**: Target the exact issues mentioned in the feedback
2. **Improve Effectiveness**: Make the action more likely to succeed
3. **Maintain Task Progress**: Ensure the refined action still moves toward the goal
4. **Consider Context**: Take into account the current page state and available elements
5. **Be More Specific**: Provide clearer, more actionable instructions

Think about:
- What went wrong with the previous action?
- How can the action be made more precise?
- Are there alternative approaches suggested in the feedback?
- What additional context or information is needed?

Generate a refined action that incorporates these improvements.

Response Format: Provide ONLY a valid JSON object with the refined action details."""

def build_webspherd_evaluation_prompt(state: Union[PageState, Dict[str, Any]], 
                                    action: Union[Action, Dict[str, Any]], 
                                    intent: str, 
                                    meta_data: Dict[str, Any]) -> str:
    """Build WebSpherd-style evaluation prompt"""
    
    url = state.get('url', 'Unknown')
    title = state.get('title', 'Unknown')
    text_observation = state.get('text_observation', 'No AXTree information available')
    
    action_type = action.get('action_type', 'unknown')
    coordinates = action.get('coordinates', 'N/A')
    text = action.get('text', 'N/A')
    element_id = action.get('element_id', 'N/A')
    thought = action.get('thought', 'No thought provided')
    
    return f"""You are an expert evaluator of web agent.
Your task is to assess how helpful a given agent's THOUGHT and ACTION is in making
progress toward the user's goal, based on the current state of the webpage.

# Action space:
- click: Click on an element
- type: Type text into an input field  
- scroll: Scroll the page
- wait: Wait for a condition
- navigate: Navigate to a URL

# Task Description
Evaluate how helpful the given thought and action is for achieving the goal.
Use the following scale:

**Scoring Criteria (1 to 5):**
- **5 (Very Helpful)**: The action directly and effectively moves toward fulfilling a key part
of the goal.
- **4 (Helpful)**: The action contributes meaningfully to progress, though it may require
follow-up actions.
- **3 (Somewhat Helpful)**: The action is partially relevant or a preparatory step, but doesn't
make immediate progress.
- **2 (Slightly Helpful)**: The action is weakly related to the goal or might only indirectly
help.
- **1 (Not Helpful)**: The action is unrelated, redundant, or distracts from the goal.

# Given Information
## User Instruction
{intent}

## Current State
### Current URL
{url}

### AXTREE
Note: [bid] is the unique alpha-numeric identifier at the beginning of lines for each element
in the AXTree. Always use bid to refer to elements in your actions.
{text_observation}

### SOM Image Screenshot
Here is the current image screenshot of the page, annotated with bounding boxes and
corresponding bids:
<IMAGE_PLACEHOLDER>

## Agent's Response
THOUGHT:
{thought}

ACTION:
{action_type} - {text} at {coordinates} (element: {element_id})

# Output Format
Please return your response in the following format:

REASON:
[Your explanation for the score]

SCORE:
[1-5]

ADDITIONAL_FEEDBACK:
[Any additional insights, suggestions, or concerns]"""

def build_web_shepherd_checklist_generation_prompt(intent: str, start_url: str, meta_data: Dict[str, Any]) -> str:
    """Build WebShepherd checklist generation prompt"""
    
    return f"""You are an AI assistant tasked with generating structured checklists that highlight key
subgoals necessary to complete a task.

## Task Description
User Instruction (Goal): {intent}
Start Website URL: {start_url}

## Guidelines for Checklist Generation
1. **Identify Essential High-Level Subgoals:**
   - A subgoal should represent a significant step involving user interaction that leads to
     noticeable page transitions or meaningful changes in system state.
   - Consolidate closely related user actions (such as applying multiple filters or selecting
     several options) into a single subgoal, rather than separate checklist items for each action.
   - Prioritize only the most critical interactions necessary for meaningful progression,
     avoiding the inclusion of minor or unnecessary steps (e.g., scroll, hover).

2. **Provide a Concise Subgoal Analysis:**
   - Before creating the checklist, offer a brief paragraph summarizing the main subgoals,
     emphasizing significant transitions or page-level interactions.

3. **Ensure Clear Goal:**
   - If multiple related interactions occur (e.g., setting filters 1, 2, and 3), combine them into one
     subgoal with clear criteria verifying all required conditions.
   - The checklist should contain only essential steps, explicitly excluding unnecessary
     actions, and should not exceed five critical subgoals. It is not necessary to use all five
     checklist items if fewer steps adequately represent the essential subgoals.

### Output Format
Before generating the checklist, first produce a concise subgoal analysis in a single
paragraph summarizing the required interactions. Then, based on this, generate the checklist
following the format below:

[SUBGOAL ANALYSIS]
[One-paragraph summary explaining the key subgoals and their logical
sequence in task completion.]

[CHECKLISTS]
Checklist X:
[Short title of the action/goal]
- Goal:
[Brief description of the subgoal at this stage, emphasizing the
purpose of the action.]"""

def build_context_analysis_prompt(state: Union[PageState, Dict[str, Any]], intent: str, meta_data: Dict[str, Any]) -> str:
    """Build context analysis prompt"""
    
    url = state.get('url', 'Unknown')
    title = state.get('title', 'Unknown')
    elements = state.get('elements', [])
    
    element_types = {}
    for element in elements:
        elem_type = element.get('type', 'unknown')
        element_types[elem_type] = element_types.get(elem_type, 0) + 1
    
    return f"""Task: {intent}

Current Web Page Analysis:
- URL: {url}
- Page Title: {title}
- Page Structure: {dict(element_types)}

Context Analysis:
Analyze the current page context to understand:

1. **Page Purpose**: What is this page designed for?
2. **Available Actions**: What can the user do here?
3. **Task Relevance**: How does this page relate to the goal?
4. **Navigation Options**: What are the next possible steps?
5. **Potential Obstacles**: What might prevent task completion?

Strategic Planning:
Based on this analysis, what should be the next logical action?

Consider:
- Is this the right page for the task?
- What elements are most relevant?
- What sequence of actions makes sense?
- Are there any dependencies or prerequisites?

Provide a strategic analysis that guides action selection."""

def build_error_recovery_prompt(state: Union[PageState, Dict[str, Any]], 
                              error_info: Dict[str, Any], 
                              intent: str, 
                              meta_data: Dict[str, Any]) -> str:
    """Build error recovery prompt"""
    
    url = state.get('url', 'Unknown')
    title = state.get('title', 'Unknown')
    error_type = error_info.get('type', 'unknown')
    error_message = error_info.get('message', 'No error details')
    
    return f"""Task: {intent}

Current Web Page State:
- URL: {url}
- Page Title: {title}

Error Encountered:
- Type: {error_type}
- Message: {error_message}

Recovery Strategy:
The previous action failed. You need to generate a recovery action that:

1. **Addresses the Error**: Understand what went wrong
2. **Provides Alternative**: Suggest a different approach
3. **Maintains Progress**: Keep moving toward the goal
4. **Prevents Recurrence**: Avoid the same error

Consider:
- What caused the failure?
- Are there alternative ways to achieve the same goal?
- Should we wait for the page to stabilize?
- Do we need to refresh or navigate differently?

Generate a recovery action that handles this error gracefully.

Response Format: Provide ONLY a valid JSON object with the recovery action details."""



# ============================================================================
# PROMPT MANAGER FUNCTIONS - Prompt management functions
# ============================================================================

def get_system_prompt(role: AgentRole) -> Dict[str, str]:
    """Get system prompt for specified role"""
    if role not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown role: {role}")
    return SYSTEM_PROMPTS[role]

def get_user_prompt(prompt_type: PromptType, **kwargs) -> Dict[str, str]:
    """Get user prompt based on type"""
    
    if prompt_type == PromptType.ACTION_GENERATION.value:
        content = build_action_generation_prompt(
            kwargs.get('state', {}),
            kwargs.get('intent', ''),
            kwargs.get('meta_data', {})
        )
    elif prompt_type == PromptType.ACTION_REFINEMENT.value:
        content = build_action_refinement_prompt(
            kwargs.get('state', {}),
            kwargs.get('intent', ''),
            kwargs.get('meta_data', {}),
            kwargs.get('feedback', {}),
            kwargs.get('refinement_step', 1)
        )
    elif prompt_type == PromptType.WEBSPHERD_EVALUATION.value:
        content = build_webspherd_evaluation_prompt(
            kwargs.get('state', {}),
            kwargs.get('action', {}),
            kwargs.get('intent', ''),
            kwargs.get('meta_data', {})
        )
    elif prompt_type == PromptType.WEB_SHEPHERD_CHECKLIST_GENERATION.value:
        content = build_web_shepherd_checklist_generation_prompt(
            kwargs.get('intent', ''),
            kwargs.get('start_url', ''),
            kwargs.get('meta_data', {})
        )
    elif prompt_type == PromptType.CONTEXT_ANALYSIS.value:
        content = build_context_analysis_prompt(
            kwargs.get('state', {}),
            kwargs.get('intent', ''),
            kwargs.get('meta_data', {})
        )
    elif prompt_type == PromptType.ERROR_RECOVERY.value:
        content = build_error_recovery_prompt(
            kwargs.get('state', {}),
            kwargs.get('error_info', {}),
            kwargs.get('intent', ''),
            kwargs.get('meta_data', {})
        )
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return {
        "role": "user",
        "content": content
    }

def get_conversation_messages(prompt_type: PromptType, role: AgentRole, **kwargs) -> List[Dict[str, str]]:
    """Get complete conversation messages including system and user prompts"""
    messages = [
        get_system_prompt(role),
        get_user_prompt(prompt_type, **kwargs)
    ]
    return messages



def get_available_prompts() -> List[str]:
    """Get list of available prompt types"""
    return [pt.value for pt in PromptType]

def get_available_roles() -> List[str]:
    """Get list of available roles"""
    return list(SYSTEM_PROMPTS.keys())

# ============================================================================
# SIMPLE USAGE EXAMPLES - Simple usage examples
# ============================================================================

def example_usage():
    """Usage example"""
    
    print("📚 Prompts System Usage Examples:")
    print("-" * 40)
    
    # 1. Get system prompt
    system_prompt = get_system_prompt("action_generator")
    print(f"1. System prompt for action_generator: {system_prompt['role']}")
    
    # 2. Get user prompt
    mock_state = {"url": "https://example.com", "title": "Example", "elements": []}
    user_prompt = get_user_prompt("action_generation", 
                                 state=mock_state, 
                                 intent="Click the button", 
                                 meta_data={})
    print(f"2. User prompt for action_generation: {user_prompt['role']}")
    
    # 3. Get complete conversation messages
    messages = get_conversation_messages("action_generation", 
                                       "action_generator",
                                       state=mock_state, 
                                       intent="Click the button", 
                                       meta_data={})
    print(f"3. Full conversation: {len(messages)} messages")
    
    # 4. View available prompt types and roles
    print(f"4. Available prompt types: {len(get_available_prompts())}")
    print(f"5. Available roles: {len(get_available_roles())}")
    
    print("\n✅ Example usage completed!")

if __name__ == "__main__":
    example_usage()
