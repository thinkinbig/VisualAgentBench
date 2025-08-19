batch_reward_evaluation_prompt = """You are an expert evaluator of web agent.
Your task is to assess how helpful multiple given agent actions are in making progress toward the user's goal, based on the current state of the webpage.

# Action space:
The actions you can perform fall into several categories:

Page Operation Actions:
```click [id]```: This action clicks on an element with a specific id on the webpage.
```type [id] [content]```: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0, i.e., ```type [id] [content] [0]```.
```hover [id]```: Hover over an element with id.
```press [key_comb]```:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
```scroll [down]``` or ```scroll [up]```: Scroll the page up or down.

Tab Management Actions:
```new_tab```: Open a new, empty browser tab.
```tab_focus [tab_index]```: Switch the browser's focus to a specific tab using its index.
```close_tab```: Close the currently active tab.

URL Navigation Actions:
```goto [url]```: Navigate to a specific URL.
```go_back```: Navigate to the previously viewed page. Use this strategically when you need to return to a previous page to access different information or continue a multi-step task. For example, after examining a user's profile, use go_back to return to the user list and select the next user for analysis.
```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
```stop [answer]```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket.

Progress Reporting Action:
```send_msg("message")```: Send a progress report or summary to the user. Use this to report intermediate findings, summarize collected information, or provide status updates. After sending a message, continue executing the task - do NOT stop. This is useful for multi-step tasks where you need to communicate progress while continuing to work toward the goal.

# Task Description
Evaluate how helpful each of the given actions is for achieving the goal.
Use the following scale:
**Scoring Criteria (1 to 5):**
- **5 (Very Helpful)**: The action directly and effectively moves toward fulfilling a key part of the goal.
- **4 (Helpful)**: The action contributes meaningfully to progress, though it may require follow-up actions.
- **3 (Somewhat Helpful)**: The action is partially relevant or a preparatory step, but doesn't make immediate progress.
- **2 (Slightly Helpful)**: The action is weakly related to the goal or might only indirectly help.
- **1 (Not Helpful)**: The action is unrelated, redundant, or distracts from the goal.

# Given Information

## User Instruction
{intent}

## Trajectory
{trajectory}

## Current State
### Current URL
{current_url}

### AXTREE
Note: [id] is the unique numeric identifier at the beginning of lines for each element in the accessibility tree. Always use id to refer to elements in your actions.
{text_observation}

## Multiple Actions to Evaluate
{multiple_actions}

## Agent's Response
THOUGHT:
{thought}

ACTION:
{action}

# Output Format
For each action i, reply on a separate line strictly as:
i: REASON: [explanation]
SCORE: [1-5]"""
