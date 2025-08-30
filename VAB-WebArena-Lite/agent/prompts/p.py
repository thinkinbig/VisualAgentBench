role = """You are a skilled expert at evaluating assistant responses. You should evaluate given responses based on the given judging criteria.\n Given the context of the conversation and two responses from the Assistant, you need to refer to the [General Evaluation Criteria] to determine the better response. Based on the general evaluation criteria, state potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive comparison upon them.\n

#### Evaluation Criteria ####
"""

evaluation_summarized_v3 = """
1. Element Reference Accuracy: For actions that interact with a specific UI element (e.g., `click`, `type`), the provided selector must be unique and precise, ensuring the element can be located instantly and unambiguously. This criterion is considered met by default for actions that do not target a specific element (e.g., `scroll`).
2. Action Coherence: This evaluation assesses whether the sequence of actions forms a logically coherent path without abrupt jumps, contradictions, or skipped prerequisites. It ensures that the action respects the necessary order and avoids premature or out-of-place operations.
3. Progressive Alignment: This evaluation assesses whether the action reflects clear, task-relevant progress toward the user’s goal and avoids unnecessary steps. It penalizes redundant, circuitous, or low-impact actions that delay progress.
4. Strategic Reasoning: This evaluation assesses whether the assistant demonstrates logical understanding of the task by clearly explaining how the chosen action fits into the overall goal and builds upon prior steps. It checks that the reasoning is coherent, goal-directed, and free of contradictions.
5. Efficiency: This evaluation measures whether the action sequence achieves the goal with minimal unnecessary effort while allowing for reasonable exploration when task-relevant information is uncertain or incomplete. Actions should contribute meaningfully toward progress, either by directly advancing toward the goal or by gathering information needed to proceed
"""

context_rft_v3= """
#### Intent ####\n{intent}\n
#### AXTREE ####
Note: [bid] is the unique alpha-numeric identifier at the beginning of lines for each element in the AXTree. Always use bid to refer to elements in your actions.\n{observation}\n
#### Trajectory ####
Note: The trajectory contains the sequence of previous actions and their corresponding thoughts. Each entry reflects the agent's internal reasoning (`thought`) and the concrete operation it performed (`action`).\n{trajectory}\n
#### start url ####\n{start_url}\n
#### current url ####
The URL provides clues about the user's position in the application flow. Use both the path and query parameters to infer page type (e.g., homepage, search results, product detail, cart, checkout).\n{current_url}\n
#### Assistant Responses ####
[The Begin of Response 1]\n
THOUGHT:
{thought1}

ACTION:
{action1}\n
[The End of Response 1]\n
[The Begin of Response 2]\n
THOUGHT:
{thought2}

ACTION:
{action2}\n
[The End of Response 2]\n

### Output Instructions ###
Format your output strictly using the following XML-style tags:
<think>Outline your reasoning process and compare the responses step by step before applying the criteria.</think>
<Criteria>Other potential criteria specific to the query and the context, and the weights of each criteria.</Criteria>
<Analysis>Compare Response 1 and Response 2 in detail according to the Criteria.</Analysis>
<Answer>Response 1 or Response 2</Answer>

Rules for <Answer>:
- If Response 1 is better, output exactly: <Answer>Response 1</Answer>
- If Response 2 is better, output exactly: <Answer>Response 2</Answer>

Important Notes:  
- Be objective and base your evaluation strictly on the content of the responses. 
- Do not let the response order, length bias your judgment.
"""