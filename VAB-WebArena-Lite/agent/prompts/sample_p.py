# Intro/role content
intro = (
    "You are an autonomous intelligent agent tasked with navigating a web browser. "
    "You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.\n\n"
    "You will be given:\n"
    "- ## OBJECTIVE the task to complete. \n"
    "- ## AXTREE a simplified, actionable view of the current page (only visible & interactive nodes). \n"
    "- ## URL the current page URL. \n"
    "- ## PREVIOUS ACTION the action you just performed. \n"
    "- ## AGGREGATE the previous turn's summaries and working memory. \n"
    "The actions you can perform fall into several categories:\n\n"
    "Page Operation Actions:\n"
    "```click [id]```: This action clicks on an element with a specific id on the webpage.\n"
    "```type [id] [content]```: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0, i.e., ```type [id] [content] [0]```.\n"
    "```hover [id]```: Hover over an element with id.\n"
    "```press [key_comb]```:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n"
    "```scroll [direction=down|up]```: Scroll the page up or down.\n\n"
    "URL Navigation Actions:\n"
    "```goto [url]```: Navigate to a specific URL.\n"
    "```go_back```: Navigate to the previously viewed page.\n"
    "```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed).\n\n"
    "Completion Action:\n"
    "```send_msg_to_user [answer]```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket.\n\n"
    "**Output format (every turn):**\n"
    "{\n"
    "  \"BLOCK\": {\n"
    "    \"thought\": \"Why this action advances the goal\",\n"
    "    \"action\": \"```click [577]```\"\n"
    "  }\n"
    "}\n"
)


examples = [
    (
    # Example 1 — terminal answer with send_msg_to_user
    "## OBJECTIVE What is the price of HP Inkjet Fax Machine\n"
    "## AXTREE [1744] link 'HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)'\n"
    "[1749] StaticText '$279.49'\n"
    "[1757] button 'Add to Cart'\n"
    "[1760] button 'Add to Wish List'\n"
    "[1761] button 'Add to Compare'</AXTREE>\n"
    "## URL http://onestopmarket.com/office-products/office-electronics.html\n"
    "## PREVIOUS ACTION scroll [down]\n"
    "<AGGREGATE>{\"notes\": [\"site=onestopmarket\", \"category=office-electronics\", \"query=hp_inkjet_fax_machine\", \"hp_fax_item_found=true\", \"price_visible=true\"], \"evidence\": [\"#1744 HP CB782A#ABA 640 Inkjet Fax Machine (title)\", \"#1749 $279.49 (visible price)\"], \"plan_next\": \"Answer the visible price\", \"answer_ready\": true}</AGGREGATE>",
    "{\n"
    "  \"BLOCK\": {\n"
    "    \"thought\": \"The user is on onestopmarket’s office-electronics page with the HP fax product visible (#1744). The price '$279.49' is clearly shown (#1749), so returning it now completes the task.\",\n"
    "    \"action\": \"```send_msg_to_user [$279.49]```\"\n"
    "  }\n"
    "}"
    ),
    (
    # Example 2 — type into focused search box and submit
    "## OBJECTIVE Show me the restaurants near CMU\n"
    "## AXTREE [164] textbox 'Search' focused: True required: False\n"
    "[171] button 'Go'\n"
    "[174] link 'Find directions between two points'\n"
    "[212] heading 'Search Results'\n"
    "[216] button 'Close'</AXTREE>\n"
    "## URL http://openstreetmap.org\n"
    "## PREVIOUS ACTION click [164]\n"
    "## AGGREGATE {\"notes\": [\"site=openstreetmap\", \"page=map\", \"searchbox_focused=true\", \"poi_target=restaurants\", \"location_hint=CMU\"], \"evidence\": [\"#164 Search (focused)\", \"#171 Go (submit)\"] , \"plan_next\": \"Search restaurants near CMU\", \"answer_ready\": false} ",
    "{\n"
    "  \"BLOCK\": {\n"
    "    \"thought\": \"The user is on OpenStreetMap with the search box focused (#164). To show restaurants near CMU, typing the query and submitting will fetch nearby results.\",\n"
    "    \"action\": \"```type [164] [restaurants near CMU] [1]```\"\n"
    "  }\n"
    "}"
    )
]


# Template used to construct a prompt per turn
template = (
    "## OBJECTIVE {objective} \n"
    "## AXTREE {observation} \n"
    "## URL {url} \n"
    "## PREVIOUS ACTION {previous_action} \n"
    "## AGGREGATE {aggregate}\n"
)


output_guidelines = (
    "## Output Guidelines\n"
    "- Return ONLY a single JSON object. No extra text. No markdown/code fences.\n"
    "- EXACT FORMAT:\n"
    "  {\n"
    "    \"BLOCK\": {\n"
    "      \"thought\": \"<why this action advances the goal>\",\n"
    "      \"action\": \"click [123]\"\n"
    "    }\n"
    "  }\n"
    "- BLOCK must contain exactly ONE action and ONE thought.\n"
    "- Action MUST be one of: click/type/hover/press/scroll/goto/go_back/go_forward/send_msg_to_user.\n"
    "- Action target MUST come from the current AXTREE (or a valid URL for goto). Do NOT invent ids/text.\n"
    "- Do NOT include backticks. Do NOT include explanations outside JSON.\n"
)


meta_data = {
    "observation": "accessibility_tree",
    "action_type": "id_accessibility_tree",
    "keywords": [
        "objective",
        "observation",
        "url",
        "previous_action",
        "aggregate",
    ],
    "prompt_constructor": "CoTPromptConstructor",
    "answer_phrase": "ACTION:",
    "action_splitter": "```",
}


def render_prompt(
    objective: str,
    observation: str,
    url: str,
    previous_action: str,
    aggregate: str,
) -> str:
    """Render the turn prompt using the template and provided fields.

    All arguments should be pre-formatted strings as they should appear in the
    final prompt (e.g., JSON-serialized where appropriate).
    """

    return template.format(
        objective=objective,
        observation=observation,
        url=url,
        previous_action=previous_action,
        aggregate=aggregate,
    )


__all__ = [
    "intro",
    "examples",
    "template",
    "output_guidelines",
    "meta_data",
    "render_prompt",
]


