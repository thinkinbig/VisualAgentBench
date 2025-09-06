from typing import Any, Dict, List, Union

from beartype import beartype

# TypedDicts imported at runtime to avoid heavy dependencies at import time
from browser_env.actions import Action, create_send_message_to_user_action
from browser_env.utils import StateInfo

from .runtime_manager import RuntimeManager
from .types import ObservationData


@beartype
def build_eval_trajectory(
    runtime: RuntimeManager,
    final_answer: str,
) -> List[Union[Action, StateInfo]]:
    """Construct an evaluator-compatible trajectory without images.

    The evaluator expects: [..., StateInfo, Action]
    - StateInfo: a TypedDict with keys {"observation", "info"}
    - Action: must include an "answer" field for string_match tasks
    """
    # Build minimal StateInfo from runtime meta
    obs_text = ObservationData.compose_observation_from_nodes(runtime.get_obs_nodes_info())
    state: StateInfo = {
        "observation": {
            # Only provide text; no image support needed
            "text": obs_text,
        },
        "info": {
            # Provide minimal metadata that might be useful to downstream tools
            "observation_metadata": {
                "current_url": runtime.get_current_url() or "",
                "start_url": runtime.get_start_url() or "",
                "obs_nodes_info": runtime.get_obs_nodes_info() or {},
            },
        },
    }

    # Build final Action carrying the answer for string evaluators
    action: Action = create_send_message_to_user_action(final_answer)

    return [state, action]
