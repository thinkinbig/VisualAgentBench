from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from llms.types import ThoughtActionPair  # type: ignore
else:
    ThoughtActionPair = object  # fallback to break circular import at runtime

Trajectory = List[ThoughtActionPair]