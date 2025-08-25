# Shared utilities for page capture and prompt inputs
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

# Ensure project root import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_env_from_dotenv() -> None:
    """Load environment variables from a nearby .env if exists."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / ".env",
    ]
    chosen: Optional[Path] = None
    for p in candidates:
        if p.exists():
            chosen = p
            break
    if not chosen:
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=chosen, override=False)
    except Exception:
        for line in chosen.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def import_capture_module():
    """Import the capture_actree module by file path to avoid package name issues."""
    vab_root = (Path(__file__).parent.parent / "VAB-WebArena-Lite").resolve()
    if str(vab_root) not in sys.path:
        sys.path.insert(0, str(vab_root))
    capture_path = (vab_root / "scripts" / "capture_actree.py").resolve()
    if not capture_path.exists():
        raise FileNotFoundError(f"capture_actree.py not found at {capture_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("_capture_actree_module", str(capture_path))
    if spec is None or spec.loader is None:
        raise ImportError("Failed to create module spec for capture_actree.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_capture_actree_module"] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def capture_observation(
    url: str,
    *,
    render: bool = False,
    wait_time: float = 0.5,
    viewport_width: int = 1280,
    viewport_height: int = 2048,
    current_viewport_only: bool = False,
    post_action: str | None = None,
) -> Dict[str, Any]:
    mod = import_capture_module()
    return mod.capture_actree_observation(  # type: ignore[attr-defined]
        url=url,
        render=render,
        wait_time=wait_time,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        current_viewport_only=current_viewport_only,
        post_action=post_action,
    )


def extract_visible_elements(text_observation: str, max_items: int = 24) -> str:
    """Extract a concise list of visible UI elements from the accessibility text."""
    pattern = re.compile(
        r"\[\d+\]\s+(button|link|textbox|input|combobox|checkbox|radio|menuitem|option|tab|switch|image)\s+'([^']+)'",
        re.IGNORECASE,
    )
    items = []
    seen = set()
    for line in text_observation.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        t = m.group(1).lower()
        label = m.group(2).strip()
        key = (t, label)
        if key in seen:
            continue
        seen.add(key)
        items.append(f"{t}: {label}")
        if len(items) >= max_items:
            break
    return ", ".join(items)


def summarize_page_state(text_observation: str, max_chars: int = 500) -> str:
    snippet = text_observation.strip()
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3] + "..."
    return f"Top accessibility snippet: {snippet}"


def get_element_context(text_observation: str, element_id: str, context_lines: int = 2) -> str:
    """Return the matching line for [element_id] and a small window of context lines around it.

    Args:
        text_observation: Full ACTree text
        element_id: Numeric id string, e.g., "2253"
        context_lines: Number of lines before/after to include
    """
    lines = text_observation.splitlines()
    target_prefix = f"[{element_id}]"
    idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(target_prefix):
            idx = i
            break
    if idx == -1:
        return ""
    start = max(0, idx - context_lines)
    end = min(len(lines), idx + context_lines + 1)
    ctx = lines[start:end]
    return "\n".join(ctx)


# --- Navigation-safe rejected action helpers ---

_LINK_LINE_RE = re.compile(r"^\[(\d+)\]\s+link\s+'([^']+)'\s*,\s*url='([^']+)'", re.IGNORECASE)
_CLICK_ACTION_RE = re.compile(r"^click\((?:'|\")(?P<id>\d+)(?:'|\")\)$", re.IGNORECASE)

def _is_navigation_url(url: str, current_url: str) -> bool:
    url = (url or "").strip()
    if not url or url.startswith("javascript:") or url.startswith("mailto:"):
        return False
    if url.startswith("#"):
        return False
    if "#" in url:
        # Exclude same-page anchors
        base = url.split("#", 1)[0]
        if base == (current_url or "").split("#", 1)[0]:
            return False
    # Heuristic: absolute http(s) and not identical to current
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    if url == current_url:
        return False
    return True


def find_navigation_link_ids(
    text_observation: str,
    current_url: str,
    prefer_path_keywords: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """Return candidate link elements that likely trigger navigation.

    Each item is a tuple: (element_id, label, url). Results are ordered: preferred path keywords first,
    then other absolute links. Filters out quick views and anchors.
    """
    prefer_path_keywords = prefer_path_keywords or []
    candidates_pref: List[Tuple[str, str, str]] = []
    candidates_other: List[Tuple[str, str, str]] = []
    for line in text_observation.splitlines():
        m = _LINK_LINE_RE.match(line.strip())
        if not m:
            continue
        eid, label, url = m.group(1), m.group(2), m.group(3)
        label_lower = label.lower()
        if "quick view" in label_lower:
            continue
        if not _is_navigation_url(url, current_url):
            continue
        # Prefer product/detail pages that typically end with .html
        score_pref = url.endswith(".html")
        # Prefer known path keywords, if provided
        score_kw = any(kw in url for kw in prefer_path_keywords)
        if score_pref or score_kw:
            candidates_pref.append((eid, label, url))
        else:
            candidates_other.append((eid, label, url))
    return candidates_pref + candidates_other


def pick_navigation_rejected_action(
    text_observation: str,
    current_url: str,
    prefer_path_keywords: Optional[List[str]] = None,
) -> Optional[str]:
    """Pick a click('ID') that is highly likely to navigate (URL change)."""
    items = find_navigation_link_ids(text_observation, current_url, prefer_path_keywords)
    if not items:
        return None
    eid, _label, _url = items[0]
    return f"click('{eid}')"


def parse_click_id(action: str) -> Optional[str]:
    """Return element id if action is exactly click('ID') or click("ID")."""
    if not isinstance(action, str):
        return None
    m = _CLICK_ACTION_RE.match(action.strip())
    return m.group("id") if m else None


def get_link_url_by_id(text_observation: str, element_id: str) -> Optional[str]:
    """Find the link URL for a given [element_id] line in the observation."""
    target_prefix = f"[{element_id}]"
    for line in text_observation.splitlines():
        line_stripped = line.strip()
        if not line_stripped.startswith(target_prefix):
            continue
        m = _LINK_LINE_RE.match(line_stripped)
        if m:
            return m.group(3)
        # If it's not a link line, it's not a guaranteed navigation
        return None
    return None


def record_has_nav_rejected(record: Dict[str, Any]) -> bool:
    """Check if record has a rejected click('ID') whose ID corresponds to a link with a URL.

    This is a strong indicator of navigation (URL change) upon execution.
    """
    rejected = record.get("rejected")
    if not isinstance(rejected, dict):
        return False
    action = rejected.get("action")
    eid = parse_click_id(action or "")
    if not eid:
        return False
    text = str(record.get("text_observation", "") or "")
    if not text:
        return False
    url = get_link_url_by_id(text, eid)
    if not url:
        return False
    # Must be absolute http(s)
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    return True
