import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, TypedDict, Union, Optional

import numpy as np
import numpy.typing as npt
from beartype import beartype
from PIL import Image

try:
    from vertexai.preview.generative_models import Image as VertexImage
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image')


@dataclass
class DetachedPage:
    url: str
    content: str  # html


@beartype
def png_bytes_to_numpy(png: bytes) -> npt.NDArray[np.uint8]:
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))


def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64


def pil_to_vertex(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_vertex = VertexImage.from_bytes(byte_data)
    return img_vertex


class DOMNode(TypedDict):
    nodeId: str
    nodeType: str
    nodeName: str
    nodeValue: str
    attributes: str
    backendNodeId: str
    parentId: str
    childIds: list[str]
    cursor: int
    union_bound: list[float] | None
    center: list[float] | None


class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: int
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None
    center: list[float] | None


class BrowserConfig(TypedDict):
    win_upper_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig


AccessibilityTree = list[AccessibilityTreeNode]
DOMTree = list[DOMNode]

Observation = str | npt.NDArray[np.uint8]


class StateInfo(TypedDict):
    """State information passed between environment and agent.

    - observation: modality-keyed observation (e.g., {"text": str, "image": np.ndarray})
    - info: auxiliary data including page handle and observation metadata
    """
    observation: Dict[str, Observation]
    info: Dict[str, Any]


@beartype
def extract_current_url(info: Dict[str, Any], fallback: Optional[str] = None) -> str:
    """Extract current URL from environment info dictionary.
    
    Args:
        info: Environment info dictionary containing page information
        fallback: Fallback URL if extraction fails
        
    Returns:
        Current page URL or fallback URL
    """
    try:
        page = info.get("page")
        if hasattr(page, "url"):
            return page.url
    except Exception:
        pass
    return fallback or ""


@beartype
def extract_obs_nodes_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract obs_nodes_info from environment info dictionary.
    
    Args:
        info: Environment info dictionary containing observation metadata
        
    Returns:
        Dictionary containing observation nodes information
    """
    try:
        om = info.get("observation_metadata", {})
        if isinstance(om.get("obs_nodes_info"), dict):
            return om.get("obs_nodes_info")
        if isinstance(om.get("text", {}), dict) and isinstance(om.get("text", {}).get("obs_nodes_info"), dict):
            return om.get("text", {}).get("obs_nodes_info")
        if isinstance(om.get("image", {}), dict) and isinstance(om.get("image", {}).get("obs_nodes_info"), dict):
            return om.get("image", {}).get("obs_nodes_info")
    except Exception:
        pass
    return {}

