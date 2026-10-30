"""SceneRef - walk through Blender / Daz Studio scenes inside ComfyUI and use the
view as a reference (colour, depth, normal, mask) for image generation."""
from .scene_ref.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .scene_ref import routes  # noqa: F401  (registers the HTTP routes)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
