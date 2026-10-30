"""Scene Reference Viewer - a Blender / Daz scene you walk through on the node,
turned into reference images (colour, depth, normal, mask) for generation.

The 3D view is drawn in the browser (three.js). When Run is pressed, the
graphToPrompt hook in web/scene_ref.js renders every requested pass at the
output size, uploads the PNGs to ComfyUI's temp folder and puts their names
in the hidden `scene_ref_capture` input. This node only loads those files.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import torch
from PIL import Image

import folder_paths

from . import jobs

CAPTURE_SUBFOLDER = "scene_ref"
HIDDEN = "scene_ref_capture"
# Exactly what the browser upload produces; /prompt is unauthenticated, so the
# name is checked before any path is built from it.
CAPTURE_RE = re.compile(r"^scene_ref/[A-Za-z0-9_-]{8,100}\.png \[temp\]$")
PASSES = ("image", "depth", "normal", "mask")
FIGURE_PASSES = ("composite", "pose", "figmap")  # composite always; pose + figmap with figures
SHOT_TYPE = "SCENEREF_SHOT"

NO_CAPTURE = (
    "Scene Reference Viewer: no picture arrived from the browser. The view is rendered in the "
    "ComfyUI page when you press Run, so queue the workflow from the page, and make sure the node "
    "shows the loaded scene (not still converting)."
)


def _capture_path(name):
    if not isinstance(name, str) or not CAPTURE_RE.match(name):
        return None
    try:
        path = folder_paths.get_annotated_filepath(name)
    except Exception:
        return None
    root = os.path.realpath(os.path.join(folder_paths.get_temp_directory(), CAPTURE_SUBFOLDER))
    real = os.path.realpath(path)
    if not real.startswith(root + os.sep) or not os.path.isfile(real):
        return None
    return real


def _fit(im, size, resample):
    """Resize to `size`; when the shape differs (the capture was framed for another size),
    crop the centre to the new shape first, so nothing gets stretched."""
    if not size or im.size == size:
        return im
    w, h = im.size
    want = size[0] / size[1]
    if abs(w / h - want) > 0.01:
        if w / h > want:
            cw = round(h * want)
            im = im.crop(((w - cw) // 2, 0, (w - cw) // 2 + cw, h))
        else:
            ch = round(w / want)
            im = im.crop((0, (h - ch) // 2, w, (h - ch) // 2 + ch))
    return im.resize(size, resample)


def _load_rgb(path, size=None):
    with Image.open(path) as im:
        im = _fit(im.convert("RGB"), size, Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _load_grey(path, size=None):
    with Image.open(path) as im:
        im = _fit(im.convert("L"), size, Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _load_figmap(path, size):
    """figmap PNG: R = full-silhouette bits (bit i = figure i), G = visible figure id."""
    with Image.open(path) as im:
        im = _fit(im.convert("RGB"), size, Image.NEAREST)
        arr = np.asarray(im)
    return arr[..., 0].astype(np.int32), arr[..., 1].astype(np.int32)


def _parse(raw):
    try:
        st = json.loads(raw) if isinstance(raw, str) and raw else {}
    except ValueError:
        st = {}
    return st if isinstance(st, dict) else {}


class SceneRefViewer:
    DESCRIPTION = (
        "Opens a Blender (.blend) or Daz Studio (.duf) scene from disk, converts it once in the "
        "background (Blender / Daz run minimized; the result is cached), and shows it in a 3D view "
        "on the node. Walk through it like a game (hold right mouse + WASD), orbit, jump to the "
        "scene's own cameras, save your own shots. On Run the exact framed view comes out as a "
        "colour picture, a depth map and a normal map for ControlNet, plus a mask and the camera."
    )
    CATEGORY = "SceneRef"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "STRING", "STRING", "INT", "INT",
                    "IMAGE", "IMAGE", "MASK", "MASK", "STRING", SHOT_TYPE)
    RETURN_NAMES = ("image", "depth", "normal", "mask", "camera_json", "glb_path", "width", "height",
                    "composite", "pose", "figure_masks", "figure_silhouettes", "figures_json", "shot")
    OUTPUT_TOOLTIPS = (
        "The view as seen on the node (colour or clay look), at width x height. A batch when "
        "'render all saved shots' is on.",
        "Depth: near is white, far is black (inverse depth, like MiDaS / Depth Anything).",
        "Normal map in camera space (OpenGL convention: red right, green up, blue towards you).",
        "White where there is geometry, black where you see background.",
        "The camera(s) used, as JSON (position, rotation quaternion, vertical fov, Y-up metres).",
        "The converted scene file (GLB) in input/3d/scene_ref, for other 3D nodes.",
        "Picture width.",
        "Picture height.",
        "The view with the coloured mannequins in it (what the image model copies placement, size and "
        "occlusion from). Same as image when there are no figures.",
        "OpenPose skeletons (COCO-18) of the figures; joints hidden behind furniture are left out. "
        "Black when there are no figures.",
        "One mask per figure: the part of it you actually see (a figure behind a table has no legs here). "
        "Shot-major order with several shots.",
        "One mask per figure: its whole silhouette, ignoring what is in front of it.",
        "Per figure: colour, height, pose, pixel boxes (visible, full, head), hidden body parts, facing, "
        "keypoints. Feeds prompts and the face pass.",
        "Everything above in one wire, for the SceneRef Studio node.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_path": ("STRING", {"default": "", "tooltip": "Full path of a .blend, .duf or .glb file."}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
            },
            "hidden": {
                HIDDEN: ("STRING", {"default": "{}"}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, scene_path, **kwargs):
        if not isinstance(scene_path, str) or not scene_path.strip():
            return "Scene Reference Viewer: no scene picked. Use Browse on the node."
        return True

    def run(self, scene_path, width, height, **kwargs):
        st = _parse(kwargs.get(HIDDEN))
        shots = st.get("shots")
        if not isinstance(shots, list) or not shots:
            raise ValueError(NO_CAPTURE)
        size = (int(width), int(height))
        out = {p: [] for p in PASSES + ("composite", "pose")}
        figure_masks, figure_full, figure_info = [], [], []
        for shot in shots[:64]:
            if not isinstance(shot, dict):
                raise ValueError(NO_CAPTURE)
            paths = {p: _capture_path(shot.get(p)) for p in PASSES}
            if not all(paths.values()):
                raise ValueError(NO_CAPTURE)
            out["image"].append(_load_rgb(paths["image"], size))
            out["depth"].append(_load_rgb(paths["depth"], size))
            out["normal"].append(_load_rgb(paths["normal"], size))
            out["mask"].append(_load_grey(paths["mask"], size))
            comp = _capture_path(shot.get("composite"))
            out["composite"].append(_load_rgb(comp, size) if comp else out["image"][-1])
            pose = _capture_path(shot.get("pose"))
            out["pose"].append(_load_rgb(pose, size) if pose else torch.zeros(size[1], size[0], 3))
            cam = shot.get("camera") if isinstance(shot.get("camera"), dict) else {}
            figs = cam.get("figures") if isinstance(cam.get("figures"), list) else []
            figure_info.append(figs)
            figmap = _capture_path(shot.get("figmap"))
            if figs and figmap:
                full, vis = _load_figmap(figmap, size)
                for i in range(min(len(figs), 8)):
                    figure_masks.append(torch.from_numpy((vis == i + 1).astype(np.float32)))
                    figure_full.append(torch.from_numpy(((full >> i) & 1).astype(np.float32)))
            else:
                figure_masks.append(torch.zeros(size[1], size[0]))
                figure_full.append(torch.zeros(size[1], size[0]))
        cams = [s.get("camera") for s in shots if isinstance(s, dict)]
        n_figs = [min(len(f), 8) if f else 0 for f in figure_info]
        shot = {"image": torch.stack(out["image"]), "composite": torch.stack(out["composite"]),
                "depth": torch.stack(out["depth"]), "normal": torch.stack(out["normal"]),
                "pose": torch.stack(out["pose"]), "figure_masks": torch.stack(figure_masks),
                "figure_silhouettes": torch.stack(figure_full), "figures": figure_info,
                "figures_per_shot": n_figs, "cameras": cams, "width": size[0], "height": size[1]}
        key = st.get("key")
        glb = ""
        if isinstance(key, str) and jobs.KEY_RE.match(key):
            p = jobs.paths_for(key)[0]
            glb = p if os.path.isfile(p) else ""
        # The size it ran at goes back to the viewer, which frames to it next time when
        # width / height come from a wire it can't read.
        return {"ui": {"sr_size": [size[0], size[1]]}, "result": (
            torch.stack(out["image"]),
            torch.stack(out["depth"]),
            torch.stack(out["normal"]),
            torch.stack(out["mask"]),
            json.dumps(cams if len(cams) > 1 else cams[0], indent=1),
            glb,
            size[0],
            size[1],
            torch.stack(out["composite"]),
            torch.stack(out["pose"]),
            torch.stack(figure_masks),
            torch.stack(figure_full),
            json.dumps(figure_info if len(figure_info) > 1 else figure_info[0], indent=1),
            shot,
        )}


from .characters import SceneRefCharacter  # noqa: E402
from .loras import SceneRefLoraStack  # noqa: E402
from .studio import SceneRefLoraFile, SceneRefStudio  # noqa: E402
from .expander import SceneRefPromptExpander  # noqa: E402
from .charmaker import SceneRefCharacterMaker  # noqa: E402
from .refmodmaker import SceneRefRefModMaker  # noqa: E402
from .outputs import SceneRefOutput, SceneRefResolution  # noqa: E402
from .scenes import SceneRefScene, SceneRefSceneMaker  # noqa: E402

NODE_CLASS_MAPPINGS = {"SceneRefViewer": SceneRefViewer, "SceneRefCharacter": SceneRefCharacter,
                       "SceneRefLoraStack": SceneRefLoraStack, "SceneRefStudio": SceneRefStudio,
                       "SceneRefLoraFile": SceneRefLoraFile, "SceneRefPromptExpander": SceneRefPromptExpander,
                       "SceneRefCharacterMaker": SceneRefCharacterMaker, "SceneRefOutput": SceneRefOutput,
                       "SceneRefResolution": SceneRefResolution, "SceneRefRefModMaker": SceneRefRefModMaker,
                       "SceneRefScene": SceneRefScene, "SceneRefSceneMaker": SceneRefSceneMaker}
NODE_DISPLAY_NAME_MAPPINGS = {"SceneRefViewer": "Scene Reference Viewer (Blender / Daz)",
                              "SceneRefCharacter": "SceneRef Character (.char / RefMod / pictures)",
                              "SceneRefLoraStack": "SceneRef LoRA Stack",
                              "SceneRefStudio": "SceneRef Studio (control panel)",
                              "SceneRefLoraFile": "SceneRef character LoRA (internal)",
                              "SceneRefPromptExpander": "SceneRef Prompt Expander (vision)",
                              "SceneRefCharacterMaker": "SceneRef Character Maker (.char)",
                              "SceneRefOutput": "SceneRef Preview & Save",
                              "SceneRefResolution": "SceneRef Resolution",
                              "SceneRefRefModMaker": "SceneRef RefMod Maker (MiniMax H3)",
                              "SceneRefScene": "SceneRef Scene (pictures)",
                              "SceneRefSceneMaker": "SceneRef Scene Maker (.scene)"}
