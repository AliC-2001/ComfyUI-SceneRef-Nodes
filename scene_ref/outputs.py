"""SceneRef Preview & Save and SceneRef Resolution.

Preview & Save shows what the Studio made. auto saves every result to ComfyUI/output;
manual keeps it in the temp folder until the Save button on the node copies it to output
(POST /scene_ref/keep), so experiments don't fill the output folder.

Resolution gives width / height from an aspect ratio and a size in megapixels; the viewer
reads it to frame its camera to that shape while you edit.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths

SAVE_MODES = ("auto", "manual")


def _save_images(images, prefix, out_dir, prompt=None, extra_pnginfo=None):
    full, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, out_dir, images[0].shape[1], images[0].shape[0])
    results = []
    for b, image in enumerate(images):
        img = Image.fromarray(np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8))
        meta = PngInfo()
        if prompt is not None:
            meta.add_text("prompt", json.dumps(prompt))
        for k, v in (extra_pnginfo or {}).items():
            meta.add_text(k, json.dumps(v))
        file = f"{filename.replace('%batch_num%', str(b))}_{counter:05}_.png"
        img.save(os.path.join(full, file), pnginfo=meta, compress_level=4)
        results.append({"filename": file, "subfolder": subfolder})
        counter += 1
    return results


def _save_video(video, prefix, out_dir, prompt=None, extra_pnginfo=None):
    from comfy_api.latest import Types
    width, height = video.get_dimensions()
    full, filename, counter, subfolder, _ = folder_paths.get_save_image_path(prefix, out_dir, width, height)
    meta = dict(extra_pnginfo or {})
    if prompt is not None:
        meta["prompt"] = prompt
    file = f"{filename}_{counter:05}_.mp4"
    video.save_to(os.path.join(full, file), format=Types.VideoContainer("mp4"), codec=Types.VideoCodec("auto"),
                  metadata=meta or None)
    return [{"filename": file, "subfolder": subfolder}]


class SceneRefOutput:
    DESCRIPTION = (
        "Shows the Studio's picture or clip. save_mode auto saves every result to ComfyUI/output; manual "
        "keeps it as a preview until you press 'Save to output' on the node, so trying things out doesn't "
        "fill the output folder."
    )
    CATEGORY = "SceneRef"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "save_mode": (list(SAVE_MODES), {"default": "manual", "tooltip": "auto: every result is saved "
                              "to ComfyUI/output. manual: only shown (kept in the temp folder) until you press "
                              "'Save to output'."}),
                "filename_prefix": ("STRING", {"default": "SceneRef/studio", "tooltip": "Saved to ComfyUI/output "
                                    "under this name."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "The Studio's 'image'."}),
                "video": ("VIDEO", {"tooltip": "The Studio's 'video' (movie mode). Shown instead of the image "
                          "when both carry something."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def run(self, save_mode, filename_prefix, image=None, video=None, prompt=None, extra_pnginfo=None):
        auto = save_mode == "auto"
        out_dir = folder_paths.get_output_directory() if auto else folder_paths.get_temp_directory()
        kind = "output" if auto else "temp"
        prefix = (filename_prefix.strip() or "SceneRef/studio") if auto else "SceneRef/preview"
        if video is not None:
            files = _save_video(video, prefix, out_dir, prompt, extra_pnginfo)
            ui = {"images": [dict(f, type=kind) for f in files], "animated": (True,)}
        elif image is not None:
            files = _save_images(image, prefix, out_dir, prompt, extra_pnginfo)
            ui = {"images": [dict(f, type=kind) for f in files]}
        else:
            return {"ui": {}}
        ui["sr_saved"] = [auto]
        return {"ui": ui}


def keep(files: list, prefix: str) -> list[str]:
    """Copies previews from the temp folder to output under `prefix`; returns the new names."""
    temp = os.path.realpath(folder_paths.get_temp_directory())
    out_dir = folder_paths.get_output_directory()
    saved = []
    for f in files:
        name, sub = str(f.get("filename") or ""), str(f.get("subfolder") or "")
        src = os.path.realpath(os.path.join(temp, sub, name))
        if not src.startswith(temp + os.sep) or not os.path.isfile(src):
            raise ValueError(f"not a preview: {name}")
        ext = os.path.splitext(name)[1].lower()
        if ext not in (".png", ".mp4"):
            raise ValueError(f"not a preview: {name}")
        full, filename, counter, subfolder, _ = folder_paths.get_save_image_path(prefix.strip() or "SceneRef/studio",
                                                                                 out_dir)
        dst = f"{filename}_{counter:05}_{ext}"
        shutil.copyfile(src, os.path.join(full, dst))
        saved.append(f"{subfolder}/{dst}" if subfolder else dst)
    return saved


ASPECTS = ("1:1", "5:4", "4:3", "3:2", "16:10", "16:9", "2:1", "21:9", "4:5", "3:4", "2:3", "10:16", "9:16", "9:21")


def resolution(aspect: str, megapixels: float, multiple: int) -> tuple[int, int]:
    """Width x height for `aspect` at about `megapixels` (1 MP = 1024 x 1024), multiples of `multiple`.
    web/scene_ref.js has the same formula, to frame the viewer before the graph runs."""
    rw, rh = (float(x) for x in aspect.split(":"))
    s = (megapixels * 1024 * 1024 / (rw * rh)) ** 0.5
    return (max(multiple, round(rw * s / multiple) * multiple), max(multiple, round(rh * s / multiple) * multiple))


class SceneRefResolution:
    DESCRIPTION = (
        "Width and height from an aspect ratio and a size. Wire both into the Scene Reference Viewer: its "
        "camera frame follows the shape at once. The size is shown on the node."
    )
    CATEGORY = "SceneRef"
    FUNCTION = "run"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (list(ASPECTS), {"default": "16:9", "tooltip": "Landscape first, then portrait "
                                 "(4:5 ... 9:21)."}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.05, "tooltip":
                               "Size: 1.0 = 1024 x 1024 worth of pixels."}),
                "multiple_of": ([8, 16, 32, 64], {"default": 16, "tooltip": "Both sides are rounded to this "
                                "(16 suits Klein / Qwen, 32 suits MiniMax H3)."}),
            },
        }

    def run(self, aspect_ratio, megapixels, multiple_of):
        return resolution(aspect_ratio, megapixels, int(multiple_of))
