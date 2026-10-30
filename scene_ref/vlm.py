"""The vision language model (Qwen3-VL) that writes text: the prompt expander's scene
directions and the Character Maker's descriptions.

ComfyUI loads it as a text encoder; it can only write when the file carries its lm_head
weights (many Qwen3-VL files are cut down to the encoder for image models, e.g. Krea2's
4B or MiniMax H3's 32B). Files are listed from models/text_encoders and models/LLM, and
only those that can write.
"""
from __future__ import annotations

import json
import os
import re
import struct

import folder_paths

AUTO = "auto"
# What auto takes when there is a choice: an 8B file, smaller builds first. Any file in the list
# can be picked instead.
PREFER = [r"qwen3vl_8b_w4a8\.", r"qwen3vl_8b_int8", r"qwen3vl_8b",
          r"qwen3.?vl.*8b", r"qwen3.?vl"]
FOLDERS = ("text_encoders", "LLM")

_CAN_WRITE: dict = {}


def _can_write(path: str) -> bool:
    """A Qwen3-VL safetensors file with lm_head (read from the header only)."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _CAN_WRITE:
        ok = False
        try:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                if n < 64 * 1024 * 1024:
                    keys = json.loads(f.read(n)).keys()
                    # deepstack layers are Qwen3-VL's own (Qwen2.5-VL has lm_head too, but the
                    # prompts and loading here are written for Qwen3-VL)
                    ok = any(k.startswith("lm_head.") for k in keys) and any("deepstack" in k for k in keys)
        except (OSError, ValueError, struct.error):
            ok = False
        _CAN_WRITE[key] = ok
    return _CAN_WRITE[key]


def _folder_files(kind: str) -> list[str]:
    try:
        return folder_paths.get_filename_list(kind)
    except Exception:
        return []


def vlm_files() -> list[str]:
    """Files that can write text. Ones from models/LLM carry an 'LLM/' prefix."""
    out = []
    for kind in FOLDERS:
        for name in _folder_files(kind):
            if not name.lower().endswith(".safetensors"):
                continue
            path = folder_paths.get_full_path(kind, name)
            if path and _can_write(path):
                out.append(name if kind == "text_encoders" else f"LLM/{name}")
    return out


def full_path(name: str) -> str:
    if name.startswith("LLM/"):
        return folder_paths.get_full_path_or_raise("LLM", name[4:])
    return folder_paths.get_full_path_or_raise("text_encoders", name)


def pick_encoder(value: str) -> str:
    files = vlm_files()
    if value and value != AUTO:
        if value not in files:
            raise ValueError(f"SceneRef: '{value}' can't write text (no lm_head) or is not in models/text_encoders "
                             f"or models/LLM. Pick one from the list.")
        return value
    for pat in PREFER:
        for f in files:
            if re.search(pat, f, re.I):
                return f
    if files:
        return files[0]
    raise ValueError("SceneRef: no vision language model that can write text was found in models/text_encoders "
                     "or models/LLM (a Qwen3-VL 8B file with its lm_head, e.g. qwen3vl_8b_int8_convrot.safetensors).")


def clean(text: str) -> str:
    text = text.split("</think>")[-1]
    return re.sub(r"\s+", " ", text).strip().strip('"').strip()


def vlm_text(questions, system: str, encoder_file: str, max_length: int = 300, seed: int = 0,
             unload_after: bool = True, greedy: bool = False) -> list[str]:
    """Answers, one per (text, image, a list of images or None) question; the model is
    unloaded afterwards (by default). greedy = always the likeliest next word (BFL's FLUX.3
    upsampler is run that way) instead of sampling."""
    import comfy.model_management as mm
    import comfy.sd

    path = full_path(encoder_file)
    out = []
    for k, (user, image) in enumerate(questions):
        # A fresh load per question: a second generate() on the same loaded encoder replays
        # ComfyUI's cached memory graph from the first, differently sized input, and the
        # GPU asserts (index out of bounds) and takes the server down.
        clip = comfy.sd.load_clip(ckpt_paths=[path], embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                  clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
        try:
            pictures = {"images": list(image)} if isinstance(image, (list, tuple)) else {"image": image}
            tokens = clip.tokenize(user, min_length=1, thinking=False, system_prompt=system, **pictures)
            if greedy:
                ids = clip.generate(tokens, do_sample=False, max_length=max_length, seed=seed)
            else:
                ids = clip.generate(tokens, do_sample=True, max_length=max_length, temperature=0.7, top_k=40,
                                    top_p=0.9, min_p=0.0, repetition_penalty=1.05, seed=seed)
            out.append(clean(clip.decode(ids)))
        finally:
            if unload_after or k < len(questions) - 1:
                mm.unload_model_and_clones(clip.patcher)
                del clip
                mm.soft_empty_cache()
    return out
