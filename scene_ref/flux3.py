"""FLUX.3 Klein: the settings of Black Forest Labs' own ComfyUI workflows, and its prompt upsampler.

FLUX.3 Klein (5.6B, distilled) runs like BFL's flux-3-image-klein-comfy workflows: Qwen3-VL 4B
as the text encoder (CLIPLoader type flux3), the FLUX.2 VAE, the references as reference
latents, 16 steps, cfg 1, euler / simple, and a timestep shift of ln(16) (ModelSamplingFlux
with max_shift = base_shift, so it does not change with the picture size).

The model is trained on full scene captions and BFL says not to prompt it without its prompt
upsampler (Qwen3.5 9B, qwen3.5_9b_flux3_prompt_upsampler_*.safetensors). The Studio's
'Write the prompt' sends the upsampler what SceneRef knows (image 1 is the scene, which
mannequin becomes whom, the action) with the scene and the cast pictures attached, and puts
its caption in the prompt box. Without the upsampler the expander model writes it, with the
same system prompt (BFL: "any LLM with a system prompt").

data/flux3_upsampler_prompts.json is BFL's system_prompts.json from
huggingface.co/black-forest-labs/flux-3-image-klein-pup (Apache-2.0), unchanged.
"""
from __future__ import annotations

import json
import math
import os
import re

ALPHA = 16.0  # BFL's workflows: alpha 16 (the model's own reference is 6.93)
SHIFT = math.log(ALPHA)
REF_MEGAPIXELS = 1.0  # BFL resizes every reference to 1 MP
UPSAMPLER_MEGAPIXELS = 0.6  # what the upsampler looks at, per picture
UPSAMPLER_TOKENS = 1536  # BFL allows 3584; its captions run 50 to 350 words
# BFL's wrapper around the request for image-to-image (the 'i2i request (plain)' node).
I2I_REQUEST = ("Reference images follow in order (the first image, the second image, ...).\n"
               "User request: {request}\nOutput the caption.")
UPSAMPLER_PATTERNS = (r"flux3_prompt_upsampler_w4a8", r"flux3_prompt_upsampler_int8", r"flux3_prompt_upsampler",
                      r"flux.?3.*upsampler")
AUTO = "auto"
# The Studio's other choice: its expander model (any Qwen3-VL), with BFL's system prompt.
EXPANDER = "expander model (BFL's instructions)"

_PROMPTS = None


def system_prompt(kind: str = "i2i") -> str:
    global _PROMPTS
    if _PROMPTS is None:
        path = os.path.join(os.path.dirname(__file__), "data", "flux3_upsampler_prompts.json")
        with open(path, encoding="utf-8") as f:
            _PROMPTS = json.load(f)
    return _PROMPTS[kind]


def _upsampler_files() -> list[tuple[str, str]]:
    """(list name, full path) of every upsampler file, in pattern order (smallest build first).
    Files from models/LLM are listed as LLM/…, as in the expander list."""
    import folder_paths
    files = []
    for kind in ("text_encoders", "LLM"):
        try:
            for f in folder_paths.get_filename_list(kind):
                files.append((f if kind == "text_encoders" else "LLM/" + f.replace("\\", "/"),
                              folder_paths.get_full_path(kind, f)))
        except Exception:
            pass
    out = []
    for pat in UPSAMPLER_PATTERNS:
        for name, path in files:
            low = name.lower()
            if (re.search(pat, name.replace("\\", "/"), re.I) and "mmproj" not in low
                    and low.endswith((".safetensors", ".gguf")) and (name, path) not in out):
                out.append((name, path))
    return out


def upsampler_choices() -> list[str]:
    """The Studio's klein3_upsampler list: auto, every upsampler file (.safetensors run in
    ComfyUI, .gguf in llama.cpp), and the expander model."""
    return [AUTO] + [name for name, _ in _upsampler_files()] + [EXPANDER]


def upsampler_file() -> str | None:
    """auto: BFL's upsampler ComfyUI can load (.safetensors), the smallest build first."""
    for name, _ in _upsampler_files():
        if name.lower().endswith(".safetensors"):
            return name
    return None


def upsampler_name(choice: str, fallback_encoder: str) -> str:
    """What actually writes the caption for a klein3_upsampler choice (also the cache key)."""
    from .vlm import pick_encoder
    if choice == EXPANDER:
        return pick_encoder(fallback_encoder)
    if choice and choice != AUTO:
        if choice not in [name for name, _ in _upsampler_files()]:
            raise ValueError(f"SceneRef Studio: prompt upsampler {choice!r} not found in models/text_encoders or "
                             "models/LLM.")
        return choice
    return upsampler_file() or pick_encoder(fallback_encoder)


def upsample(request: str, pictures: list, fallback_encoder: str, seed: int = 0, choice: str = AUTO) -> str:
    """The caption FLUX.3 Klein gets: the chosen upsampler (auto = BFL's, else the expander
    model) with BFL's i2i system prompt, the pictures attached in order. A .gguf upsampler
    runs in llama.cpp (llama-server), greedy and with thinking off like the others."""
    from .vlm import vlm_text
    name = upsampler_name(choice, fallback_encoder)
    text = I2I_REQUEST.format(request=request.strip())
    if name.lower().endswith(".gguf"):
        from . import llamacpp
        path = dict(_upsampler_files())[name]
        return llamacpp.ask(path, system_prompt("i2i"), text, pictures, UPSAMPLER_TOKENS, seed)
    return vlm_text([(text, pictures)], system_prompt("i2i"), name, UPSAMPLER_TOKENS, seed, greedy=True)[0]
