"""SceneRef LoRA Stack - a list of LoRAs with on/off, strength and trigger words.

The list lives on the node as JSON (edited by web/scene_ref_panels.js). Trigger
words are filled in from what is found next to or inside the LoRA: LoRA Manager /
Civitai sidecars (`.metadata.json`, `.civitai.info`, `.json`), rgthree's
`.rgthree-info.json`, and the trainer's own notes in the file (`modelspec.trigger_phrase`,
`ss_trained_words`, the caption tag every training picture shared). They can always be
typed in. The stack goes to the Studio, or straight onto
a MODEL when one is wired in.
"""
from __future__ import annotations

import json
import os
import struct

import folder_paths

LORAS_TYPE = "SCENEREF_LORAS"
MAX_LORAS = 32


def _json(p: str) -> dict:
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _sidecars(path: str) -> list[dict]:
    base = os.path.splitext(path)[0]
    names = (base + ".metadata.json", base + ".civitai.info", base + ".json", path + ".rgthree-info.json")
    return [_json(p) for p in names if os.path.isfile(p)]


def _header(path: str) -> dict:
    """The __metadata__ of a .safetensors file (the header only, a few KB)."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 64 * 1024 * 1024:
                return {}
            meta = json.loads(f.read(n)).get("__metadata__") or {}
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError, struct.error):
        return {}


def _words(value) -> list[str]:
    """trainedWords in its several shapes: ["a", "b"], [{"word": "a"}], "a, b"."""
    if isinstance(value, str):
        return [w for w in value.split(",")]
    out = []
    for w in value or []:
        if isinstance(w, dict):
            w = w.get("word")
        if isinstance(w, str):
            out.append(w)
    return out


def _tag_triggers(raw: str) -> list[str]:
    """kohya's ss_tag_frequency: the tags (nearly) every training caption had, which is
    what a trigger word is. Ordinary tags appear in some captions only."""
    try:
        sets = json.loads(raw)
    except (TypeError, ValueError):
        return []
    counts: dict[str, int] = {}
    for tags in sets.values() if isinstance(sets, dict) else []:
        for tag, n in (tags or {}).items():
            tag = str(tag).strip()
            if tag and isinstance(n, (int, float)):
                counts[tag] = counts.get(tag, 0) + int(n)
    if not counts:
        return []
    top = max(counts.values())
    if top < 2:
        return []  # one caption tells nothing
    return [t for t, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= 0.9 * top][:3]


def triggers_for(path: str) -> list[str]:
    words = []
    for meta in _sidecars(path):
        words += _words(meta.get("trainedWords"))
        if isinstance(meta.get("civitai"), dict):
            words += _words(meta["civitai"].get("trainedWords"))
    if path.lower().endswith(".safetensors"):
        head = _header(path)
        for key in ("modelspec.trigger_phrase", "ss_trained_words", "trigger_word", "trigger_words",
                    "krea_forge_trigger"):
            words += _words(head.get(key))
        if not any(w.strip() for w in words):
            words += _tag_triggers(head.get("ss_tag_frequency"))
    out = []
    for w in words:
        w = w.strip()
        if w and w.lower() not in (x.lower() for x in out):
            out.append(w)
    return out[:8]


_TRIGGERS: dict = {}


def _cached_triggers(path: str) -> list[str]:
    try:
        st = os.stat(path)
    except OSError:
        return []
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _TRIGGERS:
        _TRIGGERS[key] = triggers_for(path)
    return _TRIGGERS[key]


def list_loras() -> list[dict]:
    out = []
    for name in folder_paths.get_filename_list("loras"):
        path = folder_paths.get_full_path("loras", name)
        folder = os.path.dirname(name.replace("\\", "/"))
        out.append({"name": name, "folder": folder, "triggers": _cached_triggers(path) if path else []})
    return out


def parse_stack(raw) -> list[dict]:
    try:
        items = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
    except ValueError:
        items = []
    known = set(folder_paths.get_filename_list("loras"))
    stack = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not it.get("on", True):
            continue
        name = it.get("name")
        if name not in known:
            continue
        try:
            strength = float(it.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        if strength == 0:
            continue
        stack.append({"name": name, "strength": max(-10.0, min(10.0, strength)),
                      "trigger": str(it.get("trigger") or "").strip()})
    return stack[:MAX_LORAS]


class SceneRefLoraStack:
    DESCRIPTION = (
        "A list of LoRAs: add them with the button, switch each on or off, set its strength, and keep "
        "its trigger words next to it. Filter by folder (e.g. flux9 for Klein, qwen2.1 for Qwen). "
        "Wire 'loras' into the SceneRef Studio, or wire a model in to get it back with the LoRAs applied."
    )
    CATEGORY = "SceneRef"
    FUNCTION = "run"
    RETURN_TYPES = (LORAS_TYPE, "STRING", "MODEL")
    RETURN_NAMES = ("loras", "triggers", "model")
    OUTPUT_TOOLTIPS = (
        "The enabled LoRAs, for the SceneRef Studio.",
        "The trigger words of the enabled LoRAs, comma-separated (the Studio adds them to the prompt).",
        "The wired model with the LoRAs applied (only when a model is wired in).",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loras_json": ("STRING", {"default": "[]", "multiline": True,
                                          "tooltip": "Edited with the list on the node."}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Optional: a model to apply the LoRAs to."}),
            },
        }

    def run(self, loras_json, model=None):
        stack = parse_stack(loras_json)
        triggers = ", ".join(x["trigger"] for x in stack if x["trigger"])
        if model is not None and stack:
            import comfy.sd
            import comfy.utils
            for x in stack:
                lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", x["name"]),
                                                   safe_load=True)
                model, _ = comfy.sd.load_lora_for_models(model, None, lora, x["strength"], 0)
        return (stack, triggers, model)
