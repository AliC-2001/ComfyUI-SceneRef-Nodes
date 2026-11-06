"""SceneRef Studio - the control panel.

One node takes the shot from the viewer, the cast, the LoRAs and a sentence about
what happens, and builds the whole native generation graph behind the scenes
(node expansion). ComfyUI runs those nodes itself, so models stay cached between
runs and progress shows as usual.

Backends:
* FLUX.2 Klein 9B: the composite (scene + coloured figures) is reference image 1,
  each character's pictures follow; the prompt says which figure is whom.
* FLUX.3 Klein: the same references, sampled as in BFL's own workflows (see flux3.py); its
  prompt is a full scene caption from BFL's prompt upsampler.
* Qwen-Image 2.1: the same references through its own encoder, plus the Fun
  ControlNet Union fed the depth (bodies included) and the pose skeletons, so the
  people land exactly where the figures stand.
* Krea2 + the krea2_identity_edit LoRA (needs the comfyui-krea2edit node pack): one
  edit pass per character, "replace the red mannequin with the person in image 2",
  over the composite. The edit keeps image 1 exactly, so room and pose stay put.
* MOSS-Image (native in ComfyUI 0.43+): the scene reaches it as geometry, not as a picture:
  the depth (bodies included) and a flat-colour map of which object is where go through its
  colour/depth control adapter, so the room, the furniture and every body keep their exact
  place and size while the picture is drawn fresh (no render look carried over). The
  people are named in the text and the face pass gives them their faces with Klein (MOSS's
  edit LoRA loses the likeness under the control adapter, see moss.py). A scene from
  pictures goes through the edit LoRA instead: the photo is image 1, the cast follow.

After the picture: the face pass.

Movie mode is MiniMax H3 ref2va only: the scene as framed (with or without mannequins) is
<Picture 1>, the characters' pictures follow, and the prompt directs the rest. H3 makes no
pictures here.

The prompt box is used exactly as written. The Studio adds nothing to it at Run; its
'Write the prompt' button (the expander) fills it in beforehand, in the backend's format.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np
import torch

import comfy.samplers
import folder_paths

from .characters import (CAST_TYPE, FIGURE_COLORS, _upper1, add_to_cast, cache_dir, cast_from_json,
                         character_line, grow_inputs, grown, refs_text, resolve_refs)
from . import flux3, h3prompt, moss, scenes, voice
from .loras import LORAS_TYPE, parse_stack
from .vlm import vlm_files
from .nodes import SHOT_TYPE

KLEIN = "FLUX.2 Klein 9B"
KLEIN3 = "FLUX.3 Klein"
QWEN = "Qwen-Image 2.1 + ControlNet"
KREA2 = "Krea2 + identity edit"
MOSS = "MOSS-Image + scene control"
H3_BACKEND = "MiniMax H3 ref2v"  # movie mode's model; no longer an image backend
BACKENDS = (KLEIN, KLEIN3, QWEN, KREA2, MOSS)
IMAGE, MOVIE = "image", "movie"
# Settings of the removed H3 picture paths (backend, identity pass, picture + clip). Kept as
# hidden inputs so saved workflows still load; their values are ignored.
IDENTITY_PASSES = ("off", "MiniMax H3")
AUTO = "auto"
MAX_SHOTS = 8
CONTROL_MODES = ("depth + pose", "depth", "pose")
# Face pass: img2img strength. The crop is only the starting point, never a reference:
# given as image 1, Klein copied the face in it and the identity barely moved (0.7 and
# 0.85 came back unchanged).
FACE_DENOISE = 0.75
FACE_REF_MP = 0.75
H3_MAX_REFS = 9  # MiniMaxH3ReferenceToVideo takes at most 9 pictures

# Clip size (MP). 5 s at 0.4 MP is ComfyUI's own H3 template; time and VRAM grow with both.
H3_RESOLUTIONS = {"auto": None, "0.4 MP (fast)": 0.4, "0.6 MP": 0.6, "0.9 MP (720p)": 0.92, "1.1 MP": 1.1,
                  "1.5 MP": 1.5, "2.1 MP (1080p)": 2.07}
VIDEO_MP = 0.4
# Picture size: 'as framed' = the viewer's own width x height (cropped to the aspect).
IMAGE_RESOLUTIONS = {"as framed": None, "0.5 MP": 0.5, "1 MP": 1.0, "1.5 MP": 1.5, "2 MP": 2.0, "3 MP": 3.0,
                     "4 MP": 4.0}
ASPECTS = {"as framed": None, "16:9": 16 / 9, "3:2": 3 / 2, "4:3": 4 / 3, "1:1": 1.0, "4:5": 4 / 5, "3:4": 3 / 4,
           "2:3": 2 / 3, "9:16": 9 / 16, "21:9": 21 / 9}
H3_FPS = 24  # H3 is made for 24 fps; its clips are 17k + 5 frames long

# Klein's own schedule (Flux2Scheduler: the sigmas follow the picture size), offered next to
# ComfyUI's schedulers.
FLUX2_SCHEDULER = "flux2"
SAMPLERS = list(comfy.samplers.KSampler.SAMPLERS)
SCHEDULERS = list(comfy.samplers.KSampler.SCHEDULERS) + [FLUX2_SCHEDULER]
# Each model's usual settings, from its reference workflow (Klein: ComfyUI's Flux2 template;
# FLUX.3 Klein: BFL's flux-3-image-klein-comfy workflows;
# Qwen-Image 2.1 / Krea2: KSampler euler + simple; H3: ComfyUI's ref2v template). The panel
# fills them in when the backend or mode changes; any can be changed.
DEFAULTS = {KLEIN: {"steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": FLUX2_SCHEDULER},
            KLEIN3: {"steps": 16, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"},
            QWEN: {"steps": 25, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"},
            KREA2: {"steps": 10, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"},
            # ComfyUI's MOSS-Image colour/depth control template (25 steps, CFG 3, er_sde, simple).
            MOSS: {"steps": 25, "cfg": 3.0, "sampler_name": "er_sde", "scheduler": "simple"},
            MOVIE: {"steps": 20, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple"}}

# How "auto" picks a file: first match, in this order of preference.
PICKS = {
    "klein_model": ("diffusion_models", [r"klein.?9b.*\.safetensors$", r"klein.*9b"]),
    "klein_text_encoder": ("text_encoders", [r"qwen_?3_?8b", r"qwen3.?8b(?!.*vl)"]),
    "klein_vae": ("vae", [r"flux2.?vae", r"flux.?2.*vae"]),
    # The distilled checkpoint, not Klein Base (50 steps, for training).
    "klein3_model": ("diffusion_models", [r"flux3-image-klein-int8", r"flux3-image-klein-(?!base)",
                                          r"flux-?3.*klein(?!.*base)"]),
    "klein3_text_encoder": ("text_encoders", [r"qwen3vl_4b_int8",
                                              r"qwen3vl_4b_bf16", r"qwen3vl_4b(?!.*fp8)", r"qwen3vl_4b"]),
    "qwen_model": ("diffusion_models", [r"qwen_image_2\.1_int8", r"qwen.?image.?2\.1(?!.*edit)", r"qwen2\.1"]),
    "qwen_text_encoder": ("text_encoders", [r"qwen3vl_8b_int8", r"qwen3vl_8b"]),
    "qwen_vae": ("vae", [r"qwen_image_2\.1_vae", r"qwen.?image.?2\.1.*vae"]),
    "qwen_controlnet": ("model_patches", [r"2\.1.*fun.*union", r"fun.*controlnet.*union"]),
    "krea2_model": ("diffusion_models", [r"krea2_turbo_fp8_scaled", r"krea-?2.?turbo", r"krea2.*turbo"]),
    "krea2_text_encoder": ("text_encoders", [r"qwen3vl_4b_fp8", r"qwen3vl_4b"]),
    "krea2_vae": ("vae", [r"(^|/)qwen_image_vae\.safetensors$", r"qwen_image_vae(?!_diffusers)"]),
    "krea2_identity_lora": ("loras", [r"krea2_identity_edit_v1_2\.safetensors$", r"krea2_identity_edit_v1_2",
                                      r"krea2_identity_edit"]),
    "h3_model": ("diffusion_models", [r"minimaxH3ComfyNativeRef2va", r"ref2va.*w4a8", r"minimax_h3_ref2va",
                                      r"ref2va"]),
    "h3_text_encoder": ("text_encoders", [r"qwen3vl_32b_minimax_h3", r"qwen3vl_32b.*h3"]),
    "h3_vae": ("vae", [r"minimax_h3_video_vae"]),
    "h3_turbo_lora": ("loras", [r"ref2v_turbo_8step.*comfyui", r"ref2v_turbo_8step", r"ref2v_turbo"]),
    "h3_audio_vae": ("vae", [r"minimax_h3_audio_vae"]),
    # The SFT checkpoint (Base is for training); its Qwen3.5 9B encoder with the vision tower
    # (not FLUX.3's prompt upsampler, which is a Qwen3.5 9B too); the small FLUX.2 VAE.
    "moss_model": ("diffusion_models", [r"moss_image_sft_int8", r"moss_image_sft", r"moss.?image(?!.*base)"]),
    "moss_text_encoder": ("text_encoders", [r"qwen3\.5_9b_w4a8_gptq", r"qwen3\.5_9b_int8_convrot",
                                            r"qwen3\.5.?9b(?!.*(upsampler|abliterated))"]),
    "moss_vae": ("vae", [r"flux2-vae_small", r"flux2.?vae.?small"]),
    "moss_edit_lora": ("loras", [r"moss_image_edit_lora", r"moss.?image.?edit"]),
    "moss_control_lora": ("loras", [r"moss_image_color_depth_control", r"moss.*(colou?r|depth).*control"]),
}
# Character LoRAs inside .char files, per backend.
CHAR_LORA_ARCHS = {KLEIN: ("flux2-klein", "flux2", "flux2-klein-9b"), KLEIN3: ("flux3-klein", "flux3"),
                   QWEN: ("qwen-image", "qwen-image-2.1"),
                   KREA2: ("krea2",), MOSS: ("moss-image", "moss_image"), H3_BACKEND: ("minimax-h3", "h3")}
# What "auto" looks for, for the tooltips.
EXPECTED = {
    "klein_model": "flux-2-klein-9b.safetensors (diffusion_models)",
    "klein_text_encoder": "qwen_3_8b_fp8mixed.safetensors (text_encoders; Qwen3 8B, not a Qwen3-VL file)",
    "klein_vae": "flux2-vae.safetensors (vae; FLUX.3 Klein uses it too)",
    "klein3_model": "flux3-image-klein-int8-convrot.safetensors (diffusion_models; the distilled one, not -base-)",
    "klein3_text_encoder": "a Qwen3-VL 4B .safetensors such as qwen3vl_4b_int8_convrot or qwen3vl_4b_bf16 "
                           "(text_encoders)",
    "qwen_model": "qwen_image_2.1 (diffusion_models)",
    "qwen_text_encoder": "qwen3vl_8b_int8 (text_encoders)",
    "qwen_vae": "qwen_image_2.1_vae (vae)",
    "qwen_controlnet": "Qwen-Image 2.1 Fun ControlNet Union (model_patches)",
    "krea2_model": "krea2_turbo_fp8_scaled (diffusion_models)",
    "krea2_text_encoder": "qwen3vl_4b_fp8_scaled (text_encoders)",
    "krea2_vae": "qwen_image_vae.safetensors (vae)",
    "krea2_identity_lora": "krea2_identity_edit_v1_2 (loras)",
    "h3_model": "minimaxH3ComfyNativeRef2vaW4a8 (diffusion_models)",
    "h3_text_encoder": "qwen3vl_32b_minimax_h3 (text_encoders)",
    "h3_vae": "minimax_h3_video_vae_fp16 (vae)",
    "h3_turbo_lora": "minimax_h3_ref2v_turbo_8step (loras)",
    "h3_audio_vae": "minimax_h3_audio_vae_fp32 (vae)",
    "moss_model": "moss_image_sft_int8_convrot (diffusion_models; the SFT one, not -base-)",
    "moss_text_encoder": "qwen3.5_9b_w4a8_gptq (text_encoders; Comfy-Org/MOSS-Image)",
    "moss_vae": "flux2-vae_small (vae; Comfy-Org/MOSS-Image)",
    "moss_edit_lora": "moss_image_edit_lora_rank_64_bf16 (loras; Comfy-Org/MOSS-Image)",
    "moss_control_lora": "moss_image_color_depth_control (loras; Comfy-Org/MOSS-Image)",
}


def _files(kind: str) -> list[str]:
    try:
        return folder_paths.get_filename_list(kind)
    except Exception:
        return []


# FLUX.3 also takes GGUF files: ComfyUI-GGUF lists them in folders of its own.
GGUF_KINDS = {"klein3_model": "unet_gguf", "klein3_text_encoder": "clip_gguf"}


def widget_files(widget: str) -> list[str]:
    kind, _ = PICKS[widget]
    files = _files(kind)
    if widget in GGUF_KINDS:
        files += [f for f in _files(GGUF_KINDS[widget]) if f.lower().endswith(".gguf") and f not in files]
    return files


def pick(widget: str, value: str) -> str:
    kind, patterns = PICKS[widget]
    files = widget_files(widget)
    if value and value != AUTO:
        if value not in files:
            raise ValueError(f"SceneRef Studio: {widget} '{value}' is not in models/{kind}.")
        return value
    for pat in patterns:
        for f in files:
            if re.search(pat, f.replace("\\", "/"), re.I) and not f.lower().endswith(".gguf"):
                return f
    raise ValueError(f"SceneRef Studio: could not find a file for {widget} in models/{kind}. "
                     f"Pick one on the node.")


def resolved_picks() -> dict:
    """What 'auto' resolves to for every model widget (None = nothing found), so the panel
    can show the real file names instead of 'auto'."""
    out = {}
    for widget in PICKS:
        try:
            out[widget] = pick(widget, AUTO)
        except ValueError:
            out[widget] = None
    try:
        from .vlm import pick_encoder
        out["expander_model"] = pick_encoder(AUTO)
    except ValueError:
        out["expander_model"] = None
    return out


# ── prompt ────────────────────────────────────────────────────────────────────
PART_WORDS = {"head": "head", "torso": "body", "left_arm": "left arm", "right_arm": "right arm",
              "left_leg": "left leg", "right_leg": "right leg"}


def _hidden(parts: list) -> str:
    words = [PART_WORDS.get(p, str(p).replace("_", " ")) for p in parts or []]
    legs = {"left leg", "right leg"} <= set(words)
    arms = {"left arm", "right arm"} <= set(words)
    words = [w for w in words if not ((legs and w.endswith("leg")) or (arms and w.endswith("arm")))]
    if legs:
        words.append("legs")
    if arms:
        words.append("arms")
    return " and ".join(words) if len(words) <= 2 else ", ".join(words[:-1]) + " and " + words[-1]


def _pose_phrase(f: dict, marked: bool = True) -> str:
    if f.get("kind") == "daz":
        return "in exactly the pose shown"
    label = str(f.get("pose_label") or f.get("pose") or "").split("(")[0].strip().lower()
    if not marked:
        return f"pose: {label}" if label else "in the pose the control image shows"
    return f"posed exactly like the figure ({label})" if label else "posed exactly like the figure"


POSE_WORDS = {"sit": "sitting", "stand": "standing", "lie": "lying down", "kneel": "kneeling", "walk": "walking",
              "crouch": "crouching", "lean": "leaning"}


def _pose_word(f: dict) -> str:
    """'sit' -> 'sitting' for prose prompts; '' for Daz figures (their pose is their own)."""
    label = str(f.get("pose_label") or f.get("pose") or "").split("(")[0].strip().lower()
    if not label or "daz" in label:  # 'daz', 'posed in daz studio': their pose is their own
        return ""
    return POSE_WORDS.get(label, label)


def _place_phrase(f: dict, width: int, height: int) -> str:
    """Where a figure is in the frame, for prompts where the picture has no coloured figures."""
    box = f.get("bbox_visible") or f.get("bbox_full")
    if not box or not width:
        return "where the control image shows a person"
    cx = (box[0] + box[2]) / 2 / width
    side = "on the left" if cx < 0.36 else "on the right" if cx > 0.64 else "in the middle"
    near = (box[3] - box[1]) / max(height, 1)
    depth = " in the foreground" if near > 0.75 else " in the background" if near < 0.3 else ""
    return f"{side} of the picture{depth}"


def build_prompt(figures: list, cast: list, body: str, backend: str,
                 triggers: str, marked: bool = True, size: tuple = (0, 0), photo: bool = False,
                 angles: int = 0) -> str:
    """A Klein / Qwen prompt: which picture is what, who replaces which figure, then `body`
    (the scene as the expander wrote it) and the LoRA trigger words. marked = image 1 shows
    the coloured figures (the composite); otherwise it is the empty scene and people are
    placed by the ControlNet, so the prompt says where they are instead of which colour.
    photo = image 1 is a picture of the place (SceneRef Scene), with `angles` pictures of it
    from other angles right after it."""
    ref_style = "tag" if backend == QWEN else "words" if backend == KLEIN3 else "ordinal"
    scene = refs_text([1], ref_style)
    if photo:
        lines = [f"{_upper1(scene)} is a picture of the place. Turn it into the picture described below, keeping "
                 f"exactly its camera angle, framing, layout, furniture and objects."]
        if angles:
            nums = _upper1(refs_text(list(range(2, 2 + angles)), ref_style))
            lines.append(f"{nums} {'shows' if angles == 1 else 'show'} the same place from "
                         f"{'another angle' if angles == 1 else 'other angles'}: use {'it' if angles == 1 else 'them'} "
                         f"only to get the place right, and keep the camera of {scene}.")
    elif marked:
        lines = [f"{_upper1(scene)} is a 3D render of the scene. Turn it into the picture described below, keeping exactly "
                 f"its camera angle, framing, room layout, furniture and objects."]
    else:
        lines = [f"{_upper1(scene)} is a 3D render of the empty scene. Turn it into the picture described below, with "
                 f"people in it, keeping exactly its camera angle, framing, room layout, furniture and objects."]
    by_figure = {c.get("figure"): c for c in cast if c.get("figure")}
    pos = 2 + angles
    # Characters first (their pictures follow image 1 in this order).
    for ch in cast:
        lines.append(character_line(ch if marked else dict(ch, figure=0), pos, ref_style))
        pos += len(ch["refs"])
    # Then what every figure in this shot is: where, how big, in what pose, what hides it.
    for f in figures or []:
        idx = int(f.get("index") or 0)
        color = f.get("color") or FIGURE_COLORS[(idx - 1) % len(FIGURE_COLORS)]
        ch = by_figure.get(idx)
        textured = f.get("kind") == "daz" and f.get("look") == "textured"
        if ch:
            who = ch["name"]
        else:
            # Without a character the model borrows the cast's looks ("a person" came out as a
            # second Emma), so say who it is not, and use the mannequin's sex.
            who = {"male": "a man", "female": "a woman"}.get(f.get("sex"), "a person")
            if textured:
                who = "a real person who looks exactly like it"
            names = [c["name"] for c in cast]
            if names:
                who += f" who is clearly a different person from {' and '.join(names)}"
        what = f"the {color} figure" if not textured else f"the 3D character {f.get('name') or ''}".strip()
        if marked:
            sentence = (f"Replace {what} in {scene} with {who}, at exactly the same place and size, "
                        f"{_pose_phrase(f)}")
        else:
            sentence = f"{who[0].upper() + who[1:]} is {_place_phrase(f, *size)}, {_pose_phrase(f, False)}"
        facing = f.get("facing")
        if facing:
            sentence += f", {facing}"
        hidden = f.get("hidden_parts") or []
        if hidden:
            sentence += (f"; keep the {_hidden(hidden)} hidden behind what covers them in {scene}" if marked
                         else f"; the {_hidden(hidden)} are hidden behind the furniture")
        lines.append(sentence + ".")
    if figures and marked:
        lines.append("Every figure becomes a real person; no mannequins, coloured figures or markers remain.")
    head = " ".join(x for x in lines if x)
    return "\n\n".join(x for x in (head, body.strip(), triggers.strip()) if x)


def _figure_color(f: dict) -> str:
    return f.get("color") or FIGURE_COLORS[(int(f.get("index") or 1) - 1) % len(FIGURE_COLORS)]


def krea2_jobs(figures: list, cast: list) -> list:
    """Krea2's passes, in order: a figure played by a character, then each character without
    a figure (added where the text says); [None] = one pass that only turns the render real."""
    by_figure = {c.get("figure"): c for c in cast if c.get("figure") and c.get("refs")}
    cast_figs = [f for f in figures or [] if int(f.get("index") or 0) in by_figure]
    free = [c for c in cast if c.get("refs") and not c.get("figure")]
    return cast_figs + free or [None]


def krea2_prompts(figures: list, cast: list, body: str, triggers: str, photo: bool = False) -> list[str]:
    """One edit prompt per pass (see SceneRefStudio._krea2): which mannequin becomes the
    person from image 2, what stays, then `body` (the scene as the expander wrote it) and the
    trigger words."""
    by_figure = {c.get("figure"): c for c in cast if c.get("figure") and c.get("refs")}
    others = [f for f in figures or [] if int(f.get("index") or 0) not in by_figure]
    names = [c["name"] for c in cast]
    jobs = krea2_jobs(figures, cast)

    def place(f, who):
        label = str(f.get("pose_label") or f.get("pose") or "").split("(")[0].strip().lower()
        text = (f"Replace the {_figure_color(f)} mannequin with {who}, at exactly the place and size of the "
                f"{_figure_color(f)} mannequin, in exactly its pose" + (f" ({label})" if label and "daz" not in label else ""))
        if f.get("facing"):
            text += f", {f['facing']}"
        if f.get("hidden_parts"):
            text += f"; the {_hidden(f['hidden_parts'])} stay hidden behind what covers them"
        return text + "."

    def stranger(f):
        who = {"male": "a man", "female": "a woman"}.get(f.get("sex"), "a person")
        return who + (f" who is clearly a different person from {' and '.join(names)}" if names else "")

    first = ("Image 1 is a " + ("picture of the place" if photo else "3D render of a room") + ". Turn it into "
             "the picture described below, keeping exactly its camera angle, framing, room layout, furniture and "
             "objects.")
    prompts = []
    for k, f in enumerate(jobs):
        lines = [first if k == 0 else "Image 1 is a photograph."]
        if k == 0:
            lines += [place(o, stranger(o)) for o in others]
        if f is not None and "refs" in f:  # a character without a figure
            ch = f
            lines.append(f"Add {ch['name']}, the person from image 2, as the text describes: exactly "
                         f"{ch['name']}'s face, age, hair, skin and body as in image 2. Dress {ch['name']} as the "
                         f"text says, not as in image 2.")
            if ch.get("description"):
                lines.append(" ".join(ch["description"].split()))
            lines.append("Keep everything else exactly as in image 1: the room, the furniture, the light, the "
                         "camera" + (" and the people already there." if k else "."))
        else:
            if f is not None:
                ch = by_figure[int(f["index"])]
                sex = {"male": "man", "female": "woman"}.get(f.get("sex"), "person")
                lines.append(place(f, f"the {sex} from image 2: exactly {ch['name']}'s face, age, hair, skin and "
                                      f"body as in image 2"))
                lines.append(f"The {sex} from image 2 is {ch['name']}. Dress {ch['name']} as the text says, not "
                             f"as in image 2.")
                if ch.get("description"):
                    lines.append(" ".join(ch["description"].split()))
            later = [_figure_color(x) for x in jobs[k + 1:] if x is not None and "refs" not in x]
            if later:
                lines.append(f"Leave the {' and '.join(later)} mannequin{'s' if len(later) > 1 else ''} exactly as "
                             f"{'they are' if len(later) > 1 else 'it is'}.")
            lines.append("Keep everything else exactly as in image 1: the room, the furniture, the light, the camera"
                         + (" and the people already there." if k else "."))
            if not later:
                lines.append("No mannequins or coloured figures remain.")
        prompts.append("\n\n".join(x for x in (" ".join(lines), body.strip(), triggers.strip()) if x))
    return prompts


PASS_MARK = re.compile(r"^\s*\[pass\s+(\d+)\][^\n]*\n?", re.I | re.M)


def join_passes(prompts: list[str]) -> str:
    return "\n\n".join(f"[pass {i + 1}]\n{p}" for i, p in enumerate(prompts))


def split_passes(text: str, n: int) -> list[str]:
    """The prompt box for Krea2: '[pass 1] ... [pass 2] ...' (one section per pass), or one
    text used for every pass."""
    marks = list(PASS_MARK.finditer(text))
    if not marks:
        return [text.strip()] * n
    parts = [text[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(text))].strip()
             for i, m in enumerate(marks)]
    if len(parts) != n:
        raise ValueError(f"SceneRef Studio: the prompt has {len(parts)} [pass] sections, but this shot needs {n} "
                         f"Krea2 pass{'es' if n > 1 else ''} (one per character). Press 'Write the prompt' "
                         f"again, or give it {n} sections (or none: one text for every pass).")
    return parts


def moss_people(figures: list, cast: list, size: tuple) -> list[tuple[dict, str, str]]:
    """Per figure: (figure, who, what they do) for MOSS's prompt. The control map holds the
    body; the text says who it is and, for the layout's boxes, how they stand."""
    by_figure = {c.get("figure"): c for c in cast if c.get("figure")}
    names = [c["name"] for c in cast]
    out = []
    for f in figures or []:
        ch = by_figure.get(int(f.get("index") or 0))
        if ch:
            who = ch["name"]
        else:
            who = {"male": "a man", "female": "a woman"}.get(f.get("sex"), "a person")
            if f.get("kind") == "daz" and f.get("look") == "textured":
                who = "a real person who looks exactly like the 3D character there"
            elif names:
                who += f" who is clearly a different person from {' and '.join(names)}"
        what = ", ".join(x for x in (_pose_word(f), f.get("facing") or "") if x)
        out.append((f, who, what))
    return out


def moss_prompt(figures: list, cast: list, body: str, triggers: str, size: tuple = (0, 0), photo: bool = False,
                angles: int = 0, layout: bool = False, out_size: tuple = (0, 0)) -> str:
    """A MOSS-Image prompt: which reference picture shows whom (see moss.ref_plan; a scene
    from pictures only), who each figure is and where (a named character with their sex, so
    the face pass finds the right kind of person to redo), then `body` (the scene as the
    expander wrote it) and the trigger words.
    layout = MOSS's JSON layout prompt instead: all of that as the caption, plus one
    'character' box per figure in the picture's pixels (out_size; boxes in `size` are scaled),
    compiled as OpenMOSS's prompt-enhancer skill does: one line, abs_centralized yxyx boxes
    (left / top floored, right / bottom ceiled, then centred), no canvas size in it."""
    n_scene, plan = moss.ref_plan(cast, angles, photo)
    lines = []
    pictured = set()
    if photo:
        # An edit instruction, the way MOSS's edit LoRA was trained: add the people to image 1.
        added = []
        for ch, idx, first in plan:
            pictured.add(id(ch))
            nums = refs_text(list(range(first, first + len(idx)))).lower()
            added.append(f"{ch['name']}, the person from {nums},")
        if added:
            lines.append(f"Add {' and '.join(added)} to image 1.")
        else:
            lines.append("Edit image 1 as described below.")
        if n_scene > 1:
            nums = refs_text(list(range(2, n_scene + 1))).lower()
            lines.append(f"{_upper1(nums)} {'shows' if n_scene == 2 else 'show'} the same place from "
                         f"{'another angle' if n_scene == 2 else 'other angles'}, only to get it right.")
        lines.append("Keep the place, the camera angle, the framing and the light of image 1 exactly as they are.")
    for ch in cast:  # characters whose pictures did not fit: the text alone describes them
        if id(ch) not in pictured and ch.get("description"):
            lines.append(f"{ch['name']}: {' '.join(ch['description'].split())}")
    people = moss_people(figures, cast, size)
    named = {c["name"] for c in cast}
    for f, who, what in people:
        sex = {"male": "a man", "female": "a woman"}.get(f.get("sex"))
        subject = f"{who}, {sex}," if who in named and sex else who
        sentence = f"{subject[0].upper() + subject[1:]} is {_place_phrase(f, *size)}" + (f", {what}" if what else "")
        if f.get("hidden_parts"):
            sentence += f"; the {_hidden(f['hidden_parts'])} are hidden behind the furniture"
        lines.append(sentence + ".")
    head = " ".join(x for x in lines if x)
    text = "\n\n".join(x for x in (head, body.strip(), triggers.strip()) if x)
    boxed = [(f, who, what) for f, who, what in people if f.get("bbox_visible") or f.get("bbox_full")]
    if not (layout and boxed and size[0] and out_size[0]):
        return text
    W, H = out_size
    sx, sy = W / size[0], H / size[1]
    by_figure = {c.get("figure"): c for c in cast if c.get("figure")}
    items = []
    for f, who, what in boxed:
        x0, y0, x1, y1 = f.get("bbox_visible") or f.get("bbox_full")
        # Pixel edges inside the canvas, then centred.
        x0, y0 = max(0, math.floor(x0 * sx)), max(0, math.floor(y0 * sy))
        x1, y1 = min(W, math.ceil(x1 * sx)), min(H, math.ceil(y1 * sy))
        caption = f"{who}" + (f", {what}" if what else "")
        desc = (by_figure.get(int(f.get("index") or 0)) or {}).get("description")
        if desc:
            caption += f". {' '.join(desc.split())}"
        items.append({"label": "character", "caption": caption,
                      "bbox": [y0 - H // 2, x0 - W // 2, y1 - H // 2, x1 - W // 2]})
    return json.dumps({"summary": {"canvas": "single", "caption": text}, "items": items,
                       "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"},
                      ensure_ascii=False, separators=(",", ":"))


def picture_size(cw: int, ch_: int, image_resolution: str) -> tuple[int, int]:
    """The picture's size from the (cropped) frame: about the megapixels asked for in its
    aspect, or the frame itself ('as framed'); both sides multiples of 16."""
    mp = IMAGE_RESOLUTIONS.get(image_resolution)
    return h3_size(cw, ch_, mp, 16) if mp else (cw // 16 * 16, ch_ // 16 * 16)


# ── images ────────────────────────────────────────────────────────────────────
def _ref_tensor(im) -> torch.Tensor:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None]


def _soft_mask(side: int, inner: float = 0.6, outer: float = 0.95) -> torch.Tensor:
    """A round mask, full inside `inner`, fading out by `outer` (fractions of the half side),
    so a redone face blends into the picture."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, side), torch.linspace(-1, 1, side), indexing="ij")
    d = torch.sqrt(xx * xx + yy * yy)
    return ((outer - d) / (outer - inner)).clamp(0, 1)[None]


def _shots(shot: dict) -> int:
    return min(int(shot["composite"].shape[0]), MAX_SHOTS)


def h3_frames(seconds: float) -> int:
    """Seconds -> an H3 clip length: at least 5 frames, rounded up to 17k + 5."""
    n = max(5, round(float(seconds) * H3_FPS))
    return n + (5 - n % 17) % 17


def h3_size(width: int, height: int, megapixels: float, grid: int = 32) -> tuple[int, int]:
    """The shot's aspect at about `megapixels`, both sides multiples of `grid` (32 = H3's patch grid)."""
    s = (megapixels * 1e6 / (width * height)) ** 0.5
    return max(grid, round(width * s / grid) * grid), max(grid, round(height * s / grid) * grid)


# Heads with a smaller radius (in picture pixels) are left out of the face pass.
FACE_MIN_RADIUS = 48


def aspect_crop(width: int, height: int, ratio):
    """The centred (left, top, w, h) of the shot at `ratio` (None = the whole shot)."""
    if not ratio:
        return 0, 0, width, height
    if width / height > ratio:
        w = int(height * ratio) // 16 * 16
        return (width - w) // 2, 0, w, height
    h = int(width / ratio) // 16 * 16
    return 0, (height - h) // 2, width, h


def _shift_figures(figures: list, left: int, top: int) -> list:
    """Figures with their boxes moved into a cropped picture."""
    out = []
    for f in figures or []:
        f = dict(f)
        for k in ("bbox_visible", "bbox_full", "head_box"):
            if f.get(k):
                b = f[k]
                f[k] = [b[0] - left, b[1] - top, b[2] - left, b[3] - top]
        out.append(f)
    return out


def h3_people(people, first=2, slots=H3_MAX_REFS - 1):
    """The characters' pictures from <Picture first> on, and who they show
    ([{name, tags, description}], for h3prompt.build)."""
    per = max(1, slots // max(1, len(people)))
    pictures, tagged, n = [], [], first
    for ch in people:
        refs = ch["refs"][:per]
        if n + len(refs) - first > slots:
            break
        tagged.append({"name": ch["name"], "tags": [f"<Picture {i}>" for i in range(n, n + len(refs))],
                       "description": ch.get("description") or "", "voice": ch.get("voice") is not None})
        pictures += [_ref_tensor(im) for im, _role in refs]
        n += len(refs)
    return pictures, tagged


def h3_voices(prompt: str, voiced: list) -> tuple[str, list]:
    """The voices a movie prompt cites. voiced = the characters with a voice, in the order
    h3prompt.build numbered them (<Audio 1> ...). Only the tags still in the prompt are wired
    (deleting the line in the prompt box drops that voice), renumbered from 1 in the order
    MiniMaxH3ReferenceToVideo presents its audio inputs."""
    keep = [k for k in range(1, min(len(voiced), h3prompt.MAX_VOICES) + 1) if f"<Audio {k}>" in prompt]
    renumber = {k: i + 1 for i, k in enumerate(keep)}
    prompt = re.sub(r"<Audio (\d+)>", lambda m: f"<Audio {renumber.get(int(m.group(1)), m.group(1))}>", prompt)
    return prompt, [voiced[k - 1]["voice"] for k in keep]


def movie_layout(figures: list, cast: list, width: int, height: int, angles: int = 0):
    """For an H3 prompt: (pictures, tagged, places, marked). places = where each figure's
    person is in <Picture 1>; marked = coloured mannequins mark the places. angles = pictures
    of the place from other angles, <Picture 2> on; the cast's follow them."""
    people = [c for c in cast if c.get("refs")]
    pictures, tagged = h3_people(people, first=2 + angles, slots=H3_MAX_REFS - 1 - angles)
    by_figure = {c.get("figure"): c for c in people if c.get("figure")}
    names = [c["name"] for c in people]
    places, marked = [], False
    for f in figures or []:
        ch = by_figure.get(int(f.get("index") or 0))
        textured = f.get("kind") == "daz" and f.get("look") == "textured"
        who = ch["name"] if ch else {"male": "a man", "female": "a woman"}.get(f.get("sex"), "a person")
        if not ch and names:
            who += f" who is clearly a different person from {' and '.join(names)}"
        where = _place_phrase(f, width, height)
        if textured:
            where += f", where the 3D character{' ' + f['name'] if f.get('name') else ''} is"
        else:
            # No box (figure at the frame's edge): the colour alone says where.
            marked = True
            where = (f"{where}, where the {_figure_color(f)} figure is" if f.get("bbox_visible") or f.get("bbox_full")
                     else f"where the {_figure_color(f)} figure is")
        label = _pose_word(f)
        places.append(f"{who[0].upper() + who[1:]} is {where}" + (f", {label}" if label else "") + ".")
    return pictures, tagged, places, marked


def _sigmas(g, model, scheduler, steps, width, height, denoise=1.0):
    if scheduler == FLUX2_SCHEDULER:
        sigmas = g.node("Flux2Scheduler", steps=steps, width=width, height=height).out(0)
        return g.node("SplitSigmasDenoise", sigmas=sigmas, denoise=denoise).out(1) if denoise < 1.0 else sigmas
    return g.node("BasicScheduler", model=model, scheduler=scheduler, steps=steps, denoise=denoise).out(0)


def _sample(g, model, pos, neg, latent, seed, steps, cfg, sampler, scheduler, width, height, denoise=1.0):
    """KSampler's job in its parts (CFG guider, noise, sampler, sigmas), so Klein's own
    schedule can be picked as well as ComfyUI's."""
    return g.node("SamplerCustomAdvanced", noise=g.node("RandomNoise", noise_seed=seed).out(0),
                  guider=g.node("CFGGuider", model=model, positive=pos, negative=neg, cfg=cfg).out(0),
                  sampler=g.node("KSamplerSelect", sampler_name=sampler).out(0),
                  sigmas=_sigmas(g, model, scheduler, steps, width, height, denoise),
                  latent_image=latent).out(0)


def _clear_after_run():
    """Ask ComfyUI to unload all models and drop its node cache once this prompt is done:
    the same flags as its Free memory / Unload models buttons, which the queue worker reads
    after each prompt, so this run still finishes with everything loaded."""
    try:
        from server import PromptServer
        q = PromptServer.instance.prompt_queue
        q.set_flag("unload_models", True)
        q.set_flag("free_memory", True)
    except Exception:  # not running inside the ComfyUI server (tests, scripts)
        pass


class SceneRefStudio:
    DESCRIPTION = (
        "The control panel: wire the viewer's 'shot', pick image or movie at the top, set who is in it in the "
        "Cast list, add LoRAs, write the prompt (or have the vision model write it from the action) and Run. "
        "Images: FLUX.2 Klein, FLUX.3 Klein, Qwen-Image 2.1, Krea2 or MOSS-Image, built as a full graph inside. Movies: MiniMax H3 ref2va "
        "from the scene as framed and the characters' pictures. Wire 'image' / 'video' into a SceneRef Preview & "
        "Save node to see and keep the result."
    )
    CATEGORY = "SceneRef"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING", "VIDEO")
    RETURN_NAMES = ("image", "prompt", "video")
    OUTPUT_TOOLTIPS = ("The finished picture(s), one per shot. Nothing in movie mode.",
                       "The prompt that was used: the prompt box as written.",
                       "Movie mode: the clip with its sound. Nothing in image mode.")

    @classmethod
    def INPUT_TYPES(cls):
        def models(widget):
            kind, _ = PICKS[widget]
            return ([AUTO] + widget_files(widget), {"default": AUTO, "tooltip": f"From models/{kind}. Filled in with "
                    f"the file found for it ({EXPECTED.get(widget, 'the usual file')}). Pick another file only if "
                    f"yours is named differently: a file of the wrong kind fails or gives noise."})

        def legacy(spec):
            return (spec[0], {**spec[1], "tooltip": "Not used any more (kept so older workflows load)."})
        return {
            "required": {
                "shot": (SHOT_TYPE, {"tooltip": "The 'shot' output of the Scene Reference Viewer, or of SceneRef "
                                     "Scene (a place from pictures)."}),
                "backend": (list(BACKENDS), {"default": KLEIN, "sr_defaults": DEFAULTS, "tooltip": "Image mode: "
                            "which model makes the picture. FLUX.2 Klein 9B: fast (4 steps), most photographic. "
                            "FLUX.3 Klein: 16 steps, follows long prompts best; its prompt is written by BFL's "
                            "prompt upsampler (press 'Write the prompt'). "
                            "Qwen-Image 2.1 + ControlNet: slower (25 steps), depth / pose ControlNet holds "
                            "every body exactly in place. Krea2 + identity edit: one pass per character, "
                            "best likeness, keeps the layout (needs comfyui-krea2edit). MOSS-Image + scene control: "
                            "25 steps; the room, furniture and bodies are held by the scene's depth and object map "
                            "(not copied from the render, so no 3D look carries over); faces come from the face "
                            "pass (needs ComfyUI 0.43+ and the Comfy-Org/MOSS-Image files)."}),
                "action": ("STRING", {"multiline": True, "default": "", "tooltip": "For the vision model only "
                           "(it is not part of the prompt): what happens and how it should look, e.g. 'Emma "
                           "sleeps on the couch under a blanket. Jack pours coffee at the table. Late night, "
                           "light from the TV.' Name the people as in the Cast list. For a movie: in order, "
                           "with speech in quotes after says / asks ('Emma says: \"Coming?\"') and any camera "
                           "move. Press 'Write the prompt' to turn it into the prompt."}),
                "expand": legacy(("BOOLEAN", {"default": False})),
                "room_details": ("BOOLEAN", {"default": True, "tooltip": "Write the prompt: let the vision model "
                                 "add decoration and small objects and describe the light."}),
                "style": legacy(("STRING", {"multiline": True, "default": ""})),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True,
                                 "tooltip": "Same seed + same settings = the same result. Set 'control after "
                                 "generate' to fixed to keep a result while you change the text."}),
                "steps": ("INT", {"default": 4, "min": 0, "max": 200, "tooltip": "Sampling steps. Filled in "
                          "with the usual value when you change backend or mode (Klein 4, Qwen 25, Krea2 10, "
                          "movie 20, as ComfyUI's H3 template; a turbo model or LoRA needs fewer, e.g. 8). 0 = the usual."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1,
                        "tooltip": "How strictly the prompt is followed. The distilled / turbo backends here "
                        "work at 1.0; higher burns the picture. 0 = the backend's usual. Pictures only: H3 "
                        "samples without CFG (as in ComfyUI's own H3 template)."}),
                "control_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05,
                                     "tooltip": "Qwen only: how hard the depth + pose ControlNet holds the "
                                     "layout. 0 = off."}),
                "control_mode": (list(CONTROL_MODES), {"default": "depth", "tooltip": "Qwen only: what "
                                 "the ControlNet follows. depth = room and bodies as shapes; pose = the skeletons."}),
                "ref_megapixels": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 2.0, "step": 0.05,
                                   "tooltip": "Size of each character picture (Klein). Smaller = faster, "
                                   "less likeness."}),
                "face_pass": ("BOOLEAN", {"default": True, "tooltip": "After the picture is made, redo every "
                              "visible face that belongs to a character on a figure (heads big enough to redo), "
                              "close up, from that "
                              "character's face pictures only (FLUX.2 Klein, about 3 s per face). Klein, Qwen and "
                              "MOSS-Image only: Krea2 does each person from their pictures already. With MOSS-Image "
                              "in a 3D scene this is where the faces come from: keep it on."}),
                "filename_prefix": legacy(("STRING", {"default": "SceneRef/studio"})),
                "klein_model": models("klein_model"),
                "klein_text_encoder": models("klein_text_encoder"),
                "klein_vae": models("klein_vae"),
                "qwen_model": models("qwen_model"),
                "qwen_text_encoder": models("qwen_text_encoder"),
                "qwen_vae": models("qwen_vae"),
                "qwen_controlnet": models("qwen_controlnet"),
                "expander_model": ([AUTO] + vlm_files(), {"default": AUTO, "tooltip": "The vision language "
                                   "model that writes the prompt from the action. Only files that can write "
                                   "text are listed (Qwen3-VL with its lm_head, from models/text_encoders or "
                                   "models/LLM)."}),
                "cast_json": ("STRING", {"default": "[]", "multiline": True, "tooltip": "Edited with the Cast "
                              "list on the node."}),
                "loras_json": ("STRING", {"default": "[]", "multiline": True, "tooltip": "Edited with the "
                               "LoRA list on the node."}),
                # New settings go last, so workflows saved earlier keep their values in place.
                "identity_pass": legacy((list(IDENTITY_PASSES), {"default": "off"})),
                "identity_strength": legacy(("FLOAT", {"default": 0.4, "min": 0.1, "max": 0.9, "step": 0.05})),
                "ref_boost": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 10.0, "step": 0.5, "tooltip":
                              "Krea2: how hard the edit copies the person's picture (the LoRA author's "
                              "recommendation: 4; above 10 breaks)."}),
                "krea2_model": models("krea2_model"),
                "krea2_text_encoder": models("krea2_text_encoder"),
                "krea2_vae": models("krea2_vae"),
                "krea2_identity_lora": models("krea2_identity_lora"),
                "h3_model": models("h3_model"),
                "h3_text_encoder": models("h3_text_encoder"),
                "h3_vae": models("h3_vae"),
                "h3_turbo_lora": legacy(models("h3_turbo_lora")),
                "clear_cache_after": ("BOOLEAN", {"default": True, "label_on": "clear cache when done",
                                      "label_off": "keep models loaded", "tooltip": "When the run is done, "
                                      "ComfyUI unloads every model and forgets every cached node result (its own "
                                      "Free memory). Off = faster re-runs: models stay loaded."}),
                "h3_strength": legacy(("FLOAT", {"default": 0.85, "min": 0.5, "max": 1.0, "step": 0.05})),
                "h3_resolution": legacy((list(H3_RESOLUTIONS), {"default": AUTO})),
                "aspect_ratio": (list(ASPECTS), {"default": "as framed", "tooltip": "Movie: crop the camera "
                                 "view to this shape around its centre. 'as framed' = the viewer's frame."}),
                "h3_ref_size": (["match", "max"], {"default": "match", "tooltip": "Movie: how big the "
                                "character pictures go in. match = scaled down to the clip's size (fast); "
                                "max = up to 2048 px for the most likeness, several times slower."}),
                "video": legacy(("BOOLEAN", {"default": False})),
                "video_seconds": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 15.0, "step": 0.5,
                                  "tooltip": "Clip length at 24 fps, rounded up to H3's frame steps (5 s = 124 "
                                  "frames). H3 was trained on about 5-15 s; time and VRAM grow with it."}),
                "video_resolution": ([k for k in H3_RESOLUTIONS if k != AUTO], {"default": "0.4 MP (fast)",
                                     "tooltip": "The clip's size, in the aspect above. 0.4 MP is ComfyUI's own "
                                     "H3 template default; 0.9 MP is about 720p and much slower."}),
                "video_motion": legacy(("STRING", {"multiline": True, "default": ""})),
                "h3_audio_vae": models("h3_audio_vae"),
                "create": ([IMAGE, MOVIE], {"default": IMAGE, "tooltip": "image: a picture with FLUX.2 Klein, "
                           "FLUX.3 Klein, Qwen-Image or Krea2. movie: a clip with sound from MiniMax H3 (ref2va): the scene as "
                           "framed and the characters' pictures are its references; the prompt directs the rest."}),
                "expanded": ("STRING", {"multiline": True, "default": "", "tooltip": "The prompt, exactly as "
                             "the model gets it. Write it yourself, or press 'Write the prompt' to have the vision "
                             "model write it from the action, then edit it. Nothing is added at Run."}),
                "image_resolution": (list(IMAGE_RESOLUTIONS), {"default": "as framed", "tooltip": "The picture's "
                                     "size. 'as framed' = the viewer's width x height; otherwise about this many "
                                     "megapixels in the aspect below. The viewer's width / height only set how "
                                     "the reference is rendered."}),
                "image_aspect": (list(ASPECTS), {"default": "as framed", "tooltip": "The picture's shape: the "
                                 "camera view is cropped around its centre. 'as framed' = the viewer's frame."}),
                "sampler_name": (SAMPLERS, {"default": "euler", "tooltip": "The sampler. Filled in with the "
                                 "model's usual one when you change backend or mode (Klein, Qwen and Krea2: euler; "
                                 "movie: res_multistep)."}),
                "scheduler": (SCHEDULERS, {"default": FLUX2_SCHEDULER, "tooltip": "The noise schedule. Filled in "
                              "with the model's usual one (Klein: flux2, its own schedule that follows the picture "
                              "size; Qwen, Krea2 and movie: simple)."}),
                "face_steps": ("INT", {"default": 8, "min": 1, "max": 200, "tooltip": "Face pass (Klein): "
                               "sampling steps for each face."}),
                "face_denoise": ("FLOAT", {"default": FACE_DENOISE, "min": 0.05, "max": 1.0, "step": 0.05,
                                 "tooltip": "Face pass: how much of each face is redrawn. Lower keeps more of the "
                                 "picture's expression and light, higher more of the character's face."}),
                "face_sampler": (SAMPLERS, {"default": "euler", "tooltip": "Face pass: the sampler."}),
                "face_scheduler": (SCHEDULERS, {"default": FLUX2_SCHEDULER, "tooltip": "Face pass: the noise "
                                   "schedule (flux2 = Klein's own)."}),
                "klein3_model": models("klein3_model"),
                "klein3_text_encoder": models("klein3_text_encoder"),
                "klein3_upsampler": (flux3.upsampler_choices(), {"default": flux3.AUTO, "tooltip": "FLUX.3 Klein: "
                                     "what writes the caption when you press Write the prompt (not used at Run). "
                                     "auto = BFL's prompt upsampler (qwen3.5_9b_flux3_prompt_upsampler, "
                                     ".safetensors). A .gguf upsampler (in models/LLM with its "
                                     "mmproj file next to it) runs in llama.cpp's llama-server. 'expander model' uses "
                                     "the expander_model above with BFL's instructions. All get BFL's system prompt "
                                     "and run greedy, so the same choice gives the same caption."}),
                "moss_model": models("moss_model"),
                "moss_text_encoder": models("moss_text_encoder"),
                "moss_vae": models("moss_vae"),
                "moss_edit_lora": models("moss_edit_lora"),
                "moss_control_lora": models("moss_control_lora"),
                "moss_control_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 3.0, "step": 0.05,
                                          "tooltip": "MOSS-Image: how hard the scene's depth and object map hold "
                                          "the room and the bodies. 0.7 keeps everything in place and lets bodies "
                                          "look real; 1.0 (ComfyUI's template) copies the mannequins' segmented "
                                          "limbs in close shots; 0 = off (the text alone places everything)."}),
                "moss_control_end": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 1.0, "step": 0.05,
                                     "tooltip": "MOSS-Image: the part of the run the map holds (0.4 = the first 40% "
                                     "of the steps; ComfyUI's template uses 0.6). Higher pins the shapes to the last "
                                     "step; lower frees surfaces, light and bodies."}),
                "moss_regions": ("BOOLEAN", {"default": True, "label_on": "depth + object map", "label_off":
                                 "depth only", "tooltip": "MOSS-Image: also give the control adapter the flat-colour "
                                 "map of which object is where (the viewer's 'regions' output), so each piece of "
                                 "furniture keeps its outline, not just its depth."}),
                "moss_layout": ("BOOLEAN", {"default": False, "tooltip": "MOSS-Image, Write the prompt: write it as "
                                "MOSS's JSON layout (the text as the caption, plus a 'character' box per figure in "
                                "the picture's pixels). Experimental in MOSS itself. Write the prompt again after "
                                "changing the picture size: the boxes are in its pixels."}),
            },
            "optional": {
                "cast": (CAST_TYPE, {"tooltip": "Optional: more characters from SceneRef Character nodes, "
                         "added to the Cast list on the node."}),
                "loras": (LORAS_TYPE, {"tooltip": "Optional: more LoRAs from a SceneRef LoRA Stack, added "
                          "to the LoRA list on the node."}),
                "action_text": ("STRING", {"forceInput": True, "tooltip": "Wired text is used as the prompt, "
                                "instead of the prompt box (e.g. from the SceneRef Prompt Expander)."}),
                **grow_inputs("cast", (CAST_TYPE, {"tooltip": "More characters: one Character node or a chain "
                                                   "of them. A new cast input appears when this one is wired."})),
            },
        }

    def run(self, shot, backend, action, style, seed, steps, cfg, control_strength, control_mode, ref_megapixels,
            face_pass, expand=False, room_details=True, expander_model=AUTO, cast_json="[]", loras_json="[]",
            cast=None, loras=None, action_text=None, ref_boost=4.0, clear_cache_after=True,
            aspect_ratio="as framed", h3_ref_size="match", video_seconds=5.0, video_resolution="0.4 MP (fast)",
            create=IMAGE, expanded="", image_resolution="as framed", image_aspect="as framed",
            sampler_name="euler", scheduler=FLUX2_SCHEDULER, face_steps=8, face_denoise=FACE_DENOISE,
            face_sampler="euler", face_scheduler=FLUX2_SCHEDULER, moss_control_strength=0.7, moss_control_end=0.4,
            moss_regions=True, moss_layout=False, **models):
        from comfy_execution.graph_utils import GraphBuilder

        movie = create == MOVIE
        if not movie and backend not in BACKENDS:
            raise ValueError(f"SceneRef Studio: {backend!r} is not an image backend. MiniMax H3 makes movies: "
                             f"set 'create' to movie.")
        # The prompt as written. The action only feeds the 'Write the prompt' button; style and
        # the expand switch are unused (kept so older workflows load).
        prompt = (action_text if action_text and action_text.strip() else expanded).strip()
        if not prompt:
            raise ValueError("SceneRef Studio: the prompt is empty. Write it in the prompt box, or write the action "
                             "and press 'Write the prompt' to have the vision model write it.")
        # The panel's own lists first, then anything wired in from separate nodes.
        panel_cast = cast_from_json(cast_json)
        for ch in [c for wired in grown("cast", cast, models) for c in wired]:
            panel_cast = add_to_cast(panel_cast, ch)
        cast = scenes.plain_cast(shot, panel_cast)
        prompt = resolve_refs(prompt, cast)
        loras = parse_stack(loras_json) + list(loras or [])
        steps = steps or DEFAULTS[MOVIE if movie else backend]["steps"]
        cfg = cfg or DEFAULTS[backend]["cfg"]
        sampling = {"sampler": sampler_name, "scheduler": scheduler}

        g = GraphBuilder()
        arch = H3_BACKEND if movie else backend

        def add_loras(model):
            for x in loras:
                model = g.node("LoraLoaderModelOnly", model=model, lora_name=x["name"],
                               strength_model=x["strength"]).out(0)
            for ch in cast:
                for lo in ch.get("loras") or []:
                    if lo["arch"] in CHAR_LORA_ARCHS[arch]:
                        model = g.node("SceneRefLoraFile", model=model, path=lo["path"],
                                       strength=lo["strength"]).out(0)
            return model

        if clear_cache_after:
            _clear_after_run()
        if movie:
            _, video = self._movie(
                g, shot, cast, prompt, seed, steps, sampling, video_seconds,
                H3_RESOLUTIONS.get(video_resolution) or VIDEO_MP, ASPECTS.get(aspect_ratio), h3_ref_size, models,
                add_loras)
            # A movie is only the clip: no frames on `image`.
            return {"ui": {}, "result": (None, prompt, video), "expand": g.finalize()}

        if backend == KLEIN:
            model = g.node("UNETLoader", unet_name=pick("klein_model", models.get("klein_model")),
                           weight_dtype="default").out(0)
            clip = g.node("CLIPLoader", clip_name=pick("klein_text_encoder", models.get("klein_text_encoder")),
                          type="flux2", device="default").out(0)
            vae = g.node("VAELoader", vae_name=pick("klein_vae", models.get("klein_vae"))).out(0)
        elif backend == KLEIN3:
            model, clip, vae = self._klein3_models(g, models)
        elif backend == MOSS:
            moss_plan = self._moss_plan(shot, cast, moss_control_strength)
            model, clip, vae = self._moss_models(g, models, control=moss_plan["control"], edit=moss_plan["edit"])
        elif backend == KREA2:
            import nodes as comfy_nodes
            if "Krea2EditModelPatch" not in comfy_nodes.NODE_CLASS_MAPPINGS:
                raise ValueError("SceneRef Studio: the Krea2 backend needs the comfyui-krea2edit node pack "
                                 "(github.com/lbouaraba/comfyui-krea2edit, Apache-2.0). Install it in "
                                 "custom_nodes and restart ComfyUI.")
            model = g.node("UNETLoader", unet_name=pick("krea2_model", models.get("krea2_model")),
                           weight_dtype="default").out(0)
            model = g.node("LoraLoaderModelOnly", model=model, strength_model=1.0, lora_name=pick(
                "krea2_identity_lora", models.get("krea2_identity_lora"))).out(0)
            clip = g.node("CLIPLoader", clip_name=pick("krea2_text_encoder", models.get("krea2_text_encoder")),
                          type="krea2", device="default").out(0)
            vae = g.node("VAELoader", vae_name=pick("krea2_vae", models.get("krea2_vae"))).out(0)
        else:
            model = g.node("UNETLoader", unet_name=pick("qwen_model", models.get("qwen_model")),
                           weight_dtype="default").out(0)
            clip = g.node("CLIPLoader", clip_name=pick("qwen_text_encoder", models.get("qwen_text_encoder")),
                          type="qwen_image", device="default").out(0)
            vae = g.node("VAELoader", vae_name=pick("qwen_vae", models.get("qwen_vae"))).out(0)
        model = add_loras(model)
        if backend == KLEIN3:
            model = self._klein3_shift(g, model)
        if backend == MOSS:
            # MOSS-Image's own guidance: skip layer 22 on the negative branch (its user guide).
            model = g.node("SkipLayerGuidanceDiTSimple", model=model, double_layers="", single_layers=moss.SKIP_LAYER,
                           start_percent=0.0, end_percent=1.0).out(0)

        # A picture scene's other angles (images 2..n, as 'Write the prompt' numbers them), then the
        # character pictures; encoded once and shared by every shot. Krea2 takes no extra pictures.
        refs = [(None, i, t) for i, t in enumerate(scenes.angles_of(shot))] if backend != KREA2 else []
        refs += [(ch, i, _ref_tensor(ch["refs"][i][0])) for ch in cast for i in range(len(ch["refs"]))]

        # The picture's own size: the viewer's frame cropped to the aspect, then scaled to the
        # megapixels asked for ('as framed' = as rendered).
        sw, sh = int(shot["width"]), int(shot["height"])
        left, top, cw, ch_ = aspect_crop(sw, sh, ASPECTS.get(image_aspect))
        width, height = picture_size(cw, ch_, image_resolution)
        cropped = (cw, ch_) != (sw, sh)

        def view(t):
            return g.node("ImageCrop", image=t, width=cw, height=ch_, x=left, y=top).out(0) if cropped else t

        face = {"steps": face_steps, "denoise": face_denoise, "sampler": face_sampler, "scheduler": face_scheduler}
        images = []
        for s in range(_shots(shot)):
            figures = shot["figures"][s] if s < len(shot["figures"]) else []
            if cropped:
                figures = _shift_figures(figures, left, top)
            composite = view(shot["composite"][s:s + 1])
            if backend in (KLEIN, KLEIN3):
                # FLUX.3: every reference at 1 MP at most, as in BFL's workflows.
                img = self._klein(g, model, clip, vae, composite, refs, prompt, width, height, seed + s, steps, cfg,
                                  ref_megapixels if backend == KLEIN else flux3.REF_MEGAPIXELS, sampling)
            elif backend == KREA2:
                img = self._krea2(g, model, clip, vae, composite, figures, cast, prompt, width, height, seed + s,
                                  steps, cfg, sampling, ref_boost)
            elif backend == MOSS:
                depth = colours = None
                if moss_plan["control"]:
                    depth = g.node("ImageScale", image=view(shot["depth"][s:s + 1]), upscale_method="bilinear",
                                   width=width, height=height, crop="disabled").out(0)
                    if moss_regions and shot.get("regions") is not None:
                        ids = shot["regions"][s].numpy()[top:top + ch_, left:left + cw]
                        colours = g.node("ImageScale", image=torch.from_numpy(moss.colour_map(ids))[None],
                                         upscale_method="nearest-exact", width=width, height=height,
                                         crop="disabled").out(0)
                pictures = ([composite] + moss_plan["angles"] if moss_plan["photo"] else []) + moss_plan["people"]
                img = self._moss(g, model, clip, vae, pictures, prompt, width, height, seed + s, steps, cfg, sampling,
                                 depth, colours, moss_control_strength, moss_control_end, place=moss_plan["photo"])
            else:
                # A picture scene has no depth: Qwen keeps image 1 by itself.
                depth = view(shot["depth"][s:s + 1]) if control_mode != "pose" and shot["depth"] is not None else None
                pose = view(shot["pose"][s:s + 1]) if figures and control_mode != "depth" else None
                # With the ControlNet on, image 1 is the empty scene: Qwen keeps whatever image 1
                # shows, so a composite there came back with the coloured mannequins still in it.
                # The bodies reach it through the depth / pose instead.
                marked = not (control_strength > 0 and figures and (depth is not None or pose is not None))
                scene = composite if marked else view(shot["image"][s:s + 1])
                img = self._qwen(g, model, clip, vae, scene, refs, prompt, width, height, seed + s,
                                 steps, cfg, sampling, control_strength, depth, pose, models)
            # The face pass only after Klein / Qwen / MOSS. Krea2 draws people at real-body size, so
            # heads sit away from the mannequins' heads and faces redone there landed on the wall;
            # MOSS's control map keeps every body on its mannequin, so the head boxes still fit.
            if face_pass and backend in (KLEIN, KLEIN3, QWEN, MOSS):
                own = backend in (KLEIN, KLEIN3)
                img = self._faces(g, img, figures, cast, (width, height), (cw, ch_), seed + s, models,
                                  model if own else None, clip if own else None, vae if own else None, backend, face)
            images.append(img)
        out = images[0]
        for img in images[1:]:
            out = g.node("ImageBatch", image1=out, image2=img).out(0)
        return {"ui": {}, "result": (out, prompt, None), "expand": g.finalize()}

    def _faces(self, g, img, figures, cast, size, shot_size, seed, models, klein_model, clip, vae, backend, face):
        """Face pass: per figure tied to a character with a visible face, crop a square around the
        head, regenerate it with Klein from the character's face pictures, blend it back."""
        by_figure = {c.get("figure"): c for c in cast if c.get("figure") and c.get("refs")}
        jobs = []
        for f in figures or []:
            ch = by_figure.get(int(f.get("index") or 0))
            if ch and f.get("face_visible") and f.get("head_box"):
                jobs.append((f, ch))
        if not jobs:
            return img
        try:
            flux2 = pick("klein_model", models.get("klein_model")) if backend != KLEIN else None
        except ValueError:
            if backend != KLEIN3 or klein_model is None:
                raise
            # No FLUX.2 Klein: FLUX.3 redoes the faces itself, on its own schedule.
            flux2 = None
            face = dict(face, scheduler="simple" if face["scheduler"] == FLUX2_SCHEDULER else face["scheduler"])
        if flux2:
            klein_model = g.node("UNETLoader", unet_name=flux2, weight_dtype="default").out(0)
            for ch in cast:
                for lo in ch.get("loras") or []:
                    if lo["arch"] in CHAR_LORA_ARCHS[KLEIN]:
                        klein_model = g.node("SceneRefLoraFile", model=klein_model, path=lo["path"],
                                             strength=lo["strength"]).out(0)
            clip = g.node("CLIPLoader", clip_name=pick("klein_text_encoder", models.get("klein_text_encoder")),
                          type="flux2", device="default").out(0)
            vae = g.node("VAELoader", vae_name=pick("klein_vae", models.get("klein_vae"))).out(0)
        W, H = size
        sx, sy = W / max(shot_size[0], 1), H / max(shot_size[1], 1)
        for k, (f, ch) in enumerate(jobs):
            x0, y0, x1, y1 = f["head_box"]
            r = (x1 - x0) / 2 * sx
            if r < FACE_MIN_RADIUS:
                # A small head is blown up many times to the working size, and Klein then paints a new
                # close-up into the blurry crop: a face twice the head's size pasted over it. Klein's
                # first pass already drew these faces from the references.
                continue
            cx, cy = (x0 + x1) / 2 * sx, (y0 + y1) / 2 * sy + 0.2 * r
            side = int(min(max(r * 5, 96), W, H)) // 8 * 8
            left = int(min(max(cx - side / 2, 0), W - side))
            top = int(min(max(cy - side / 2, 0), H - side))
            faces = [t for t, role in ch["refs"] if role == "face"] or [ch["refs"][0][0]]
            refs = [(ch, i, _ref_tensor(t)) for i, t in enumerate(faces[:3])]
            nums = refs_text(list(range(1, 1 + len(refs)))).lower()
            # The references are the only images; the crop is just where sampling starts
            # (img2img), so pose, expression and light come from it and the face from them.
            # With the crop as image 1 Klein reproduced its face and the identity hardly moved.
            prompt = (f"A close-up photograph of {ch['name']}, the person shown in {nums}: exactly "
                      f"{ch['name']}'s face, face shape, age, skin and hair (glasses only if {nums} show them), "
                      f"with natural skin detail, the same expression and head angle as before.")
            if ch.get("description"):
                prompt += f" {' '.join(ch['description'].split())}"
            crop = g.node("ImageCrop", image=img, width=side, height=side, x=left, y=top).out(0)
            work = 1024 if side < 1024 else side // 16 * 16
            new = self._klein(g, klein_model, clip, vae, crop, refs, prompt, work, work, seed + 101 * (k + 1),
                              face["steps"], 1.0, FACE_REF_MP, face, denoise=face["denoise"], scene_ref=False)
            new = g.node("ImageScale", image=new, upscale_method="lanczos", width=side, height=side,
                         crop="disabled").out(0)
            # Only head and hair go back (the crop is 5 head radii wide; hair reaches about 1.4).
            img = g.node("ImageCompositeMasked", destination=img, source=new, x=left, y=top, resize_source=False,
                         mask=_soft_mask(side, 0.55, 0.8)).out(0)
        return img

    @staticmethod
    def _moss_plan(shot, cast, strength):
        """What a MOSS run uses: the control (a 3D scene's depth and object map, unless switched
        off) and, for a scene from pictures, the reference pictures (moss.ref_plan: the place and
        its angles, then the characters') through the edit LoRA."""
        photo = scenes.is_pictures(shot)
        angles = scenes.angles_of(shot)
        n_scene, plan = moss.ref_plan(cast, len(angles), photo)
        people = [_ref_tensor(ch["refs"][i][0]) for ch, idx, _first in plan for i in idx]
        control = strength > 0 and shot.get("depth") is not None
        return {"photo": photo, "angles": angles[:max(0, n_scene - 1)], "people": people, "control": control,
                "edit": bool(people) or photo}

    @staticmethod
    def _moss_models(g, models, control, edit):
        """MOSS-Image (SFT), its Qwen3.5 encoder and the small FLUX.2 VAE (all native in ComfyUI
        0.43+), with the colour/depth control adapter and the edit LoRA when the run uses them."""
        import nodes as comfy_nodes
        missing = [n for n in ("MOSSImageControl", "TextEncodeMOSSImageEdit") if n not in comfy_nodes.NODE_CLASS_MAPPINGS]
        if missing:
            raise ValueError("SceneRef Studio: the MOSS-Image backend needs ComfyUI 0.43 or newer (its native "
                             f"MOSS-Image nodes; missing: {', '.join(missing)}). Update ComfyUI.")
        model = g.node("UNETLoader", unet_name=pick("moss_model", models.get("moss_model")),
                       weight_dtype="default").out(0)
        if control:
            model = g.node("LoraLoaderModelOnly", model=model, strength_model=1.0,
                           lora_name=pick("moss_control_lora", models.get("moss_control_lora"))).out(0)
        if edit:
            model = g.node("LoraLoaderModelOnly", model=model, strength_model=1.0,
                           lora_name=pick("moss_edit_lora", models.get("moss_edit_lora"))).out(0)
        clip = g.node("CLIPLoader", clip_name=pick("moss_text_encoder", models.get("moss_text_encoder")),
                      type="moss_image", device="default").out(0)
        vae = g.node("VAELoader", vae_name=pick("moss_vae", models.get("moss_vae"))).out(0)
        return model, clip, vae

    @staticmethod
    def _moss(g, model, clip, vae, pictures, prompt, width, height, seed, steps, cfg, sampling, depth, colours,
              strength, end, place=False):
        """MOSS-Image: the reference pictures (if any) through the edit LoRA's encoder, the depth
        and object map through the colour/depth control, a fresh latent at the picture's size."""
        if pictures:
            # 1 MP for all of them together (see moss.REF_MEGAPIXELS), never enlarged.
            pictures = pictures[:moss.MAX_REFS]
            if place and len(pictures) > 1:  # a scene from pictures: the place keeps most of it
                shares = [moss.REF_MEGAPIXELS * moss.PLACE_SHARE] +                          [moss.REF_MEGAPIXELS * (1 - moss.PLACE_SHARE) / (len(pictures) - 1)] * (len(pictures) - 1)
            else:
                shares = [moss.REF_MEGAPIXELS / len(pictures)] * len(pictures)

            def scale(t, mp):
                if isinstance(t, torch.Tensor):
                    mp = min(mp, t.shape[1] * t.shape[2] / 1e6)
                return g.node("ImageScaleToTotalPixels", image=t, upscale_method="lanczos", megapixels=round(mp, 3),
                              resolution_steps=16).out(0)
            scaled = {f"images.image_{k}": scale(t, mp) for k, (t, mp) in enumerate(zip(pictures, shares))}
            pos = g.node("TextEncodeMOSSImageEdit", clip=clip, prompt=prompt, vae=vae, edit_format=moss.EDIT_FORMAT,
                         image_strength=1.0, **scaled).out(0)
            neg = g.node("TextEncodeMOSSImageEdit", clip=clip, prompt=moss.NEGATIVE, vae=vae,
                         edit_format=moss.EDIT_FORMAT, image_strength=1.0, **scaled).out(0)
        else:
            pos = g.node("CLIPTextEncode", clip=clip, text=prompt).out(0)
            neg = g.node("CLIPTextEncode", clip=clip, text=moss.NEGATIVE).out(0)
        if depth is not None or colours is not None:
            maps = {k: v for k, v in (("depth_map", depth), ("color_map", colours)) if v is not None}
            ctl = g.node("MOSSImageControl", positive=pos, negative=neg, vae=vae, strength=strength, start_percent=0.0,
                         end_percent=end, **maps)
            pos, neg = ctl.out(0), ctl.out(1)
        latent = g.node("EmptyFlux2LatentImage", width=width, height=height, batch_size=1).out(0)
        sampled = _sample(g, model, pos, neg, latent, seed, steps, cfg, sampling["sampler"], sampling["scheduler"],
                          width, height)
        return g.node("VAEDecode", samples=sampled, vae=vae).out(0)

    @staticmethod
    def _klein3_models(g, models):
        """FLUX.3 Klein, its Qwen3-VL 4B text encoder (CLIP type flux3) and the FLUX.2 VAE. A .gguf
        picked for the model or the encoder loads through ComfyUI-GGUF (city96)."""
        import nodes as comfy_nodes
        unet = pick("klein3_model", models.get("klein3_model"))
        encoder = pick("klein3_text_encoder", models.get("klein3_text_encoder"))
        if (unet + encoder).lower().count(".gguf") and "UnetLoaderGGUF" not in comfy_nodes.NODE_CLASS_MAPPINGS:
            raise ValueError("SceneRef Studio: a .gguf FLUX.3 file needs the ComfyUI-GGUF node pack "
                             "(github.com/city96/ComfyUI-GGUF). Install it, or pick a .safetensors file.")
        if unet.lower().endswith(".gguf"):
            model = g.node("UnetLoaderGGUF", unet_name=unet).out(0)
        else:
            model = g.node("UNETLoader", unet_name=unet, weight_dtype="default").out(0)
        if encoder.lower().endswith(".gguf"):
            clip = g.node("CLIPLoaderGGUF", clip_name=encoder, type="flux3").out(0)
        else:
            clip = g.node("CLIPLoader", clip_name=encoder, type="flux3", device="default").out(0)
        vae = g.node("VAELoader", vae_name=pick("klein_vae", models.get("klein_vae"))).out(0)
        return model, clip, vae

    @staticmethod
    def _klein3_shift(g, model):
        """BFL's timestep shift: ln(alpha) for both ends, so the picture size does not change it."""
        return g.node("ModelSamplingFlux", model=model, max_shift=round(flux3.SHIFT, 5),
                      base_shift=round(flux3.SHIFT, 5), width=1024, height=1024).out(0)

    @staticmethod
    def _klein(g, model, clip, vae, composite, refs, prompt, width, height, seed, steps, cfg, ref_mp, sampling,
               denoise=1.0, scene_ref=True):
        """Klein reference edit. denoise < 1 starts from image 1 itself (img2img) instead of noise.
        scene_ref=False: image 1 is only the img2img start, not a reference (the refs are images 1..n)."""
        pos = g.node("CLIPTextEncode", clip=clip, text=prompt).out(0)
        neg = g.node("ConditioningZeroOut", conditioning=pos).out(0)
        scene = g.node("ImageScaleToTotalPixels", image=composite, upscale_method="lanczos",
                       megapixels=round(width * height / 1e6, 3), resolution_steps=16).out(0)
        start = g.node("VAEEncode", pixels=scene, vae=vae).out(0)
        # Shrunk to ref_mp, never enlarged (OmniChar's Klein policy): an enlarged small photo
        # only adds reference tokens, no detail.
        pictures = [g.node("ImageScaleToTotalPixels", image=t, upscale_method="lanczos",
                           megapixels=round(min(ref_mp, t.shape[1] * t.shape[2] / 1e6), 3),
                           resolution_steps=16).out(0) for _c, _i, t in refs]
        lats = ([start] if scene_ref else []) + [g.node("VAEEncode", pixels=p, vae=vae).out(0) for p in pictures]
        for lat in lats:
            pos = g.node("ReferenceLatent", conditioning=pos, latent=lat).out(0)
            neg = g.node("ReferenceLatent", conditioning=neg, latent=lat).out(0)
        if denoise < 1.0:
            latent = start
        else:
            latent = g.node("EmptyFlux2LatentImage", width=width, height=height, batch_size=1).out(0)
        sampled = _sample(g, model, pos, neg, latent, seed, steps, cfg, sampling["sampler"], sampling["scheduler"],
                          width, height, denoise)
        return g.node("VAEDecode", samples=sampled, vae=vae).out(0)

    @staticmethod
    def _krea2(g, model, clip, vae, composite, figures, cast, prompt, width, height, seed, steps, cfg, sampling,
               boost):
        """Krea2 identity edit, one pass per character: image 1 = the picture so far (the
        composite at first), image 2 = that character's first picture, "replace the red
        mannequin with the person from image 2". The edit LoRA keeps image 1 exactly, so the
        room and every pose stay; it takes one person picture per pass, hence the chain.
        Figures without a character become people in the first pass. The prompt box holds one
        section per pass ([pass 1] ...), or one text for all of them."""
        s = min(1.0, (1.5e6 / (width * height)) ** 0.5)  # the LoRA works up to about 1.5 MP for two people
        W, H = int(width * s) // 16 * 16, int(height * s) // 16 * 16
        lat = g.node("EmptySD3LatentImage", width=W, height=H, batch_size=1).out(0)
        by_figure = {c.get("figure"): c for c in cast if c.get("figure") and c.get("refs")}
        jobs = krea2_jobs(figures, cast)
        prompts = split_passes(prompt, len(jobs))

        def run_pass(current, text, person, k):
            src = g.node("VAEEncode", pixels=current, vae=vae).out(0)
            patch = {"model": model, "source_latent": src, "ref_boost": boost, "ref_boost_a": 1.0,
                     "fit_mode": "fit", "vae": vae, "source_image": current, "target_latent": lat,
                     "ref_position": "aligned (trained)", "ref_position_a": "aligned (trained)"}
            enc = {"image": current, "grounding_px": 768, "system_prompt": ""}
            if person is not None:
                patch.update(source_latent_b=g.node("VAEEncode", pixels=person, vae=vae).out(0),
                             source_image_b=person)
                enc["image_b"] = person
            m = g.node("Krea2EditModelPatch", **patch).out(0)
            pos = g.node("Krea2EditGroundedEncode", clip=clip, prompt=text, **enc).out(0)
            neg = g.node("Krea2EditGroundedEncode", clip=clip, prompt="", **enc).out(0)
            sampled = _sample(g, m, pos, neg, lat, seed + 7 * k, steps, cfg, sampling["sampler"],
                              sampling["scheduler"], W, H)
            return g.node("VAEDecode", samples=sampled, vae=vae).out(0)

        current = composite
        for k, (f, text) in enumerate(zip(jobs, prompts)):
            if f is None:
                person = None
            elif "refs" in f:  # a character without a figure
                person = _ref_tensor(f["refs"][0][0])
            else:
                person = _ref_tensor(by_figure[int(f["index"])]["refs"][0][0])
            current = run_pass(current, text, person, k)
        if (W, H) != (width, height):
            current = g.node("ImageScale", image=current, upscale_method="lanczos", width=width, height=height,
                             crop="disabled").out(0)
        return current

    @staticmethod
    def _h3_models(g, models, add_loras=None):
        # No built-in turbo LoRA: the LoRA list holds whatever the user wants on it.
        model = g.node("UNETLoader", unet_name=pick("h3_model", models.get("h3_model")), weight_dtype="default").out(0)
        if add_loras:
            model = add_loras(model)
        clip = g.node("CLIPLoader", clip_name=pick("h3_text_encoder", models.get("h3_text_encoder")), type="minimax",
                      device="default").out(0)
        vae = g.node("VAELoader", vae_name=pick("h3_vae", models.get("h3_vae"))).out(0)
        return model, clip, vae

    @staticmethod
    def _movie(g, shot, cast, prompt, seed, steps, sampling, seconds, megapixels, aspect, ref_size, models,
               add_loras):
        """MiniMax H3 ref2va: <Picture 1> is the scene as framed (with mannequins, Daz people or
        nobody), the characters' pictures follow, and the prompt directs everything else. No
        first frame is pinned and nothing is drawn beforehand. Returns (frames, video); with
        several saved shots only the first is used."""
        sw, sh = int(shot["width"]), int(shot["height"])
        left, top, cw, ch_ = aspect_crop(sw, sh, aspect)
        scene = shot["composite"][0:1]
        if (cw, ch_) != (sw, sh):
            scene = g.node("ImageCrop", image=scene, width=cw, height=ch_, x=left, y=top).out(0)
        W, H = h3_size(cw, ch_, megapixels)
        model, clip, vae = SceneRefStudio._h3_models(g, models, add_loras)
        audio_vae = g.node("VAELoader", vae_name=pick("h3_audio_vae", models.get("h3_audio_vae"))).out(0)
        people = [c for c in cast if c.get("refs")]
        angles = scenes.angles_of(shot, movie=True)
        pictures, tagged = h3_people(people, first=2 + len(angles), slots=H3_MAX_REFS - 1 - len(angles))
        refs = {f"ref_images.ref_image_{i}": t for i, t in enumerate([scene] + angles + pictures)}
        prompt, voices = h3_voices(prompt, [c for c, tg in zip(people, tagged) if tg["voice"]])
        refs.update({f"ref_audios.ref_audio_{i}": voice.to_audio(v) for i, v in enumerate(voices)})
        r2v = g.node("MiniMaxH3ReferenceToVideo", clip=clip, vae=vae, audio_vae=audio_vae, prompt=prompt, width=W,
                     height=H, length=h3_frames(seconds), ref_image_size=ref_size, **refs)
        sampled = g.node("SamplerCustomAdvanced", noise=g.node("RandomNoise", noise_seed=seed).out(0),
                         guider=g.node("BasicGuider", model=model, conditioning=r2v.out(0)).out(0),
                         sampler=g.node("KSamplerSelect", sampler_name=sampling["sampler"]).out(0),
                         sigmas=_sigmas(g, model, sampling["scheduler"], int(steps or 20), W, H),
                         latent_image=r2v.out(1)).out(0)
        frames = g.node("VAEDecode", samples=sampled, vae=vae).out(0)
        audio = g.node("VAEDecodeAudio", samples=sampled, vae=audio_vae).out(0)
        video = g.node("CreateVideo", images=frames, fps=float(H3_FPS), audio=audio).out(0)
        return frames, video

    @staticmethod
    def _qwen(g, model, clip, vae, composite, refs, prompt, width, height, seed, steps, cfg, sampling, strength,
              depth, pose, models):
        if strength > 0 and (depth is not None or pose is not None):
            patch = g.node("ModelPatchLoader", name=pick("qwen_controlnet", models.get("qwen_controlnet"))).out(0)
            # Depth (the bodies are in it) holds layout and occlusion; pose holds the limbs.
            if depth is not None:
                model = g.node("ZImageFunControlnet", model=model, model_patch=patch, vae=vae, strength=strength,
                               image=depth, start_percent=0.0, end_percent=0.8).out(0)
            if pose is not None:
                model = g.node("ZImageFunControlnet", model=model, model_patch=patch, vae=vae,
                               strength=strength * 0.8, image=pose, start_percent=0.0, end_percent=0.6).out(0)
        resolution = int(round((width * height) ** 0.5 / 32) * 32)
        inputs = {"images.image_1": composite}
        for k, (_c, _i, t) in enumerate(refs[:15]):
            inputs[f"images.image_{k + 2}"] = t
        enc = g.node("TextEncodeQwenImage21", clip=clip, prompt=prompt, negative_prompt="", vae=vae,
                     resolution=resolution, **inputs)
        sampled = _sample(g, model, enc.out(0), enc.out(1), enc.out(2), seed, steps, cfg, sampling["sampler"],
                          sampling["scheduler"], width, height)
        return g.node("VAEDecode", samples=sampled, vae=vae).out(0)


class SceneRefLoraFile:
    """Internal: applies a character LoRA extracted from a .char file (they live in
    input/scene_ref_chars, not in models/loras)."""
    CATEGORY = "SceneRef/internal"
    FUNCTION = "run"
    RETURN_TYPES = ("MODEL",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "path": ("STRING", {"default": ""}),
                             "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0})}}

    def run(self, model, path, strength):
        import comfy.sd
        import comfy.utils
        root = os.path.realpath(cache_dir())
        real = os.path.realpath(path)
        if not real.startswith(root + os.sep) or not real.endswith(".safetensors") or not os.path.isfile(real):
            raise ValueError("SceneRef: character LoRA not found (it is extracted from the .char file on load).")
        lora = comfy.utils.load_torch_file(real, safe_load=True)
        model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
        return (model,)
