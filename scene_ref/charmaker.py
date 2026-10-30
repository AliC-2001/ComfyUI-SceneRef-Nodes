"""SceneRef Character Maker - writes OmniChar `.char` files.

The file is the one OmniChar Studio writes (layout after the published omnichar-sdk,
Apache-2.0; no code taken from it): a zip with manifest.json first (its signature at
byte 0), the original pictures in refs/ with their roles, text/description.md, a
reference set per model in payloads/ (FLUX.2 Klein: at most 1 MP on a 16 grid;
MiniMax H3: short edge up to 2048 on a 32 grid, H3's own limit) and optionally a voice
(reserved.voice: the recording in voice/samples/, a mono 16-bit WAV for H3 in
voice/payload/). OmniChar Studio and SceneRef both read it; payloads go stale (and are
rebuilt by OmniChar) only if the pictures change.

Unlike OmniChar's H3 policy, pictures are never enlarged for a payload: H3 itself never
enlarges a reference, so an upscaled payload only carries a blurred copy.

Optional: a vision model (Qwen3-VL 8B, the expander's) looks at the pictures and writes
a face and a body description into the file. The picture check (characters.refine:
faces, blur, someone else, duplicates) is reported, not applied: the file keeps the
originals and SceneRef crops faces when it uses them. Not done here: LoRA training.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
import uuid
import wave
import zipfile

from PIL import Image, ImageOps

import folder_paths

from .characters import (CHAR_PREFIX, NONE, PICTURE_ROLES, REFMOD_PREFIX, ROLES, _resolve, _roots, _slug,
                         _to_pil, cap_picture, contact_sheet, guess_roles, input_picture, list_sources,
                         read_char, read_refmod, refine, refmod_vae_file)

MAGIC = "INLINECHAR"
FORMAT_VERSION = 1
KEY_ORDER = ("magic", "format_version", "char_id", "name", "created_at", "modified_at", "app", "app_version",
             "refs", "derived", "text", "payloads", "scoring", "hints", "apply", "reserved")
ENCODER = {"id": "flux2-klein-refset", "version": "1"}  # the SDK's id for every reference set
POLICIES = {
    "flux2-klein": {"max_pixels": 1024 ** 2, "multiple_of": 16},
    "minimax-h3": {"short_edge": 2048, "multiple_of": 32, "max_aspect": 4.0},
}
VOICE_ENCODER = {"id": "h3-voice-wav", "version": "sceneref-1"}  # not OmniChar's: it rebuilds its own
MAX_SAMPLE_SECONDS = 120.0

DESCRIBE_SYSTEM = ("You describe people's appearance precisely for a character sheet that artists use to draw "
                   "the same person again. Plain sentences only. No names, no guesses about identity, no "
                   "comments on the photos, their quality or the background.")
DESCRIBE = {
    "face": ("These photos all show the same person. Describe the face so that someone could recognise it: "
             "apparent age range, face shape, skin tone and complexion, eyes (colour, shape), eyebrows, nose, lips "
             "and smile, hair (colour, length, texture, parting, style), facial hair, glasses, piercings and any "
             "distinctive marks. At most 90 words."),
    "body": ("These photos all show the same person. Describe the body: overall height impression, build and "
             "weight, proportions, shoulders, chest, waist, hips, legs, posture, skin tone, body hair, tattoos or "
             "marks. Do not describe clothing. At most 80 words."),
    "cloth": "Describe this outfit and its accessories: garments, colours, materials, fit. At most 60 words.",
}
LABEL = {"face": "Face", "body": "Body", "cloth": "Outfit"}


def _png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalise(im: Image.Image, policy: dict) -> Image.Image:
    """A picture sized for one model, as the SDK's policies define it."""
    grid = int(policy["multiple_of"])
    w, h = im.width, im.height
    limit = policy.get("max_aspect")
    if limit and max(w / h, h / w) > float(limit):
        # Too long and thin for H3: crop the middle rather than refuse the whole character.
        if w > h:
            nw = int(h * limit)
            im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
        else:
            nh = int(w * limit)
            im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
        w, h = im.width, im.height
    if "short_edge" in policy:
        s = min(1.0, int(policy["short_edge"]) / min(w, h))  # shrink only
        w, h = max(grid, int(w * s) // grid * grid), max(grid, int(h * s) // grid * grid)
    else:
        if w * h > int(policy["max_pixels"]):
            s = (int(policy["max_pixels"]) / (w * h)) ** 0.5
            w, h = int(w * s), int(h * s)
        w, h = max(grid, w // grid * grid), max(grid, h // grid * grid)
    return im if (w, h) == im.size else im.resize((w, h), Image.LANCZOS)


def fingerprint(refs: list[dict], policy: dict) -> str:
    parts = [r["sha256"] for r in refs if r.get("origin", "original") == "original"]
    parts.append(json.dumps(policy, sort_keys=True, separators=(",", ":")))
    return _sha("\x1f".join(parts).encode())


def _voice_members(sample, clip) -> tuple[dict, dict]:
    """voice/samples/000.wav (the recording, 16-bit, its own rate and channels, at most
    MAX_SAMPLE_SECONDS) and voice/payload/000.wav (mono 32 kHz, cleaned up), with the
    reserved.voice record as the SDK writes it."""
    import torch
    from . import voice
    wav, sr = sample
    wav = wav.float()
    while wav.ndim > 2:
        wav = wav[0]
    if wav.ndim == 1:
        wav = wav[None]
    wav = wav[:2, :int(MAX_SAMPLE_SECONDS * sr)].clamp(-1, 1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(wav.shape[0])
        f.setsampwidth(2)
        f.setframerate(int(sr))
        f.writeframes((wav.t() * 32767).round().to(torch.int16).numpy().tobytes())
    sample_bytes, payload = buf.getvalue(), voice.wav_bytes(clip)
    members = {"voice/samples/000.wav": sample_bytes, "voice/payload/000.wav": payload}
    record = {"version": 1,
              "samples": [{"path": "voice/samples/000.wav", "sha256": _sha(sample_bytes), "source_name": "voice.wav",
                           "bytes": len(sample_bytes)}],
              "payload": {"payload_version": 1, "type": "voice", "encoder": dict(VOICE_ENCODER),
                          "source_sha256": _sha(sample_bytes), "path": "voice/payload/000.wav",
                          "sha256": _sha(payload), "sample_rate": voice.VOICE_SR, "channels": 1,
                          "seconds": round(voice.seconds(clip), 3)},
              "scoring": {}}
    return members, record


def build_char(name: str, description: str, pictures: list[tuple[Image.Image, str]],
               voice_sample=None, voice_clip=None) -> tuple[bytes, dict]:
    """The .char bytes and its manifest. pictures: [(PIL, role)], role in face / body / cloth.
    voice_sample = (waveform, sample rate) of the recording, voice_clip its cleaned-up clip."""
    if not pictures:
        raise ValueError("SceneRef Character Maker: add at least one picture.")
    now = int(time.time())
    members: dict[str, bytes] = {}
    refs, ordered = [], []
    # Face, then body, then outfit: prompts number the pictures in this order.
    for i, (im, role) in enumerate(sorted(pictures, key=lambda p: ROLES.index(p[1]))):
        rgb = cap_picture(ImageOps.exif_transpose(im).convert("RGB"))
        data = _png(rgb)
        path = f"refs/{i:03d}.png"
        members[path] = data
        refs.append({"path": path, "sha256": _sha(data), "width": rgb.width, "height": rgb.height,
                     "origin": "original", "role": role})
        ordered.append(rgb)
    manifest = {"magic": MAGIC, "format_version": FORMAT_VERSION, "char_id": str(uuid.uuid4()),
                "name": name.strip() or "Character", "created_at": now, "modified_at": now, "app": "sceneref",
                "app_version": "2", "refs": refs, "derived": [], "text": {}, "payloads": {}, "scoring": {},
                "hints": [], "apply": {}, "reserved": {"adapters": {}, "video_payloads": {}, "members": []}}
    text = description.strip().encode("utf-8")
    if text:
        members["text/description.md"] = text
        manifest["text"] = {"path": "text/description.md", "sha256": _sha(text)}
    for arch, policy in POLICIES.items():
        files = []
        for slot, im in enumerate(ordered):
            data = _png(normalise(im, policy))
            path = f"payloads/{arch}/ref_{slot:03d}.png"
            members[path] = data
            files.append({"path": path, "sha256": _sha(data), "role": refs[slot]["role"]})
        manifest["payloads"][arch] = {"payload_version": 1, "type": "ref", "encoder": dict(ENCODER),
                                      "source_sha256": fingerprint(refs, policy), "policy": dict(policy),
                                      "harvested_count": 0, "files": files}
    if voice_clip is not None:
        from . import voice
        if voice.seconds(voice_clip) < voice.MIN_SECONDS:
            raise ValueError(f"SceneRef Character Maker: the voice has {voice.seconds(voice_clip):.1f} s of sound; "
                             f"it needs at least {voice.MIN_SECONDS:.0f} s (5 to 30 s of clear speech works best).")
        vm, record = _voice_members(voice_sample, voice_clip)
        members.update(vm)
        manifest["reserved"]["voice"] = record
    head = json.dumps({k: manifest[k] for k in KEY_ORDER}, ensure_ascii=True, separators=(",", ":")).encode()
    assert head.startswith(f'{{"magic":"{MAGIC}","format_version":'.encode())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", head)  # first, so the file is recognised from its first bytes
        for path in sorted(members):
            z.writestr(path, members[path], compress_type=zipfile.ZIP_STORED if path.endswith(".png")
                       else zipfile.ZIP_DEFLATED)
    return buf.getvalue(), manifest


def describe(pictures: list[tuple[Image.Image, str]], encoder: str, seed: int = 0) -> str:
    """Face / body / outfit descriptions from the pictures (all pictures of a role at once,
    as one contact sheet), with the expander's vision model, unloaded afterwards."""
    from .expander import _small
    from .vlm import pick_encoder, vlm_text

    by_role: dict[str, list] = {}
    for im, role in pictures:
        by_role.setdefault(role, []).append((im, role))
    questions, roles = [], []
    for role in ROLES:
        if by_role.get(role):
            sheet = contact_sheet(by_role[role][:4], cell=448)
            questions.append((DESCRIBE[role], _small(sheet)))
            roles.append(role)
    answers = vlm_text(questions, DESCRIBE_SYSTEM, pick_encoder(encoder), 260, seed)
    return "\n\n".join(f"{LABEL[r]}: {a}" for r, a in zip(roles, answers) if a)


def characters_folder() -> str:
    root = _roots("characters")[0]
    os.makedirs(root, exist_ok=True)
    return root


class SceneRefCharacterMaker:
    DESCRIPTION = (
        "Makes an OmniChar .char character file from pictures (face, body, outfit) or from a RefMod / "
        ".char, with a name and a description. Optionally a vision model writes a detailed face and body "
        "description from the pictures. The file lands in models/characters, ready for the Studio's Cast "
        "list and for OmniChar Studio."
    )
    CATEGORY = "SceneRef"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("file", "description", "pictures")
    OUTPUT_TOOLTIPS = ("The .char file written (relative to models/characters).",
                       "The description stored in the file.", "The pictures stored, side by side.")

    @classmethod
    def INPUT_TYPES(cls):
        from .vlm import vlm_files
        return {
            "required": {
                "name": ("STRING", {"default": "", "tooltip": "The character's name, also the file name."}),
                "description": ("STRING", {"multiline": True, "default": "", "tooltip": "Your own text: "
                                "age, looks, anything the pictures don't show. Stored in the file."}),
                "describe_with_ai": ("BOOLEAN", {"default": False, "label_on": "write face / body text with AI",
                                     "label_off": "only my text", "tooltip": "A vision model (Qwen3-VL 8B) "
                                     "looks at the pictures and writes a detailed face, body and outfit "
                                     "description, added after yours. Unloaded afterwards."}),
                "source": (list_sources(), {"tooltip": "Optional: start from a RefMod or another .char "
                           "(its pictures and text are copied), or pictures only."}),
                "max_pictures": ("INT", {"default": 8, "min": 1, "max": 24, "tooltip": "At most this many "
                                 "pictures go into the file (faces first)."}),
                "overwrite": ("BOOLEAN", {"default": False, "tooltip": "Replace a .char with the same name. "
                              "Off = a new file name (_2, _3 ...)."}),
                "describe_model": (["auto"] + vlm_files(), {"default": "auto", "tooltip": "With "
                                   "describe_with_ai: the vision language model. Only files that can write "
                                   "text are listed (from models/text_encoders or models/LLM)."}),
                "pictures_json": ("STRING", {"default": "[]", "multiline": True, "tooltip": "Edited with the "
                                  "picture list on the node."}),
            },
            "optional": {
                "face": ("IMAGE", {"tooltip": "Face picture(s)."}),
                "body": ("IMAGE", {"tooltip": "Full body picture(s)."}),
                "outfit": ("IMAGE", {"tooltip": "Outfit picture(s)."}),
                "refmod_vae": (["auto"] + folder_paths.get_filename_list("vae"), {"default": "auto",
                               "tooltip": "Only for a RefMod source: the MiniMax H3 video VAE (auto = the "
                               "minimax_h3_video VAE in models/vae)."}),
                "voice": ("AUDIO", {"tooltip": "Optional: a recording of this person speaking (5 to 30 s of clear "
                          "speech, no music), stored in the file for MiniMax H3 movies (OmniChar's voice slot). "
                          "Without it, a voice in the source .char / RefMod is kept."}),
            },
        }

    def run(self, name, description, describe_with_ai, source, max_pictures, overwrite, describe_model,
            pictures_json="[]", face=None, body=None, outfit=None, refmod_vae="", voice=None):
        pictures = []
        try:
            rows = json.loads(pictures_json or "[]")
        except ValueError:
            rows = []
        for r in rows if isinstance(rows, list) else []:
            if isinstance(r, dict) and r.get("file"):
                pictures.append((input_picture(r["file"]), PICTURE_ROLES.get(r.get("role"), "face")))
        for batch, role in ((face, "face"), (body, "body"), (outfit, "cloth")):
            if batch is not None:
                pictures.extend((_to_pil(img), role) for img in batch)
        text = description.strip()
        from . import voice as voices
        sample = clip = None
        if voice is not None:
            sample = (voice["waveform"], voice["sample_rate"])
            clip, _notes = voices.prepare(*sample)
        if source != NONE:
            # The source's own pictures as they are (no face crops), its text and its voice.
            if source.startswith(CHAR_PREFIX):
                ch = read_char(_resolve("characters", source[len(CHAR_PREFIX):]))
            else:
                ch = read_refmod(source[len(REFMOD_PREFIX):], refmod_vae_file(refmod_vae))
            pictures = guess_roles(ch["refs"]) + pictures
            name = name.strip() or ch["name"]
            if not text:
                text = ch.get("description", "")
            if clip is None and ch.get("voice") is not None:
                clip = ch["voice"]
                sample = (clip[None], voices.VOICE_SR)
        pictures = sorted(pictures, key=lambda p: ROLES.index(p[1]))[:max_pictures]
        _used, notes = refine(pictures, max_pictures)
        if not pictures:
            raise ValueError("SceneRef Character Maker: add pictures (＋ picture on the node, or wire face / "
                             "body / outfit), or pick a RefMod / .char as the source.")
        if not name.strip():
            raise ValueError("SceneRef Character Maker: give the character a name.")
        if describe_with_ai:
            ai = describe(pictures, describe_model)
            text = f"{text}\n\n{ai}".strip() if ai else text
        data, manifest = build_char(name, text, pictures, sample, clip)
        folder = characters_folder()
        base = _slug(name)
        path = os.path.join(folder, base + ".char")
        n = 2
        while os.path.exists(path) and not overwrite:
            path = os.path.join(folder, f"{base}_{n}.char")
            n += 1
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        rel = os.path.basename(path)
        summary = (f"Saved {rel} ({len(pictures)} pictures: "
                   + ", ".join(f"{sum(1 for _p, r in pictures if r == role)} {role}" for role in ROLES
                               if any(r == role for _p, r in pictures))
                   + (f", voice {voices.seconds(clip):.1f} s" if clip is not None else ", no voice")
                   + f"). In the Cast list it is '{CHAR_PREFIX}{rel}'."
                   + "".join(f"\n• {n}" for n in notes if "left out" not in n) + f"\n\n{text}")
        return {"ui": {"text": [summary]}, "result": (rel, text, contact_sheet(pictures))}
