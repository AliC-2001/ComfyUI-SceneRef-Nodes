"""SceneRef Character - who plays which figure.

A character comes from one of three places and always ends up as the same thing:
a name, a description, a few reference pictures (each with a role: face, body or
outfit) and optionally a LoRA. Characters are chained into a cast; each one is
tied to a figure in the scene (figure 1 = red, 2 = blue, ...), so the Studio can
tell the model "the red figure is Emma" and hand it Emma's pictures.

Sources:
* OmniChar `.char` files (a zip: manifest.json + refs/ + text/ + payloads/), read
  after the published format (omnichar-sdk, Apache-2.0); no code taken from it.
* DaSiWa RefMods (`models/refmods/*.safetensors`): MiniMax H3 latents plus their
  descriptions. The latents are decoded to pictures once with the H3 video VAE and
  cached, so Klein / Qwen can use them.
* Plain face / body / outfit pictures wired into the node.

On the way in (refine): faces are found (scene_ref.faces), a face picture whose face is
small is cropped to the face, a character with only body pictures gets a face crop, RefMod
pictures get their role from the face size, and notes say what looks wrong (no face,
blurred, a different person, the same photo twice). max_refs is shared out face : body :
outfit = 2 : 1 : 1, best pictures first. A voice (scene_ref.voice) comes from the .char's
voice record, a RefMod's audio member or an uploaded recording.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths

CAST_TYPE = "SCENEREF_CAST"
# Inputs that grow (web/scene_ref_panels.js growInputs): `name`, then name_2 ... name_N, each
# shown once the one before it is wired.
GROW_INPUTS = 8


def grown(name: str, first, kwargs: dict, n: int = GROW_INPUTS) -> list:
    """The values wired to a growing input, in slot order (taken out of kwargs)."""
    return [x for x in [first] + [kwargs.pop(f"{name}_{i}", None) for i in range(2, n + 1)] if x is not None]


def grow_inputs(name: str, spec: tuple, n: int = GROW_INPUTS) -> dict:
    return {f"{name}_{i}": spec for i in range(2, n + 1)}
ROLES = ("face", "body", "cloth")
ROLE_WORDS = {"face": "face", "body": "full body and build", "cloth": "outfit"}
FIGURE_COLORS = ("red", "blue", "green", "yellow", "purple", "orange", "cyan", "pink")
NONE = "(pictures only)"
CHAR_PREFIX = "char: "
REFMOD_PREFIX = "refmod: "

# .char limits: a manifest is small, pictures are at most a few MB, adapters can be big.
MAX_MANIFEST = 4 * 2**20
MAX_IMAGE = 64 * 2**20
MAX_ADAPTER = 4 * 2**30
MAX_VOICE = 50 * 2**20
CHAR_FORMAT_VERSION = 1
# Pictures are kept up to a 2048 short edge (MiniMax H3's own limit for references with
# ref_image_size "max"); each backend shrinks them to what it uses (Klein to ref_megapixels,
# Qwen to its resolution, Krea2 to the target). Nothing is ever enlarged: a resampled small
# photo has no more detail, only more pixels.
MAX_REF_SHORT = 2048
MAX_REF_LONG = 4096
# Role weights when max_refs has to leave pictures out (OmniChar's split).
ROLE_WEIGHT = {"face": 2, "body": 1, "cloth": 1}
FACE_FILL = 0.28  # a face this tall (fraction of the picture's height) makes a face picture
CROP_MIN = 384  # a face crop smaller than this (pixels a side) is not worth using
AUTO_ROLE = "auto"  # RefMod pictures carry no role: decided from the face size


# ── folders ───────────────────────────────────────────────────────────────────
def _roots(kind: str) -> list[str]:
    """Folders for a model kind: ComfyUI's registered ones, models/<kind>, and for
    characters OmniChar's INLINE_CHARACTERS_DIR."""
    out = []
    try:
        out.extend(folder_paths.get_folder_paths(kind))
    except Exception:
        pass
    out.append(os.path.join(folder_paths.models_dir, kind))
    if kind == "characters" and os.environ.get("INLINE_CHARACTERS_DIR"):
        out.append(os.environ["INLINE_CHARACTERS_DIR"])
    seen, roots = set(), []
    for p in out:
        ident = os.path.normcase(os.path.realpath(p))
        if ident not in seen:
            seen.add(ident)
            roots.append(os.path.abspath(p))
    return roots


def _list(kind: str, ext: str) -> list[str]:
    names = set()
    for root in _roots(kind):
        if not os.path.isdir(root):
            continue
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if not x.startswith(".") and x not in ("graph_presets", "__pycache__")]
            for f in files:
                if f.lower().endswith(ext) and not f.startswith("."):
                    names.add(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/"))
    return sorted(names, key=str.lower)


def _resolve(kind: str, rel: str) -> str:
    """A listed name back to a file, refusing anything outside the roots."""
    parts = rel.replace("\\", "/").split("/")
    if not rel or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"SceneRef Character: invalid name {rel!r}")
    for root in _roots(kind):
        path = os.path.realpath(os.path.join(root, *parts))
        if path.startswith(os.path.realpath(root) + os.sep) and os.path.isfile(path):
            return path
    raise ValueError(f"SceneRef Character: {rel} not found in the {kind} folders.")


def list_sources() -> list[str]:
    return ([NONE] + [CHAR_PREFIX + n for n in _list("characters", ".char")]
            + [REFMOD_PREFIX + n[:-len(".safetensors")] for n in _list("refmods", ".safetensors")])


def cache_dir() -> str:
    d = os.path.join(folder_paths.get_input_directory(), "scene_ref_chars")
    os.makedirs(d, exist_ok=True)
    return d


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:60] or "char"


def _to_tensor(im: Image.Image) -> torch.Tensor:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None]


def _to_pil(t: torch.Tensor) -> Image.Image:
    a = (t.clamp(0, 1).mul(255).round().byte().cpu().numpy())
    return Image.fromarray(a)


def cap_picture(im: Image.Image) -> Image.Image:
    """Shrink (never enlarge) to a short edge of at most MAX_REF_SHORT and a long edge of at
    most MAX_REF_LONG, keeping the aspect ratio."""
    s = min(1.0, MAX_REF_SHORT / min(im.width, im.height), MAX_REF_LONG / max(im.width, im.height))
    if s >= 1.0:
        return im
    return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)


# ── .char ─────────────────────────────────────────────────────────────────────
def _char_description(manifest: dict, names: set, member) -> str:
    # The SDK writes text = {"path": "text/description.md", "sha256": ...}.
    text = manifest.get("text") if isinstance(manifest.get("text"), dict) else {}
    desc_path = text.get("path") if isinstance(text.get("path"), str) else "text/description.md"
    return member(desc_path, MAX_MANIFEST).decode("utf-8", "replace").strip() if desc_path in names else ""


def _char_voice(manifest: dict, names: set, member):
    """The voice in a .char (reserved.voice): the mono payload, else the first sample.
    Returns (clip or None, notes)."""
    from . import voice
    reserved = manifest.get("reserved") if isinstance(manifest.get("reserved"), dict) else {}
    rec = reserved.get("voice") if isinstance(reserved.get("voice"), dict) else {}
    payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
    paths = [payload.get("path")] + [s.get("path") for s in rec.get("samples") or [] if isinstance(s, dict)]
    for path in paths:
        if isinstance(path, str) and path in names:
            wav, sr = voice.load_bytes(member(path, MAX_VOICE), os.path.splitext(path)[1].lower() or ".wav")
            return voice.prepare(wav, sr)
    return None, []


def read_char(path: str) -> dict:
    """Name, description, refs [(PIL, role)] and LoRAs [{arch, base, strength, path}]."""
    with zipfile.ZipFile(path) as z:
        def member(name, limit):
            info = z.getinfo(name)
            if info.file_size > limit:
                raise ValueError(f"SceneRef Character: {name} in {os.path.basename(path)} is too large.")
            return z.read(info)

        try:
            manifest = json.loads(member("manifest.json", MAX_MANIFEST))
        except KeyError:
            raise ValueError(f"SceneRef Character: {os.path.basename(path)} has no manifest.json; not a .char file.")
        if not isinstance(manifest, dict) or manifest.get("magic") != "INLINECHAR":
            raise ValueError(f"SceneRef Character: {os.path.basename(path)} is not a .char file.")
        version = manifest.get("format_version")
        if not isinstance(version, int) or version > CHAR_FORMAT_VERSION:
            raise ValueError(f"SceneRef Character: {os.path.basename(path)} uses .char format "
                             f"{version}; this node reads up to {CHAR_FORMAT_VERSION}. Update SceneRef.")
        names = set(z.namelist())
        name = clean_name(str(manifest.get("name") or os.path.splitext(os.path.basename(path))[0]))

        desc = _char_description(manifest, names, member)

        def role_of(entry):
            r = entry.get("role") if isinstance(entry, dict) else None
            return r if r in ROLES else "face"  # the spec reads a missing role as face

        # The originals: every backend sizes them itself. The payloads are only a fallback,
        # H3's first (the larger one; OmniChar's Encode node writes it at 512 by default).
        entries = []
        for r in manifest.get("refs") or []:
            if isinstance(r, dict) and r.get("path") in names and r.get("origin", "original") == "original":
                entries.append((r["path"], role_of(r)))
        payloads = manifest.get("payloads") if isinstance(manifest.get("payloads"), dict) else {}
        for arch in ("minimax-h3", "flux2-klein"):
            if entries:
                break
            pay = payloads.get(arch) if isinstance(payloads.get(arch), dict) else {}
            for f in pay.get("files") or []:
                if isinstance(f, dict) and f.get("path") in names:
                    entries.append((f["path"], role_of(f)))
        refs = []
        for p, role in entries:
            with Image.open(io.BytesIO(member(p, MAX_IMAGE))) as im:
                refs.append((cap_picture(ImageOps.exif_transpose(im).convert("RGB")), role))
        clip, notes = _char_voice(manifest, names, member)

        loras = []
        char_id = _slug(str(manifest.get("char_id") or name))
        for arch, entry in payloads.items():
            if not isinstance(entry, dict) or entry.get("type") != "lora":
                continue
            adapter = f"payloads/{arch}/adapter.safetensors"
            if adapter not in names:
                continue
            info = z.getinfo(adapter)
            if info.file_size > MAX_ADAPTER:
                continue
            # Extracted once per content (crc in the name), then reused.
            out = os.path.join(cache_dir(), f"{char_id}_{_slug(arch)}_{info.CRC:08x}.safetensors")
            if not os.path.isfile(out):
                tmp = out + ".part"
                with z.open(info) as src, open(tmp, "wb") as dst:
                    while True:
                        chunk = src.read(16 * 2**20)
                        if not chunk:
                            break
                        dst.write(chunk)
                os.replace(tmp, out)
            strength = entry.get("strength")
            loras.append({"arch": arch[:-5] if arch.endswith("-lora") else arch,
                          "base": str(entry.get("base") or ""),
                          "strength": float(strength) if isinstance(strength, (int, float)) else 1.0,
                          "path": out})
    return {"name": name, "description": desc, "refs": refs, "loras": loras, "voice": clip, "notes": notes}


# ── RefMods ───────────────────────────────────────────────────────────────────
def _refmod_meta(path: str) -> dict:
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        header = f.metadata() or {}
    for key in ("refmod_meta",):
        if key in header:
            meta = json.loads(header[key])
            if isinstance(meta, dict):
                return meta
    raise ValueError(f"SceneRef Character: {os.path.basename(path)} has no refmod_meta; not a RefMod.")


# DaSiWa Forge roles that are not the person: a style, a first frame or a pose to copy.
NOT_PERSON_ROLES = ("style", "keyframe", "pose")


def _refmod_items(meta: dict, kind: str = "image") -> list[tuple[str, str]]:
    """(tensor name, description) for every enabled member of one kind ("image" or "audio").
    Members whose Forge role is a style, keyframe or pose are not the person and are skipped;
    videos carry no single picture and are skipped too."""
    if meta.get("kind") == "bundle":
        out = []
        for i, m in enumerate(meta.get("members") or []):
            if (isinstance(m, dict) and m.get("kind") == kind and m.get("source_enabled", True) is not False
                    and m.get("source_role") not in NOT_PERSON_ROLES):
                out.append((f"ref_{i}", str(m.get("description") or "")))
        return out
    if meta.get("kind") == kind:
        return [("latent", str(meta.get("description") or ""))]
    return []


def _refmod_description(meta: dict) -> str:
    # The longest member description is usually the full one; RefMod bundles repeat
    # the name in every member.
    descs = [d for _t, d in _refmod_items(meta) if d]
    return max(descs, key=len) if descs else ""


def stored_text(source: str) -> str:
    """The description stored in a .char / RefMod, read without its pictures (no decoding),
    for the cast panel's 'also use the text stored in the file'."""
    if source.startswith(CHAR_PREFIX):
        path = _resolve("characters", source[len(CHAR_PREFIX):])
        with zipfile.ZipFile(path) as z:
            def member(name, limit):
                info = z.getinfo(name)
                if info.file_size > limit:
                    raise ValueError(f"SceneRef Character: {name} in {os.path.basename(path)} is too large.")
                return z.read(info)
            try:
                manifest = json.loads(member("manifest.json", MAX_MANIFEST))
            except KeyError:
                raise ValueError(f"SceneRef Character: {os.path.basename(path)} has no manifest.json; not a .char "
                                 f"file.")
            return _char_description(manifest if isinstance(manifest, dict) else {}, set(z.namelist()), member)
    if source.startswith(REFMOD_PREFIX):
        return _refmod_description(_refmod_meta(_resolve("refmods", source[len(REFMOD_PREFIX):] + ".safetensors")))
    return ""


def default_refmod_vae(kind: str = "video") -> str:
    vaes = folder_paths.get_filename_list("vae")
    for v in vaes:
        if f"minimax_h3_{kind}" in v.lower():
            return v
    return vaes[0] if vaes and kind == "video" else ""


def refmod_vae_file(name: str) -> str:
    """The refmod_vae choice as a file: 'auto' (or nothing) = the MiniMax H3 video VAE in models/vae.
    Saved as 'auto', a workflow validates on installs without that file when no RefMod is used."""
    return default_refmod_vae() if not name or name == "auto" else name


def _refmod_name(meta: dict, rel: str, desc: str) -> str:
    m = re.match(r"\s*([A-Z][\w-]+),", desc)
    if m:
        return m.group(1)  # "Emma, a woman in her thirties" -> Emma
    name = str(meta.get("name") or os.path.basename(rel))
    return clean_name(re.sub(r"_(single|multi.*)$", "", name, flags=re.I))


def clean_name(name: str) -> str:
    """A stored name as a person's name: 'Emma-sceneref-v1' -> 'Emma'. With a version
    tag at the end, everything after the first - or _ goes; other names stay as they are."""
    m = re.match(r"^\s*([^\W\d_][^-_]*?)\s*[-_](?:.*[-_\s])?v\d+(\.\d+)*\s*$", name, re.I)
    return m.group(1) if m else name.strip()


def read_refmod(rel: str, vae_name: str) -> dict:
    path = _resolve("refmods", rel + ".safetensors")
    meta = _refmod_meta(path)
    items = _refmod_items(meta)
    if not items:
        raise ValueError(f"SceneRef Character: RefMod {rel} holds no pictures (video / audio only).")
    st = os.stat(path)
    folder = os.path.join(cache_dir(), f"refmod_{_slug(rel)}_{st.st_mtime_ns:x}_{st.st_size:x}")
    files = [os.path.join(folder, f"{t}.png") for t, _d in items]
    if not all(os.path.isfile(f) for f in files):
        _decode_refmod(path, [t for t, _d in items], files, vae_name)
    refs = []
    for f in files:
        with Image.open(f) as im:
            refs.append((im.convert("RGB"), AUTO_ROLE))
    desc = _refmod_description(meta)
    name = _refmod_name(meta, rel, desc)
    clip, notes = _refmod_voice(path, meta, folder)
    return {"name": name, "description": desc, "refs": refs, "loras": [], "voice": clip, "notes": notes}


@torch.inference_mode()
def _refmod_voice(path: str, meta: dict, folder: str):
    """The first audio member of a RefMod as the character's voice, decoded once with the H3
    audio VAE and cached as a WAV next to the decoded pictures. Returns (clip or None, notes)."""
    from . import voice
    items = _refmod_items(meta, "audio")
    if not items:
        return None, []
    out = os.path.join(folder, f"{items[0][0]}.wav")
    if not os.path.isfile(out):
        vae_name = default_refmod_vae("audio")
        if not vae_name:
            return None, ["the RefMod has a voice, but the MiniMax H3 audio VAE is not in models/vae"]
        import comfy.model_management as mm
        import comfy.sd
        import comfy.utils
        from comfy_extras.nodes_audio import vae_decode_audio
        from safetensors import safe_open
        sd, md = comfy.utils.load_torch_file(folder_paths.get_full_path("vae", vae_name), return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=md)
        del sd
        try:
            with safe_open(path, framework="pt") as f:
                lat = f.get_tensor(items[0][0]).float()
            audio = vae_decode_audio(vae, {"samples": lat})
        finally:
            del vae
            mm.soft_empty_cache()
        clip, _notes = voice.prepare(audio["waveform"], audio["sample_rate"])
        os.makedirs(folder, exist_ok=True)
        with open(out + ".part", "wb") as fh:
            fh.write(voice.wav_bytes(clip))
        os.replace(out + ".part", out)
    with open(out, "rb") as fh:
        return voice.prepare(*voice.load_bytes(fh.read()))


@torch.inference_mode()  # also when called outside a prompt (the cast check): no autograd buffers
def _decode_refmod(path: str, tensors: list[str], files: list[str], vae_name: str):
    import comfy.model_management as mm
    import comfy.sd
    import comfy.utils
    from safetensors import safe_open

    vae_path = folder_paths.get_full_path("vae", vae_name) if vae_name else None
    if not vae_path:
        raise ValueError("SceneRef Character: RefMods are decoded with the MiniMax H3 video VAE "
                         "(minimax_h3_video_vae_fp16.safetensors in models/vae). Pick it in refmod_vae.")
    sd, md = comfy.utils.load_torch_file(vae_path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=md)
    del sd
    try:
        os.makedirs(os.path.dirname(files[0]), exist_ok=True)
        with safe_open(path, framework="pt") as f:
            for name, out in zip(tensors, files):
                lat = f.get_tensor(name).float()
                if lat.ndim == 4:
                    lat = lat[:, :, None]
                if lat.ndim != 5 or lat.shape[1] != 24:
                    raise ValueError(f"SceneRef Character: {os.path.basename(path)}:{name} is not an H3 "
                                     f"image latent (shape {tuple(lat.shape)}).")
                img = vae.decode(lat)
                img = img.reshape(-1, *img.shape[-3:])[0]
                _to_pil(img).save(out + ".part.png")
                os.replace(out + ".part.png", out)
    finally:
        # Decoding is a one-off per RefMod: give the VRAM back right away.
        del vae
        mm.soft_empty_cache()


# ── cast ──────────────────────────────────────────────────────────────────────
def contact_sheet(refs: list[tuple[Image.Image, str]], cell: int = 384) -> torch.Tensor:
    if not refs:
        return torch.zeros(1, 64, 64, 3)
    sheet = Image.new("RGB", (cell * len(refs), cell), (24, 24, 24))
    for i, (im, _role) in enumerate(refs):
        t = ImageOps.contain(im, (cell, cell))
        sheet.paste(t, (i * cell + (cell - t.width) // 2, (cell - t.height) // 2))
    return _to_tensor(sheet)


def figure_label(figure: int) -> str:
    return f"the {FIGURE_COLORS[(figure - 1) % len(FIGURE_COLORS)]} figure" if figure > 0 else ""


ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")


def _upper1(text: str) -> str:
    return text[:1].upper() + text[1:]


def refs_text(nums: list[int], style: str = "ordinal") -> str:
    """How a prompt points at reference pictures: "Image 2" / "Images 2 and 3" for
    FLUX.2 Klein, "<image2>" / "<image2> and <image3>" for Qwen-Image 2.1, whose
    encoder tags every reference that way, "the first image" / "the second and third images"
    for FLUX.3 Klein, whose prompt upsampler names its references by ordinal."""
    if style == "tag":
        tags = [f"<image{n}>" for n in nums]
        return tags[0] if len(tags) == 1 else f"{', '.join(tags[:-1])} and {tags[-1]}"
    if style == "words":
        words = [ORDINALS[n - 1] if n <= len(ORDINALS) else f"{n}th" for n in nums]
        if len(words) == 1:
            return f"the {words[0]} image"
        return f"the {', '.join(words[:-1])} and {words[-1]} images"
    if len(nums) == 1:
        return f"Image {nums[0]}"
    return f"Images {', '.join(map(str, nums[:-1]))} and {nums[-1]}"


def character_line(ch: dict, first_position: int, style: str = "ordinal", scene_ref: int = 1) -> str:
    """One character: which pictures show them, which figure they play, what they look like."""
    name = ch["name"]
    n = len(ch["refs"])
    parts = []
    if n:
        nums = list(range(first_position, first_position + n))
        parts.append(f"{_upper1(refs_text(nums, style))} {'shows' if n == 1 else 'show'} {name}, "
                     f"the same person in every picture.")
        roles = {}
        for i, (_im, role) in enumerate(ch["refs"]):
            roles.setdefault(role, []).append(first_position + i)
        if set(roles) - {"face"}:
            for role in ROLES:
                if role in roles:
                    r = roles[role]
                    parts.append(f"{_upper1(refs_text(r, style))} {'shows' if len(r) == 1 else 'show'} "
                                 f"{name}'s {ROLE_WORDS[role]}.")
    label = figure_label(ch.get("figure", 0))
    if label:
        parts.append(f"{name} is {label} in {refs_text([scene_ref], style).lower() if style != 'tag' else refs_text([scene_ref], style)}.")
    desc = _upper1(" ".join(ch.get("description", "").split()))
    if desc:
        parts.append(desc if desc[-1] in ".!?" else desc + ".")
    return " ".join(parts)


def cast_prompt(cast: list[dict], first_position: int, style: str = "ordinal") -> str:
    """Character lines for a whole cast; references numbered in cast order from first_position."""
    lines, pos = [], first_position
    for ch in cast:
        lines.append(character_line(ch, pos, style))
        pos += len(ch["refs"])
    return " ".join(x for x in lines if x)


class SceneRefCharacter:
    DESCRIPTION = (
        "One character for the scene: from an OmniChar .char file, a DaSiWa RefMod, or face / body / "
        "outfit pictures. Pick which figure it plays (1 = red, 2 = blue, ...) and chain several "
        "Character nodes through 'cast' for several people. The Studio node turns the cast into "
        "reference pictures, LoRAs and prompt text."
    )
    CATEGORY = "SceneRef"
    FUNCTION = "run"
    RETURN_TYPES = (CAST_TYPE, "IMAGE", "STRING")
    RETURN_NAMES = ("cast", "refs", "prompt")
    OUTPUT_TOOLTIPS = (
        "The cast so far, with this character added. Chain into the next Character node or the Studio.",
        "This character's reference pictures side by side (a preview).",
        "The prompt text this character adds (references counted as if the scene were image 1).",
    )

    @classmethod
    def INPUT_TYPES(cls):
        vaes = folder_paths.get_filename_list("vae")
        return {
            "required": {
                "source": (list_sources(), {"tooltip": "A .char file (models/characters), a RefMod "
                                            "(models/refmods), or pictures only (wire face / body / outfit)."}),
                "figure": ("INT", {"default": 0, "min": 0, "max": 8, "tooltip": "Which figure in the scene "
                                   "this character plays: 1 = red, 2 = blue, 3 = green, 4 = yellow, 5 = purple, "
                                   "6 = orange, 7 = cyan, 8 = pink. 0 = not tied to a figure."}),
                "name": ("STRING", {"default": "", "tooltip": "Name used in the prompt. Empty = the name "
                                    "stored in the file (or 'Figure N')."}),
                "description": ("STRING", {"default": "", "multiline": True, "tooltip": "Looks, clothes, age, "
                                           "added to the prompt. Empty = the pictures speak for themselves "
                                           "(see use_file_text)."}),
                "max_refs": ("INT", {"default": 3, "min": 1, "max": 8, "tooltip": "At most this many "
                                     "reference pictures (face first). Each one costs memory and time."}),
            },
            "optional": {
                "cast": (CAST_TYPE, {"tooltip": "The cast from the previous Character node."}),
                "face": ("IMAGE", {"tooltip": "Face picture(s), added to the file's pictures."}),
                "body": ("IMAGE", {"tooltip": "Full body picture(s)."}),
                "outfit": ("IMAGE", {"tooltip": "Outfit picture(s)."}),
                "refmod_vae": (["auto"] + vaes, {"default": "auto", "tooltip": "Only for RefMods: the "
                                      "MiniMax H3 video VAE that decodes their latents to pictures (once; "
                                      "cached in input/scene_ref_chars). auto = the minimax_h3_video VAE."}),
                "use_file_text": ("BOOLEAN", {"default": False, "tooltip": "Also put the description stored "
                                  "in the .char / RefMod into the prompt. Off by default: a stored text that "
                                  "disagrees with the pictures (age, hair) wins over them."}),
                "voice": ("AUDIO", {"tooltip": "Optional: a recording of this person speaking (5 to 30 s of "
                          "clear speech, no music), for MiniMax H3 movies. Replaces a voice stored in the "
                          ".char / RefMod. Cleaned up first: mono, 32 kHz, silence trimmed, level evened out."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, source, **kwargs):
        try:
            if source.startswith(CHAR_PREFIX):
                p = _resolve("characters", source[len(CHAR_PREFIX):])
            elif source.startswith(REFMOD_PREFIX):
                p = _resolve("refmods", source[len(REFMOD_PREFIX):] + ".safetensors")
            else:
                return ""
            st = os.stat(p)
            return f"{st.st_mtime_ns}:{st.st_size}"
        except Exception:
            return ""

    def run(self, source, figure, name, description, max_refs, cast=None, face=None, body=None,
            outfit=None, refmod_vae="", use_file_text=False, voice=None):
        pictures = []
        for batch, role in ((face, "face"), (body, "body"), (outfit, "cloth")):
            if batch is not None:
                pictures.extend((_to_pil(img), role) for img in batch)
        ch = make_character(source, figure, name, description, max_refs, pictures, refmod_vae, use_file_text)
        if voice is not None:
            from . import voice as voices
            ch["voice"], notes = voices.prepare(voice["waveform"], voice["sample_rate"])
            ch["notes"] = ch["notes"] + notes
        new_cast = add_to_cast(cast, ch)
        return {"ui": {"text": [character_report(ch)]},
                "result": (new_cast, contact_sheet(ch["refs"]), cast_prompt([ch], 2))}


def character_report(ch: dict) -> str:
    """What the character is made of and what looks wrong, for the node and the panel."""
    from . import voice
    roles = ", ".join(f"{sum(1 for _i, r in ch['refs'] if r == role)} {ROLE_WORDS[role]}" for role in ROLES
                      if any(r == role for _i, r in ch["refs"]))
    lines = [f"{ch['name']}: {len(ch['refs'])} pictures ({roles or 'none'})"
             + (f", voice {voice.seconds(ch['voice']):.1f} s" if ch.get("voice") is not None else ", no voice")]
    lines += [f"• {n}" for n in ch.get("notes") or []]
    return "\n".join(lines)


def make_character(source: str, figure: int, name: str = "", description: str = "", max_refs: int = 3,
                   pictures=(), refmod_vae: str = "", use_file_text: bool = False) -> dict:
    """One character from a .char / RefMod source and/or extra pictures [(PIL, role)].
    Shared by the Character node and the Studio's cast panel.

    ch["description"] is what goes into the image prompt: what the user typed, or the file's
    own text only when use_file_text is on (or there are no pictures). A stored text that
    disagrees with the photos wins over them (a stored "curly hair" beats photos with straight
    hair: Klein draws the text). The file text is kept
    as ch["file_description"] for the prompt expander, which only needs a rough idea."""
    if source.startswith(CHAR_PREFIX):
        ch = read_char(_resolve("characters", source[len(CHAR_PREFIX):]))
    elif source.startswith(REFMOD_PREFIX):
        ch = read_refmod(source[len(REFMOD_PREFIX):], refmod_vae_file(refmod_vae))
    else:
        ch = {"name": "", "description": "", "refs": [], "loras": []}
    ch.setdefault("voice", None)
    ch.setdefault("notes", [])
    ch["refs"].extend((cap_picture(im), role) for im, role in pictures)
    who = name.strip() or ch["name"] or (f"figure {figure}" if figure else "a character")
    if not ch["refs"] and not ch["loras"]:
        raise ValueError(f"SceneRef: {who} has no pictures. Pick a .char or RefMod as the source, or add "
                         "a face / body / outfit picture.")
    ch["refs"], notes = refine(ch["refs"], max_refs)
    ch["notes"] = ch["notes"] + notes
    ch["name"] = name.strip() or ch["name"] or (f"Figure {figure}" if figure else "the person")
    ch["file_description"] = ch.get("description", "")
    if description.strip():
        ch["description"] = description.strip()
    elif ch["refs"] and not use_file_text:
        ch["description"] = ""
    ch["figure"] = int(figure)
    ch["source"] = source
    return ch


def allocate(counts: dict, n: int) -> dict:
    """How many pictures of each role to keep when only n fit: face : body : outfit =
    2 : 1 : 1 by largest remainder, slots a role can't fill going to the others."""
    present = [r for r in ROLES if counts.get(r)]
    if sum(counts[r] for r in present) <= n:
        return {r: counts[r] for r in present}
    total = sum(ROLE_WEIGHT[r] for r in present)
    quota = {r: n * ROLE_WEIGHT[r] / total for r in present}
    take = {r: min(counts[r], int(quota[r])) for r in present}
    order = sorted(present, key=lambda r: (-(quota[r] - int(quota[r])), ROLES.index(r)))
    while sum(take.values()) < n:
        grew = False
        for r in order:
            if take[r] < counts[r] and sum(take.values()) < n:
                take[r] += 1
                grew = True
        if not grew:
            break
    return take


def guess_roles(refs: list) -> list:
    """RefMod pictures (role "auto") as face or body pictures, from the face size."""
    if not any(r == AUTO_ROLE for _i, r in refs):
        return list(refs)
    from . import faces
    out = []
    for im, role in refs:
        if role == AUTO_ROLE:
            f = next(iter(faces.detect(im)), None)
            role = "face" if not faces.available() or (f and (f["box"][3] - f["box"][1]) / im.height >= FACE_FILL)                 else "body"
        out.append((im, role))
    return out


def refine(refs: list, max_refs: int) -> tuple[list, list[str]]:
    """The pictures a character is drawn from, best first per role, in face / body / outfit
    order, at most max_refs; and notes on what looks wrong. Without the face models the
    pictures stay as given (RefMod pictures count as face pictures)."""
    from . import faces
    if not faces.available():
        refs = [(im, "face" if role == AUTO_ROLE else role) for im, role in refs]
        take = allocate({r: sum(1 for _i, x in refs if x == r) for r in ROLES}, max_refs)
        out = []
        for role in ROLES:
            out += [x for x in refs if x[1] == role][:take.get(role, 0)]
        return out, []
    info = faces.inspect(refs)
    notes, items = [], []  # items: {im, role, score, feat, n, src}
    for n, ((im, role), i) in enumerate(zip(refs, info), 1):
        f = i["face"]
        fill = i["face_px"] / im.height if f else 0.0
        if role == AUTO_ROLE:
            role = "face" if fill >= FACE_FILL else "body"
        if not f:
            if role == "face":
                notes.append(f"picture {n}: no face found")
            items.append({"im": im, "role": role, "score": 0.0, "feat": None, "n": n, "src": None})
            continue
        if i["faces"] > 1:
            notes.append(f"picture {n}: {i['faces']} faces, the biggest is used")
        if i["sharp"] < 25:
            notes.append(f"picture {n}: the face is blurred")
        score = min(i["face_px"], 512) / 512 * min(i["sharp"] / 60, 1.0)
        src = (im, f)
        if role == "face" and fill < FACE_FILL:
            # A face picture whose face is small (a half-body selfie, a group photo): the crop
            # puts many more of the reference's pixels on the face.
            crop = faces.face_crop(im, f)
            if crop.width >= CROP_MIN:
                notes.append(f"picture {n}: cropped to the face (it filled {fill:.0%} of the picture)")
                im = crop
            elif i["face_px"] < 96:
                notes.append(f"picture {n}: the face is only {i['face_px']} pixels tall")
        items.append({"im": im, "role": role, "score": score, "feat": i["feat"], "n": n, "src": src})
    # A different person, or the same photo twice, among the pictures with a face.
    with_face = [x for x in items if x["feat"] is not None]
    # SFace tells people apart only roughly across real photos (glasses, angles and lighting
    # move one person's own scores as far apart as two people's), so only a picture
    # that looks like none of the others is called out.
    if len(with_face) >= 3:
        for x in with_face:
            best = max(faces.similarity(x["feat"], y["feat"]) for y in with_face if y is not x)
            if best < faces.DIFFERENT:
                notes.append(f"picture {x['n']}: may show a different person (best likeness {best:.2f})")
                x["score"] -= 2.0
    for a, x in enumerate(with_face):
        for y in with_face[a + 1:]:
            if faces.similarity(x["feat"], y["feat"]) >= faces.DUPLICATE:
                notes.append(f"pictures {x['n']} and {y['n']}: the same photo, or nearly")
                y["score"] -= 1.0
    # Body or outfit pictures only: a face crop from the best of them is the face picture.
    if not any(x["role"] == "face" for x in items):
        best = max((x for x in items if x["src"] and x["score"] > -1.0), key=lambda x: x["score"], default=None)
        if best:
            crop = faces.face_crop(*best["src"])
            if crop.width >= CROP_MIN // 2:
                notes.append(f"face picture made from picture {best['n']}")
                items.append({"im": crop, "role": "face", "score": best["score"], "feat": best["feat"],
                              "n": best["n"], "src": None})
    take = allocate({r: sum(1 for x in items if x["role"] == r) for r in ROLES}, max_refs)
    out = []
    for role in ROLES:
        mine = sorted((x for x in items if x["role"] == role), key=lambda x: -x["score"])
        out += [(x["im"], role) for x in mine[:take.get(role, 0)]]
    left = len(items) - len(out)
    if left > 0:
        notes.append(f"{left} picture{'s' if left > 1 else ''} left out (max {max_refs})")
    return out, notes


def add_to_cast(cast, ch: dict) -> list:
    new_cast = list(cast or [])
    figure = ch.get("figure")
    if figure and any(c.get("figure") == figure for c in new_cast):
        raise ValueError(f"SceneRef: two characters play figure {figure}. Give each one its own figure.")
    new_cast.append(ch)
    return new_cast


PICTURE_ROLES = {"face": "face", "body": "body", "outfit": "cloth", "cloth": "cloth"}


def input_picture(rel: str):
    """A picture the cast panel uploaded into ComfyUI/input (kept inside that folder)."""
    with Image.open(_input_file(rel)) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _input_file(rel: str) -> str:
    root = os.path.realpath(folder_paths.get_input_directory())
    path = os.path.realpath(os.path.join(root, rel))
    if not path.startswith(root + os.sep) or not os.path.isfile(path):
        raise ValueError(f"SceneRef: {rel!r} is not in ComfyUI/input.")
    return path


def input_voice(rel: str):
    """A voice recording the cast panel uploaded into ComfyUI/input: (clip, notes)."""
    from . import voice
    return voice.prepare(*voice.load_file(_input_file(rel)))


# @1, @2 ... in the action or the prompt: the cast's characters in order (the Studio's Cast
# list first, then its cast, cast_2 ... sockets), replaced by their names.
CAST_REF = re.compile(r"(?<![\w@])@(\d+)\b")


def resolve_refs(text: str, cast: list) -> str:
    def name(m):
        k = int(m.group(1))
        if not 1 <= k <= len(cast):
            who = ", ".join(f"@{i + 1} = {c['name']}" for i, c in enumerate(cast)) or "nobody"
            raise ValueError(f"SceneRef: the text says @{k}, but the cast has {len(cast)} "
                             f"character{'' if len(cast) == 1 else 's'} ({who}).")
        return cast[k - 1]["name"]
    return CAST_REF.sub(name, text or "")


def character_name(source: str, figure: int = 0, name: str = "") -> str:
    """The name make_character gives, without loading any pictures (for the Studio's
    'In the prompt' line)."""
    if name.strip():
        return name.strip()
    stored = ""
    try:
        if source.startswith(CHAR_PREFIX):
            path = _resolve("characters", source[len(CHAR_PREFIX):])
            with zipfile.ZipFile(path) as z:
                manifest = json.loads(z.read("manifest.json")) if "manifest.json" in z.namelist() else {}
            stored = clean_name(str(manifest.get("name") or os.path.splitext(os.path.basename(path))[0]))
        elif source.startswith(REFMOD_PREFIX):
            rel = source[len(REFMOD_PREFIX):]
            meta = _refmod_meta(_resolve("refmods", rel + ".safetensors"))
            stored = _refmod_name(meta, rel, _refmod_description(meta))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        pass
    return stored or (f"Figure {figure}" if figure else "the person")


def cast_names(raw) -> list:
    """For each row of a cast list (as cast_from_json reads it): its name, or None for a row
    the Run skips."""
    try:
        rows = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
    except ValueError:
        rows = []
    out = []
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict) or r.get("on") is False or (
                str(r.get("source") or NONE) == NONE and not any(
                    isinstance(p, dict) and p.get("file") for p in r.get("pictures") or [])):
            out.append(None)
            continue
        out.append(character_name(str(r.get("source") or NONE), int(r.get("figure") or 0), str(r.get("name") or "")))
    return out


def cast_from_json(raw) -> list:
    """The Studio panel's cast: [{source, figure, name, description, max_refs, pictures: [{file, role}]}]."""
    try:
        rows = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
    except ValueError:
        rows = []
    cast = []
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict) or r.get("on") is False:
            continue
        pics = [(input_picture(p["file"]), PICTURE_ROLES.get(p.get("role"), "face"))
                for p in r.get("pictures") or [] if isinstance(p, dict) and p.get("file")]
        source = str(r.get("source") or NONE)
        if source == NONE and not pics:
            continue  # an empty row
        ch = make_character(source, int(r.get("figure") or 0), str(r.get("name") or ""),
                            str(r.get("description") or ""), max(1, min(8, int(r.get("max_refs") or 3))),
                            pics, str(r.get("refmod_vae") or ""), bool(r.get("file_text")))
        # voice: "" = the file's own (if any), "off" = none, else an uploaded recording.
        v = str(r.get("voice") or "")
        if v == "off":
            ch["voice"] = None
        elif v:
            ch["voice"], notes = input_voice(v)
            ch["notes"] = ch["notes"] + notes
        cast = add_to_cast(cast, ch)
    return cast
