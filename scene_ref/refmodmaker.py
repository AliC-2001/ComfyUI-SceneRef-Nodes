"""SceneRef RefMod Maker - writes DaSiWa RefMods for MiniMax H3.

A RefMod (DaSiWa's format, read by its H3 Director and by SceneRef) is one .safetensors
file in models/refmods: each member is already a MiniMax H3 latent, so H3 skips encoding
the pictures on every run. Layout as DaSiWa's v5 bundle writes it (learned from its files
and its documentation; no code taken from it): tensors ref_0 ... ref_{n-1}, image members
[1, 24, 1, H/16, W/16] from the H3 video VAE, an audio member [1, 32, 2, T] from the H3
audio VAE, and a JSON header "refmod_meta" describing every member.

What goes in: the character's pictures after the picture check (characters.refine): best
first, face : body : outfit 2 : 1 : 1, small faces cropped to the face, which DaSiWa does
not do; sized as H3's ref_image_size "max" (short edge up to 2048 on a 32 grid, never
enlarged). Optionally a voice: cleaned up (scene_ref.voice) and cut to 15 s, DaSiWa's
limit for an audio member.
"""
from __future__ import annotations

import json
import os

import torch
from PIL import Image

import folder_paths

from .characters import (CHAR_PREFIX, NONE, PICTURE_ROLES, REFMOD_PREFIX, _resolve, _roots, _slug,
                         _to_pil, _to_tensor, cap_picture, character_report, contact_sheet, default_refmod_vae,
                         input_picture, list_sources, read_char, read_refmod, refine)

FORMAT_VERSION = 5
MEMBER_VERSION = 4
GRID = 32
SHORT_EDGE = 2048
MAX_AUDIO_SECONDS = 15.0
MAX_PICTURES = 8  # H3 takes 9 pictures, and the scene is one of them


def h3_size(w: int, h: int) -> tuple[int, int]:
    """MiniMaxH3ReferenceToVideo's size for ref_image_size "max": the short edge scaled down
    (never up) to 2048, each side rounded to the 32 grid."""
    s = min(1.0, SHORT_EDGE / min(w, h))
    return max(GRID, round(w * s / GRID) * GRID), max(GRID, round(h * s / GRID) * GRID)


def _vae(name: str):
    import comfy.sd
    import comfy.utils
    path = folder_paths.get_full_path("vae", name) if name else None
    if not path:
        raise ValueError(f"SceneRef RefMod Maker: VAE {name!r} not found in models/vae.")
    sd, md = comfy.utils.load_torch_file(path, return_metadata=True)
    return comfy.sd.VAE(sd=sd, metadata=md)


@torch.inference_mode()
def build_refmod(name: str, description: str, pictures: list[tuple[Image.Image, str]], video_vae: str,
                 clip=None, audio_vae: str = "", notes=()) -> tuple[dict, dict]:
    """(tensors, refmod_meta) for a bundle. clip = a cleaned-up voice (scene_ref.voice)."""
    import comfy.model_management as mm
    from . import voice
    tensors, members = {}, []
    desc = " ".join(description.split())
    vae = _vae(video_vae)
    try:
        for i, (im, role) in enumerate(pictures):
            w, h = h3_size(im.width, im.height)
            pic = im if (w, h) == im.size else im.resize((w, h), Image.LANCZOS)
            lat = vae.encode(_to_tensor(pic)).float().cpu()
            if lat.ndim == 4:
                lat = lat[:, :, None]
            tensors[f"ref_{i}"] = lat.to(torch.float16).contiguous()
            members.append({"kind": "image", "_format_version": MEMBER_VERSION, "mode": "encode", "latent_t": 1,
                            "latent_h": int(lat.shape[3]), "latent_w": int(lat.shape[4]),
                            "source": f"picture {i + 1} ({role})", "source_item_id": f"sceneref-{i}",
                            "source_enabled": True, "description": desc, "source_media_mode": "video",
                            "trim_start": 0.0, "trim_end": None, "source_role": "subject", "source_audio": None})
    finally:
        del vae
        mm.soft_empty_cache()
    if clip is not None:
        clip = clip[:int(MAX_AUDIO_SECONDS * voice.VOICE_SR)]
        avae = _vae(audio_vae)
        try:
            rate = getattr(avae, "audio_sample_rate", voice.VOICE_SR)
            wav = clip.reshape(1, 1, -1)
            if rate != voice.VOICE_SR:
                import comfy.audio
                wav = comfy.audio.resample(wav, voice.VOICE_SR, rate)
            lat = avae.encode(wav.movedim(1, -1)).float().cpu()
        finally:
            del avae
            mm.soft_empty_cache()
        i = len(members)
        tensors[f"ref_{i}"] = lat.to(torch.float16).contiguous()
        members.append({"kind": "audio", "_format_version": MEMBER_VERSION, "mode": "encode",
                        "latent_t": int(lat.shape[-1]), "sample_rate": voice.VOICE_SR, "source": "voice",
                        "source_item_id": f"sceneref-{i}", "source_enabled": True,
                        "description": f"{name}'s voice", "source_media_mode": "audio", "trim_start": 0.0,
                        "trim_end": round(voice.seconds(clip), 3), "source_role": "subject", "source_audio": None})
    meta = {"_format_version": FORMAT_VERSION, "kind": "bundle", "name": name, "members": members,
            "description": desc, "concept_type": "identity", "dasiwa_warnings": [],
            "sceneref_notes": list(notes)}
    return tensors, meta


def refmods_folder() -> str:
    root = _roots("refmods")[0]
    os.makedirs(root, exist_ok=True)
    return root


class SceneRefRefModMaker:
    DESCRIPTION = (
        "Makes a DaSiWa RefMod (models/refmods) for MiniMax H3 from pictures (face, body, outfit) or from a "
        ".char / RefMod, with a description and optionally a voice. The pictures are checked and picked as "
        "for the Studio (best first, small faces cropped to the face) and stored as H3 latents, so H3 movies "
        "start faster. Read by the Studio's Cast list and by DaSiWa's H3 Director."
    )
    CATEGORY = "SceneRef"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("file", "pictures")
    OUTPUT_TOOLTIPS = ("The RefMod written (its name in the Cast list).", "The pictures stored, side by side.")

    @classmethod
    def INPUT_TYPES(cls):
        vaes = folder_paths.get_filename_list("vae")
        return {
            "required": {
                "name": ("STRING", {"default": "", "tooltip": "The character's name, also the file name."}),
                "description": ("STRING", {"multiline": True, "default": "", "tooltip": "Stored with every "
                                "picture: name first, then looks the pictures don't show ('Emma, a woman of about "
                                "40, wavy brown hair, glasses')."}),
                "source": (list_sources(), {"tooltip": "Optional: start from a .char or another RefMod (its "
                           "pictures, text and voice), or pictures only."}),
                "max_pictures": ("INT", {"default": 4, "min": 1, "max": MAX_PICTURES, "tooltip": "At most this "
                                 "many pictures (face : body : outfit = 2 : 1 : 1). H3 takes 9 pictures in all, "
                                 "the scene included, shared by everyone in the movie."}),
                "overwrite": ("BOOLEAN", {"default": False, "tooltip": "Replace a RefMod with the same name. "
                              "Off = a new file name (_2, _3 ...)."}),
                "h3_vae": (vaes, {"default": default_refmod_vae("video"), "tooltip": "The MiniMax H3 video VAE "
                           "(minimax_h3_video_vae): encodes the pictures."}),
                "h3_audio_vae": (vaes, {"default": default_refmod_vae("audio") or (vaes[0] if vaes else ""),
                                 "tooltip": "The MiniMax H3 audio VAE (minimax_h3_audio_vae): encodes the voice."}),
                "pictures_json": ("STRING", {"default": "[]", "multiline": True, "tooltip": "Edited with the "
                                  "picture list on the node."}),
            },
            "optional": {
                "face": ("IMAGE", {"tooltip": "Face picture(s)."}),
                "body": ("IMAGE", {"tooltip": "Full body picture(s)."}),
                "outfit": ("IMAGE", {"tooltip": "Outfit picture(s)."}),
                "voice": ("AUDIO", {"tooltip": "Optional: a recording of this person speaking (5 to 15 s of clear "
                          "speech, no music). Stored as an audio member (cut to 15 s)."}),
            },
        }

    def run(self, name, description, source, max_pictures, overwrite, h3_vae, h3_audio_vae, pictures_json="[]",
            face=None, body=None, outfit=None, voice=None):
        from safetensors.torch import save_file
        from . import voice as voices
        pictures = []
        try:
            rows = json.loads(pictures_json or "[]")
        except ValueError:
            rows = []
        for r in rows if isinstance(rows, list) else []:
            if isinstance(r, dict) and r.get("file"):
                pictures.append((cap_picture(input_picture(r["file"])), PICTURE_ROLES.get(r.get("role"), "face")))
        for batch, role in ((face, "face"), (body, "body"), (outfit, "cloth")):
            if batch is not None:
                pictures.extend((cap_picture(_to_pil(img)), role) for img in batch)
        text = description.strip()
        clip = None
        if voice is not None:
            clip, _n = voices.prepare(voice["waveform"], voice["sample_rate"])
        if source != NONE:
            if source.startswith(CHAR_PREFIX):
                ch = read_char(_resolve("characters", source[len(CHAR_PREFIX):]))
            else:
                ch = read_refmod(source[len(REFMOD_PREFIX):], h3_vae)
            pictures = ch["refs"] + pictures
            name = name.strip() or ch["name"]
            text = text or ch.get("description", "")
            if clip is None:
                clip = ch.get("voice")
        if not pictures:
            raise ValueError("SceneRef RefMod Maker: add pictures (＋ picture on the node, or wire face / body / "
                             "outfit), or pick a .char / RefMod as the source.")
        if not name.strip():
            raise ValueError("SceneRef RefMod Maker: give the character a name.")
        used, notes = refine(pictures, max_pictures)
        if clip is not None and voices.seconds(clip) > MAX_AUDIO_SECONDS:
            notes.append(f"voice cut to {MAX_AUDIO_SECONDS:.0f} s (DaSiWa's limit for an audio member)")
        tensors, meta = build_refmod(name.strip(), text, used, h3_vae, clip, h3_audio_vae, notes)
        folder = refmods_folder()
        base = _slug(name)
        path = os.path.join(folder, base + ".safetensors")
        n = 2
        while os.path.exists(path) and not overwrite:
            path = os.path.join(folder, f"{base}_{n}.safetensors")
            n += 1
        tmp = path + ".part"
        save_file(tensors, tmp, metadata={"refmod_meta": json.dumps(meta, ensure_ascii=False)})
        os.replace(tmp, path)
        rel = os.path.splitext(os.path.basename(path))[0]
        report = character_report({"name": name.strip(), "refs": used, "notes": notes, "voice": clip})
        sizes = ", ".join(f"{m['latent_w'] * 16}x{m['latent_h'] * 16}" for m in meta["members"] if m["kind"] == "image")
        summary = (f"Saved {rel}.safetensors in {folder}: H3 latents at {sizes}"
                   + (f" + voice {min(voices.seconds(clip), MAX_AUDIO_SECONDS):.1f} s" if clip is not None else "")
                   + f". In the Cast list it is '{REFMOD_PREFIX}{rel}'.\n\n{report}")
        return {"ui": {"text": [summary]}, "result": (rel, contact_sheet(used))}

