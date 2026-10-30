"""SceneRef Scene - a place from pictures, in place of a 3D scene.

One picture is the camera (the main view): the result keeps its angle, framing and what is
in it. Pictures of the same place from other angles go to MiniMax H3 movies right after it,
where the camera can move. Picture models only get them with angles_in_pictures on: FLUX.2
Klein took the camera of an angle instead of the main view's (tested on a courtyard, two
seeds; the same runs without angles kept the camera), whatever the prompt said and at any
reference size. Nobody is placed: there are no mannequins, so the action text says who is
where.

The scene comes out as the same 'shot' the Scene Reference Viewer gives (kind "pictures"),
so it plugs into the Studio and the Prompt Expander in its place. Its picture order is the
one every backend uses: image 1 = the main view, then the other angles, then the cast.

.scene files (SceneRef's own, in models/scenes): a zip with manifest.json first, the
pictures in pictures/ with their role (main / angle) and text/description.md. SceneRef
Scene Maker writes them, optionally with a vision model's description of the place.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
import uuid
import zipfile

import torch
from PIL import Image

from .characters import (MAX_IMAGE, MAX_MANIFEST, _list, _resolve, _roots, _slug, _to_pil, _to_tensor, cap_picture,
                         contact_sheet, grow_inputs, grown, input_picture)
from .nodes import SHOT_TYPE, _fit

PICTURES = "pictures"  # shot["kind"] of a scene made from pictures
NONE = "(pictures only)"
SCENE_PREFIX = "scene: "
MAGIC = "SCENEREFSCENE"
FORMAT_VERSION = 1
MAIN, ANGLE = "main", "angle"
# Other angles given to the model (each one costs reference tokens / H3 picture slots).
MAX_ANGLES = 4
H3_ANGLES = 3  # H3 takes 9 pictures in all: the main view, these, then the cast
MAX_FILE_PICTURES = 12

DESCRIBE_SYSTEM = ("You describe places precisely for artists who have to draw the same place again from any "
                   "angle. Plain sentences only. No people, no comments on the photos or their quality.")
DESCRIBE_PLACE = ("These pictures all show the same place from different angles. Describe the place: what kind of "
                  "place it is, its layout, the walls, floor and ceiling (materials and colours), windows and doors "
                  "and where they are, the furniture and where it stands, notable objects, and the light. At most "
                  "130 words.")


# ── .scene files ──────────────────────────────────────────────────────────────
def _register_folder():
    """models/scenes next to every models/characters (Comfy Desktop's shared models folder
    registers 'characters' but knows nothing of 'scenes'), then the install's own."""
    import folder_paths
    try:
        chars = folder_paths.get_folder_paths("characters")
    except Exception:
        chars = []
    for p in [os.path.join(os.path.dirname(os.path.normpath(c)), "scenes") for c in chars] + [
            os.path.join(folder_paths.models_dir, "scenes")]:
        folder_paths.add_model_folder_path("scenes", p)


_register_folder()


def list_sources() -> list[str]:
    return [NONE] + [SCENE_PREFIX + n for n in _list("scenes", ".scene")]


def scenes_folder() -> str:
    root = _roots("scenes")[0]
    os.makedirs(root, exist_ok=True)
    return root


def source_path(source: str) -> str | None:
    if not source or source == NONE:
        return None
    if not source.startswith(SCENE_PREFIX):
        raise ValueError(f"SceneRef Scene: unknown source {source!r}.")
    return _resolve("scenes", source[len(SCENE_PREFIX):])


def read_scene(path: str) -> dict:
    """{name, description, pictures: [(PIL, role)]} from a .scene file, main view first."""
    with zipfile.ZipFile(path) as z:
        def member(name, limit):
            info = z.getinfo(name)
            if info.file_size > limit:
                raise ValueError(f"SceneRef Scene: {name} in {os.path.basename(path)} is too big.")
            return z.read(name)

        manifest = json.loads(member("manifest.json", MAX_MANIFEST))
        if manifest.get("magic") != MAGIC:
            raise ValueError(f"SceneRef Scene: {os.path.basename(path)} is not a SceneRef .scene file.")
        names = set(z.namelist())
        pictures = []
        for p in manifest.get("pictures") or []:
            if isinstance(p, dict) and p.get("path") in names:
                with Image.open(io.BytesIO(member(p["path"], MAX_IMAGE))) as im:
                    pictures.append((im.convert("RGB"), MAIN if p.get("role") == MAIN else ANGLE))
        text = manifest.get("text") or {}
        desc = member(text["path"], MAX_MANIFEST).decode("utf-8", "replace") if text.get("path") in names else ""
    return {"name": str(manifest.get("name") or ""), "description": desc.strip(), "pictures": pictures}


def _png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def build_scene(name: str, description: str, pictures: list[tuple[Image.Image, str]]) -> bytes:
    """The .scene bytes. pictures: [(PIL, role)] with exactly one main view."""
    now = int(time.time())
    members, entries = {}, []
    for i, (im, role) in enumerate(pictures):
        data = _png(cap_picture(im.convert("RGB")))
        path = f"pictures/{i:03d}.png"
        members[path] = data
        entries.append({"path": path, "sha256": hashlib.sha256(data).hexdigest(), "width": im.width,
                        "height": im.height, "role": role})
    manifest = {"magic": MAGIC, "format_version": FORMAT_VERSION, "scene_id": str(uuid.uuid4()),
                "name": name.strip() or "Scene", "created_at": now, "modified_at": now, "app": "sceneref",
                "app_version": "2", "pictures": entries, "text": {}}
    text = description.strip().encode("utf-8")
    if text:
        members["text/description.md"] = text
        manifest["text"] = {"path": "text/description.md", "sha256": hashlib.sha256(text).hexdigest()}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=True, separators=(",", ":")))
        for path in sorted(members):
            z.writestr(path, members[path], compress_type=zipfile.ZIP_STORED if path.endswith(".png")
                       else zipfile.ZIP_DEFLATED)
    return buf.getvalue()


def describe_place(pictures: list[tuple[Image.Image, str]], encoder: str, seed: int = 0) -> str:
    """The place in words, from up to four of its pictures (main view first), by the
    expander's vision model, unloaded afterwards."""
    from .expander import _small
    from .vlm import pick_encoder, vlm_text
    images = [_small(_to_tensor(im)) for im, _role in pictures[:4]]
    return vlm_text([(DESCRIBE_PLACE, images)], DESCRIBE_SYSTEM, pick_encoder(encoder), 300, seed)[0].strip()


# ── gathering the pictures ────────────────────────────────────────────────────
def _rows(pictures_json) -> list[dict]:
    try:
        rows = json.loads(pictures_json or "[]")
    except ValueError:
        rows = []
    return [r for r in rows if isinstance(r, dict) and r.get("file")] if isinstance(rows, list) else []


def gather(source: str, pictures_json: str = "[]", main_view=None, other_angles=None) -> dict:
    """Every picture of the scene with exactly one main view: {name, file_text, pictures:
    [(PIL, role)] main first, from: where the main view came from}. The main view is the
    wired main_view, else the one marked main on the node, else the file's, else the first
    picture; all others are other angles (the file's, the node's, the wired ones)."""
    found = []  # (PIL, role, priority of a main, where)
    name, file_text = "", ""
    path = source_path(source)
    if path:
        sc = read_scene(path)
        name, file_text = sc["name"], sc["description"]
        found += [(im, role, 1, os.path.basename(path)) for im, role in sc["pictures"]]
    for r in _rows(pictures_json):
        found.append((input_picture(r["file"]), MAIN if r.get("role") == MAIN else ANGLE, 2, r["file"]))
    for im in main_view if main_view is not None else []:
        found.append((_to_pil(im), MAIN, 3, "main_view"))
    for im in other_angles if other_angles is not None else []:
        found.append((_to_pil(im), ANGLE, 0, "other_angles"))
    if not found:
        raise ValueError("SceneRef Scene: no pictures. Add some on the node (＋ picture), wire main_view, or pick "
                         "a .scene file.")
    mains = [i for i, f in enumerate(found) if f[1] == MAIN]
    main = max(mains, key=lambda i: found[i][2]) if mains else 0  # the first of the highest priority
    pictures = [(found[main][0], MAIN)] + [(f[0], ANGLE) for i, f in enumerate(found) if i != main]
    return {"name": name, "file_text": file_text, "pictures": pictures, "from": found[main][3]}


def output_size(width: int, height: int, megapixels: float) -> tuple[int, int]:
    """The main view shrunk (never enlarged) to `megapixels`, both sides multiples of 16.
    web/scene_ref_panels.js (sceneSize) works it out the same way."""
    s = min(1.0, (float(megapixels) * 1e6 / (width * height)) ** 0.5)
    return max(16, int(width * s) // 16 * 16), max(16, int(height * s) // 16 * 16)


def make_shot(pictures: list[tuple[Image.Image, str]], megapixels: float, place: str = "", name: str = "",
              angles_in_pictures: bool = False) -> dict:
    """The shot the Studio reads: the main view at its output size is both 'image' and
    'composite'; no depth, pose or figures; 'angles' = the other angles (at most MAX_ANGLES,
    capped like character pictures)."""
    main = pictures[0][0]
    size = output_size(main.width, main.height, megapixels)
    t = _to_tensor(_fit(main, size, Image.LANCZOS))
    angles = [_to_tensor(cap_picture(im)) for im, _role in pictures[1:1 + MAX_ANGLES]]
    w, h = size
    return {"kind": PICTURES, "image": t, "composite": t, "depth": None, "normal": None, "pose": None,
            "figure_masks": torch.zeros(1, h, w), "figure_silhouettes": torch.zeros(1, h, w), "figures": [[]],
            "figures_per_shot": [0], "cameras": [], "width": w, "height": h, "angles": angles,
            "place": " ".join(place.split()), "name": name, "angles_in_pictures": bool(angles_in_pictures)}


def place_text(description: str, file_text: str, use_file_text: bool) -> str:
    """The user's text, then the file's; each ends a sentence (they are read as one line)."""
    parts = [x.strip() for x in (description, file_text if use_file_text else "") if x and x.strip()]
    return "\n\n".join(x if x[-1] in ".!?" else x + "." for x in parts)


def report(scene: dict, shot: dict) -> str:
    extra = len(scene["pictures"]) - 1 - len(shot["angles"])
    used = ("movies and pictures" if shot["angles_in_pictures"] else "movies only (angles_in_pictures is off: "
            "picture models take the camera of an angle)")
    lines = [f"Main view (image 1): {scene['from']}, comes out at {shot['width']} × {shot['height']}.",
             f"Other angles: {len(shot['angles'])}" + (f" ({extra} more left out: at most {MAX_ANGLES} go to the "
                                                         f"model)" if extra > 0 else "")
             + (f", used for {used}" if shot["angles"] else "") + "."]
    if shot["place"]:
        lines.append(f"The place: {shot['place']}")
    lines.append("Nobody is placed: say in the action who is where.")
    return "\n".join(lines)


# ── for the Studio and the expander ──────────────────────────────────────────
def is_pictures(shot) -> bool:
    return isinstance(shot, dict) and shot.get("kind") == PICTURES


def angles_of(shot, movie: bool = False) -> list:
    """The other angles that go to the model, right after image 1 (fewer for H3); none for a
    picture model unless the Scene's angles_in_pictures is on."""
    if not is_pictures(shot) or not (movie or shot.get("angles_in_pictures")):
        return []
    return list(shot.get("angles") or [])[:H3_ANGLES if movie else MAX_ANGLES]


def plain_cast(shot, cast: list) -> list:
    """With a picture scene nobody plays a figure: there are none to play, and a figure number
    would leave the character out of the prompt's people."""
    return [dict(ch, figure=0) for ch in cast] if is_pictures(shot) else cast


def _input_name(rel) -> str:
    """A Load Image widget's value as a file in ComfyUI/input ('x.png' or 'x.png [input]')."""
    rel = str(rel or "").strip()
    if rel.endswith(" [input]"):
        rel = rel[:-len(" [input]")]
    if not rel or rel.endswith("]"):
        raise ValueError(f"SceneRef Scene: {rel!r} is not a picture in ComfyUI/input.")
    return rel


def shot_from_request(d) -> dict:
    """The Studio's 'Write the prompt' with a SceneRef Scene wired: the node's settings as the
    panel sends them, and the files of Load Image nodes wired to main_view / other_angles."""
    if not isinstance(d, dict):
        raise ValueError("bad scene request")
    main = [input_picture(_input_name(d["main_file"]))] if d.get("main_file") else None
    angles = [input_picture(_input_name(f)) for f in d.get("angle_files") or []]
    scene = gather(str(d.get("source") or NONE), str(d.get("pictures_json") or "[]"),
                   [_to_tensor(im)[0] for im in main] if main else None,
                   [_to_tensor(im)[0] for im in angles] or None)
    place = place_text(str(d.get("description") or ""), scene["file_text"], bool(d.get("use_file_text", True)))
    try:
        mp = float(d.get("max_megapixels", 1.0))
    except (TypeError, ValueError):
        mp = 1.0
    return make_shot(scene["pictures"], mp, place, scene["name"], bool(d.get("angles_in_pictures")))


# ── nodes ─────────────────────────────────────────────────────────────────────
def _picture_inputs(main_tip: str) -> dict:
    angles = ("IMAGE", {"tooltip": "Optional: the same place from other angles (one picture or a batch), for "
                        "movies (and pictures with angles_in_pictures on). A new input appears when this one is "
                        "wired, so pictures of different sizes each get their own."})
    return {"main_view": ("IMAGE", {"tooltip": main_tip}), "other_angles": angles, **grow_inputs("other_angles", angles)}


def _angles(other_angles, more: dict) -> list | None:
    """The pictures wired to other_angles, other_angles_2 ... as one list (sizes may differ)."""
    return [im for batch in grown("other_angles", other_angles, more) for im in batch] or None


class SceneRefScene:
    DESCRIPTION = (
        "A scene from pictures instead of a 3D file: one main view (the camera: the result keeps its angle, "
        "framing, furniture and objects) and optionally the same place from other angles, which MiniMax H3 "
        "movies get as extra references (picture models only with angles_in_pictures on). Plug 'shot' into the Studio (or the Prompt Expander) instead of the Scene "
        "Reference Viewer. There are no mannequins: the action says who is where."
    )
    CATEGORY = "SceneRef"
    FUNCTION = "run"
    RETURN_TYPES = (SHOT_TYPE, "IMAGE", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("shot", "image", "angles", "width", "height", "description")
    OUTPUT_TOOLTIPS = (
        "The scene for the SceneRef Studio's (or the Prompt Expander's) 'shot' input.",
        "The main view as the model gets it, at width x height.",
        "The other angles side by side (a preview).",
        "Picture width (the main view shrunk to max_megapixels).",
        "Picture height.",
        "The description of the place that goes to the prompt writer.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (list_sources(), {"tooltip": "A .scene file from models/scenes (made with SceneRef Scene "
                           "Maker), or pictures only (add them on the node or wire them)."}),
                "description": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional: the place in a "
                                "few words ('a small Paris kitchen, morning'). Goes to the prompt writer (✎), "
                                "with the .scene's own text if use_file_text is on."}),
                "use_file_text": ("BOOLEAN", {"default": True, "tooltip": "Also give the prompt writer the "
                                  "description stored in the .scene file."}),
                "max_megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "tooltip":
                                   "The main view is shrunk to at most this size (never enlarged), sides on a 16 "
                                   "grid. It is the picture size when the Studio's picture_size is 'as framed'."}),
                "pictures_json": ("STRING", {"default": "[]", "multiline": True, "tooltip": "Edited with the "
                                  "picture list on the node."}),
                "angles_in_pictures": ("BOOLEAN", {"default": False, "tooltip": "Also give the other angles to the "
                                       "picture models (Klein, FLUX.3, Qwen). Off: only MiniMax H3 movies get them. "
                                       "Klein tends to take the camera of an angle instead of the main view's."}),
            },
            "optional": _picture_inputs("Optional: the main view (the camera). Wins over a main view marked on "
                                        "the node or stored in the .scene file."),
        }

    @classmethod
    def IS_CHANGED(cls, source, **kwargs):
        try:
            p = source_path(source)
            if not p:
                return ""
            st = os.stat(p)
            return f"{st.st_mtime_ns}:{st.st_size}"
        except Exception:
            return ""

    def run(self, source, description, use_file_text, max_megapixels, pictures_json="[]", angles_in_pictures=False,
            main_view=None, other_angles=None, **more):
        scene = gather(source, pictures_json, main_view, _angles(other_angles, more))
        place = place_text(description, scene["file_text"], use_file_text)
        shot = make_shot(scene["pictures"], max_megapixels, place, scene["name"], angles_in_pictures)
        sheet = contact_sheet([(_to_pil(t[0]), ANGLE) for t in shot["angles"]])
        return {"ui": {"sr_size": [shot["width"], shot["height"]], "text": [report(scene, shot)]},
                "result": (shot, shot["image"], sheet, shot["width"], shot["height"], place)}


class SceneRefSceneMaker:
    DESCRIPTION = (
        "Makes a .scene file from pictures of a place: one main view (the camera) and the same place from "
        "other angles, with a name and a description. Optionally a vision model describes the place from the "
        "pictures. The file lands in models/scenes, ready for the SceneRef Scene node."
    )
    CATEGORY = "SceneRef"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("file", "description", "pictures")
    OUTPUT_TOOLTIPS = ("The .scene file written (relative to models/scenes).",
                       "The description stored in the file.", "The pictures stored, main view first.")

    @classmethod
    def INPUT_TYPES(cls):
        from .vlm import vlm_files
        return {
            "required": {
                "name": ("STRING", {"default": "", "tooltip": "The scene's name, also the file name."}),
                "description": ("STRING", {"multiline": True, "default": "", "tooltip": "Your own text about the "
                                "place. Stored in the file."}),
                "describe_with_ai": ("BOOLEAN", {"default": False, "label_on": "describe the place with AI",
                                     "label_off": "only my text", "tooltip": "A vision model (Qwen3-VL 8B) looks "
                                     "at the pictures and describes the place (layout, materials, furniture, "
                                     "light), added after your text. Unloaded afterwards."}),
                "source": (list_sources(), {"tooltip": "Optional: start from another .scene (its pictures and "
                           "text are copied), or pictures only."}),
                "overwrite": ("BOOLEAN", {"default": False, "tooltip": "Replace a .scene with the same name. "
                              "Off = a new file name (_2, _3 ...)."}),
                "describe_model": (["auto"] + vlm_files(), {"default": "auto", "tooltip": "With "
                                   "describe_with_ai: the vision language model (models/text_encoders or "
                                   "models/LLM)."}),
                "pictures_json": ("STRING", {"default": "[]", "multiline": True, "tooltip": "Edited with the "
                                  "picture list on the node."}),
            },
            "optional": _picture_inputs("Optional: the main view (the camera)."),
        }

    def run(self, name, description, describe_with_ai, source, overwrite, describe_model, pictures_json="[]",
            main_view=None, other_angles=None, **more):
        scene = gather(source, pictures_json, main_view, _angles(other_angles, more))
        pictures = scene["pictures"][:MAX_FILE_PICTURES]
        name = name.strip() or scene["name"]
        if not name:
            raise ValueError("SceneRef Scene Maker: give the scene a name.")
        text = description.strip() or scene["file_text"]
        if describe_with_ai:
            ai = describe_place(pictures, describe_model)
            text = f"{text}\n\n{ai}".strip() if ai else text
        data = build_scene(name, text, pictures)
        folder = scenes_folder()
        base = _slug(name)
        path = os.path.join(folder, base + ".scene")
        n = 2
        while os.path.exists(path) and not overwrite:
            path = os.path.join(folder, f"{base}_{n}.scene")
            n += 1
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        rel = os.path.basename(path)
        left = len(scene["pictures"]) - len(pictures)
        summary = (f"Saved {rel}: the main view ({scene['from']}) and {len(pictures) - 1} other angle"
                   f"{'' if len(pictures) == 2 else 's'}" + (f" ({left} left out: at most {MAX_FILE_PICTURES} "
                                                             f"pictures)" if left else "")
                   + f". In the SceneRef Scene node it is '{SCENE_PREFIX}{rel}'.\n\n{text}")
        return {"ui": {"text": [summary]}, "result": (rel, text, contact_sheet(pictures))}
