"""MOSS-Image helpers: the colour region map and which pictures go in as references.

MOSS-Image's colour/depth control adapter (ComfyUI's MOSSImageControl) takes a depth map
(near = white) and a colour map: one flat colour per region, the colours only region ids,
black = unspecified. The viewer renders which object owns each pixel (its 'regions' pass:
24-bit ids in RGB, 0 = no surface); colour_map() turns that into the adapter's map.

The edit LoRA takes at most four reference pictures (ComfyUI's MOSS-Image Edit template and
the LoRA's own card); ref_plan() decides which, the same way for the prompt and the Run, so
"Image 2" means the same picture in both. Only a scene from pictures uses it (the place is
image 1, the edit LoRA's own job). In a 3D scene the people get no references: tested on a
16 GB card, the control adapter's weights alone (no map, strength 0) wipe out the likeness
the references give, and without control two references blend into one look (both people
got one woman's bob and glasses). The control keeps the bodies, the text names them and the
face pass gives each one their face.
"""
from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image

MAX_REFS = 4
# All references together, in megapixels. ComfyUI's edit template uses one 1 MP reference; MOSS keeps
# every reference's keys and values for all 42 blocks, on the positive and the negative (about 2 GB
# per conditioning per MP), so several pictures share that one megapixel instead of 1 MP each, which
# on a 16 GB card pushed the model out of VRAM (about 60 s per step) or ran out of memory.
REF_MEGAPIXELS = 1.0
# A scene from pictures: the place keeps this share of it (the people share the rest). With an
# even split the edit moved the camera in; with the place at 3/4 it kept the photo's framing.
PLACE_SHARE = 0.75
NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"  # MOSS-Image's user guide
SKIP_LAYER = "22"  # the user guide's skip-layer guidance layer (negative branch)
EDIT_FORMAT = "Moss-Image-Edit"  # the layout the official edit LoRA was trained with
# Regions smaller than this share of the picture are left black ('unspecified'): a speck of
# its own colour says nothing useful and only adds noise to the map.
MIN_REGION = 0.0004


def decode_regions(rgb: np.ndarray) -> np.ndarray:
    """The viewer's regions PNG (H x W x 3, uint8) -> region ids (int32, 0 = nothing)."""
    rgb = rgb.astype(np.int32)
    return rgb[..., 0] | (rgb[..., 1] << 8) | (rgb[..., 2] << 16)


def load_regions(path: str, size=None) -> np.ndarray:
    """Region ids from a regions PNG, resized (nearest: ids are labels, never blended) and
    centre-cropped to `size` like the other passes."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size and im.size != tuple(size):
            w, h = im.size
            want = size[0] / size[1]
            if abs(w / h - want) > 0.01:
                if w / h > want:
                    cw = round(h * want)
                    im = im.crop(((w - cw) // 2, 0, (w - cw) // 2 + cw, h))
                else:
                    ch = round(w / want)
                    im = im.crop((0, (h - ch) // 2, w, (h - ch) // 2 + ch))
            im = im.resize(tuple(size), Image.NEAREST)
        return decode_regions(np.asarray(im))


def palette(n: int) -> np.ndarray:
    """n clearly different, never dark colours (n x 3, 0..1). Hues step by the golden ratio,
    so neighbours in the list are far apart; saturation and value alternate a little."""
    out = np.zeros((n, 3), np.float32)
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        s = (0.95, 0.65, 0.8)[i % 3]
        v = (1.0, 0.8, 0.9, 0.7)[(i // 3) % 4]
        out[i] = colorsys.hsv_to_rgb(h, s, v)
    return out


def colour_map(ids: np.ndarray, min_share: float = MIN_REGION) -> np.ndarray:
    """Region ids (H x W) -> the control adapter's colour map (H x W x 3 float, 0..1): one
    flat colour per region, the biggest regions first in the palette, background and specks
    black. Same ids in, same colours out."""
    out = np.zeros(ids.shape + (3,), np.float32)
    labels, counts = np.unique(ids, return_counts=True)
    keep = [(c, lab) for lab, c in zip(labels.tolist(), counts.tolist())
            if lab != 0 and c >= min_share * ids.size]
    keep.sort(key=lambda x: (-x[0], x[1]))
    colours = palette(len(keep))
    lut = np.zeros((int(labels.max()) + 1 if labels.size else 1, 3), np.float32)
    for k, (_c, lab) in enumerate(keep):
        lut[lab] = colours[k]
    return lut[ids] if keep else out


def ref_plan(cast: list, angles: int = 0, photo: bool = False, slots: int = MAX_REFS):
    """Which reference pictures MOSS gets, in order: for a scene from pictures the place
    (image 1) and its other angles, then each character's first pictures, shared out evenly.
    Returns (scene_count, [(character, [picture index in ch['refs'], ...], first image number)]).
    A 3D scene gets none (see the module notes)."""
    if not photo:
        return 0, []
    scene = 1 + min(angles, slots - 1)
    people = [c for c in cast if c.get("refs")]
    free = slots - scene
    per = max(1, free // max(1, len(people))) if people else 0
    plan, n = [], scene + 1
    for ch in people:
        take = min(per, len(ch["refs"]), slots - (n - 1))
        if take <= 0:
            break
        plan.append((ch, list(range(take)), n))
        n += take
    return scene, plan
