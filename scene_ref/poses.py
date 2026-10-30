"""Poses for the viewer's mannequins, from several sources, as one kind of data: 3D points.

Every source ends up as the same "pose points": figure-local positions of the joints
(+Y up, +Z where the body faces, +X the figure's own left; centimetres, pelvis at the
origin). The viewer fits its mannequin to them (web/scene_ref_figures.mjs fitPose), so
only directions matter, never the source's proportions.

Sources:
  * the bundled library (data/poses_cmu.json): frames of the CMU Graphics Lab Motion
    Capture Database (mocap.cs.cmu.edu), from the cgspeed BVH conversion; made by
    tools/make_pose_library.py;
  * Daz Studio pose presets (.duf, preset_pose): read directly, with the figure's own
    skeleton from the content library (forward kinematics as Daz does it, checked
    against Daz Studio itself);
  * OpenPose keypoints (.json: pose_keypoints_2d, COCO-18 or BODY-25): 2D, lifted to 3D
    from bone lengths (the depth of each bone is a guess: knees forward, hands in front);
  * BVH files (.bvh): a few distinct frames each;
  * your own saved poses (.srpose.json): joint angles from the viewer, used as they are.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIBRARY = os.path.join(HERE, "data", "poses_cmu.json")
POSE_EXTS = (".srpose.json", ".json", ".duf", ".bvh")

# ── shared ────────────────────────────────────────────────────────────────────────
KEYS = ("pelvis", "spine", "chest", "neck", "head", "head_top", "nose", "l_eye", "r_eye",
        "l_shoulder", "l_elbow", "l_wrist", "l_hand", "r_shoulder", "r_elbow", "r_wrist", "r_hand",
        "l_hip", "l_knee", "l_ankle", "l_toe", "r_hip", "r_knee", "r_ankle", "r_toe")


def normalise(points: dict) -> dict:
    """Pelvis to the origin, turned about the vertical so the hips' left-right line is +X
    (so the body faces +Z). Values rounded to millimetres."""
    pts = {k: np.asarray(v, dtype=float) for k, v in points.items() if v is not None}
    if "l_hip" in pts and "r_hip" in pts:
        pelvis = pts.get("pelvis", (pts["l_hip"] + pts["r_hip"]) / 2)
        side = pts["l_hip"] - pts["r_hip"]
    elif "l_shoulder" in pts and "r_shoulder" in pts:
        pelvis = pts.get("pelvis", (pts["l_shoulder"] + pts["r_shoulder"]) / 2)
        side = pts["l_shoulder"] - pts["r_shoulder"]
    else:
        raise ValueError("a pose needs both hips or both shoulders")
    pts.setdefault("pelvis", pelvis)
    # Turn about +Y so the hip line lands on +X: x' = c x + s z = |side|, z' = c z - s x = 0.
    a = math.atan2(side[2], side[0])
    c, s = math.cos(a), math.sin(a)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    out = {}
    for k, v in pts.items():
        if k in KEYS:
            out[k] = [round(float(x), 1) for x in rot @ (v - pelvis)]
    return out


def anchor_for(points: dict) -> str:
    """'seat' for sitting on something (pelvis well above the lowest point, thighs near
    level), else 'lowest' (standing, kneeling, sitting on the floor, lying)."""
    p = {k: np.asarray(v) for k, v in points.items()}
    low = min(v[1] for v in p.values())
    try:
        shin = np.linalg.norm(p["l_ankle"] - p["l_knee"])
        leg = np.linalg.norm(p["l_knee"] - p["l_hip"]) + shin
        up = p["pelvis"][1] - low
        thigh = (p["l_knee"] - p["l_hip"]) + (p["r_knee"] - p["r_hip"])
        level = abs(thigh[1]) < 0.6 * np.linalg.norm(thigh)
        # On a chair the knees are well off the ground; kneeling or sitting on the heels puts them on it.
        knees_up = min(p["l_knee"][1], p["r_knee"][1]) - low > 0.6 * shin
        # ...and the feet are out in front (facing +Z); in a squat they are under the body.
        feet_out = ((p["l_ankle"] + p["r_ankle"]) / 2 - p["pelvis"])[2] > 0.35 * np.linalg.norm(p["l_knee"] - p["l_hip"])
    except KeyError:
        return "lowest"
    return "seat" if level and knees_up and feet_out and 0.3 * leg < up < 0.85 * leg else "lowest"


def _load_json(path: str):
    with open(path, "rb") as f:
        b = f.read()
    if b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b)
    return json.loads(b.decode("utf-8-sig"))


# ── bundled library ───────────────────────────────────────────────────────────────
_library = None


def library() -> list[dict]:
    global _library
    if _library is None:
        try:
            _library = _load_json(LIBRARY)["poses"]
        except FileNotFoundError:
            _library = []
    return _library


# ── Daz pose presets ──────────────────────────────────────────────────────────────
# Daz bone names per pose key: Genesis 3 / 8 first, then Genesis 2, then Genesis 9.
DAZ_BONES = {
    "pelvis": ["pelvis"], "spine": ["abdomenUpper", "abdomen2", "spine2"], "chest": ["chestUpper", "chest", "spine4"],
    "neck": ["neckLower", "neck", "neck1"], "head": ["head"],
    "l_shoulder": ["lShldrBend", "lShldr", "l_upperarm"], "r_shoulder": ["rShldrBend", "rShldr", "r_upperarm"],
    "l_elbow": ["lForearmBend", "lForeArm", "l_forearm"], "r_elbow": ["rForearmBend", "rForeArm", "r_forearm"],
    "l_wrist": ["lHand", "l_hand"], "r_wrist": ["rHand", "r_hand"],
    "l_hip": ["lThighBend", "lThigh", "l_thigh"], "r_hip": ["rThighBend", "rThigh", "r_thigh"],
    "l_knee": ["lShin", "l_shin"], "r_knee": ["rShin", "r_shin"],
    "l_ankle": ["lFoot", "l_foot"], "r_ankle": ["rFoot", "r_foot"],
    "l_toe": ["lToe", "l_toes"], "r_toe": ["rToe", "r_toes"],
    "l_eye": ["lEye", "l_eye"], "r_eye": ["rEye", "r_eye"],
}
# Figure skeletons, by what the pose's folder says, relative to a content library.
SKELETONS = [
    (re.compile(r"genesis\s*9|\bg9", re.I), "data/DAZ 3D/Genesis 9/Base/Genesis9.dsf"),
    (re.compile(r"genesis\s*8(\.|_)1\s*male|\bg8\.?1m", re.I), "data/DAZ 3D/Genesis 8/Male 8_1/Genesis8_1Male.dsf"),
    (re.compile(r"genesis\s*8(\.|_)1|\bg8\.?1", re.I), "data/DAZ 3D/Genesis 8/Female 8_1/Genesis8_1Female.dsf"),
    (re.compile(r"genesis\s*8\s*male|\bg8m", re.I), "data/DAZ 3D/Genesis 8/Male/Genesis8Male.dsf"),
    (re.compile(r"genesis\s*8|\bg8", re.I), "data/DAZ 3D/Genesis 8/Female/Genesis8Female.dsf"),
    (re.compile(r"genesis\s*3\s*male|\bg3m", re.I), "data/DAZ 3D/Genesis 3/Male/Genesis3Male.dsf"),
    (re.compile(r"genesis\s*3|\bg3", re.I), "data/DAZ 3D/Genesis 3/Female/Genesis3Female.dsf"),
    (re.compile(r"genesis\s*2\s*male|\bg2m", re.I), "data/DAZ 3D/Genesis 2/Male/Genesis2Male.dsf"),
    (re.compile(r"genesis\s*2|\bg2", re.I), "data/DAZ 3D/Genesis 2/Female/Genesis2Female.dsf"),
]
DEFAULT_SKELETON = "data/DAZ 3D/Genesis 8/Female/Genesis8Female.dsf"


def content_root(path: str) -> str | None:
    """The Daz content library a file lives in: the nearest folder holding data/DAZ 3D."""
    d = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.isdir(os.path.join(d, "data", "DAZ 3D")):
            return d
        up = os.path.dirname(d)
        if up == d:
            return None
        d = up


def content_roots(near: str | None) -> list[str]:
    """Libraries to look for skeletons in: the file's own, then libraries beside it."""
    roots = []
    own = content_root(near) if near else None
    if own:
        roots.append(own)
        parent = os.path.dirname(own)
        try:
            for name in sorted(os.listdir(parent)):
                d = os.path.join(parent, name)
                if d != own and os.path.isdir(os.path.join(d, "data", "DAZ 3D")):
                    roots.append(d)
        except OSError:
            pass
    return roots


_skeletons: dict[str, dict] = {}


def _rot(axis: str, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _euler(order: str, ang: dict) -> np.ndarray:
    # Daz's rotation order "YZX" means Y is applied first to a vector: R = Rx Rz Ry.
    m = np.eye(3)
    for ax in order.lower()[::-1]:
        m = m @ _rot(ax, ang.get(ax, 0.0))
    return m


def skeleton(dsf: str) -> dict:
    """Bones of a Genesis figure by name: parent, centre, end, orientation, rotation order."""
    if dsf in _skeletons:
        return _skeletons[dsf]
    data = _load_json(dsf)
    nodes = {}
    for n in data["node_library"]:
        val = lambda k: [float(c["value"]) for c in n[k]] if k in n else [0.0, 0.0, 0.0]  # noqa: E731
        nodes[n.get("name", n["id"])] = {"id": n["id"], "parent": str(n.get("parent", ""))[1:],
                                         "c": np.array(val("center_point")), "e": np.array(val("end_point")),
                                         "o": val("orientation"), "ro": n.get("rotation_order", "XYZ")}
    by_id = {v["id"]: k for k, v in nodes.items()}
    for v in nodes.values():
        v["parent"] = by_id.get(v["parent"])
    _skeletons[dsf] = nodes
    return nodes


def find_skeleton(pose_path: str, bones: set[str], named: str | None = None) -> str:
    """The .dsf of the figure a pose was made for: the one the preset names, else from the
    folder names and the bone names (Genesis 9 uses l_thigh...), else Genesis 8 Female."""
    rel = None
    if named:
        rel = named
    elif any(b.startswith(("l_", "r_")) for b in bones):
        rel = SKELETONS[0][1]
    else:
        for rx, r in SKELETONS[1:]:
            if rx.search(pose_path.replace("\\", "/")):
                rel = r
                break
    for r in ([rel] if rel else []) + [DEFAULT_SKELETON]:
        for root in content_roots(pose_path):
            p = os.path.join(root, r)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError("no Genesis figure found next to this pose (looked for %s in the content library)"
                            % (rel or DEFAULT_SKELETON))


_SEL_RX = re.compile(r"name://@selection/([^:]+):\?(rotation|translation)/([xyz])/value")
# Hierarchical presets: "hip:/data/DAZ%203D/Genesis%208/Female/Genesis8Female.dsf#hip?rotation/x/value".
_HIER_RX = re.compile(r"[^:]+:(/[^#]*\.dsf)#([^?]+)\?(rotation|translation)/([xyz])/value")


def read_daz_pose(path: str) -> tuple[dict, dict, str | None]:
    """bone -> {x, y, z} degrees, bone -> translation (cm), and the figure file the preset
    names (hierarchical presets do; relative to the content library), from a pose preset."""
    from urllib.parse import unquote
    data = _load_json(path)
    kind = (data.get("asset_info") or {}).get("type", "")
    if kind and kind not in ("preset_pose", "preset_hierarchical_pose"):
        raise ValueError(f"not a pose preset ({kind})")
    ang, trans, figs = {}, {}, {}
    hier = []
    for a in (data.get("scene") or {}).get("animations", []):
        if not a.get("keys"):
            continue
        v = a["keys"][0][1]
        if not isinstance(v, (int, float)):
            continue
        url = str(a.get("url", ""))
        m = _SEL_RX.match(url)
        if m:
            bone, ch, ax = m.groups()
            (ang if ch == "rotation" else trans).setdefault(bone, {})[ax] = float(v)
            continue
        m = _HIER_RX.match(url)
        if m:
            dsf, bone, ch, ax = m.groups()
            hier.append((unquote(dsf), unquote(bone), ch, ax, float(v)))
            figs[unquote(dsf)] = figs.get(unquote(dsf), 0) + 1
    figure = None
    if hier:
        # Clothing and props in the same preset have their own bones: keep the figure's.
        genesis = [f for f in figs if re.search(r"/Genesis[^/]*\.dsf$", f, re.I)] or list(figs)
        figure = max(genesis, key=lambda f: figs[f]).lstrip("/")
        for dsf, bone, ch, ax, v in hier:
            if dsf.lstrip("/") == figure:
                (ang if ch == "rotation" else trans).setdefault(bone, {})[ax] = v
    if not ang:
        raise ValueError("this preset holds no bone rotations")
    return ang, trans, figure


# Mannequin joints a Daz bone moves (left side; right is the same with r).
DAZ_JOINT = [(re.compile(r"^(hip|pelvis)$"), "hips"), (re.compile(r"^(abdomen|spine[12])"), "spine"),
             (re.compile(r"^(chest|spine[34])"), "chest"), (re.compile(r"^neck"), "neck"), (re.compile(r"^head$"), "head"),
             (re.compile(r"^([lr]Collar|[lr]Shldr|[lr]_shoulder|[lr]_upperarm)"), "_shoulder"),
             (re.compile(r"^([lr]Fore[Aa]rm|[lr]_forearm)"), "_elbow"), (re.compile(r"^([lr]Hand|[lr]_hand)$"), "_wrist"),
             (re.compile(r"^([lr]Thigh|[lr]_thigh)"), "_hip"), (re.compile(r"^([lr]Shin|[lr]_shin)"), "_knee"),
             (re.compile(r"^([lr]Foot|[lr]Toe|[lr]_foot|[lr]_toes)$"), "_ankle")]


def daz_touched(ang: dict) -> list[str]:
    """The mannequin joints a preset turns (a partial preset only poses a leg or the arms)."""
    out = set()
    for bone, a in ang.items():
        if not any(abs(v) > 0.01 for v in a.values()):
            continue
        for rx, j in DAZ_JOINT:
            if rx.match(bone):
                out.add((bone[0].lower() + j) if j.startswith("_") else j)
                break
    return sorted(out)


def daz_points(path: str) -> dict:
    """Pose points of a Daz pose preset, by forward kinematics on the figure's skeleton:
    each bone turns about its centre, in its own orientation frame (both Euler orders
    applied last-letter-outermost, as Daz Studio does)."""
    ang, trans, named = read_daz_pose(path)
    nodes = skeleton(find_skeleton(path, set(ang), named))
    world: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get(name):
        if name in world:
            return world[name]
        n = nodes[name]
        o = _euler("XYZ", dict(zip("xyz", n["o"])))
        r = o @ _euler(n["ro"], ang.get(name, {})) @ o.T
        t = np.array([trans.get(name, {}).get(a, 0.0) for a in "xyz"])
        local_t = -r @ n["c"] + n["c"] + t
        if n["parent"] and n["parent"] in nodes:
            pr, pt = get(n["parent"])
            world[name] = (pr @ r, pr @ local_t + pt)
        else:
            world[name] = (r, local_t)
        return world[name]

    def at(name, end=False):
        m, t = get(name)
        return m @ (nodes[name]["e"] if end else nodes[name]["c"]) + t

    pts = {}
    for key, names in DAZ_BONES.items():
        name = next((b for b in names if b in nodes), None)
        if name:
            pts[key] = at(name)
    if "head" in pts:
        pts["head_top"] = at("head", end=True)
    hand = {"l": next((b for b in ("lHand", "l_hand") if b in nodes), None),
            "r": next((b for b in ("rHand", "r_hand") if b in nodes), None)}
    for s, b in hand.items():
        if b:
            pts[s + "_hand"] = at(b, end=True)
    eyes = [next((b for b in (s + "Eye", s + "_eye") if b in nodes), None) for s in "lr"]
    if all(eyes):
        # The eye bones point along the gaze: their ends sit in front of the face.
        pts["nose"] = (at(eyes[0], True) + at(eyes[1], True)) / 2 - np.array([0, 2.5, 0]) @ get("head")[0].T
    return pts


# ── OpenPose ──────────────────────────────────────────────────────────────────────
COCO18 = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
          "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]
BODY25 = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "pelvis",
          "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear",
          "l_toe", "l_small_toe", "l_heel", "r_toe", "r_small_toe", "r_heel"]
# Bone lengths as fractions of the height (the mannequin's own proportions).
BONE_LEN = {("l_shoulder", "l_elbow"): 0.182, ("l_elbow", "l_wrist"): 0.142, ("l_hip", "l_knee"): 0.245,
            ("l_knee", "l_ankle"): 0.246, ("neck", "pelvis"): 0.288, ("l_shoulder", "r_shoulder"): 0.188,
            ("l_hip", "r_hip"): 0.098, ("neck", "nose"): 0.1, ("l_ankle", "l_toe"): 0.12}


def _bone_len(a: str, b: str) -> float | None:
    for (x, y), v in BONE_LEN.items():
        for p, q in ((x, y), (x.replace("l_", "r_"), y.replace("l_", "r_"))):
            if {p, q} == {a, b}:
                return v
    return None


def openpose_people(data) -> list[tuple[list, int, int]]:
    """[(keypoints [[x, y, conf]...], width, height)] from the JSON shapes in use: OpenPose's
    own {"people": [{"pose_keypoints_2d": [...]}]}, ComfyUI's POSE_KEYPOINT (a list of
    those with canvas_width / canvas_height), and the openpose-editor exports."""
    frames = data if isinstance(data, list) else [data]
    out = []
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        w, h = int(fr.get("canvas_width") or fr.get("width") or 0), int(fr.get("canvas_height") or fr.get("height") or 0)
        for p in fr.get("people") or []:
            flat = p.get("pose_keypoints_2d") or p.get("pose_keypoints") or []
            pts = [flat[i:i + 3] for i in range(0, len(flat) - 2, 3)]
            if len(pts) in (18, 25):
                out.append((pts, w, h))
    return out


def openpose_points(kps: list, width: int = 0, height: int = 0) -> dict:
    """3D pose points from 2D OpenPose keypoints. The image gives each bone's direction on
    screen; its length on screen against its real length gives how far it points towards
    or away from the camera, and the side is the natural guess (knees forward, feet back,
    elbows back, hands forward). Fine-tune the result with the bone handles."""
    names = COCO18 if len(kps) == 18 else BODY25
    # Normalised (0-1) coordinates are scaled by the canvas, so a wide picture keeps its shape.
    norm = all(0 <= float(x) <= 1.5 and 0 <= float(y) <= 1.5 for x, y, c in kps if c > 0)
    sx = width if norm and width else 1
    sy = height if norm and height else 1
    p2 = {n: np.array([float(x) * sx, -float(y) * sy]) for n, (x, y, c) in zip(names, kps) if c and c > 0.05}
    if "pelvis" not in p2 and "l_hip" in p2 and "r_hip" in p2:
        p2["pelvis"] = (p2["l_hip"] + p2["r_hip"]) / 2
    need = {"l_shoulder", "r_shoulder"} | {"l_hip", "r_hip"}
    if len(need & set(p2)) < 2:
        raise ValueError("too few body keypoints in this OpenPose file")
    # Pixels per body height: the least foreshortened of the long bones.
    scale = 0.0
    for (a, b), frac in list(BONE_LEN.items()) + [((x.replace("l_", "r_"), y.replace("l_", "r_")), f)
                                                  for (x, y), f in BONE_LEN.items() if x.startswith("l_")]:
        if a in p2 and b in p2:
            scale = max(scale, np.linalg.norm(p2[a] - p2[b]) / frac)
    if scale <= 0:
        raise ValueError("cannot size this OpenPose figure")
    # Facing the camera when the figure's left shoulder is on the image's right.
    front = 1.0
    if "l_shoulder" in p2 and "r_shoulder" in p2:
        front = 1.0 if p2["l_shoulder"][0] >= p2["r_shoulder"][0] else -1.0
    elif "l_hip" in p2 and "r_hip" in p2:
        front = 1.0 if p2["l_hip"][0] >= p2["r_hip"][0] else -1.0
    centre = p2.get("pelvis", next(iter(p2.values())))
    pts = {n: np.array([(v[0] - centre[0]) / scale, (v[1] - centre[1]) / scale, 0.0]) for n, v in p2.items()}
    # Depth along each chain: towards the camera is +z in this (camera) frame.
    chains = [("l_shoulder", "l_elbow", -1), ("l_elbow", "l_wrist", 1), ("r_shoulder", "r_elbow", -1),
              ("r_elbow", "r_wrist", 1), ("l_hip", "l_knee", 1), ("l_knee", "l_ankle", -1),
              ("r_hip", "r_knee", 1), ("r_knee", "r_ankle", -1), ("neck", "nose", 1)]
    for a, b, side in chains:
        if a in pts and b in pts:
            ln = _bone_len(a, b) or 0
            d = np.linalg.norm(pts[b][:2] - pts[a][:2])
            pts[b][2] = pts[a][2] + side * front * math.sqrt(max(0.0, ln * ln - d * d))
    for k in ("l_eye", "r_eye", "l_ear", "r_ear"):
        if k in pts and "nose" in pts:
            pts[k][2] = pts["nose"][2] - (0.02 if "eye" in k else 0.08) * front
    if "neck" in pts and "nose" in pts:
        pts["head"] = pts["neck"] + (pts["nose"] - pts["neck"]) * 0.45
        pts["head"][2] -= 0.04 * front
    out = {k: v * 170.0 for k, v in pts.items() if k in KEYS}  # height fractions -> cm of a 170 cm figure
    return out


# ── BVH ───────────────────────────────────────────────────────────────────────────
BVH_BONES = {
    "pelvis": ["Hips", "hip", "pelvis"], "spine": ["Spine", "spine", "abdomen", "Chest"],
    "chest": ["Spine1", "Spine2", "chest", "Chest2", "Chest"], "neck": ["Neck", "neck"], "head": ["Head", "head"],
    "l_shoulder": ["LeftArm", "LeftUpArm", "lShldr", "LeftShoulder"], "r_shoulder": ["RightArm", "RightUpArm", "rShldr", "RightShoulder"],
    "l_elbow": ["LeftForeArm", "LeftLowArm", "lForeArm"], "r_elbow": ["RightForeArm", "RightLowArm", "rForeArm"],
    "l_wrist": ["LeftHand", "lHand"], "r_wrist": ["RightHand", "rHand"],
    "l_hand": ["LeftFingerBase", "LeftHandIndex1", "LeftHandMiddle1"], "r_hand": ["RightFingerBase", "RightHandIndex1", "RightHandMiddle1"],
    "l_hip": ["LeftUpLeg", "LeftHip", "lThigh"], "r_hip": ["RightUpLeg", "RightHip", "rThigh"],
    "l_knee": ["LeftLeg", "LeftLowLeg", "LeftKnee", "lShin"], "r_knee": ["RightLeg", "RightLowLeg", "RightKnee", "rShin"],
    "l_ankle": ["LeftFoot", "LeftAnkle", "lFoot"], "r_ankle": ["RightFoot", "RightAnkle", "rFoot"],
    "l_toe": ["LeftToeBase", "LeftToe", "lToe"], "r_toe": ["RightToeBase", "RightToe", "rToe"],
}


def read_bvh(path: str):
    """(joints, frames): joints in file order {name, parent, offset, channels, end}; frames
    an (n, channels) array."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        tokens = f.read().split()
    i = 0
    joints, stack = [], []
    while tokens[i] != "MOTION":
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            joints.append({"name": tokens[i + 1], "parent": stack[-1] if stack else None, "offset": None,
                           "channels": [], "end": None})
            i += 2
        elif t == "End":
            j = joints[stack[-1]]
            k = tokens.index("OFFSET", i)
            j["end"] = np.array([float(x) for x in tokens[k + 1:k + 4]])
            i = tokens.index("}", k) + 1
        elif t == "{":
            stack.append(len(joints) - 1)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            joints[-1]["offset"] = np.array([float(x) for x in tokens[i + 1:i + 4]])
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            joints[-1]["channels"] = tokens[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    n = int(tokens[tokens.index("Frames:", i) + 1])
    k = tokens.index("Time:", i) + 2
    width = sum(len(j["channels"]) for j in joints)
    data = np.array(tokens[k:k + n * width], dtype=float).reshape(n, width)
    return joints, data


def bvh_frame_points(joints, row) -> dict:
    """World positions of every joint (and Head's end) for one frame."""
    pos, rot = {}, {}
    c = 0
    for idx, j in enumerate(joints):
        vals = row[c:c + len(j["channels"])]
        c += len(j["channels"])
        r = np.eye(3)
        t = j["offset"].copy()
        for ch, v in zip(j["channels"], vals):
            if ch.endswith("rotation"):
                r = r @ _rot(ch[0].lower(), v)
            elif ch.endswith("position"):
                t["XYZ".index(ch[0])] = v
        if j["parent"] is None:
            rot[idx], pos[idx] = r, t
        else:
            p = j["parent"]
            rot[idx] = rot[p] @ r
            pos[idx] = pos[p] + rot[p] @ t
    out = {j["name"]: pos[i] for i, j in enumerate(joints)}
    for i, j in enumerate(joints):
        if j["end"] is not None:
            out[j["name"] + "_end"] = pos[i] + rot[i] @ j["end"]
        out[j["name"] + "_rot"] = rot[i]
    return out


def bvh_points(world: dict, rest_forward: np.ndarray | None = None) -> dict:
    pts = {}
    for key, names in BVH_BONES.items():
        name = next((n for n in names if n in world), None)
        if name is not None:
            pts[key] = world[name]
    head = next((n for n in BVH_BONES["head"] if n in world), None)
    if head and head + "_end" in world:
        pts["head_top"] = world[head + "_end"]
        if rest_forward is not None:
            fwd = world[head + "_rot"] @ rest_forward
            pts["nose"] = pts["head"] + (pts["head_top"] - pts["head"]) * 0.45 + fwd * np.linalg.norm(
                pts["head_top"] - pts["head"]) * 0.8
    return pts


def bvh_rest_forward(joints) -> np.ndarray:
    """Which way the BVH skeleton faces in its rest pose (+Y up): left hip x up."""
    names = [j["name"] for j in joints]
    li = next((names.index(n) for n in BVH_BONES["l_hip"] if n in names), None)
    ri = next((names.index(n) for n in BVH_BONES["r_hip"] if n in names), None)
    if li is None or ri is None:
        return np.array([0.0, 0.0, 1.0])

    def rest(i):
        p = np.zeros(3)
        while i is not None:
            p = p + joints[i]["offset"]
            i = joints[i]["parent"]
        return p
    left = rest(li) - rest(ri)
    f = np.cross(left, [0.0, 1.0, 0.0])
    return f / (np.linalg.norm(f) or 1)


def _features(pts: dict) -> np.ndarray:
    pairs = [("pelvis", "neck"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"),
             ("r_elbow", "r_wrist"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("r_hip", "r_knee"),
             ("r_knee", "r_ankle"), ("neck", "head_top")]
    v = []
    for a, b in pairs:
        if a in pts and b in pts:
            d = np.asarray(pts[b]) - np.asarray(pts[a])
            v.append(d / (np.linalg.norm(d) or 1))
        else:
            v.append(np.zeros(3))
    return np.concatenate(v)


def is_tpose(pts: dict) -> bool:
    try:
        arms = [np.asarray(pts[s + "_wrist"]) - np.asarray(pts[s + "_shoulder"]) for s in "lr"]
    except KeyError:
        return False
    return all(abs(a[1]) < 0.3 * np.linalg.norm(a) for a in arms)


def distinct_frames(frames: list[dict], k: int, min_change: float = 0.35, skip_tpose: bool = True) -> list[int]:
    """Up to k frame indices that look clearly different from each other (farthest-point
    sampling on bone directions)."""
    feats = [(i, _features(p)) for i, p in enumerate(frames) if not (skip_tpose and is_tpose(p))]
    if not feats:
        return []
    mid = feats[len(feats) // 2]
    chosen = [mid]
    while len(chosen) < k:
        best, best_d = None, 0.0
        for i, f in feats:
            d = min(np.linalg.norm(f - g) / math.sqrt(len(f) / 3) for _, g in chosen)
            if d > best_d:
                best, best_d = (i, f), d
        if best is None or best_d < min_change:
            break
        chosen.append(best)
    return sorted(i for i, _ in chosen)


def bvh_poses(path: str, k: int = 4) -> list[tuple[int, dict]]:
    joints, data = read_bvh(path)
    fwd = bvh_rest_forward(joints)
    step = max(1, len(data) // 400)
    frames = [(i, bvh_points(bvh_frame_points(joints, data[i]), fwd)) for i in range(0, len(data), step)]
    pick = distinct_frames([p for _, p in frames], k)
    return [(frames[i][0], frames[i][1]) for i in pick]


# ── user poses ────────────────────────────────────────────────────────────────────
def user_dir() -> str:
    import folder_paths
    try:
        paths = folder_paths.get_folder_paths("poses")
        if paths:
            return paths[0]
    except Exception:
        pass
    return os.path.join(folder_paths.models_dir, "poses")


def safe_name(name: str) -> str:
    n = re.sub(r"[^\w\- .()]+", "_", str(name or "").strip())[:80].strip(" .")
    if not n:
        raise ValueError("give the pose a name")
    return n


def list_user() -> list[dict]:
    root = user_dir()
    out = []
    if not os.path.isdir(root):
        return out
    for dp, _dirs, files in os.walk(root):
        for f in sorted(files):
            low = f.lower()
            if not low.endswith(POSE_EXTS):
                continue
            full = os.path.join(dp, f)
            rel = os.path.relpath(full, root).replace("\\", "/")
            base = re.sub(r"(\.srpose)?\.(json|duf|bvh)$", "", f, flags=re.I)
            thumb = next((os.path.join(dp, base + e) for e in (".png", ".jpg", ".duf.png") if
                          os.path.isfile(os.path.join(dp, base + e))), None)
            kind = pose_kind(f)
            out.append({"id": "mine:" + rel, "name": base, "kind": kind, "folder": os.path.dirname(rel),
                        "thumb": bool(thumb)})
    return out


def user_path(rel: str) -> str:
    root = os.path.realpath(user_dir())
    p = os.path.realpath(os.path.join(root, rel))
    if not p.startswith(root + os.sep):
        raise ValueError("not in the poses folder")
    return p


def save_user(name: str, bones: dict, anchor: str, png: bytes | None) -> str:
    root = user_dir()
    os.makedirs(root, exist_ok=True)
    base = safe_name(name)
    clean = {}
    for k, v in (bones or {}).items():
        if isinstance(k, str) and isinstance(v, (list, tuple)) and len(v) == 3:
            clean[k[:32]] = [round(float(x), 2) for x in v]
    if not clean:
        raise ValueError("no joint angles to save")
    with open(os.path.join(root, base + ".srpose.json"), "w", encoding="utf-8") as f:
        json.dump({"format": "sceneref-pose", "version": 1, "name": base, "anchor": anchor if anchor == "seat"
                   else "lowest", "bones": clean}, f, indent=1)
    if png:
        with open(os.path.join(root, base + ".png"), "wb") as f:
            f.write(png)
    return "mine:" + base + ".srpose.json"


# ── one entry point for the viewer ────────────────────────────────────────────────
def thumb_for(path: str) -> str | None:
    base = re.sub(r"(\.srpose)?\.(json|duf|bvh)$", "", path, flags=re.I)
    for p in (path + ".png", base + ".png", base + ".jpg", base + ".tip.png"):
        if os.path.isfile(p):
            return p
    return None


def get(pose_id: str) -> dict:
    """{name, anchor, points} or {name, anchor, bones} (or {name, poses: [...]} for a file
    with several, e.g. a BVH clip or an OpenPose picture of a group)."""
    src, _, ref = pose_id.partition(":")
    if src == "lib":
        for p in library():
            if p["id"] == pose_id:
                return {"name": p["name"], "anchor": p.get("anchor", "lowest"), "points": p["points"]}
        raise KeyError(pose_id)
    if src == "mine":
        path = user_path(ref)
    elif src == "file":
        path = os.path.abspath(ref)
        if not path.lower().endswith(POSE_EXTS) or not os.path.isfile(path):
            raise ValueError("not a pose file")
    else:
        raise ValueError("unknown pose source")
    return from_file(path)


def from_file(path: str) -> dict:
    low = path.lower()
    name = re.sub(r"(\.srpose)?\.(json|duf|bvh)$", "", os.path.basename(path), flags=re.I)
    if low.endswith(".duf"):
        touched = daz_touched(read_daz_pose(path)[0])
        if not touched:
            raise ValueError("this preset turns no body joints (only the face, eyes or fingers)")
        pts = normalise(daz_points(path))
        out = {"name": name, "anchor": anchor_for(pts), "points": pts}
        if len(touched) < 12:
            out["only"] = touched  # a partial preset: the rest of the pose stays as it is
        return out
    if low.endswith(".bvh"):
        poses = []
        for i, p in bvh_poses(path, k=6):
            pts = normalise(p)
            poses.append({"name": f"{name} · frame {i}", "anchor": anchor_for(pts), "points": pts})
        if not poses:
            raise ValueError("no usable frames in this BVH file")
        return poses[0] if len(poses) == 1 else {"name": name, "poses": poses}
    return from_json(_load_json(path), name)


def from_json(data, name: str = "OpenPose") -> dict:
    """Our saved pose, or OpenPose keypoints (one pose per person)."""
    if isinstance(data, dict) and data.get("format") == "sceneref-pose":
        return {"name": data.get("name") or name, "anchor": data.get("anchor", "lowest"), "bones": data["bones"]}
    people = openpose_people(data)
    if not people:
        raise ValueError("no OpenPose people (pose_keypoints_2d) in this file")
    poses = []
    for n, (kps, w, h) in enumerate(people):
        pts = normalise(openpose_points(kps, w, h))
        poses.append({"name": name if len(people) == 1 else f"{name} · person {n + 1}", "anchor": anchor_for(pts),
                      "points": pts, "keypoints": [[round(float(c), 4) for c in k] for k in kps], "canvas": [w, h]})
    return poses[0] if len(poses) == 1 else {"name": name, "poses": poses}


def pose_kind(name: str) -> str:
    low = name.lower()
    return ("srpose" if low.endswith(".srpose.json") else "openpose" if low.endswith(".json")
            else "daz" if low.endswith(".duf") else "bvh")


def folder(path: str, limit: int = 3000) -> dict:
    """A folder anywhere on disk: sub-folders and the pose files in it (Daz presets, OpenPose
    JSON, BVH, saved poses). One level only: a Daz library holds hundreds of thousands."""
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise ValueError("not a folder: %s" % path)
    dirs, files = [], []
    with os.scandir(path) as it:
        for i, e in enumerate(it):
            if i > limit:
                break
            try:
                if e.is_dir(follow_symlinks=False):
                    if not e.name.startswith(("$", ".")):
                        dirs.append(e.name)
                elif e.name.lower().endswith(POSE_EXTS) and not e.name.lower().endswith((".tip.json",)):
                    base = re.sub(r"(\.srpose)?\.(json|duf|bvh)$", "", e.name, flags=re.I)
                    files.append({"id": "file:" + e.path, "name": base, "kind": pose_kind(e.name),
                                  "thumb": bool(thumb_for(e.path))})
            except OSError:
                continue
    dirs.sort(key=str.lower)
    files.sort(key=lambda f: f["name"].lower())
    parent = os.path.dirname(path.rstrip("\\/"))
    if parent == path.rstrip("\\/") or not parent:
        parent = ""
    elif len(parent) == 2 and parent[1] == ":":
        parent += os.sep
    return {"path": path, "parent": parent, "dirs": dirs, "poses": files}


def default_folder(scene_path: str | None) -> str:
    """Where the file tab opens: the People folder of the open scene's content library."""
    for root in content_roots(scene_path) if scene_path else []:
        p = os.path.join(root, "People")
        return p if os.path.isdir(p) else root
    return ""


def thumb_path(pose_id: str) -> str | None:
    src, _, ref = pose_id.partition(":")
    if src == "mine":
        return thumb_for(user_path(ref))
    if src == "file":
        path = os.path.abspath(ref)
        if path.lower().endswith(POSE_EXTS):
            return thumb_for(path)
    return None
