"""Builds scene_ref/data/poses_cmu.json, the viewer's bundled pose library.

Source: the CMU Graphics Lab Motion Capture Database (http://mocap.cs.cmu.edu), in the
cgspeed BVH conversion (https://github.com/Shriinivas/cmubvh mirrors it). CMU: "This data
is free for use in research and commercial projects worldwide"; the BVH conversion adds no
restrictions. Acknowledgment: "The data used in this project was obtained from
mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217."

Usage: python tools/make_pose_library.py <folder with the .bvh files named like 13_04.bvh>

For each clip below, a few frames that look clearly different from each other are taken
(farthest-point sampling on bone directions, T-poses skipped), turned into pose points
(scene_ref/poses.py) and stored with a name and a category.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scene_ref"))
import poses  # noqa: E402  (plain import: no ComfyUI needed)

# trial: (name, category, frames to take)
CLIPS = {
    "82_01": ("Standing, relaxed", "Standing", 1),
    "82_18": ("Standing, at ease", "Standing", 1),
    "77_03": ("Ready stance", "Standing", 2),
    "77_01": ("Looking around", "Standing", 3),
    "18_09": ("Talking with hand gestures", "Gestures", 4),
    "13_14": ("Laughing", "Gestures", 3),
    "79_71": ("Sad", "Gestures", 3),
    "13_26": ("Waving", "Gestures", 3),
    "13_27": ("Pointing", "Gestures", 3),
    "79_36": ("On the phone", "Everyday", 3),
    "79_37": ("Dialling a phone", "Everyday", 2),
    "13_09": ("Drinking", "Everyday", 3),
    "80_19": ("Sweeping", "Everyday", 3),
    "79_77": ("Vacuuming", "Everyday", 2),
    "70_03": ("Carrying a suitcase", "Everyday", 2),
    "81_06": ("Pushing a heavy box", "Everyday", 2),
    "26_09": ("Bending to pick up", "Bending and reaching", 3),
    "69_70": ("Squatting to pick up", "Bending and reaching", 3),
    "15_06": ("Leaning forward, reaching", "Bending and reaching", 3),
    "13_10": ("Reaching up, on tiptoe", "Bending and reaching", 3),
    "42_01": ("Stretching", "Exercise", 4),
    "13_29": ("Exercise: jumping jacks, twists, squats", "Exercise", 5),
    "22_14": ("Squats", "Exercise", 2),
    "49_18": ("Balancing on one leg", "Exercise", 2),
    "13_17": ("Boxing", "Action", 4),
    "79_86": ("Shooting a bow", "Action", 2),
    "02_01": ("Walking", "Walking and running", 2),
    "17_03": ("Walking stealthily", "Walking and running", 2),
    "82_10": ("Sad walk", "Walking and running", 2),
    "82_12": ("Happy walk", "Walking and running", 2),
    "09_01": ("Running", "Walking and running", 2),
    "13_04": ("Sitting on a stool, chin in hand", "Sitting", 3),
    "13_05": ("Sitting, hands at the chin", "Sitting", 3),
    "13_06": ("Sitting, elbow out", "Sitting", 2),
    "14_30": ("Sitting, ankle on the other knee", "Sitting", 3),
    "13_01": ("Sitting on a high stool", "Sitting", 2),
    "86_09": ("Sitting and looking around", "Sitting", 3),
    "86_15": ("Sitting, talking with the hands", "Sitting", 3),
    "22_03": ("Sitting, face in the hands", "Sitting", 2),
    "01_06": ("Sitting, legs dangling", "Sitting", 2),
    "82_05": ("Sitting on the ground, relaxed", "Sitting on the floor", 4),
    "23_03": ("Kneeling, comforting", "Kneeling and crouching", 3),
    "77_16": ("Lying down, getting up", "Lying", 4),
    "77_18": ("Lying on the back, getting up", "Lying", 4),
    "05_12": ("Dance: arms held high", "Dance", 3),
    "60_01": ("Salsa", "Dance", 3),
    "93_03": ("Charleston", "Dance", 3),
    "79_17": ("Playing the violin", "Music", 2),
}


POSTURES = {"Sitting": "seat", "Sitting on the floor": "floor", "Kneeling and crouching": "floor", "Lying": "lying"}


def posture(p) -> str:
    """lying / floor (sitting on the floor, kneeling, crouching) / seat / standing."""
    import numpy as np
    v = {k: np.asarray(x) for k, x in p.items()}
    torso = v["neck"] - v["pelvis"]
    if abs(torso[1]) < 0.5 * np.linalg.norm(torso):
        return "lying"
    if poses.anchor_for(p) == "seat":
        return "seat"
    low = min(x[1] for x in v.values())
    leg = np.linalg.norm(v["l_knee"] - v["l_hip"]) + np.linalg.norm(v["l_ankle"] - v["l_knee"])
    return "floor" if v["pelvis"][1] - low < 0.6 * leg else "standing"


def main(folder):
    out = []
    for trial, (name, cat, k) in CLIPS.items():
        path = os.path.join(folder, trial + ".bvh")
        if not os.path.isfile(path):
            print("missing", path)
            continue
        picked = poses.bvh_poses(path, k=k + 2)
        want = POSTURES.get(cat)
        picked = [(f, pts) for f, pts in picked if not want or posture(poses.normalise(pts)) == want][:k]
        for n, (frame, pts) in enumerate(picked, 1):
            p = poses.normalise(pts)
            label = name if len(picked) == 1 else f"{name} {n}"
            out.append({"id": f"lib:{trial}:{frame}", "name": label, "category": cat, "source": f"CMU {trial}",
                        "anchor": poses.anchor_for(p), "points": p})
        print(trial, name, len(picked))
    dst = os.path.join(HERE, "..", "scene_ref", "data", "poses_cmu.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"source": "CMU Graphics Lab Motion Capture Database, mocap.cs.cmu.edu (cgspeed BVH conversion)",
                   "acknowledgment": "The data used in this project was obtained from mocap.cs.cmu.edu. The database "
                                     "was created with funding from NSF EIA-0196217.",
                   "poses": out}, f, separators=(",", ":"))
    print(len(out), "poses ->", dst)


if __name__ == "__main__":
    main(sys.argv[1])
