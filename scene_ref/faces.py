"""Faces in reference pictures: find them, crop them, judge them, compare them.

Uses OpenCV's YuNet detector and SFace recogniser (OpenCV Zoo, Apache-2.0) when cv2 and
the two .onnx files are present (face_detection_yunet_*.onnx, face_recognition_sface_*.onnx,
anywhere under ComfyUI/models). Without them every function degrades to "no face found"
and callers keep the pictures as they are.

What it is for:
* a face crop from a body or group photo, so the face reference has many pixels on the
  face instead of a small face in a big picture (FLUX.2 and H3 both see a reference at
  about 1 MP: a 120-pixel face in a 4000-pixel photo is a few tokens);
* quality notes: no face, several faces, a small or blurred face, near-duplicates;
* a likeness score (cosine similarity of SFace features, 0.363 is OpenCV's "same person"
  threshold) between the character's pictures and a result.
"""
from __future__ import annotations

import os
import re

import numpy as np
from PIL import Image

import folder_paths

SAME_PERSON = 0.363  # SFace cosine threshold from the OpenCV model card
DUPLICATE = 0.92  # two pictures this close are the same photo (or a crop of it)
DIFFERENT = 0.20  # a picture this far from all the others is most likely someone else
_cache: dict = {}


def _find(pattern: str) -> str:
    rx = re.compile(pattern, re.I)
    roots = [folder_paths.models_dir]
    for kind in ("ultralytics", "onnx", "detection", "facedetection", "insightface"):
        try:
            roots.extend(folder_paths.get_folder_paths(kind))
        except Exception:
            pass
    # Shared model folders (extra_model_paths.yaml): the parent of any registered folder.
    for paths, _exts in list(getattr(folder_paths, "folder_names_and_paths", {}).values()):
        roots.extend(os.path.dirname(os.path.normpath(p)) for p in paths)
    seen = set()
    for root in roots:
        root = os.path.realpath(root)
        if root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        for d, dirs, files in os.walk(root):
            # Model folders can be huge; these never hold the face models.
            dirs[:] = [x for x in dirs if x.lower() not in ("diffusion_models", "unet", "checkpoints", "loras",
                                                             "text_encoders", "clip", "llm", "vae")]
            for f in files:
                if rx.match(f):
                    return os.path.join(d, f)
    return ""


def _models():
    if "models" not in _cache:
        try:
            import cv2
            det, rec = _find(r"face_detection_yunet.*\.onnx$"), _find(r"face_recognition_sface.*\.onnx$")
            _cache["models"] = (cv2, det, rec) if det else None
        except Exception:
            _cache["models"] = None
    return _cache["models"]


def available() -> bool:
    return _models() is not None


def _bgr(im: Image.Image):
    return np.ascontiguousarray(np.asarray(im.convert("RGB"))[:, :, ::-1])


def detect(im: Image.Image, min_score: float = 0.7) -> list[dict]:
    """Faces, biggest first: {box: (x0, y0, x1, y1), score, row (YuNet's 15 numbers)}.
    Large pictures are searched at about 1280 px; boxes are in the picture's pixels."""
    faces = _detect(im, min_score)
    if not faces:
        # YuNet misses a face that fills the frame (a close-up cut at the forehead): try again
        # with the picture in the middle of a grey canvas twice its size.
        pad_x, pad_y = im.width // 2, im.height // 2
        canvas = Image.new("RGB", (im.width + 2 * pad_x, im.height + 2 * pad_y), (127, 127, 127))
        canvas.paste(im.convert("RGB"), (pad_x, pad_y))
        faces = _detect(canvas, min_score)
        for f in faces:
            x0, y0, x1, y1 = f["box"]
            f["box"] = (max(0.0, x0 - pad_x), max(0.0, y0 - pad_y), min(im.width, x1 - pad_x), min(im.height, y1 - pad_y))
            f["row"][0:14:2] -= pad_x
            f["row"][1:14:2] -= pad_y
    return faces


def _detect(im: Image.Image, min_score: float) -> list[dict]:
    m = _models()
    if not m:
        return []
    cv2, det_path, _rec = m
    s = min(1.0, 1280 / max(im.width, im.height))
    small = im if s == 1.0 else im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
    img = _bgr(small)
    det = cv2.FaceDetectorYN.create(det_path, "", (img.shape[1], img.shape[0]), min_score, 0.3, 50)
    _n, rows = det.detect(img)
    faces = []
    for r in rows if rows is not None else []:
        r = r.astype(np.float64)
        r[:14] /= s
        x, y, w, h = r[:4]
        faces.append({"box": (x, y, x + w, y + h), "score": float(r[14]), "row": r.astype(np.float32)})
    faces.sort(key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]), reverse=True)
    return faces


def sharpness(im: Image.Image, box) -> float:
    """Variance of the Laplacian over the face, at a fixed 160 px face size, so faces of any
    size compare. Below about 60 a face reads as soft, below 25 as blurred."""
    m = _models()
    if not m:
        return 0.0
    cv2 = m[0]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(im.width, x1), min(im.height, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    face = im.crop((x0, y0, x1, y1)).convert("L").resize((160, 160), Image.BILINEAR)
    return float(cv2.Laplacian(np.asarray(face), cv2.CV_64F).var())


def embedding(im: Image.Image, face: dict | None = None):
    """SFace feature (128 numbers, unit length) of the biggest face, or None."""
    m = _models()
    if not m or not m[2]:
        return None
    cv2, _det, rec_path = m
    face = face or next(iter(detect(im)), None)
    if face is None:
        return None
    if "rec" not in _cache:
        _cache["rec"] = cv2.FaceRecognizerSF.create(rec_path, "")
    rec = _cache["rec"]
    aligned = rec.alignCrop(_bgr(im), face["row"])
    feat = rec.feature(aligned).reshape(-1).astype(np.float64)
    n = np.linalg.norm(feat)
    return feat / n if n else None


def similarity(a, b) -> float:
    return float(np.dot(a, b)) if a is not None and b is not None else 0.0


def face_crop(im: Image.Image, face: dict, scale: float = 2.4) -> Image.Image:
    """A square around the face, `scale` face heights wide, a little more room above for
    hair. Stays inside the picture (shifted, then shrunk if the picture is too small)."""
    x0, y0, x1, y1 = face["box"]
    h = max(x1 - x0, y1 - y0)
    side = min(h * scale, im.width, im.height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.12 * h
    left = min(max(cx - side / 2, 0), im.width - side)
    top = min(max(cy - side / 2, 0), im.height - side)
    return im.crop((int(left), int(top), int(left + side), int(top + side)))


def inspect(pictures: list[tuple[Image.Image, str]]) -> list[dict]:
    """Per picture: {faces, face_px (height of the biggest face), sharp, feat}."""
    out = []
    for im, _role in pictures:
        faces = detect(im)
        f = faces[0] if faces else None
        out.append({"faces": len(faces), "face": f,
                    "face_px": int(f["box"][3] - f["box"][1]) if f else 0,
                    "sharp": sharpness(im, f["box"]) if f else 0.0,
                    "feat": embedding(im, f) if f else None})
    return out
