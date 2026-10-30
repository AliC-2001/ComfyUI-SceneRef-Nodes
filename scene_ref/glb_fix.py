"""Post-export repairs on a GLB file (pure Python, no Blender)."""
import json
import struct


def sanitize_glb(path):
    """Drop material references to textures that have no image.

    When the exporter cannot encode an image it still writes the texture, with
    no "source"; three.js then fails the WHOLE scene on it ("reading 'uri'").
    A missing texture is far better than a missing scene. Returns the count.
    """
    with open(path, "rb") as f:
        data = f.read()
    jlen = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + jlen])
    images = gltf.get("images", [])

    def has_image(t):
        idx = t.get("source")
        for ext in (t.get("extensions") or {}).values():
            if isinstance(ext, dict) and "source" in ext:
                idx = ext["source"]
        return isinstance(idx, int) and 0 <= idx < len(images)

    bad = {i for i, t in enumerate(gltf.get("textures", [])) if not has_image(t)}
    if not bad:
        return 0

    def scrub(obj):
        if isinstance(obj, dict):
            for k in list(obj):
                v = obj[k]
                if isinstance(v, dict) and isinstance(v.get("index"), int) and v["index"] in bad                         and k.lower().endswith("texture"):
                    del obj[k]
                else:
                    scrub(v)
        elif isinstance(obj, list):
            for v in obj:
                scrub(v)

    scrub(gltf.get("materials", []))
    new_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    new_json += b" " * ((4 - len(new_json) % 4) % 4)
    rest = data[20 + jlen:]
    total = 12 + 8 + len(new_json) + len(rest)
    out = data[:8] + struct.pack("<I", total) + struct.pack("<I", len(new_json)) + data[16:20] + new_json + rest
    with open(path, "wb") as f:
        f.write(out)
    return len(bad)
