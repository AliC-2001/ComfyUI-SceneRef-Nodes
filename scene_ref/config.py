"""Where Blender and Daz Studio live, and where converted scenes are cached.

Resolution order for each tool:
    1. scene_ref_config.json next to this pack's __init__.py ("blender" / "daz" keys)
    2. environment variables SCENEREF_BLENDER / SCENEREF_DAZ
    3. auto-detection (file associations, registry, usual install folders, PATH)
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PACK_DIR, "scene_ref_config.json")
CONVERTERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "converters")
VENDOR_DIR = os.path.join(PACK_DIR, "vendor")

DEFAULTS = {
    "blender": "",
    "daz": "",
    # Longest texture side baked into the GLB. Browsers choke on hundreds of 4K maps.
    "max_texture": 2048,
    "draco": True,
    # Daz Studio's -headless mode crashes on scenes that use GUI-bound plugins
    # (Puppeteer, aniMate...), so the default is a normal, minimized instance.
    "daz_headless": False,
    "daz_timeout_s": 3600,
    "blender_timeout_s": 3600,
    # Keep the intermediate Daz OBJ (can be close to a gigabyte) after converting.
    "keep_daz_obj": False,
    # llama.cpp's llama-server, for GGUF prompt upsamplers (FLUX.3). Empty = auto-detect.
    "llama_server": "",
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user = json.load(f)
        if isinstance(user, dict):
            cfg.update({k: v for k, v in user.items() if not k.startswith("_")})
    except FileNotFoundError:
        pass
    except Exception as e:
        print("[SceneRef] could not read %s: %s" % (CONFIG_PATH, e))
    return cfg


def _version_key(path: str):
    nums = re.findall(r"\d+", os.path.basename(os.path.dirname(path)))
    return tuple(int(n) for n in nums[:3]) or (0,)


def _reg_value(root, key, name):
    try:
        import winreg
        with winreg.OpenKey(root, key) as k:
            return winreg.QueryValueEx(k, name)[0]
    except Exception:
        return None


def _exe_from_command(cmd):
    if not cmd:
        return None
    m = re.match(r'\s*"([^"]+)"', cmd) or re.match(r"\s*(\S+)", cmd)
    return m.group(1) if m else None


def find_blender(cfg=None) -> str | None:
    cfg = cfg or load_config()
    for c in (cfg.get("blender"), os.environ.get("SCENEREF_BLENDER")):
        if c and os.path.isfile(c):
            return c
    found = []
    try:
        import winreg
        for key in (r"blendfile\shell\open\command", r"Applications\blender.exe\shell\open\command"):
            exe = _exe_from_command(_reg_value(winreg.HKEY_CLASSES_ROOT, key, ""))
            if exe and exe.lower().endswith("blender-launcher.exe"):
                exe = os.path.join(os.path.dirname(exe), "blender.exe")
            if exe and os.path.isfile(exe):
                found.append(exe)
    except ImportError:
        pass
    for pattern in (r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
                    r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe",
                    "/Applications/Blender.app/Contents/MacOS/Blender",
                    "/usr/bin/blender", "/snap/bin/blender"):
        found.extend(glob.glob(pattern))
    w = shutil.which("blender")
    if w:
        found.append(w)
    found = [f for f in found if os.path.isfile(f)]
    if not found:
        return None
    return sorted(set(found), key=_version_key)[-1]


def find_daz(cfg=None) -> str | None:
    cfg = cfg or load_config()
    for c in (cfg.get("daz"), os.environ.get("SCENEREF_DAZ")):
        if c and os.path.isfile(c):
            return c
    try:
        import winreg
        for key in (r"Software\DAZ\Studio4", r"Software\DAZ\Studio4 Public Build"):
            p = _reg_value(winreg.HKEY_CURRENT_USER, key, "InstallPath-64")
            if p and os.path.isfile(os.path.join(p, "DAZStudio.exe")):
                return os.path.join(p, "DAZStudio.exe")
    except ImportError:
        pass
    for p in (r"C:\Program Files\DAZ 3D\DAZStudio4\DAZStudio.exe",
              r"C:\Program Files\DAZ 3D\DAZStudio4 Public Build\DAZStudio.exe",
              "/Applications/DAZ 3D/DAZStudio4 64-bit/DAZStudio.app/Contents/MacOS/DAZStudio"):
        if os.path.isfile(p):
            return p
    return None


_GPU_SEEN: dict = {}


def llama_server_env() -> dict:
    """The environment llama-server runs in: a CUDA build of llama.cpp needs the CUDA runtime
    (cudart / cuBLAS) DLLs, which ComfyUI's own torch ships in torch/lib, so that folder and
    CUDA_PATH's bin go first on PATH."""
    env = dict(os.environ)
    dirs = []
    try:
        import torch
        dirs.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except Exception:
        pass
    if env.get("CUDA_PATH"):
        dirs.append(os.path.join(env["CUDA_PATH"], "bin"))
    env["PATH"] = os.pathsep.join([d for d in dirs if os.path.isdir(d)] + [env.get("PATH", "")])
    return env


def llama_server_has_gpu(exe: str) -> bool:
    """Whether a llama-server build sees a GPU (`--list-devices`); a CPU-only one is very slow."""
    if exe not in _GPU_SEEN:
        import subprocess
        try:
            out = subprocess.run([exe, "--list-devices"], capture_output=True, text=True, timeout=60,
                                 env=llama_server_env(),
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            devices = out.split("Available devices:", 1)[-1]
            _GPU_SEEN[exe] = bool(devices.strip()) and "(none)" not in devices
        except Exception:
            _GPU_SEEN[exe] = False
    return _GPU_SEEN[exe]


def find_llama_server(cfg=None) -> str | None:
    """llama.cpp's server: the config or SCENEREF_LLAMA_SERVER as given; else PATH, then
    Unsloth Studio's build, preferring one that sees a GPU."""
    cfg = cfg or load_config()
    for c in (cfg.get("llama_server"), os.environ.get("SCENEREF_LLAMA_SERVER")):
        if c and os.path.isfile(c):
            return c
    found = [c for c in (shutil.which("llama-server"),
                         os.path.expanduser(r"~\.unsloth\llama.cpp\build\bin\Release\llama-server.exe"),
                         os.path.expanduser("~/.unsloth/llama.cpp/build/bin/llama-server")) if c and os.path.isfile(c)]
    return next((c for c in found if llama_server_has_gpu(c)), found[0] if found else None)


def cache_dir() -> str:
    """input/3d/scene_ref - converted GLBs also show up in other 3D loaders' lists."""
    import folder_paths
    d = os.path.join(folder_paths.get_input_directory(), "3d", "scene_ref")
    os.makedirs(d, exist_ok=True)
    return d


def work_dir() -> str:
    """Scratch space for the intermediate Daz OBJ; ComfyUI empties temp on start."""
    import folder_paths
    d = os.path.join(folder_paths.get_temp_directory(), "scene_ref_work")
    os.makedirs(d, exist_ok=True)
    return d
