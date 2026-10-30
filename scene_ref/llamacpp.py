"""One answer from a GGUF vision model through llama.cpp's llama-server.

ComfyUI (and ComfyUI-GGUF) can't load Qwen3.5 GGUF files, such as GGUF builds of the FLUX.3
prompt upsampler, but llama.cpp can. For one question SceneRef frees ComfyUI's VRAM,
starts llama-server on a free local port with the model and its mmproj (the vision part),
asks through its OpenAI-style chat API, and stops the server again, so the image model gets
the VRAM back.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request

START_TIMEOUT_S = 600
ANSWER_TIMEOUT_S = 900


def mmproj_for(model_path: str) -> str | None:
    """The mmproj file next to a GGUF model: the one whose name shares the longest start with it."""
    folder, name = os.path.split(model_path)
    try:
        candidates = [f for f in os.listdir(folder) if f.lower().endswith(".gguf") and "mmproj" in f.lower()]
    except OSError:
        return None
    if not candidates:
        return None

    def shared(f):
        n = 0
        while n < min(len(f), len(name)) and f[n].lower() == name[n].lower():
            n += 1
        return n
    return os.path.join(folder, max(candidates, key=shared))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _png_url(picture) -> str:
    """A [1,H,W,3] or [H,W,3] 0-1 tensor as a data: URL."""
    import torch
    from PIL import Image
    a = picture[0] if picture.dim() == 4 else picture
    im = Image.fromarray((a.clamp(0, 1) * 255).round().to(dtype=torch.uint8).cpu().numpy())
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _get(url: str, timeout: float):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def ask(model_path: str, system: str, text: str, pictures: list, max_tokens: int, seed: int = 0) -> str:
    """Greedy answer (top_k 1, thinking off) to one user turn: the pictures in order, then the text."""
    from . import config
    server = config.find_llama_server()
    if not server:
        raise ValueError("SceneRef: a .gguf upsampler runs in llama.cpp, and llama-server was not found. Set "
                         "\"llama_server\" in scene_ref_config.json (or SCENEREF_LLAMA_SERVER) to llama-server.exe.")
    if not config.llama_server_has_gpu(server):
        print(f"[SceneRef] {server} sees no GPU: the upsampler runs on the CPU (minutes). Point \"llama_server\" "
              "in scene_ref_config.json at a CUDA build of llama.cpp.")
    mmproj = mmproj_for(model_path)
    if not mmproj:
        raise ValueError(f"SceneRef: no mmproj .gguf (the vision part) next to {os.path.basename(model_path)}; "
                         "it is needed to show the model the pictures.")
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass
    port = _free_port()
    import folder_paths
    log_path = os.path.join(folder_paths.get_temp_directory(), "scene_ref_llama_server.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [server, "-m", model_path, "--mmproj", mmproj, "--host", "127.0.0.1", "--port", str(port),
           "-ngl", "999", "-c", "16384", "--jinja", "-np", "1"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    print(f"[SceneRef] starting llama-server for {os.path.basename(model_path)} (log: {log_path})")
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
                                env=config.llama_server_env())
    try:
        base = f"http://127.0.0.1:{port}"
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server stopped while loading (exit {proc.returncode}); see {log_path}")
            try:
                if _get(base + "/health", 2)[0] == 200:
                    break
            except (urllib.error.URLError, OSError, ConnectionError):
                pass
            if time.time() - t0 > START_TIMEOUT_S:
                raise RuntimeError(f"llama-server did not get ready in {START_TIMEOUT_S} s; see {log_path}")
            time.sleep(1)
        content = [{"type": "image_url", "image_url": {"url": _png_url(p)}} for p in pictures]
        content.append({"type": "text", "text": text})
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "max_tokens": int(max_tokens), "temperature": 0.0, "top_k": 1, "seed": int(seed) % (2 ** 31),
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=ANSWER_TIMEOUT_S) as r:
                answer = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"llama-server answered {e.code}: {e.read()[:300]!r}") from None
        text = answer["choices"][0]["message"].get("content") or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    finally:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
