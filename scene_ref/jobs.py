"""Scene conversion jobs: .blend / .duf on disk -> cached GLB + sidecar JSON.

A job is keyed by the source file's path, size, mtime and the conversion
options, so opening the same unchanged scene again is instant, and editing it
in Blender / Daz makes the next open convert again.

    .blend  -> Blender (background)                      -> GLB
    .duf    -> Daz Studio (minimized) -> OBJ -> Blender   -> GLB
    .glb / .gltf are served as they are (no conversion).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from . import config, glb_fix

CONVERTER_VERSION = 10
SOURCE_EXTS = (".blend", ".duf", ".glb", ".gltf")
KEY_RE = re.compile(r"^[a-z0-9_-]{1,80}_[0-9a-f]{12}$")

_jobs: dict[str, "Job"] = {}
_lock = threading.Lock()
# One Daz and one Blender at a time: both are memory hungry, and a second Daz
# instance loading the same content library gains nothing.
_daz_slot = threading.Semaphore(1)
_blender_slot = threading.Semaphore(1)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return (s or "scene")[:60]


def source_key(path: str, cfg: dict) -> str:
    st = os.stat(path)
    ident = json.dumps([CONVERTER_VERSION, os.path.normcase(os.path.abspath(path)), st.st_size,
                        st.st_mtime_ns, int(cfg.get("max_texture") or 0), bool(cfg.get("draco"))])
    digest = hashlib.sha1(ident.encode("utf-8")).hexdigest()[:12]
    return "%s_%s" % (_slug(os.path.splitext(os.path.basename(path))[0]), digest)


def paths_for(key: str) -> tuple[str, str]:
    d = config.cache_dir()
    return os.path.join(d, key + ".glb"), os.path.join(d, key + ".json")


class BlenderCrash(RuntimeError):
    """Blender died (access violation etc.), as opposed to a Python error in the script."""


class Job:
    def __init__(self, key: str, source: str, cfg: dict):
        self.key = key
        self.source = source
        self.cfg = cfg
        self.state = "queued"       # queued | running | done | error | cancelled
        self.phase = "queued"
        self.message = ""
        self.log: list[str] = []
        self.started = time.time()
        self.finished = None
        self.proc: subprocess.Popen | None = None
        self.cancelled = False

    def note(self, phase: str, message: str = ""):
        self.phase = phase
        self.message = message
        line = "%6.1fs  %s %s" % (time.time() - self.started, phase, message)
        self.log.append(line.rstrip())
        del self.log[:-200]
        print("[SceneRef] %s: %s" % (self.key, line.strip()))

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "source": self.source,
            "state": self.state,
            "phase": self.phase,
            "message": self.message,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
            "log": self.log[-12:],
        }


def _creation_flags():
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def _run(job: Job, cmd: list[str], timeout: float, on_line=None, minimized=False, cwd=None) -> int:
    kwargs = {}
    if sys.platform == "win32":
        if minimized:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 7  # SW_SHOWMINNOACTIVE: never steal focus from the browser
            kwargs["startupinfo"] = si
        else:
            kwargs["creationflags"] = _creation_flags()
    job.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd,
                                stdin=subprocess.DEVNULL, **kwargs)
    deadline = time.time() + timeout

    def pump():
        for raw in iter(job.proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if on_line:
                on_line(line)

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    while job.proc.poll() is None:
        if job.cancelled:
            job.proc.kill()
            raise RuntimeError("cancelled")
        if time.time() > deadline:
            job.proc.kill()
            raise RuntimeError("timed out after %d s" % timeout)
        time.sleep(0.5)
        _tick(job)
    t.join(timeout=5)
    return job.proc.returncode


_tickers: dict[str, callable] = {}


def _tick(job: Job):
    f = _tickers.get(job.key)
    if f:
        try:
            f()
        except Exception:
            pass


def _blender(job: Job, args: dict, blend_file: str | None):
    exe = config.find_blender(job.cfg)
    if not exe:
        raise RuntimeError("Blender was not found. Set \"blender\" in scene_ref_config.json "
                           "to the full path of blender.exe.")
    script = os.path.join(config.CONVERTERS_DIR, "blender_convert.py")
    cmd = [exe, "-b", "--factory-startup"]
    if blend_file:
        cmd.append(blend_file)
    cmd += ["--python", script, "--", json.dumps(args)]
    tail: list[str] = []

    def on_line(line):
        tail.append(line)
        del tail[:-40]
        if line.startswith("SCENEREF "):
            parts = line.split(" ", 3)
            job.note("blender:" + parts[1], parts[3] if len(parts) > 3 else "")

    job.note("blender", os.path.basename(exe) + " " + os.path.dirname(exe))
    with _blender_slot:
        code = _run(job, cmd, float(job.cfg.get("blender_timeout_s") or 3600), on_line)
    if code != 0 or not os.path.isfile(args["out_glb"]):
        detail = "\n".join(l for l in tail if l.strip() and not l.startswith("DracoEncoder"))[-1500:]
        # The script exits 1 on its own Python errors; anything else is Blender crashing.
        kind = RuntimeError if code in (0, 1) else BlenderCrash
        raise kind("Blender failed (exit %s).\n%s" % (code, detail))


def _daz(job: Job, out_dir: str):
    exe = config.find_daz(job.cfg)
    if not exe:
        raise RuntimeError("Daz Studio was not found. Set \"daz\" in scene_ref_config.json "
                           "to the full path of DAZStudio.exe.")
    script = os.path.join(config.CONVERTERS_DIR, "daz_export.dsa").replace("\\", "/")
    status_path = os.path.join(out_dir, "daz_status.json")
    cmd = [exe, "-noPrompt", "-instanceName", "#"]
    if job.cfg.get("daz_headless"):
        cmd.append("-headless")
    cmd += ["-scriptArg", job.source.replace("\\", "/"), "-scriptArg", out_dir.replace("\\", "/"), script]

    last = {"state": None}

    def tick():
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                st = json.load(f)
        except Exception:
            return
        if st.get("state") != last["state"]:
            last["state"] = st.get("state")
            job.note("daz:" + str(st.get("state")), "")

    _tickers[job.key] = tick
    job.note("daz", "starting Daz Studio (minimized) - loading a big scene takes minutes")
    try:
        with _daz_slot:
            code = _run(job, cmd, float(job.cfg.get("daz_timeout_s") or 3600), None, minimized=True)
    finally:
        _tickers.pop(job.key, None)
    tick()
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            st = json.load(f)
    except Exception:
        st = {"state": "unknown", "message": "Daz Studio wrote no status (exit %s)" % code}
    obj = os.path.join(out_dir, "scene.obj")
    if st.get("state") != "ok" or not os.path.isfile(obj):
        raise RuntimeError("Daz Studio export failed: %s %s (exit %s). If Daz crashed while loading, "
                           "open the scene in Daz once to check for missing content."
                           % (st.get("state"), st.get("message", ""), code))
    return obj


def _convert(job: Job):
    glb, info = paths_for(job.key)
    # Blender's glTF exporter forces a ".glb" ending, so the temp name keeps it.
    tmp_glb, tmp_info = glb[:-4] + ".part.glb", info[:-5] + ".part.json"
    ext = os.path.splitext(job.source)[1].lower()
    common = {"out_glb": tmp_glb, "out_info": tmp_info,
              "max_tex": int(job.cfg.get("max_texture") or 0), "draco": bool(job.cfg.get("draco", True))}
    if ext == ".blend":
        try:
            _blender(job, dict(common, mode="blend"), job.source)
        except BlenderCrash:
            if job.cancelled:
                raise
            # A modifier that crashes Blender while being applied must not cost the
            # whole scene: export the base meshes instead (modifiers left out).
            job.note("blender:retry", "first attempt failed; retrying without applying modifiers")
            _blender(job, dict(common, mode="blend", safe=True), job.source)
    elif ext == ".duf":
        work = os.path.join(config.work_dir(), job.key)
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work, exist_ok=True)
        try:
            obj = _daz(job, work)
            size_mb = os.path.getsize(obj) / 1e6
            job.note("daz:exported", "%.0f MB OBJ" % size_mb)
            _blender(job, dict(common, mode="obj", obj=obj,
                               daz_info=os.path.join(work, "scene_info.json")), None)
        finally:
            if not job.cfg.get("keep_daz_obj"):
                shutil.rmtree(work, ignore_errors=True)
    else:
        raise RuntimeError("unsupported file type %s" % ext)
    dropped = glb_fix.sanitize_glb(tmp_glb)
    if dropped:
        job.note("fixup", "%d texture(s) without an image were dropped" % dropped)
    os.replace(tmp_glb, glb)
    os.replace(tmp_info, info)


def _worker(job: Job):
    job.state = "running"
    try:
        _convert(job)
        job.state = "done"
        job.note("done", "")
    except Exception as e:
        job.state = "cancelled" if job.cancelled else "error"
        job.note("error", str(e))
    finally:
        job.finished = time.time()
        job.proc = None


def _passthrough(path: str, key: str):
    """A GLB / glTF is used as it is: link it into the cache under its key."""
    glb, info = paths_for(key)
    if not os.path.isfile(glb):
        shutil.copyfile(path, glb)
    if not os.path.isfile(info):
        with open(info, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "mode": "glb", "up": "Y", "cameras": [], "viewport": None,
                       "bounds": None, "stats": {}}, f)


def open_scene(path: str, force: bool = False) -> dict:
    """Start (or join, or skip) the conversion of `path`; returns the job state."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("no scene path given")
    given = os.path.expanduser(path.strip().strip('"'))
    path = os.path.abspath(given)
    if not os.path.isabs(given) and not os.path.isfile(path):
        # A relative path that isn't there from ComfyUI's folder: the pack's own (examples/living_room.blend).
        path = os.path.abspath(os.path.join(config.PACK_DIR, given))
    if path.startswith("\\\\"):
        raise ValueError("network (UNC) paths are not supported; map the share to a drive letter")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SOURCE_EXTS:
        raise ValueError("pick a .blend, .duf, .glb or .gltf file")
    if not os.path.isfile(path):
        raise ValueError("file not found: %s" % path)
    cfg = config.load_config()
    key = source_key(path, cfg)
    glb, info = paths_for(key)
    with _lock:
        job = _jobs.get(key)
        if job and job.state in ("queued", "running"):
            return job.to_dict()
        if ext == ".gltf":
            raise ValueError(".gltf with separate files is not supported; export a .glb instead")
        if ext == ".glb":
            _passthrough(path, key)
        if not force and os.path.isfile(glb) and os.path.isfile(info):
            done = Job(key, path, cfg)
            done.state, done.phase, done.finished = "done", "cached", time.time()
            _jobs[key] = done
            return done.to_dict()
        job = Job(key, path, cfg)
        _jobs[key] = job
    threading.Thread(target=_worker, args=(job,), daemon=True, name="SceneRef-" + key).start()
    return job.to_dict()


def job_status(key: str) -> dict | None:
    job = _jobs.get(key)
    if job:
        return job.to_dict()
    glb, info = paths_for(key)
    if os.path.isfile(glb) and os.path.isfile(info):
        return {"key": key, "state": "done", "phase": "cached", "message": "", "elapsed": 0, "log": []}
    return None


def cancel(key: str) -> bool:
    job = _jobs.get(key)
    if not job or job.state not in ("queued", "running"):
        return False
    job.cancelled = True
    if job.proc and job.proc.poll() is None:
        job.proc.kill()
    return True


def tools_status() -> dict:
    cfg = config.load_config()
    return {"blender": config.find_blender(cfg), "daz": config.find_daz(cfg),
            "max_texture": cfg.get("max_texture"), "config": config.CONFIG_PATH}
