"""HTTP routes the viewer talks to.

    POST /scene_ref/open            {"path": "...", "force": false} -> job state
    GET  /scene_ref/status/{key}                                     -> job state
    POST /scene_ref/cancel/{key}
    GET  /scene_ref/model/{key}.glb | .json                          -> converted scene
    GET  /scene_ref/browse?path=...                                  -> folders + scene files
    GET  /scene_ref/tools                                            -> detected Blender / Daz
    GET  /scene_ref/vendor/...                                       -> three.js (not auto-loaded)
    GET  /scene_ref/studio_picks                                     -> the files 'auto' picks
    POST /scene_ref/expand          {capture, action, backend, ...}   -> {"text"} (Studio prompt)
    POST /scene_ref/keep            {files, prefix}                   -> previews copied to output
    POST /scene_ref/cast_names      {cast_json}                       -> each row's name (or null)
    GET  /scene_ref/char_text?source=...                             -> text stored in a .char / RefMod
    POST /scene_ref/char_check        {row}                          -> pictures used, notes, voice
    GET  /scene_ref/loras                                            -> LoRAs with trigger words
    GET  /scene_ref/poses?near=<scene>                               -> pose library + saved poses
    GET  /scene_ref/poses/folder?path=...                            -> pose files in a folder
    GET  /scene_ref/poses/get?id=...  | /poses/thumb?id=...          -> one pose's points / preview
    POST /scene_ref/poses/parse       {name, data}                   -> an OpenPose JSON as a pose
    POST /scene_ref/poses/save        {name, bones, anchor, png}     -> saved to models/poses
"""
from __future__ import annotations

import asyncio
import json
import os
import string
import sys

from aiohttp import web
from server import PromptServer

from . import config, jobs

routes = PromptServer.instance.routes

BROWSE_EXTS = (".blend", ".duf", ".glb")
MAX_ENTRIES = 4000


def _bad(msg, status=400):
    return web.json_response({"error": msg}, status=status)


@routes.post("/scene_ref/open")
async def open_scene(request):
    try:
        body = await request.json()
    except Exception:
        return _bad("expected a JSON body")
    try:
        return web.json_response(jobs.open_scene(body.get("path", ""), bool(body.get("force"))))
    except ValueError as e:
        return _bad(str(e))
    except OSError as e:
        return _bad("cannot read the file: %s" % e)


@routes.get("/scene_ref/status/{key}")
async def status(request):
    key = request.match_info["key"]
    if not jobs.KEY_RE.match(key):
        return _bad("bad key")
    st = jobs.job_status(key)
    return web.json_response(st) if st else _bad("unknown scene", 404)


@routes.post("/scene_ref/cancel/{key}")
async def cancel(request):
    key = request.match_info["key"]
    if not jobs.KEY_RE.match(key):
        return _bad("bad key")
    return web.json_response({"cancelled": jobs.cancel(key)})


@routes.get("/scene_ref/model/{name}")
async def model(request):
    name = request.match_info["name"]
    base, ext = os.path.splitext(name)
    if ext not in (".glb", ".json") or not jobs.KEY_RE.match(base):
        return _bad("bad name")
    glb, info = jobs.paths_for(base)
    path = glb if ext == ".glb" else info
    if not os.path.isfile(path):
        return _bad("not converted yet", 404)
    resp = web.FileResponse(path)
    resp.headers["Cache-Control"] = "no-cache"  # revalidate (cheap 304): a cached file can be repaired in place
    return resp


def _drives():
    if sys.platform != "win32":
        return ["/"]
    return ["%s:\\" % d for d in string.ascii_uppercase if os.path.isdir("%s:\\" % d)]


@routes.get("/scene_ref/browse")
async def browse(request):
    """Lists folders and scene files so the node can pick a file anywhere on disk.

    This exposes directory NAMES (never contents of other files) of the machine
    running ComfyUI; like ComfyUI itself, keep the server on localhost.
    """
    path = request.query.get("path", "").strip()
    if not path:
        return web.json_response({"path": "", "parent": None, "dirs": _drives(), "files": []})
    if path.startswith("\\\\") or path.startswith("//"):
        return _bad("network paths are not supported")
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return _bad("not a folder: %s" % path)
    dirs, files = [], []
    try:
        with os.scandir(path) as it:
            for i, e in enumerate(it):
                if i > MAX_ENTRIES:
                    break
                try:
                    if e.is_dir(follow_symlinks=False):
                        if not e.name.startswith(("$", ".")):
                            dirs.append(e.name)
                    elif e.name.lower().endswith(BROWSE_EXTS):
                        st = e.stat()
                        files.append({"name": e.name, "size": st.st_size, "mtime": st.st_mtime})
                except OSError:
                    continue
    except OSError as e:
        return _bad(str(e))
    parent = os.path.dirname(path.rstrip("\\/"))
    if parent == path.rstrip("\\/") or not parent:
        parent = ""
    elif sys.platform == "win32" and len(parent) == 2:
        parent += "\\"
    dirs.sort(key=str.lower)
    files.sort(key=lambda f: f["name"].lower())
    return web.json_response({"path": path, "parent": parent, "dirs": dirs, "files": files})


@routes.get("/scene_ref/tools")
async def tools(request):
    return web.json_response(jobs.tools_status())


@routes.get("/scene_ref/vendor/{tail:.+}")
async def vendor(request):
    tail = request.match_info["tail"]
    root = os.path.realpath(config.VENDOR_DIR)
    path = os.path.realpath(os.path.join(root, tail))
    if not path.startswith(root + os.sep) or not os.path.isfile(path):
        return _bad("not found", 404)
    resp = web.FileResponse(path)
    if path.endswith(".wasm"):
        resp.content_type = "application/wasm"
    elif path.endswith(".js"):
        resp.content_type = "text/javascript"
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@routes.get("/scene_ref/studio_picks")
async def studio_picks(request):
    """The file 'auto' picks for each Studio model widget, so the panel shows real names."""
    from .studio import resolved_picks
    return web.json_response(resolved_picks())


_expanding = asyncio.Lock()


def _shot_from_capture(capture, width, height):
    """The first captured view as the small shot dict the expander reads (composite + figures)."""
    from .nodes import _capture_path, _load_rgb
    shots = capture.get("shots") if isinstance(capture, dict) else None
    first = shots[0] if isinstance(shots, list) and shots and isinstance(shots[0], dict) else {}
    path = _capture_path(first.get("composite")) or _capture_path(first.get("image"))
    if not path:
        raise ValueError("the viewer has no picture yet: open a scene in it first")
    cam = first.get("camera") if isinstance(first.get("camera"), dict) else {}
    figs = cam.get("figures") if isinstance(cam.get("figures"), list) else []
    size = (int(width), int(height))
    return {"composite": _load_rgb(path, size)[None], "figures": [figs], "width": size[0], "height": size[1]}


@routes.post("/scene_ref/expand")
async def expand_text(request):
    """The Studio's 'Write the prompt' button: the vision model writes the whole prompt now,
    so it can be read and edited before Run."""
    try:
        body = await request.json()
    except Exception:
        return _bad("expected a JSON body")
    if PromptServer.instance.prompt_queue.get_tasks_remaining() > 0:
        return _bad("ComfyUI is busy running a workflow: try again when the queue is empty", 409)
    if _expanding.locked():
        return _bad("already writing a description", 409)
    from .characters import cast_from_json
    from .expander import write_prompt
    from .loras import parse_stack
    from .studio import ASPECTS, BACKENDS

    def number(key, default):
        try:
            return float(body.get(key, default))
        except (TypeError, ValueError):
            return default

    def work():
        import comfy.utils
        backend = body.get("backend")
        movie = bool(body.get("movie"))
        if not movie and backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}")
        if body.get("scene") is not None:  # a SceneRef Scene (pictures) instead of the viewer
            from .scenes import shot_from_request
            shot = shot_from_request(body["scene"])
        else:
            shot = _shot_from_capture(body.get("capture"), body.get("width") or 1024, body.get("height") or 1024)
        cast = cast_from_json(body.get("cast_json") or "[]")
        triggers = ", ".join(x["trigger"] for x in parse_stack(body.get("loras_json") or "[]") if x.get("trigger"))
        # ComfyUI's progress hook reports to the running prompt; outside one it fails (no
        # last_prompt_id before the first run), so it is off while the model writes.
        hook = comfy.utils.PROGRESS_BAR_HOOK
        comfy.utils.set_progress_bar_global_hook(None)
        try:
            return write_prompt(str(body.get("action") or ""), shot, cast, backend=backend, movie=movie,
                                room_details=bool(body.get("room_details", True)),
                                text_encoder=str(body.get("expander_model") or "auto"), triggers=triggers,
                                control_strength=number("control_strength", 0.75),
                                image_aspect=ASPECTS.get(body.get("image_aspect")),
                                movie_aspect=ASPECTS.get(body.get("aspect_ratio")),
                                video_seconds=number("video_seconds", 5.0),
                                upsampler=str(body.get("klein3_upsampler") or "auto"))
        finally:
            comfy.utils.set_progress_bar_global_hook(hook)
    async with _expanding:
        try:
            text = await asyncio.get_running_loop().run_in_executor(None, work)
        except ValueError as e:
            return _bad(str(e))
        except Exception as e:  # model loading / generation errors go back to the panel
            return _bad(f"{type(e).__name__}: {e}", 500)
    return web.json_response({"text": text})


@routes.post("/scene_ref/cast_names")
async def cast_names(request):
    """The Studio's 'In the prompt' line: the name each cast row goes by (@1, @2 ...)."""
    try:
        body = await request.json()
    except Exception:
        return _bad("expected a JSON body")
    from .characters import cast_names as names
    return web.json_response({"names": names(body.get("cast_json") or "[]")})


@routes.post("/scene_ref/keep")
async def keep_preview(request):
    """Preview & Save in manual mode: copy the shown previews into ComfyUI/output."""
    try:
        body = await request.json()
    except Exception:
        return _bad("expected a JSON body")
    from .outputs import keep
    try:
        return web.json_response({"saved": keep(body.get("files") or [], str(body.get("prefix") or ""))})
    except (ValueError, OSError) as e:
        return _bad(str(e))


@routes.get("/scene_ref/char_text")
async def char_text(request):
    """The description stored in a .char / RefMod (cast panel: 'also use the text stored in the file')."""
    from .characters import stored_text
    try:
        return web.json_response({"text": stored_text(request.query.get("source", ""))})
    except (ValueError, OSError, KeyError) as e:
        return _bad(str(e))


@routes.post("/scene_ref/char_check")
async def char_check(request):
    """One cast row read the way the Run reads it: the pictures used and what looks wrong."""
    from .characters import cast_from_json, character_report
    try:
        row = (await request.json()).get("row")
    except Exception:
        return _bad("expected a JSON body")
    if not isinstance(row, dict):
        return _bad("expected {row: {...}}")

    def work():
        cast = cast_from_json(json.dumps([{**row, "on": True}]))
        return character_report(cast[0]) if cast else "Nobody: pick a .char / RefMod or add a picture."
    try:
        report = await asyncio.get_running_loop().run_in_executor(None, work)
    except (ValueError, OSError, KeyError) as e:
        return _bad(str(e))
    return web.json_response({"report": report})


@routes.get("/scene_ref/loras")
async def loras(request):
    """Every LoRA with its folder and trigger words, for the LoRA Stack list."""
    from . import loras as lora_mod
    return web.json_response(lora_mod.list_loras())


# ── poses for the viewer's mannequins (scene_ref/poses.py) ──────────────────────────
@routes.get("/scene_ref/poses")
async def poses_index(request):
    """The bundled library (with its points, so the browser draws the previews), your saved
    poses, and where the file tab starts (the open scene's Daz content library)."""
    from . import poses
    return web.json_response({"library": poses.library(), "mine": poses.list_user(),
                              "folder": poses.default_folder(request.query.get("near") or None),
                              "user_dir": poses.user_dir()})


@routes.get("/scene_ref/poses/folder")
async def poses_folder(request):
    from . import poses
    path = request.query.get("path", "").strip()
    if not path:
        return web.json_response({"path": "", "parent": None, "dirs": _drives(), "poses": []})
    if path.startswith(("\\\\", "//")):
        return _bad("network paths are not supported")
    try:
        return web.json_response(poses.folder(path))
    except (ValueError, OSError) as e:
        return _bad(str(e))


@routes.get("/scene_ref/poses/get")
async def poses_get(request):
    from . import poses
    try:
        data = await asyncio.get_running_loop().run_in_executor(None, poses.get, request.query.get("id", ""))
        return web.json_response(data)
    except KeyError:
        return _bad("no such pose", 404)
    except (ValueError, OSError) as e:
        return _bad(str(e))
    except Exception as e:  # a file we cannot make sense of
        return _bad(f"cannot read this pose: {type(e).__name__}: {e}")


@routes.get("/scene_ref/poses/thumb")
async def poses_thumb(request):
    from . import poses
    try:
        p = poses.thumb_path(request.query.get("id", ""))
    except ValueError as e:
        return _bad(str(e))
    if not p:
        return _bad("no preview", 404)
    resp = web.FileResponse(p)
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


@routes.post("/scene_ref/poses/parse")
async def poses_parse(request):
    """An OpenPose (or saved pose) JSON dropped on the pose browser, read the same way as a file."""
    from . import poses
    try:
        body = await request.json()
        return web.json_response(poses.from_json(body.get("data"), str(body.get("name") or "OpenPose")[:80]))
    except (ValueError, TypeError, KeyError) as e:
        return _bad(str(e))


@routes.post("/scene_ref/poses/save")
async def poses_save(request):
    """Save a mannequin's joint angles (and its preview) to models/poses."""
    import base64
    from . import poses
    try:
        body = await request.json()
        png = body.get("png") or ""
        raw = base64.b64decode(png.split(",", 1)[-1]) if png else None
        if raw and raw[:8] != b"\x89PNG\r\n\x1a\n":
            raw = None
        pid = poses.save_user(str(body.get("name") or ""), body.get("bones") or {}, str(body.get("anchor") or ""), raw)
        return web.json_response({"id": pid, "mine": poses.list_user()})
    except (ValueError, TypeError, OSError) as e:
        return _bad(str(e))
