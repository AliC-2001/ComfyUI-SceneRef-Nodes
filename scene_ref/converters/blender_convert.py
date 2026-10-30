"""SceneRef - runs INSIDE Blender (background mode) and writes a browser-ready GLB.

    blender -b [scene.blend] --factory-startup? --python blender_convert.py -- '<json args>'

Args (one JSON object after "--"):
    mode        "blend"  the .blend given to Blender on the command line is exported as is
                "obj"    an OBJ (Daz export) is imported into an empty scene first
    obj         OBJ path (mode "obj")
    daz_info    scene_info.json written by daz_export.dsa (mode "obj", optional)
    out_glb     where the GLB goes
    out_info    where the sidecar JSON (cameras, bounds, stats) goes
    max_tex     longest texture side in the GLB (0 = keep)
    draco       true to Draco-compress the meshes

Progress lines start with "SCENEREF " so the ComfyUI side can follow along.
"""
import json
import math
import os
import sys
import time

import bpy
from mathutils import Matrix, Quaternion, Vector

T0 = time.time()


def log(phase, msg=""):
    print("SCENEREF %s %.1fs %s" % (phase, time.time() - T0, msg), flush=True)


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("blender_convert.py: missing '-- <json>'")
    return json.loads(argv[argv.index("--") + 1])


# Blender is Z-up, glTF (and the viewer, and the Daz OBJ) are Y-up.
# glTF exporter mapping: (x, y, z)_blender -> (x, z, -y)_gltf
def to_gltf_pos(v):
    return [v.x, v.z, -v.y]


def to_gltf_quat(q):
    # World basis change only: R_gltf = C * R, C = rot -90 deg about X. A camera's
    # local frame (look down -Z, +Y up) is the same in Blender and three.js, so no
    # C^-1 on the right (that tipped every camera 90 deg to look at the sky).
    c = Quaternion((1, 0, 0), -math.pi / 2)
    r = c @ q
    return [r.x, r.y, r.z, r.w]


def camera_entry(name, matrix_world, cam_data, render):
    loc, rot, _scale = matrix_world.decompose()
    entry = {"name": name, "position": to_gltf_pos(loc), "rotation": to_gltf_quat(rot)}
    if cam_data is not None:
        entry["type"] = cam_data.type
        if cam_data.type == "ORTHO":
            entry["ortho_scale"] = cam_data.ortho_scale
        else:
            # Vertical FOV for three.js, from the sensor fit Blender actually uses.
            aspect = render.resolution_x / max(1, render.resolution_y)
            fit = cam_data.sensor_fit
            if fit == "AUTO":
                fit = "HORIZONTAL" if aspect >= 1 else "VERTICAL"
            if fit == "VERTICAL":
                vfov = 2 * math.atan(cam_data.sensor_height / (2 * cam_data.lens))
            else:
                hfov = 2 * math.atan(cam_data.sensor_width / (2 * cam_data.lens))
                vfov = 2 * math.atan(math.tan(hfov / 2) / aspect)
            entry["fov"] = math.degrees(vfov)
            entry["aspect"] = aspect
            entry["lens"] = cam_data.lens
        entry["near"] = cam_data.clip_start
        entry["far"] = cam_data.clip_end
    return entry


def blend_cameras():
    scene = bpy.context.scene
    render = scene.render
    cams = []
    for ob in scene.objects:
        if ob.type == "CAMERA":
            e = camera_entry(ob.name, ob.matrix_world, ob.data, render)
            e["active"] = ob == scene.camera
            cams.append(e)
    cams.sort(key=lambda c: (not c.get("active"), c["name"].lower()))
    viewport = None
    # The 3D view the file was saved with, when the UI was stored in it.
    try:
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type != "VIEW_3D":
                    continue
                r3d = area.spaces.active.region_3d
                if r3d is None:
                    continue
                if r3d.view_perspective == "CAMERA":
                    continue
                inv = r3d.view_matrix.inverted()
                lens = area.spaces.active.lens
                viewport = camera_entry("Viewport (" + screen.name + ")", inv, None, render)
                # The viewport lens is on a 72 mm "sensor" diagonal-ish convention; good enough
                # as a starting FOV.
                viewport["fov"] = math.degrees(2 * math.atan(36.0 / (2 * lens)))
                raise StopIteration
    except StopIteration:
        pass
    except Exception as e:  # never fail the export over a convenience
        log("warn", "viewport camera: %s" % e)
    return cams, viewport


def daz_cameras(info_path):
    """Daz cameras are already Y-up metres (daz_export.dsa scaled them)."""
    if not info_path or not os.path.isfile(info_path):
        return [], None
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    cams = []
    for c in info.get("cameras", []):
        if "fov" not in c:
            continue  # light blockers and other camera-like helpers
        e = {"name": c["name"], "position": c["position"], "rotation": c["rotation"]}
        # Daz getFieldOfView() is in radians (vertical).
        e["fov"] = math.degrees(c["fov"]) if c["fov"] < math.pi else c["fov"]
        cams.append(e)
    vp = info.get("viewport")
    if vp:
        vp = {"name": "Viewport (" + vp.get("name", "Daz") + ")", "position": vp["position"],
              "rotation": vp["rotation"], "fov": math.degrees(vp["fov"]) if vp.get("fov") else 40}
    return cams, vp


def daz_figures(info_path):
    """Import each figure OBJ that daz_export.dsa wrote next to scene_info.json as
    one object named SRFIG_<n> (the viewer finds figures by that name)."""
    if not info_path or not os.path.isfile(info_path):
        return []
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    out = []
    base = os.path.dirname(info_path)
    for fig in info.get("figures", []):
        path = os.path.join(base, fig.get("file", ""))
        if not os.path.isfile(path):
            continue
        before = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=path, forward_axis="NEGATIVE_Z", up_axis="Y",
                              use_split_objects=False, use_split_groups=False)
        new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
        if not new:
            continue
        if len(new) > 1:
            with bpy.context.temp_override(active_object=new[0], selected_editable_objects=new):
                bpy.ops.object.join()
        ob = new[0]
        name = "SRFIG_%d" % len(out)
        ob.name = name
        ob.data.name = name
        out.append({"mesh": name, "label": fig.get("label") or fig.get("name") or name,
                    "joints": fig.get("joints") or {}})
        log("figure", "%s: %s" % (name, out[-1]["label"]))
    return out


def rebuild_images(max_side):
    """Every texture -> a fresh in-memory image, downscaled to max_side.

    Two reasons. Browsers cannot hold hundreds of 4K maps. And Blender 5.1's
    WEBP writer fails on the exporter's temporary copy of a PACKED image
    ("Could not write image: internal error"), which silently dropped those
    textures from the GLB. A generated image with its pixels set saves fine.
    """
    import numpy as np
    done = 0
    for img in list(bpy.data.images):
        if img.source not in {"FILE"} or img.type != "IMAGE" or img.users == 0:
            continue
        try:
            w, h = img.size
        except Exception:
            continue
        if not w or not h:
            continue  # missing file: nothing to rebuild
        try:
            if max_side and max(w, h) > max_side:
                s = max_side / float(max(w, h))
                img.scale(max(1, int(w * s)), max(1, int(h * s)))
                w, h = img.size
            buf = np.empty(w * h * 4, np.float32)
            img.pixels.foreach_get(buf)
            new = bpy.data.images.new(img.name + "_sr", w, h, alpha=True, float_buffer=False)
            new.colorspace_settings.name = img.colorspace_settings.name
            new.alpha_mode = img.alpha_mode
            new.pixels.foreach_set(buf)
            img.user_remap(new)
            done += 1
        except Exception as e:
            log("warn", "could not rebuild %s: %s" % (img.name, e))
    return done


def replace_auto_smooth():
    """Blender 4.1+ turns a pre-4.1 file's auto smooth into an "Auto Smooth" Geometry
    Nodes modifier. Evaluating its Edge Angle field crashes Blender 5.1.1 on some
    meshes (EXCEPTION_ACCESS_VIOLATION in AngleFieldInput, single-threaded too).
    The same shading comes from marking sharp edges directly on the mesh.
    """
    replaced = 0
    for ob in bpy.data.objects:
        for m in list(getattr(ob, "modifiers", [])):
            if not (m.type == "NODES" and m.node_group
                    and m.node_group.name.startswith(("Auto Smooth", "Smooth by Angle"))):
                continue
            angle = math.radians(30)
            for item in m.node_group.interface.items_tree:
                if getattr(item, "in_out", None) == "INPUT" and item.name == "Angle":
                    try:
                        angle = float(m[item.identifier])
                    except Exception:
                        pass
            if ob.type == "MESH":
                try:
                    ob.data.set_sharp_from_angle(angle=angle)
                except Exception as e:
                    log("warn", "sharp from angle %s: %s" % (ob.name, e))
            ob.modifiers.remove(m)
            replaced += 1
    return replaced


def drop_hidden():
    """Objects that would not render must not end up in the GLB."""
    removed = 0
    for ob in list(bpy.context.scene.objects):
        if ob.hide_render or ob.hide_get():
            if ob.type in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
                bpy.data.objects.remove(ob, do_unlink=True)
                removed += 1
    return removed


def dedupe_obj_materials():
    """A Daz OBJ repeats one material per surface per object; merge identical ones.

    Fewer materials means fewer draw calls in the browser, which is what makes a
    four-million-triangle apartment move smoothly.
    """
    def key(mat):
        if not mat or not mat.use_nodes:
            return None
        parts = []
        for n in mat.node_tree.nodes:
            if n.type == "TEX_IMAGE" and n.image:
                parts.append(("img", n.image.filepath, n.outputs[0].links[0].to_socket.name if n.outputs[0].links else ""))
            elif n.type == "BSDF_PRINCIPLED":
                bc = n.inputs["Base Color"].default_value
                parts.append(("bc", tuple(round(x, 3) for x in bc)))
                parts.append(("a", round(n.inputs["Alpha"].default_value, 3)))
        return tuple(sorted(parts))

    canon = {}
    remap = {}
    for mat in bpy.data.materials:
        k = key(mat)
        if k is None:
            continue
        if k in canon:
            remap[mat.name] = canon[k]
        else:
            canon[k] = mat
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH":
            continue
        for slot in ob.material_slots:
            if slot.material and slot.material.name in remap:
                slot.material = remap[slot.material.name]
    # Collapse now-duplicate slots so the exporter makes one primitive per material.
    for ob in bpy.context.scene.objects:
        if ob.type == "MESH" and len(ob.material_slots) > 1:
            bpy.context.view_layer.objects.active = ob
            ob.select_set(True)
            try:
                bpy.ops.object.material_slot_remove_unused()
            except Exception:
                pass
    return len(remap)


def snap_opaque():
    """Daz writes "d 0.995" for most Iray surfaces; anything that close to 1 with
    no opacity map is opaque. Left alone, glTF marks it BLEND and the browser
    sorts and blends nearly the whole scene as glass."""
    n = 0
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            a = node.inputs["Alpha"]
            if not a.is_linked and 0.98 <= a.default_value < 1.0:
                a.default_value = 1.0
                n += 1
    return n


def scene_bounds():
    lo = Vector((1e30, 1e30, 1e30))
    hi = Vector((-1e30, -1e30, -1e30))
    any_mesh = False
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH":
            continue
        for c in ob.bound_box:
            w = ob.matrix_world @ Vector(c)
            lo = Vector((min(lo.x, w.x), min(lo.y, w.y), min(lo.z, w.z)))
            hi = Vector((max(hi.x, w.x), max(hi.y, w.y), max(hi.z, w.z)))
            any_mesh = True
    if not any_mesh:
        return None
    a, b = to_gltf_pos(lo), to_gltf_pos(hi)
    return {"min": [min(a[i], b[i]) for i in range(3)], "max": [max(a[i], b[i]) for i in range(3)]}


def export_glb(path, draco, safe=False):
    op = bpy.ops.export_scene.gltf
    props = {p.identifier for p in op.get_rna_type().properties}
    want = {
        "filepath": path,
        "export_format": "GLB",
        "use_visible": True,
        "use_renderable": True,
        "export_apply": not safe,
        "export_cameras": True,
        "export_lights": True,
        "export_yup": True,
        "export_texcoords": True,
        "export_normals": True,
        "export_materials": "EXPORT",
        "export_image_format": "WEBP",
        "export_image_quality": 82,
        "export_jpeg_quality": 82,
        "export_animations": False,
        "export_skins": False,
        "export_morph": False,
        "export_extras": False,
        "export_draco_mesh_compression_enable": bool(draco),
        "export_draco_mesh_compression_level": 6,
        # 20 bits: one Draco range spans the whole merged mesh, sky domes and
        # backdrops included; 16 bits over 1.7 km is 2.6 cm of error on props.
        "export_draco_position_quantization": 20,
        "export_draco_normal_quantization": 10,
        "export_draco_texcoord_quantization": 12,
    }
    kwargs = {k: v for k, v in want.items() if k in props}
    log("export", "options=%s" % sorted(kwargs))
    op(**kwargs)


def main():
    a = parse_args()
    mode = a["mode"]
    out_glb = a["out_glb"]
    out_info = a["out_info"]
    stats = {}

    if mode == "obj":
        bpy.ops.wm.read_factory_settings(use_empty=True)
        log("import", a["obj"])
        bpy.ops.wm.obj_import(filepath=a["obj"], forward_axis="NEGATIVE_Z", up_axis="Y",
                              use_split_objects=False, use_split_groups=False)
        log("imported", "%d objects" % len(bpy.context.scene.objects))
        figures = daz_figures(a.get("daz_info"))
        stats["figures"] = len(figures)
        stats["opaque_snapped"] = snap_opaque()
        stats["merged_materials"] = dedupe_obj_materials()
        cams, viewport = daz_cameras(a.get("daz_info"))
    else:
        log("opened", bpy.data.filepath)
        # Make every object selectable / exportable state predictable.
        stats["dropped_hidden"] = drop_hidden()
        stats["auto_smooth_replaced"] = replace_auto_smooth()
        cams, viewport = blend_cameras()
        figures = []

    stats["textures"] = rebuild_images(int(a.get("max_tex") or 0))
    log("textures", "rebuilt %d" % stats["textures"])

    tris = 0
    for ob in bpy.context.scene.objects:
        if ob.type == "MESH":
            me = ob.data
            tris += sum(len(p.vertices) - 2 for p in me.polygons)
    stats["triangles_base"] = tris
    bounds = scene_bounds()

    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    export_glb(out_glb, a.get("draco", True), bool(a.get("safe")))
    stats["safe_mode"] = bool(a.get("safe"))
    stats["glb_bytes"] = os.path.getsize(out_glb)
    log("exported", "%d bytes" % stats["glb_bytes"])

    info = {
        "version": 1,
        "mode": mode,
        "up": "Y",
        "units": "m",
        "cameras": cams,
        "viewport": viewport,
        "figures": figures,
        "bounds": bounds,
        "stats": stats,
        "seconds": round(time.time() - T0, 1),
    }
    with open(out_info, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    log("done")


try:
    main()
except Exception as exc:
    import traceback
    traceback.print_exc()
    log("error", repr(exc))
    sys.exit(1)
sys.exit(0)
