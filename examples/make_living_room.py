"""The demo living room in examples/living_room.blend, built from plain shapes.

blender -b --factory-startup --python make_living_room.py  (writes living_room.blend next to this script)
"""
import bpy, math, os, random
random.seed(7)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
mats = {}
def mat(name, rgb, rough=0.6, metal=0.0):
    if name in mats: return mats[name]
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1); b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    m.diffuse_color = (*rgb, 1)
    mats[name] = m; return m
def box(name, loc, size, m, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(location=loc)
    o = bpy.context.object; o.name = name; o.scale = (size[0]/2, size[1]/2, size[2]/2)
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        mod = o.modifiers.new("bevel", "BEVEL"); mod.width = bevel; mod.segments = 3
        o.modifiers.new("ws", "WEIGHTED_NORMAL")
    o.data.materials.append(m); return o
def cyl(name, loc, r, h, m, verts=32):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=h, location=loc)
    o = bpy.context.object; o.name = name; o.data.materials.append(m); bpy.ops.object.shade_smooth(); return o
def sph(name, loc, r, m):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=24, ring_count=12)
    o = bpy.context.object; o.name = name; o.data.materials.append(m); bpy.ops.object.shade_smooth(); return o

W, D, H = 5.6, 4.6, 2.7
wall = mat("wall", (0.82, 0.79, 0.73), 0.9); floor = mat("floor", (0.45, 0.30, 0.18), 0.5)
box("floor", (0, 0, -0.05), (W, D, 0.1), floor)
box("ceiling", (0, 0, H + 0.05), (W, D, 0.1), mat("ceiling", (0.92, 0.92, 0.9), 0.9))
box("wall_back", (0, D/2 + 0.05, H/2), (W, 0.1, H), wall)
box("wall_left", (-W/2 - 0.05, 0, H/2), (0.1, D, H), wall)
box("wall_front", (0, -D/2 - 0.05, H/2), (W, 0.1, H), wall)
# right wall with a window (1.6 x 1.3 m, sill at 0.85 m)
x = W/2 + 0.05; wy0, wy1, z0, z1 = -0.6, 1.0, 0.85, 2.15
box("wall_right_a", (x, (-D/2 + wy0)/2, H/2), (0.1, wy0 + D/2, H), wall)
box("wall_right_b", (x, (wy1 + D/2)/2, H/2), (0.1, D/2 - wy1, H), wall)
box("wall_right_c", (x, (wy0 + wy1)/2, z0/2), (0.1, wy1 - wy0, z0), wall)
box("wall_right_d", (x, (wy0 + wy1)/2, (z1 + H)/2), (0.1, wy1 - wy0, H - z1), wall)
frame = mat("frame", (0.95, 0.95, 0.95), 0.4)
for i, (ly, lz, sy, sz) in enumerate([((wy0+wy1)/2, z0, wy1-wy0, 0.06), ((wy0+wy1)/2, z1, wy1-wy0, 0.06),
                                      (wy0, (z0+z1)/2, 0.06, z1-z0), (wy1, (z0+z1)/2, 0.06, z1-z0),
                                      ((wy0+wy1)/2, (z0+z1)/2, 0.04, z1-z0)]):
    box(f"window_frame{i}", (x - 0.02, ly, lz), (0.12, sy, sz), frame)
box("window_sill", (x - 0.12, (wy0+wy1)/2, z0 - 0.03), (0.26, wy1 - wy0 + 0.2, 0.04), frame)
box("skirting_back", (0, D/2 - 0.01, 0.05), (W, 0.02, 0.1), frame)
box("skirting_left", (-W/2 + 0.01, 0, 0.05), (0.02, D, 0.1), frame)

# sofa against the back wall
fab = mat("sofa", (0.25, 0.36, 0.42), 0.95); cush = mat("cushion", (0.30, 0.42, 0.48), 0.95)
sx, sy = -0.4, D/2 - 0.55
box("sofa_base", (sx, sy, 0.22), (2.2, 0.9, 0.3), fab, 0.03)
box("sofa_back", (sx, sy + 0.35, 0.6), (2.2, 0.2, 0.55), fab, 0.05)
box("sofa_arm_l", (sx - 1.0, sy, 0.45), (0.2, 0.9, 0.35), fab, 0.05)
box("sofa_arm_r", (sx + 1.0, sy, 0.45), (0.2, 0.9, 0.35), fab, 0.05)
for i in range(3):
    box(f"sofa_seat{i}", (sx - 0.6 + i * 0.6, sy - 0.05, 0.43), (0.58, 0.75, 0.13), cush, 0.04)
for i, c in enumerate([(0.85, 0.6, 0.25), (0.8, 0.8, 0.75)]):
    o = box(f"pillow{i}", (sx - 0.65 + i * 1.3, sy + 0.17, 0.66), (0.42, 0.14, 0.38), mat(f"pillow{i}", c, 0.9), 0.06)
    o.rotation_euler = (math.radians(-12), 0, math.radians(8 - 16 * i))
# rug, coffee table
box("rug", (sx, sy - 1.35, 0.005), (2.4, 1.7, 0.01), mat("rug", (0.70, 0.66, 0.58), 1.0))
oak = mat("oak", (0.62, 0.45, 0.28), 0.5); black = mat("black_metal", (0.05, 0.05, 0.05), 0.4, 1.0)
box("table_top", (sx, sy - 1.3, 0.42), (1.1, 0.6, 0.04), oak, 0.01)
for i, (dx, dy) in enumerate([(-0.5, -0.25), (0.5, -0.25), (-0.5, 0.25), (0.5, 0.25)]):
    cyl(f"table_leg{i}", (sx + dx, sy - 1.3 + dy, 0.2), 0.02, 0.4, black, 12)
cyl("mug", (sx + 0.25, sy - 1.25, 0.49), 0.04, 0.1, mat("mug", (0.9, 0.9, 0.88), 0.3))
box("book_on_table", (sx - 0.2, sy - 1.35, 0.455), (0.24, 0.17, 0.03), mat("red", (0.6, 0.12, 0.1), 0.7))
# armchair by the window
ach = mat("armchair", (0.55, 0.42, 0.32), 0.9)
ax, ay = 1.8, -0.6
box("armchair_base", (ax, ay, 0.25), (0.8, 0.8, 0.35), ach, 0.04)
box("armchair_back", (ax + 0.33, ay, 0.62), (0.15, 0.8, 0.5), ach, 0.05)
box("armchair_arm_l", (ax, ay - 0.37, 0.5), (0.8, 0.12, 0.2), ach, 0.04)
box("armchair_arm_r", (ax, ay + 0.37, 0.5), (0.8, 0.12, 0.2), ach, 0.04)
# bookshelf on the left wall
shx = -W/2 + 0.2
box("shelf_side_a", (shx, -1.4, 1.0), (0.35, 0.03, 2.0), oak)
box("shelf_side_b", (shx, -0.4, 1.0), (0.35, 0.03, 2.0), oak)
for i in range(5):
    z = 0.05 + i * 0.47
    box(f"shelf_board{i}", (shx, -0.9, z), (0.35, 1.0, 0.03), oak)
    if i < 4:
        yy = -1.36
        while yy < -0.5:
            w = random.uniform(0.03, 0.06); h = random.uniform(0.22, 0.34)
            c = random.choice([(0.6, 0.12, 0.1), (0.15, 0.25, 0.45), (0.85, 0.75, 0.5), (0.2, 0.35, 0.2), (0.3, 0.3, 0.3)])
            box(f"book{i}_{yy:.2f}", (shx, yy + w/2, z + 0.015 + h/2), (0.24, w, h), mat(f"bk{c}", c, 0.8))
            yy += w + random.uniform(0.0, 0.01)
            if random.random() < 0.12: yy += 0.12
# floor lamp, plant, picture
lx, ly = -1.85, D/2 - 0.45
cyl("lamp_base", (lx, ly, 0.02), 0.16, 0.04, black)
cyl("lamp_pole", (lx, ly, 0.8), 0.015, 1.55, black, 12)
cyl("lamp_shade", (lx, ly, 1.62), 0.22, 0.3, mat("shade", (0.95, 0.92, 0.85), 0.9))
cyl("pot", (2.35, 1.75, 0.2), 0.18, 0.4, mat("pot", (0.75, 0.42, 0.3), 0.8))
leaf = mat("leaves", (0.18, 0.4, 0.18), 0.8)
for i in range(7):
    a = i * 0.9
    sph(f"leaf{i}", (2.35 + 0.15 * math.cos(a), 1.75 + 0.15 * math.sin(a), 0.65 + 0.12 * (i % 3)), 0.17, leaf)
box("picture_frame", (sx, D/2 - 0.02, 1.65), (1.0, 0.03, 0.7), black)
box("picture", (sx, D/2 - 0.04, 1.65), (0.9, 0.01, 0.6), mat("picture", (0.55, 0.65, 0.7), 0.6))
box("tv_cabinet", (0, -D/2 + 0.25, 0.25), (1.8, 0.45, 0.5), mat("cabinet", (0.9, 0.9, 0.88), 0.5), 0.01)
# camera: from the front-right corner towards the sofa
bpy.ops.object.camera_add(location=(1.9, -2.0, 1.55))
cam = bpy.context.object; cam.name = "Camera"; cam.data.lens = 24
d = cam.location.copy(); import mathutils
tgt = mathutils.Vector((-0.6, 1.2, 0.8)); rot = (tgt - d).to_track_quat("-Z", "Y"); cam.rotation_euler = rot.to_euler()
sc.camera = cam
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.path.dirname(os.path.abspath(__file__)), "living_room.blend"))
print("saved", len(bpy.data.objects))
