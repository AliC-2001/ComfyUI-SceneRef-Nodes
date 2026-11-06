// SceneRef - the three.js side: one Viewer per node.
//
// Navigation (what a Blender / Daz / game user expects):
//   left drag      orbit around the pivot          shift + left drag / middle drag  pan
//   right drag     look around (camera stays put)   right held + WASD / QE           fly
//   wheel          dolly towards the pivot          right held + wheel               fly speed
//   double click   new pivot on the surface         F frame scene · R back to start
//
// Figures (mannequins): click to select, drag to slide over the floor / seats. Gizmos:
//   1 move (arrows, planes)   2 rotate (turn, tilt)   3 scale (height)   4 bones (pose)
//   0 off · Ctrl while dragging a gizmo snaps · Esc lets go of the bone, then the figure
//
// Capture renders the exact framed view at the output size: the colour picture
// with the current look, and one float pass holding linear depth + camera-space
// normal, from which depth, normal and mask PNGs are built on the CPU.

import { Figure, DazFigure, PALETTE, MAX_FIGURES, COCO, PARTS, drawOpenPose, poseOf, JOINTS, JOINT_LABELS, MIRROR } from "./scene_ref_figures.mjs?v=6";

const VENDOR = new URL("../../scene_ref/vendor/three/", import.meta.url).href;

let _libs = null;
export function loadLibs() {
  _libs ??= (async () => {
    const THREE = await import(VENDOR + "three.module.js");
    const { GLTFLoader } = await import(VENDOR + "GLTFLoader.js");
    const { DRACOLoader } = await import(VENDOR + "DRACOLoader.js");
    const { RoomEnvironment } = await import(VENDOR + "RoomEnvironment.js");
    const { TransformControls } = await import(VENDOR + "TransformControls.js");
    return { THREE, GLTFLoader, DRACOLoader, RoomEnvironment, TransformControls };
  })();
  return _libs;
}

// A small picture of a mannequin in a pose ({points} or {bones}, as /scene_ref/poses gives),
// three-quarter view, for the pose browser. One hidden renderer serves every viewer.
let _thumb = null;
export async function poseThumbnail(pose, { size = 120, sex = "female" } = {}) {
  const { THREE } = await loadLibs();
  if (!_thumb) {
    const r = new THREE.WebGLRenderer({ canvas: document.createElement("canvas"), antialias: true, alpha: true, preserveDrawingBuffer: true });
    r.outputColorSpace = THREE.SRGBColorSpace;
    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x4a4440, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.8);
    sun.position.set(1, 2, 2);
    scene.add(sun);
    _thumb = { r, scene, cam: new THREE.PerspectiveCamera(28, 1, 0.05, 50), figs: {} };
  }
  const T = _thumb;
  T.r.setSize(size, size, false);
  let fig = T.figs[sex];
  if (!fig) {
    fig = T.figs[sex] = new Figure(THREE, { height_cm: 170, sex, color: 0 });
    T.scene.add(fig.root);
  }
  for (const [k, g] of Object.entries(T.figs)) g.root.visible = k === sex;
  const bones = pose.bones || fig.fitPose(pose.points || {});
  fig.set({ pose: "custom", pose_label: pose.name || "", anchor: pose.anchor, bones, yaw: 0, pitch: 0, roll: 0, at: [0, 0, 0] });
  const box = new THREE.Box3().setFromObject(fig.body);
  const c = box.getCenter(new THREE.Vector3());
  const r = box.getSize(new THREE.Vector3()).length() / 2;
  const dir = new THREE.Vector3(0.55, 0.32, 1).normalize();
  T.cam.position.copy(c).addScaledVector(dir, r / Math.sin((T.cam.fov / 2) * DEG) * 0.92);
  T.cam.lookAt(c);
  T.r.setClearColor(0x000000, 0);
  T.r.render(T.scene, T.cam);
  return T.r.domElement.toDataURL("image/png");
}

export const LOOKS = ["color", "clay", "depth", "normal"];
export const LIGHTS = ["studio", "scene", "flat"];

const DEG = Math.PI / 180;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// ── shared model cache: one download + parse per converted scene ─────────────
const _models = new Map(); // key -> Promise<{gltf, info}>

export function loadModel(key, onProgress) {
  if (_models.has(key)) return _models.get(key);
  const p = (async () => {
    const { GLTFLoader, DRACOLoader } = await loadLibs();
    const base = new URL("../../scene_ref/model/", import.meta.url).href;
    const info = await (await fetch(base + key + ".json")).json();
    const draco = new DRACOLoader();
    draco.setDecoderPath(VENDOR + "draco/");
    const loader = new GLTFLoader();
    loader.setDRACOLoader(draco);
    const gltf = await new Promise((resolve, reject) =>
      loader.load(base + key + ".glb", resolve, (e) => onProgress?.(e.loaded, e.total), reject));
    draco.dispose();
    return { gltf, info };
  })();
  _models.set(key, p);
  p.catch(() => _models.delete(key));
  return p;
}

// ── data pass: linear view depth in R, camera-space normal in GBA ──────────────
const DATA_VERT = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_vertex>
varying vec3 vViewPos;
varying vec3 vNormalV;
varying vec2 vUv;
void main() {
  vUv = uv;
  vec4 p = vec4(position, 1.0);
  vec3 n = normal;
#ifdef USE_INSTANCING
  p = instanceMatrix * p;
  n = mat3(instanceMatrix) * n;
#endif
  vec4 mv = modelViewMatrix * p;
  vViewPos = mv.xyz;
  vNormalV = normalize(normalMatrix * n);
  gl_Position = projectionMatrix * mv;
  #include <logdepthbuf_vertex>
}`;

// idMode: instead of depth + normal, write which figure owns the pixel (figId,
// 0 for the scene), so occlusion by tables and sofas comes straight from the GPU.
const DATA_FRAG = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_fragment>
uniform sampler2D map;
uniform float useMap;
uniform float alphaCut;
uniform float idMode;
uniform float figId;
varying vec3 vViewPos;
varying vec3 vNormalV;
varying vec2 vUv;
void main() {
  #include <logdepthbuf_fragment>
  if (useMap > 0.5 && texture2D(map, vUv).a < alphaCut) discard;
  if (idMode > 0.5) { gl_FragColor = vec4(figId, -vViewPos.z, 0.0, 1.0); return; }
  vec3 n = normalize(vNormalV);
  // A face seen from behind still faces the camera in a normal map.
  if (dot(n, vViewPos) > 0.0) n = -n;
  gl_FragColor = vec4(-vViewPos.z, n);
}`;

// Daz glass (refraction in Iray) arrives through OBJ as an opaque surface with a
// dark diffuse map, so cabinet doors and windows turned into black panels.
const GLASS_RE = /glass|\bglas\b|crystal/i;
function makeGlass(m) {
  m.userData.srGlass = true;
  m.map = null;
  m.color?.set(0xdde6ea);
  m.transparent = true;
  m.opacity = Math.min(m.opacity ?? 1, 0.18); // < 0.5: see-through in depth / normal / clay too
  m.depthWrite = false;
  if ("roughness" in m) m.roughness = 0.05;
  if ("metalness" in m) m.metalness = 0;
  m.needsUpdate = true;
}

// Is this material something you see (for depth / normal / clay)?
function isGhost(m) {
  return m.visible === false || (m.transparent && m.opacity < 0.5 && !m.map);
}

// Robust [near, far] of a list of distances (any order); null when empty.
function depthRange(ds) {
  if (!ds.length) return null;
  ds.sort((a, b) => a - b);
  const near = ds[Math.floor(ds.length * 0.005)];
  const far = Math.max(ds[Math.floor((ds.length - 1) * 0.995)], near * 1.01);
  return [near, far];
}

// Words for how a figure stands relative to the camera; yaw 0 = facing it. A body lying
// down (torso near level) is described by which way its front points instead.
function describeFacing(yaw, f) {
  const q = f.joints?.chest?.getWorldQuaternion?.(new f.THREE.Quaternion());
  if (q && f.kind !== "daz") {
    const up = new f.THREE.Vector3(0, 1, 0).applyQuaternion(q);
    const front = new f.THREE.Vector3(0, 0, 1).applyQuaternion(q);
    if (Math.abs(up.y) < 0.5) return front.y > 0.6 ? "lying on the back" : front.y < -0.6 ? "lying face down" : "lying on the side";
  }
  const a = Math.abs(yaw);
  const side = yaw > 0 ? "frame right" : "frame left";
  if (a < 30) return "facing the camera";
  if (a < 70) return `three-quarter view, turned towards ${side}`;
  if (a < 110) return `in profile, facing ${side}`;
  if (a < 150) return `three-quarter back view, turned towards ${side}`;
  return "back to the camera";
}

function hasCutout(m) {
  return !!m.map && (m.alphaTest > 0 || m.transparent);
}

export class Viewer {
  constructor(host, { onChange, onStatus, onFigures, onSelect, onGizmo } = {}) {
    this.host = host;
    this.onChange = onChange || (() => {});
    this.onStatus = onStatus || (() => {});
    this.onFigures = onFigures || (() => {}); // figures moved / edited: persist them
    this.onSelect = onSelect || (() => {}); // selected figure index (or -1)
    this.onGizmo = onGizmo || (() => {}); // {mode, bone} after a gizmo or bone change
    this.gizmo = "off"; // off | move | rotate | scale | bones
    this.bone = null; // the joint the bones gizmo turns
    this.figures = [];
    this.selected = -1;
    this.look = "color";
    this.light = "studio";
    this.exposure = 1;
    this.aspect = 1;
    this.key = null;
    this.ready = false;
    this.keys = new Set();
    this.flySpeed = 2;
    this._raf = 0;
    this._last = 0;
    this._matCache = new Map();
    this.canvas = document.createElement("canvas");
    this.canvas.tabIndex = 0;
    this.canvas.className = "sr-canvas";
    host.appendChild(this.canvas);
    this._init = this._setup();
  }

  async _setup() {
    const { THREE, RoomEnvironment, TransformControls } = await loadLibs();
    this.THREE = THREE;
    const r = new THREE.WebGLRenderer({
      canvas: this.canvas, antialias: true, logarithmicDepthBuffer: true, preserveDrawingBuffer: false,
    });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    r.toneMapping = THREE.AgXToneMapping;
    r.toneMappingExposure = this.exposure;
    r.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer = r;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x30343a);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
    this.euler = new THREE.Euler(0, 0, 0, "YXZ");
    this.pos = new THREE.Vector3(3, 1.6, 3);
    this.dist = 4;

    const pmrem = new THREE.PMREMGenerator(r);
    this.envMap = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
    this.rig = new THREE.Group();
    this.hemi = new THREE.HemisphereLight(0xffffff, 0x5a5048, 1.1);
    this.sun = new THREE.DirectionalLight(0xffffff, 1.6);
    this.sun.position.set(0.5, 1, 0.8);
    this.ambient = new THREE.AmbientLight(0xffffff, 0.25);
    this.rig.add(this.hemi, this.sun, this.ambient);
    this.scene.add(this.rig);
    this.clay = new THREE.MeshStandardMaterial({ color: 0xb8b8b8, roughness: 0.85, metalness: 0 });
    this.figGroup = new THREE.Group();
    this.figGroup.name = "sceneref_figures";
    this.scene.add(this.figGroup);
    this._idMode = { value: 0 }; // shared by every data material
    // Gizmos and bone handles live in their own scene, drawn over the view on screen only:
    // never in a capture, a data pass or the clay look.
    this.overlay = new THREE.Scene();
    this._proxy = new THREE.Object3D(); // what the gizmo moves; changes go to the figure's spec
    this.overlay.add(this._proxy);
    const tc = new TransformControls(this.camera, null); // fed by our own pointer handlers
    tc.setSize(0.85);
    tc.addEventListener("change", () => this.draw());
    tc.addEventListener("objectChange", () => this._gizmoMoved());
    tc.addEventListener("dragging-changed", (e) => (e.value ? this._gizmoStart() : this._gizmoDone()));
    this._tc = tc;
    this.overlay.add(tc.getHelper());
    this._handles = new THREE.Group();
    this._handles.visible = false;
    this.overlay.add(this._handles);
    this._handleGeo = new THREE.SphereGeometry(1, 14, 10);
    this._handleMat = new THREE.MeshBasicMaterial({ color: 0xffd23f, depthTest: false, transparent: true, opacity: 0.9 });
    this._handleSel = new THREE.MeshBasicMaterial({ color: 0xff5a1f, depthTest: false, transparent: true, opacity: 0.95 });
    for (const j of JOINTS) {
      const m = new THREE.Mesh(this._handleGeo, this._handleMat);
      m.userData.joint = j;
      m.renderOrder = 10;
      this._handles.add(m);
    }
    this.labels = document.createElement("div");
    this.labels.className = "sr-labels";
    this.host.appendChild(this.labels);
    if (this._pendingFigures) { this.setFigures(this._pendingFigures); this._pendingFigures = null; }
    this._wire();
    this._layout();
    this._ro = new ResizeObserver(() => this._layout());
    this._ro.observe(this.host);
    this.applyLighting();
    this.draw();
  }

  // ── model ──────────────────────────────────────────────────────────────────
  async load(key, onProgress) {
    await this._init;
    if (this.key === key && this.ready) return;
    this.ready = false;
    this.key = key;
    const { gltf, info } = await loadModel(key, onProgress);
    if (this.key !== key) return; // another scene was picked meanwhile
    const THREE = this.THREE;
    if (this.model) this.scene.remove(this.model);
    this.model = gltf.scene.clone(true);
    this.info = info;
    this.sceneLights = [];
    this.model.traverse((o) => {
      if (o.isLight) this.sceneLights.push(o);
      if (o.isMesh) {
        o.frustumCulled = true;
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        for (const m of mats) {
          if (m.isMeshStandardMaterial && !m.envMap) m.envMapIntensity = 0.6;
          if (GLASS_RE.test(m.name) && !m.userData.srGlass) makeGlass(m);
          // Daz OBJ surfaces are often single-sided planes (floors, walls).
          m.side = THREE.DoubleSide;
        }
      }
    });
    this.scene.add(this.model);
    this.model.updateMatrixWorld(true);
    // Characters posed in Daz arrive as SRFIG_<n> meshes: they become figures.
    for (const f of this.figures) if (f.kind === "daz") f.dispose();
    const figData = new Map((info.figures || []).map((d) => [d.mesh, d]));
    const found = [];
    this.model.traverse((o) => { if (/^SRFIG_\d+$/.test(o.name) && !found.some((x) => x.mesh === o)) found.push({ mesh: o }); });
    this.dazMeshes = found.map((d) => ({ mesh: d.mesh, data: figData.get(d.mesh.name) || {} }));
    this._robustBounds();
    this._buildCameras(gltf);
    this._rebuildFigures();
    if (this.dazMeshes.length) this.onFigures(this.figureSpecs());
    this.ready = true;
    this.applyLighting();
    if (!this._pendingView) this.goHome();
    else {
      this.setView(this._pendingView);
      this._pendingView = null;
    }
    this.draw();
  }

  stats() {
    let tris = 0;
    let meshes = 0;
    const mats = new Set();
    this.model?.traverse((o) => {
      if (!o.isMesh) return;
      meshes++;
      const g = o.geometry;
      tris += (g.index ? g.index.count : g.attributes.position.count) / 3;
      (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => mats.add(m));
    });
    return { tris, meshes, materials: mats.size };
  }

  // Bounds from a sample of vertices, 2nd-98th percentile: a 100 km sky dome
  // must not decide the fly speed or the clipping planes.
  _robustBounds() {
    const THREE = this.THREE;
    const xs = [], ys = [], zs = [];
    const full = new THREE.Box3();
    const v = new THREE.Vector3();
    let total = 0;
    this.model.traverse((o) => { if (o.isMesh) total += o.geometry.attributes.position.count; });
    const step = Math.max(1, Math.floor(total / 60000));
    this.model.traverse((o) => {
      if (!o.isMesh) return;
      const pos = o.geometry.attributes.position;
      for (let i = 0; i < pos.count; i += step) {
        v.fromBufferAttribute(pos, i).applyMatrix4(o.matrixWorld);
        xs.push(v.x); ys.push(v.y); zs.push(v.z);
        full.expandByPoint(v);
      }
    });
    const pct = (a, p) => { a.sort((x, y) => x - y); return a[Math.floor(clamp(p, 0, 1) * (a.length - 1))] ?? 0; };
    const lo = new THREE.Vector3(pct(xs, 0.02), pct(ys, 0.02), pct(zs, 0.02));
    const hi = new THREE.Vector3(pct(xs, 0.98), pct(ys, 0.98), pct(zs, 0.98));
    this.bounds = new THREE.Box3(lo, hi);
    this.fullBounds = full;
    const size = Math.max(0.5, lo.distanceTo(hi));
    this.sceneSize = size;
    this.flySpeed = clamp(size * 0.12, 0.5, 50);
    const fullSize = full.isEmpty() ? size : full.min.distanceTo(full.max);
    this.camera.near = clamp(size * 0.0005, 0.002, 0.05);
    this.camera.far = Math.max(size * 20, fullSize * 2.5);
    this.camera.updateProjectionMatrix();
  }

  _buildCameras(gltf) {
    const THREE = this.THREE;
    const list = [];
    const info = this.info || {};
    if (info.viewport) list.push({ ...this._camFromInfo(info.viewport), group: "scene" });
    for (const c of info.cameras || []) list.push({ ...this._camFromInfo(c), group: "scene" });
    // A plain GLB carries its cameras only inside the file.
    if (!list.length) {
      for (const c of gltf.cameras || []) {
        const p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
        c.updateMatrixWorld(true);
        c.matrixWorld.decompose(p, q, s);
        list.push({ name: c.name || "Camera", pos: p.toArray(), quat: q.toArray(), fov: c.fov || 40, group: "scene" });
      }
    }
    this.sceneCameras = list;
  }

  _camFromInfo(c) {
    const r = c.rotation || [0, 0, 0, 1];
    return { name: c.name, pos: c.position, quat: [r[0], r[1], r[2], r[3]], fov: c.fov || 40 };
  }

  goHome() {
    const cams = this.sceneCameras || [];
    const active = cams.find((c) => /active/i.test(c.name)) || cams[0];
    if (active) this.setView(active);
    else this.frameAll();
  }

  frameAll() {
    const THREE = this.THREE;
    if (!this.bounds) return;
    const c = this.bounds.getCenter(new THREE.Vector3());
    const r = this.sceneSize / 2;
    const d = r / Math.tan((this.camera.fov * DEG) / 2) * 1.1;
    this.euler.set(-25 * DEG, 35 * DEG, 0);
    const fwd = this._fwd();
    this.pos.copy(c).addScaledVector(fwd, -d);
    this.dist = d;
    this._sync();
  }

  // ── view state ─────────────────────────────────────────────────────────────
  getView() {
    const q = new this.THREE.Quaternion().setFromEuler(this.euler);
    const r = (x) => Math.round(x * 1e5) / 1e5;
    return {
      pos: this.pos.toArray().map(r), quat: q.toArray().map(r),
      fov: r(this.camera.fov), dist: r(this.dist),
    };
  }

  setView(v) {
    if (!v) return;
    if (!this.THREE) { this._pendingView = v; return; }
    const THREE = this.THREE;
    if (!this.ready && !this.model) this._pendingView = v;
    this.pos.fromArray(v.pos);
    this.euler.setFromQuaternion(new THREE.Quaternion().fromArray(v.quat), "YXZ");
    if (v.fov) this.camera.fov = clamp(v.fov, 5, 140);
    if (v.dist) this.dist = v.dist;
    else this.dist = this._probeDepth(0.5, 0.5) || this.sceneSize * 0.3 || 3;
    this._sync();
  }

  setFov(fov) {
    this.camera.fov = clamp(fov, 5, 140);
    this._sync();
  }

  _fwd() {
    return new this.THREE.Vector3(0, 0, -1).applyEuler(this.euler);
  }

  _sync(notify = true) {
    this.camera.position.copy(this.pos);
    this.camera.quaternion.setFromEuler(this.euler);
    this.camera.updateProjectionMatrix();
    this.camera.updateMatrixWorld(true);
    this.draw();
    if (notify) this.onChange(this.getView());
  }

  // ── looks & light ──────────────────────────────────────────────────────────
  setLook(look) {
    this.look = LOOKS.includes(look) ? look : "color";
    this.draw();
  }

  setLight(light) {
    this.light = LIGHTS.includes(light) ? light : "studio";
    this.applyLighting();
    this.draw();
  }

  setExposure(e) {
    this.exposure = clamp(e, 0.05, 8);
    if (this.renderer) this.renderer.toneMappingExposure = this.exposure;
    this.draw();
  }

  applyLighting() {
    if (!this.scene) return;
    const studio = this.light === "studio";
    const scene = this.light === "scene";
    const flat = this.light === "flat";
    this.scene.environment = studio || scene ? this.envMap : null;
    this.hemi.visible = studio;
    this.sun.visible = studio;
    this.ambient.intensity = flat ? 3.2 : scene ? 0.35 : 0.25;
    for (const l of this.sceneLights || []) l.visible = scene;
  }

  // Swap every mesh material (scene and figures) for `pick(orig)` while fn runs.
  _withMaterials(pick, fn) {
    const saved = [];
    const swap = (o) => {
      if (!o.isMesh) return;
      saved.push([o, o.material]);
      o.material = Array.isArray(o.material) ? o.material.map(pick) : pick(o.material);
    };
    this.model?.traverse(swap);
    this.figGroup?.traverse(swap);
    try {
      return fn();
    } finally {
      for (const [o, m] of saved) o.material = m;
    }
  }

  _clayFor(m) {
    if (m.userData.srFigure) return m; // mannequins keep their colour in the clay look
    const k = "clay:" + m.uuid;
    let c = this._matCache.get(k);
    if (c) return c;
    const THREE = this.THREE;
    c = this.clay.clone();
    c.side = THREE.DoubleSide;
    c.visible = !isGhost(m);
    if (hasCutout(m)) {
      // Keep the cut-out (leaves, hair cards, lace) but drop the colours.
      c.map = m.map;
      c.alphaTest = m.alphaTest || 0.5;
      c.onBeforeCompile = (sh) => {
        sh.fragmentShader = sh.fragmentShader.replace(
          "#include <map_fragment>",
          "#ifdef USE_MAP\n diffuseColor.a *= texture2D( map, vMapUv ).a;\n#endif");
      };
      c.customProgramCacheKey = () => "sr-clay-cut";
    }
    this._matCache.set(k, c);
    return c;
  }

  _dataFor(m) {
    const k = "data:" + m.uuid;
    let d = this._matCache.get(k);
    if (d) return d;
    const THREE = this.THREE;
    const cut = hasCutout(m);
    const fig = m.userData.srFigureOwner;
    d = new THREE.ShaderMaterial({
      vertexShader: DATA_VERT, fragmentShader: DATA_FRAG,
      uniforms: {
        map: { value: cut ? m.map : null },
        useMap: { value: cut ? 1 : 0 },
        alphaCut: { value: m.alphaTest || 0.5 },
        idMode: this._idMode,
        // Figure ids are read back by index; the owner keeps it current.
        figId: { get value() { return fig ? fig._index + 1 : 0; } },
      },
      side: THREE.DoubleSide,
    });
    d.visible = !isGhost(m);
    this._matCache.set(k, d);
    return d;
  }

  // ── drawing ────────────────────────────────────────────────────────────────
  draw() {
    if (this._drawQueued || !this.renderer) return;
    this._drawQueued = true;
    requestAnimationFrame(() => {
      this._drawQueued = false;
      this._render();
    });
  }

  _render(target = null) {
    const r = this.renderer;
    if (!r) return;
    this.rig.quaternion.copy(this.camera.quaternion); // studio light follows the view
    this.rig.position.copy(this.camera.position);
    if (this.look === "clay") {
      this._withMaterials((m) => this._clayFor(m), () => r.render(this.scene, this.camera));
    } else if (this.look === "depth" || this.look === "normal") {
      this._renderPreviewData();
    } else {
      r.render(this.scene, this.camera);
    }
    if (!target) {
      if (this._tc?.object || this._handles?.visible) {
        const auto = r.autoClear;
        r.autoClear = false;
        r.clearDepth();
        if (this._handles.visible) this._sizeHandles();
        r.render(this.overlay, this.camera);
        r.autoClear = auto;
      }
      this._updateLabels();
    }
  }

  // On-screen depth / normal preview: the same data pass, shown through a quad.
  _renderPreviewData() {
    const THREE = this.THREE;
    const r = this.renderer;
    const size = r.getDrawingBufferSize(new THREE.Vector2());
    if (!this._prevRT || this._prevRT.width !== size.x || this._prevRT.height !== size.y) {
      this._prevRT?.dispose();
      this._prevRT = this._floatRT(size.x, size.y);
    }
    this._renderData(this._prevRT);
    if (!this._quad) {
      this._quadMat = new THREE.ShaderMaterial({
        uniforms: { tex: { value: null }, mode: { value: 0 }, near: { value: 1 }, far: { value: 100 } },
        vertexShader: "varying vec2 vUv; void main(){ vUv=uv; gl_Position=vec4(position.xy,0.,1.); }",
        fragmentShader: `uniform sampler2D tex; uniform float mode; uniform float near; uniform float far; varying vec2 vUv;
          void main(){ vec4 d = texture2D(tex, vUv);
            if (d.r <= 0.0) { gl_FragColor = vec4(0.,0.,0.,1.); return; }
            if (mode < 0.5) { float v = clamp((1. / d.r - 1. / far) / (1. / near - 1. / far), 0., 1.); gl_FragColor = vec4(vec3(v), 1.); }
            else { gl_FragColor = vec4(d.gba * 0.5 + 0.5, 1.); } }`,
        depthTest: false, depthWrite: false,
      });
      this._quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this._quadMat);
      this._quadScene = new THREE.Scene();
      this._quadScene.add(this._quad);
      this._quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    }
    this._quadMat.uniforms.tex.value = this._prevRT.texture;
    this._quadMat.uniforms.mode.value = this.look === "depth" ? 0 : 1;
    const [pn, pf] = this._previewRange || [this.camera.near * 50, this.camera.near * 5000];
    this._quadMat.uniforms.near.value = pn;
    this._quadMat.uniforms.far.value = Math.max(pf, pn * 1.001);
    const tm = r.toneMapping;
    r.toneMapping = THREE.NoToneMapping;
    r.render(this._quadScene, this._quadCam);
    r.toneMapping = tm;
    // Refresh the normalisation from what is on screen, for the next frame.
    if (!this._nearProbeQueued) {
      this._nearProbeQueued = true;
      setTimeout(() => {
        this._nearProbeQueued = false;
        const r = this._depthRange();
        const old = this._previewRange;
        if (r && (!old || Math.abs(r[0] - old[0]) / r[0] > 0.02 || Math.abs(r[1] - old[1]) / r[1] > 0.02)) {
          this._previewRange = r;
          if (this.look === "depth") this.draw();
        }
      }, 120);
    }
  }

  // figures: draw the mannequins; model: draw the scene; ids: write figure ids
  // (R) + depth (G) instead of depth + normal; only: draw just this figure.
  _renderData(target, { figures = true, model = true, ids = false, only = null } = {}) {
    const THREE = this.THREE;
    const r = this.renderer;
    const bg = this.scene.background;
    const env = this.scene.environment;
    this.scene.background = null;
    this.scene.environment = null;
    const prevClear = r.getClearColor(new THREE.Color());
    const prevAlpha = r.getClearAlpha();
    const vis = this._setVisible({ figures, model, only });
    this._idMode.value = ids ? 1 : 0;
    r.setRenderTarget(target);
    r.setClearColor(0x000000, 0);
    r.clear();
    try {
      this._withMaterials((m) => this._dataFor(m), () => r.render(this.scene, this.camera));
    } finally {
      this._idMode.value = 0;
      vis();
      r.setRenderTarget(null);
      r.setClearColor(prevClear, prevAlpha);
      this.scene.background = bg;
      this.scene.environment = env;
    }
  }

  // Show / hide the scene and the figures; returns a function that restores.
  _setVisible({ figures = true, model = true, only = null }) {
    const prev = [this.model?.visible, this.figGroup.visible, this.figures.map((f) => f.root.visible)];
    if (this.model) this.model.visible = model;
    this.figGroup.visible = figures || !!only;
    if (only) this.figures.forEach((f) => (f.root.visible = f === only));
    return () => {
      if (this.model) this.model.visible = prev[0];
      this.figGroup.visible = prev[1];
      this.figures.forEach((f, i) => (f.root.visible = prev[2][i] ?? true));
    };
  }

  _readFloat(rt) {
    const buf = new Float32Array(rt.width * rt.height * 4);
    this.renderer.readRenderTargetPixels(rt, 0, 0, rt.width, rt.height, buf);
    return buf;
  }

  // [near, far] of what is on screen (0.5th / 99.5th percentile distance), from one
  // small extra data render.
  _depthRange() {
    const THREE = this.THREE;
    const w = 96, h = Math.max(8, Math.round(96 / this.aspect));
    if (!this._smallRT || this._smallRT.width !== w || this._smallRT.height !== h) {
      this._smallRT?.dispose();
      this._smallRT = this._floatRT(w, h);
    }
    this._renderData(this._smallRT);
    const buf = this._readFloat(this._smallRT);
    const ds = [];
    for (let i = 0; i < buf.length; i += 4) if (buf[i] > 0) ds.push(buf[i]);
    if (!ds.length) return null;
    return depthRange(ds);
  }

  // Float targets read back exactly; nearest filtering because linear filtering
  // of float textures is an optional WebGL extension.
  _floatRT(w, h) {
    const THREE = this.THREE;
    return new THREE.WebGLRenderTarget(w, h, {
      type: THREE.FloatType, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
      depthBuffer: true, samples: 0,
    });
  }

  // Distance to the surface under a canvas point (0..1, 0..1 from top-left); GPU picking,
  // so it stays instant on an eight-million-triangle scene.
  _probeDepth(u, v, { figures = true } = {}) {
    const THREE = this.THREE;
    if (!this.renderer || !this.model) return null;
    const w = 512, h = Math.max(1, Math.round(512 / this.aspect));
    const rt = this._floatRT(1, 1);
    this.camera.setViewOffset(w, h, Math.floor(u * w), Math.floor(v * h), 1, 1);
    this.camera.updateProjectionMatrix();
    this._renderData(rt, { figures });
    this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
    const px = new Float32Array(4);
    this.renderer.readRenderTargetPixels(rt, 0, 0, 1, 1, px);
    rt.dispose();
    return px[0] > 0 ? px[0] : null;
  }

  // World point on the surface under a canvas point (0..1 from top-left).
  _surfacePoint(u, v, opts) {
    const THREE = this.THREE;
    const d = this._probeDepth(u, v, opts);
    if (!d) return null;
    const ndc = new THREE.Vector3(u * 2 - 1, -(v * 2 - 1), 0.5).unproject(this.camera);
    const dir = ndc.sub(this.camera.position).normalize();
    return this.camera.position.clone().addScaledVector(dir, d / Math.max(0.05, dir.dot(this._fwd())));
  }

  // The nearest seat to a world point: flat, facing up, 25-62 cm above the floor. Read from
  // a small top-down height map rendered on the GPU (same data pass as picking), so a
  // sitting figure dropped on a sofa's front, its backrest or the floor in front of it
  // lands on the cushion. null when there is no seat within reach.
  _seatNear(p, { radius = 0.8, n = 48 } = {}) {
    const THREE = this.THREE;
    if (!this.renderer || !this.model || !p) return null;
    const cam = new THREE.OrthographicCamera(-radius, radius, radius, -radius, 0.01, 6);
    cam.position.set(p.x, p.y + 3, p.z);
    cam.up.set(0, 0, -1); // screen right = world +x, screen up = world -z
    cam.lookAt(p.x, p.y - 1, p.z);
    cam.updateMatrixWorld(true);
    cam.updateProjectionMatrix();
    const rt = this._floatRT(n, n);
    const keep = this.camera;
    this.camera = cam;
    let buf;
    try {
      this._renderData(rt, { figures: false });
      buf = this._readFloat(rt);
    } finally {
      this.camera = keep;
      rt.dispose();
    }
    const up = (k) => buf[k * 4 + 3] > 0.9; // view-space normal z = world up here
    const hs = new Float32Array(n * n).fill(NaN);
    let floor = Infinity;
    for (let k = 0; k < n * n; k++) {
      if (buf[k * 4] > 0) hs[k] = cam.position.y - buf[k * 4];
      if (up(k) && hs[k] < floor) floor = hs[k];
    }
    if (!isFinite(floor)) return null;
    let best = null, bestD = Infinity;
    for (let j = 1; j < n - 1; j++) {
      for (let i = 1; i < n - 1; i++) {
        const k = j * n + i, h = hs[k];
        // Up to 62 cm: a backrest top (~80 cm) is not a seat.
        if (!up(k) || h - floor < 0.25 || h - floor > 0.62) continue;
        if ([-1, 1, -n, n].some((o) => !(Math.abs(hs[k + o] - h) < 0.04))) continue; // a flat patch, not an edge
        const x = p.x + ((i + 0.5) / n * 2 - 1) * radius;
        const z = p.z - ((j + 0.5) / n * 2 - 1) * radius;
        const d = (x - p.x) ** 2 + (z - p.z) ** 2;
        if (d < bestD) { bestD = d; best = new THREE.Vector3(x, h, z); }
      }
    }
    return best;
  }

  // Where a figure goes for a picked point: sitting poses snap to the nearest seat.
  _placeFor(f, p) {
    if (!p || f?.kind === "daz" || poseOf(f?.spec).anchor !== "seat") return p;
    return this._seatNear(p) || p;
  }

  // ── figures ────────────────────────────────────────────────────────────────
  // specs: the plain objects stored in the workflow (see Figure for the fields).
  // Daz figures come from the loaded scene (meshes named SRFIG_<n>); their saved
  // specs only keep the colour tag and look, matched by mesh name.
  setFigures(specs) {
    this._savedSpecs = Array.isArray(specs) ? specs : [];
    if (!this.THREE) { this._pendingFigures = specs; return; }
    this._rebuildFigures();
  }

  _rebuildFigures() {
    const specs = this._savedSpecs || [];
    for (const f of this.figures) if (f.kind !== "daz") f.dispose();
    this.figures = [];
    const daz = new Map((this.dazMeshes || []).map((d) => [d.mesh.name, d]));
    const used = new Set();
    for (const s of specs) {
      if (this.figures.length >= MAX_FIGURES) break;
      if (s?.kind === "daz") {
        const d = daz.get(s.mesh);
        if (!d || used.has(s.mesh)) continue; // another scene's figure
        used.add(s.mesh);
        this._pushFigure(d.fig || (d.fig = this._makeDaz(d, s)), s);
      } else {
        this._pushFigure(new Figure(this.THREE, s));
      }
    }
    // Figures in the scene that the workflow does not know yet.
    for (const [name, d] of daz) {
      if (used.has(name) || this.figures.length >= MAX_FIGURES) continue;
      const taken = new Set(this.figures.map((f) => f.spec.color));
      const color = [...PALETTE.keys()].find((i) => !taken.has(i)) ?? 0;
      this._pushFigure(d.fig || (d.fig = this._makeDaz(d, { color })), { color });
    }
    this.selected = Math.min(this.selected, this.figures.length - 1);
    this._highlight();
    this._syncGizmo();
    this.draw();
  }

  _makeDaz(d, spec) {
    return new DazFigure(this.THREE, d.mesh, d.data, spec, (m, hex, owner) => this._paletteMat(m, hex, owner));
  }

  _pushFigure(f, spec) {
    if (spec && f.kind === "daz") f.set(spec);
    f._index = this.figures.length;
    this.figures.push(f);
    this.figGroup.add(f.root);
    return f;
  }

  // Solid palette colour for a Daz figure surface, keeping hair-card cut-outs.
  _paletteMat(m, hex, owner) {
    if (Array.isArray(m)) return m.map((x) => this._paletteMat(x, hex, owner));
    const k = "pal:" + m.uuid + ":" + hex;
    let c = this._matCache.get(k);
    if (c) return c;
    const THREE = this.THREE;
    c = new THREE.MeshStandardMaterial({ color: hex, roughness: 0.55, metalness: 0, side: THREE.DoubleSide });
    c.visible = !isGhost(m);
    if (hasCutout(m)) {
      c.map = m.map;
      c.alphaTest = m.alphaTest || 0.5;
      c.onBeforeCompile = (sh) => {
        sh.fragmentShader = sh.fragmentShader.replace(
          "#include <map_fragment>",
          "#ifdef USE_MAP\n diffuseColor.a *= texture2D( map, vMapUv ).a;\n#endif");
      };
      c.customProgramCacheKey = () => "sr-pal-cut";
    }
    c.userData.srFigure = true;
    c.userData.srFigureOwner = owner;
    this._matCache.set(k, c);
    return c;
  }

  figureSpecs() {
    return this.figures.map((f) => ({ ...f.spec }));
  }

  _addFigure(spec) {
    return this._pushFigure(new Figure(this.THREE, spec));
  }

  // A new mannequin on the surface in the middle of the view.
  addFigure(patch = {}) {
    if (!this.ready) return -1;
    if (this.figures.length >= MAX_FIGURES) { this.onStatus(`at most ${MAX_FIGURES} figures`); return -1; }
    const at = this._surfacePoint(0.5, 0.62, { figures: false }) ||
      this.pos.clone().addScaledVector(this._fwd(), Math.min(this.dist, 3));
    const used = new Set(this.figures.map((f) => f.spec.color));
    const color = [...PALETTE.keys()].find((i) => !used.has(i)) ?? 0;
    // Face the camera.
    const toCam = this.camera.position.clone().sub(at);
    const yaw = Math.atan2(toCam.x, toCam.z) * 180 / Math.PI;
    this._addFigure({ height_cm: 168, sex: "female", build: "average", pose: "stand", ...patch, color, at: at.toArray(), yaw: Math.round(yaw) });
    this.select(this.figures.length - 1);
    this._figuresChanged();
    return this.figures.length - 1;
  }

  updateFigure(i, patch) {
    const f = this.figures[i];
    if (!f) return;
    f.set(patch);
    // Switching to (or placing) a sitting pose: onto the nearest seat.
    if ((patch.pose || patch.at) && f.kind !== "daz" && poseOf(f.spec).anchor === "seat") {
      const seat = this._seatNear(new this.THREE.Vector3(...f.spec.at));
      if (seat) f.set({ at: seat.toArray().map((x) => Math.round(x * 1000) / 1000) });
    }
    this._highlight();
    this._syncGizmo();
    this._figuresChanged();
  }

  // ── poses ──────────────────────────────────────────────────────────────────
  // A pose from the pose browser onto a mannequin: {name, anchor, points} (fitted here) or
  // {bones}; `only` (a partial Daz preset) changes just those joints.
  applyPose(i, pose) {
    const f = this.figures[i];
    if (!f || f.kind === "daz" || !pose) return;
    let bones = pose.bones ? { ...pose.bones } : f.fitPose(pose.points || {});
    if (Array.isArray(pose.only)) {
      const now = {};
      for (const j of JOINTS) now[j] = f.jointAngles(j).map((v, k) => v + f.tweakOf(j)[k]);
      bones = Object.fromEntries(JOINTS.map((j) => [j, pose.only.includes(j) ? bones[j] : now[j]]));
    }
    this.updateFigure(i, {
      pose: "custom", pose_label: pose.name || "Custom pose", anchor: pose.only ? f.anchor : pose.anchor,
      bones, head_turn: 0, lean: 0,
    });
  }

  // The selected mannequin's pose, to save: every joint's angles and how it rests.
  poseOfFigure(i) {
    const f = this.figures[i];
    if (!f || f.kind === "daz") return null;
    const bones = {};
    for (const j of JOINTS) bones[j] = f.jointAngles(j).map((v, k) => Math.round((v + f.tweakOf(j)[k]) * 10) / 10);
    return { name: f.poseLabel.replace(/ \(adjusted\)$/, ""), anchor: f.anchor, bones };
  }

  // ── gizmos ─────────────────────────────────────────────────────────────────
  setGizmo(mode) {
    this.gizmo = ["move", "rotate", "scale", "bones"].includes(mode) ? mode : "off";
    if (this.gizmo !== "bones") this.bone = null;
    const f = this.figures[this.selected];
    if (this.gizmo !== "off" && f?.kind === "daz") this.onStatus("Daz figures keep the pose and place they have in Daz");
    else if (this.gizmo === "bones") this.onStatus("click a yellow joint, then turn it with the rings (Ctrl snaps to 15°)");
    else if (this.gizmo !== "off" && !f) this.onStatus("select a figure first");
    this._syncGizmo();
    this.onGizmo({ mode: this.gizmo, bone: this.bone });
    this.draw();
  }

  selectBone(name) {
    this.bone = JOINTS.includes(name) ? name : null;
    if (this.bone && this.gizmo !== "bones") this.gizmo = "bones";
    this._syncGizmo();
    this.onGizmo({ mode: this.gizmo, bone: this.bone });
    if (this.bone) this.onStatus(JOINT_LABELS[this.bone]);
    this.draw();
  }

  // Back to the preset's angle for the chosen joint (or every joint).
  resetBones(all = false) {
    const f = this.figures[this.selected];
    if (!f || f.kind === "daz") return;
    const bones = { ...f.spec.bones };
    if (all || !this.bone) for (const k of Object.keys(bones)) delete bones[k];
    else if (f.spec.pose === "custom") bones[this.bone] = [0, 0, 0];
    else delete bones[this.bone];
    this.updateFigure(this.selected, { bones });
  }

  // Left and right swapped: the same pose, mirrored.
  mirrorPose() {
    const i = this.selected;
    const p = this.poseOfFigure(i);
    if (!p) return;
    const bones = {};
    for (const j of JOINTS) {
      const [x, y, z] = p.bones[MIRROR[j]];
      bones[j] = [x, -y, -z];
    }
    this.updateFigure(i, { bones, head_turn: 0, lean: 0, ...(this.figures[i].spec.pose === "custom" ? {} : { pose: "custom", pose_label: this.figures[i].poseLabel, anchor: this.figures[i].anchor }) });
  }

  _gizmoFig() {
    const f = this.figures[this.selected];
    return f && f.kind !== "daz" ? f : null;
  }

  // Where the rotate / scale gizmo sits: the hips.
  _pivotOf(f) {
    return f.joints.hips.getWorldPosition(new this.THREE.Vector3());
  }

  _placeHandles(f) {
    for (const m of this._handles.children) {
      f.joints[m.userData.joint].getWorldPosition(m.position);
      m.material = m.userData.joint === this.bone ? this._handleSel : this._handleMat;
    }
    this._sizeHandles();
  }

  // Handles keep about the same size on screen, however far the camera is.
  _sizeHandles() {
    const k = Math.tan((this.camera.fov * DEG) / 2) * 0.022;
    for (const m of this._handles.children) {
      const r = Math.max(0.006, m.position.distanceTo(this.camera.position) * k);
      m.scale.setScalar(m.userData.joint === this.bone ? r * 1.35 : r);
    }
    this._handles.updateMatrixWorld(true);
  }

  _syncGizmo() {
    const tc = this._tc;
    if (!tc || tc.dragging) return;
    const THREE = this.THREE;
    const f = this._gizmoFig();
    const on = !!f && this.gizmo !== "off";
    this._handles.visible = on && this.gizmo === "bones";
    if (this._handles.visible) this._placeHandles(f);
    if (!on || (this.gizmo === "bones" && !this.bone)) { tc.detach(); return; }
    const p = this._proxy;
    const s = f.spec;
    p.scale.set(1, 1, 1);
    p.quaternion.identity();
    if (this.gizmo === "move") {
      p.position.fromArray(s.at);
      tc.setMode("translate");
      tc.setSpace("world");
    } else if (this.gizmo === "rotate") {
      p.position.copy(this._pivotOf(f));
      p.quaternion.setFromEuler(new THREE.Euler(s.pitch * DEG, s.yaw * DEG, s.roll * DEG, "YXZ"));
      tc.setMode("rotate");
      tc.setSpace("local");
    } else if (this.gizmo === "scale") {
      p.position.copy(this._pivotOf(f));
      p.quaternion.setFromEuler(new THREE.Euler(0, s.yaw * DEG, 0));
      tc.setMode("scale");
      tc.setSpace("local");
    } else {
      const j = f.joints[this.bone];
      j.getWorldPosition(p.position);
      j.getWorldQuaternion(p.quaternion);
      tc.setMode("rotate");
      tc.setSpace("local");
    }
    tc.setSize(this.gizmo === "bones" ? 0.6 : 0.85);
    p.updateMatrixWorld(true);
    if (tc.object !== p) tc.attach(p);
  }

  _gizmoStart() {
    const f = this._gizmoFig();
    this._h0 = f?.spec.height_cm || 170;
  }

  // The gizmo moved the proxy: turn that into the figure's spec.
  _gizmoMoved() {
    const f = this._gizmoFig();
    if (!f) return;
    const THREE = this.THREE;
    const p = this._proxy;
    const r1 = (x) => Math.round(x * 10) / 10;
    if (this.gizmo === "move") {
      f.set({ at: p.position.toArray().map((x) => Math.round(x * 1000) / 1000) });
    } else if (this.gizmo === "rotate") {
      const e = new THREE.Euler().setFromQuaternion(p.quaternion, "YXZ");
      f.set({ yaw: r1(e.y / DEG), pitch: r1(e.x / DEG), roll: r1(e.z / DEG) });
      p.position.copy(this._pivotOf(f));
    } else if (this.gizmo === "scale") {
      // Uniform: whichever handle is dragged sets the height.
      const s = [p.scale.x, p.scale.y, p.scale.z].reduce((a, b) => (Math.abs(b - 1) > Math.abs(a - 1) ? b : a), 1);
      const h = Math.round(clamp(this._h0 * s, 60, 230));
      p.scale.setScalar(h / this._h0);
      if (h !== f.spec.height_cm) f.set({ height_cm: h });
      p.position.copy(this._pivotOf(f));
    } else if (this.gizmo === "bones" && this.bone) {
      const j = f.joints[this.bone];
      const parent = j.parent.getWorldQuaternion(new THREE.Quaternion());
      j.quaternion.copy(parent.invert().multiply(p.quaternion));
      const a = f.jointAngles(this.bone);
      // Elbows only fold forward, knees only backward (a ring dragged the wrong way stops at straight).
      if (/_elbow$/.test(this.bone)) a[0] = clamp(a[0], -160, 5);
      else if (/_knee$/.test(this.bone)) a[0] = clamp(a[0], -5, 160);
      f.set({ bones: { ...f.spec.bones, [this.bone]: a } });
      j.getWorldPosition(p.position);
      this._placeHandles(f);
    }
    this.draw();
  }

  _gizmoDone() {
    this._figuresChanged();
    this.onSelect(this.selected); // the panel's sliders show the new turn / height
    this._syncGizmo();
  }

  // Canvas point -> the {x, y} (-1..1) that TransformControls and the ray casts take.
  _ptr(e) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: ((e.clientX - rect.left) / rect.width) * 2 - 1, y: -(((e.clientY - rect.top) / rect.height) * 2 - 1) };
  }

  _pickHandle(ptr) {
    if (!this._handles.visible) return null;
    const rc = new this.THREE.Raycaster();
    rc.setFromCamera(ptr, this.camera);
    // Handles draw on top of the body, so the nearest to the ray wins, not the nearest to the eye.
    let best = null, bestD = Infinity;
    for (const m of this._handles.children) {
      const d = rc.ray.distanceToPoint(m.position);
      if (d < m.scale.x * 1.6 && d < bestD) { bestD = d; best = m.userData.joint; }
    }
    return best;
  }

  removeFigure(i) {
    const f = this.figures[i];
    if (!f) return;
    if (f.kind === "daz") { this.onStatus("Daz figures come from the scene: hide it in Daz and save"); return; }
    f.dispose();
    this.figures.splice(i, 1);
    this.figures.forEach((g, j) => (g._index = j));
    this.select(this.figures.length ? Math.min(i, this.figures.length - 1) : -1);
    this._figuresChanged();
  }

  select(i) {
    const changed = i !== this.selected;
    this.selected = i >= 0 && i < this.figures.length ? i : -1;
    if (changed) this.bone = null;
    this._highlight();
    this._syncGizmo();
    this.onSelect(this.selected);
    this.draw();
  }

  _highlight() {
    this.figures.forEach((f, i) => {
      f.material.emissive?.setHex(i === this.selected ? 0x333333 : 0x000000);
    });
  }

  _figuresChanged() {
    this.draw();
    this.onFigures(this.figureSpecs());
  }

  // Which figure is under a canvas point (CPU ray cast; mannequins are tiny).
  _pickFigure(u, v) {
    const THREE = this.THREE;
    if (!this.figures.length) return -1;
    const rc = new THREE.Raycaster();
    rc.setFromCamera(new THREE.Vector2(u * 2 - 1, -(v * 2 - 1)), this.camera);
    const hits = rc.intersectObject(this.figGroup, true);
    if (!hits.length) return -1;
    // Something in the scene in front of the figure wins.
    const scene = this._probeDepth(u, v, { figures: false });
    const camDist = hits[0].distance * rc.ray.direction.dot(this._fwd());
    if (scene && scene < camDist - 0.02) return -1;
    let o = hits[0].object;
    while (o && !this.figures.some((f) => f.root === o)) o = o.parent;
    return o ? this.figures.findIndex((f) => f.root === o) : -1;
  }

  // Number tags above the heads, on screen only (never in the capture).
  _updateLabels() {
    if (!this.labels) return;
    const THREE = this.THREE;
    const want = this.figures.length;
    while (this.labels.children.length > want) this.labels.lastChild.remove();
    while (this.labels.children.length < want) this.labels.append(document.createElement("div"));
    const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
    const ox = this.canvas.offsetLeft, oy = this.canvas.offsetTop;
    this.figures.forEach((f, i) => {
      const tag = this.labels.children[i];
      const hs = f.headSphere();
      const p = hs.center.clone().add(new THREE.Vector3(0, hs.radius * 1.6, 0)).project(this.camera);
      const hidden = p.z > 1 || p.z < -1 || Math.abs(p.x) > 1.2 || Math.abs(p.y) > 1.2;
      tag.className = "sr-tag" + (i === this.selected ? " sel" : "");
      tag.textContent = String(i + 1);
      tag.style.background = "#" + PALETTE[f.spec.color].hex.toString(16).padStart(6, "0");
      tag.style.display = hidden ? "none" : "";
      tag.style.left = Math.round(ox + (p.x + 1) / 2 * cw) + "px";
      tag.style.top = Math.round(oy + (1 - p.y) / 2 * ch) + "px";
    });
  }

  // ── layout ─────────────────────────────────────────────────────────────────
  setAspect(a) {
    if (!(a > 0) || Math.abs(a - this.aspect) < 1e-6) return;
    this.aspect = a;
    this._layout();
  }

  _layout() {
    if (!this.renderer) return;
    const W = Math.max(1, this.host.clientWidth);
    const H = Math.max(1, this.host.clientHeight);
    let w = W, h = Math.round(W / this.aspect);
    if (h > H) { h = H; w = Math.round(H * this.aspect); }
    this.canvas.style.width = w + "px";
    this.canvas.style.height = h + "px";
    this.canvas.style.left = Math.round((W - w) / 2) + "px";
    this.canvas.style.top = Math.round((H - h) / 2) + "px";
    this.renderer.setSize(w, h, false);
    this.camera.aspect = this.aspect;
    this.camera.updateProjectionMatrix();
    this.draw();
  }

  // ── input ──────────────────────────────────────────────────────────────────
  _wire() {
    const c = this.canvas;
    let drag = null;
    const stop = (e) => e.stopPropagation();
    c.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); });
    c.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      e.preventDefault();
      c.focus({ preventScroll: true });
      c.setPointerCapture(e.pointerId);
      let mode = e.button === 2 ? "look" : e.button === 1 || e.shiftKey ? "pan" : "orbit";
      if (mode === "orbit" && this._tc?.object) {
        // A gizmo handle under the cursor wins over everything else.
        const ptr = this._ptr(e);
        this._tc.pointerHover(ptr);
        if (this._tc.axis) {
          this._tc.pointerDown({ ...ptr, button: 0 });
          drag = { mode: "gizmo", x: e.clientX, y: e.clientY, id: e.pointerId, moved: false };
          return;
        }
      }
      if (mode === "orbit" && this.gizmo === "bones") {
        const j = this._pickHandle(this._ptr(e));
        if (j) {
          this.selectBone(j);
          drag = { mode: "none", x: e.clientX, y: e.clientY, id: e.pointerId, moved: false };
          return;
        }
      }
      if (mode === "orbit" && this.figures.length) {
        const rect = c.getBoundingClientRect();
        const hit = this._pickFigure((e.clientX - rect.left) / rect.width, (e.clientY - rect.top) / rect.height);
        // Daz figures keep the pose and place they have in Daz: select, then orbit. With the
        // rotate / scale / bones gizmo on, a click only selects (the gizmo does the work).
        if (hit >= 0) {
          this.select(hit);
          if (this.figures[hit].kind !== "daz" && (this.gizmo === "off" || this.gizmo === "move")) mode = "figure";
        }
      }
      drag = { mode, x: e.clientX, y: e.clientY, id: e.pointerId, moved: false };
      if (mode === "look") this._looking = true;
      this._euler0 = null;
    });
    c.addEventListener("pointermove", (e) => {
      if (!drag) {
        // Hovering: light up the gizmo handle under the cursor.
        if (this._tc?.object) {
          const before = this._tc.axis;
          this._tc.pointerHover(this._ptr(e));
          if (this._tc.axis !== before) this.draw();
        }
        return;
      }
      if (e.pointerId !== drag.id) return;
      e.stopPropagation();
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      drag.moved = true;
      if (drag.mode === "gizmo") {
        const snap = e.ctrlKey;
        this._tc.translationSnap = snap ? 0.05 : null;
        this._tc.rotationSnap = snap ? 15 * DEG : null;
        this._tc.scaleSnap = snap ? 0.05 : null;
        this._tc.pointerMove({ ...this._ptr(e), button: -1 });
        return;
      }
      if (drag.mode === "none") return;
      if (drag.mode === "figure") {
        // Slide over whatever surface is under the cursor: floor, seat, bed.
        const rect = c.getBoundingClientRect();
        const f = this.figures[this.selected];
        const p = this._placeFor(f, this._surfacePoint((e.clientX - rect.left) / rect.width, (e.clientY - rect.top) / rect.height, { figures: false }));
        if (p && f) { f.set({ at: p.toArray().map((x) => Math.round(x * 1000) / 1000) }); this.draw(); }
        return;
      }
      this._drag(drag.mode, dx, dy);
    });
    const end = (e) => {
      if (!drag || e.pointerId !== drag.id) return;
      e.stopPropagation();
      if (drag.mode === "look") { this._looking = false; this.keys.clear(); }
      if (drag.mode === "gizmo") this._tc.pointerUp({ ...this._ptr(e), button: 0 });
      if (drag.mode === "figure" && drag.moved) this._figuresChanged();
      drag = null;
      try { c.releasePointerCapture(e.pointerId); } catch (_e) { /* already released */ }
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const s = Math.sign(e.deltaY);
      if (this._looking) {
        this.flySpeed = clamp(this.flySpeed * (s < 0 ? 1.25 : 0.8), 0.05, 500);
        this.onStatus(`fly speed ${this.flySpeed.toFixed(2)} m/s`);
        return;
      }
      this._dolly(-s * (e.ctrlKey ? 0.3 : 1));
    }, { passive: false });
    c.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      const rect = c.getBoundingClientRect();
      const u = (e.clientX - rect.left) / rect.width, v = (e.clientY - rect.top) / rect.height;
      this._pivotAt(u, v);
    });
    c.addEventListener("keydown", (e) => {
      e.stopPropagation();
      const k = e.key.toLowerCase();
      if ("wasdqe".includes(k) && k.length === 1 || k.startsWith("arrow")) {
        e.preventDefault();
        this.keys.add(k);
        this._loop();
      } else if (k === "f") {
        this.frameAll();
      } else if (k === "r") {
        this.goHome();
      } else if ((k === "[" || k === "]") && this.selected >= 0 && this.figures[this.selected].kind !== "daz") {
        const f = this.figures[this.selected];
        this.updateFigure(this.selected, { yaw: Math.round(f.spec.yaw + (k === "]" ? 1 : -1) * (e.shiftKey ? 5 : 15)) });
      } else if ((k === "delete" || k === "backspace") && this.selected >= 0) {
        this.removeFigure(this.selected);
      } else if (k === "escape") {
        if (this.bone) this.selectBone(null);
        else this.select(-1);
      } else if ("01234".includes(k) && k.length === 1 && !e.ctrlKey && !e.altKey) {
        this.setGizmo(["off", "move", "rotate", "scale", "bones"][+k]);
      }
      this.keys.shift = e.shiftKey;
      this.keys.ctrl = e.ctrlKey;
    });
    c.addEventListener("keyup", (e) => {
      e.stopPropagation();
      this.keys.delete(e.key.toLowerCase());
      this.keys.shift = e.shiftKey;
      this.keys.ctrl = e.ctrlKey;
    });
    c.addEventListener("blur", () => this.keys.clear());
    for (const t of ["pointerenter", "pointerleave", "click", "mousedown", "mouseup"]) c.addEventListener(t, stop);
  }

  _drag(mode, dx, dy) {
    const THREE = this.THREE;
    if (!THREE) return;
    const rot = 0.0045;
    this.euler.z = 0; // any interaction levels the horizon
    if (mode === "orbit") {
      const pivot = this.pos.clone().addScaledVector(this._fwd(), this.dist);
      this.euler.y -= dx * rot;
      this.euler.x = clamp(this.euler.x - dy * rot, -89.5 * DEG, 89.5 * DEG);
      this.pos.copy(pivot).addScaledVector(this._fwd(), -this.dist);
    } else if (mode === "look") {
      this.euler.y -= dx * rot * 0.8;
      this.euler.x = clamp(this.euler.x - dy * rot * 0.8, -89.5 * DEG, 89.5 * DEG);
    } else {
      const h = this.canvas.clientHeight || 1;
      const s = (2 * this.dist * Math.tan((this.camera.fov * DEG) / 2)) / h;
      const right = new THREE.Vector3(1, 0, 0).applyEuler(this.euler);
      const up = new THREE.Vector3(0, 1, 0).applyEuler(this.euler);
      this.pos.addScaledVector(right, -dx * s).addScaledVector(up, dy * s);
    }
    this._sync();
  }

  _dolly(steps) {
    const minD = Math.max(this.camera.near * 4, this.sceneSize * 0.002 || 0.01);
    const step = Math.max(this.dist * 0.15, minD) * steps;
    this.pos.addScaledVector(this._fwd(), step);
    this.dist = Math.max(minD, this.dist - step);
    this._sync();
  }

  _pivotAt(u, v) {
    const THREE = this.THREE;
    const d = this._probeDepth(u, v);
    if (!d) return;
    // Turn towards the picked point, keeping the camera where it is.
    const ndc = new THREE.Vector3(u * 2 - 1, -(v * 2 - 1), 0.5).unproject(this.camera);
    const dir = ndc.sub(this.camera.position).normalize();
    const fwd = this._fwd();
    const target = this.pos.clone().addScaledVector(dir, d / Math.max(0.05, dir.dot(fwd)));
    const look = target.clone().sub(this.pos).normalize();
    this.euler.set(Math.asin(clamp(look.y, -1, 1)), Math.atan2(-look.x, -look.z), 0, "YXZ");
    this.dist = this.pos.distanceTo(target);
    this._sync();
    this.onStatus(`pivot ${this.dist.toFixed(2)} m away`);
  }

  _loop() {
    if (this._raf) return;
    this._last = performance.now();
    const tick = (now) => {
      const dt = Math.min(0.1, (now - this._last) / 1000);
      this._last = now;
      if (!this.keys.size) { this._raf = 0; return; }
      const THREE = this.THREE;
      const v = this.flySpeed * dt * (this.keys.shift ? 4 : 1) * (this.keys.ctrl ? 0.25 : 1);
      const fwd = this._fwd();
      const right = new THREE.Vector3(1, 0, 0).applyEuler(this.euler);
      const up = new THREE.Vector3(0, 1, 0);
      const k = this.keys;
      if (k.has("w") || k.has("arrowup")) this.pos.addScaledVector(fwd, v);
      if (k.has("s") || k.has("arrowdown")) this.pos.addScaledVector(fwd, -v);
      if (k.has("d") || k.has("arrowright")) this.pos.addScaledVector(right, v);
      if (k.has("a") || k.has("arrowleft")) this.pos.addScaledVector(right, -v);
      if (k.has("e")) this.pos.addScaledVector(up, v);
      if (k.has("q")) this.pos.addScaledVector(up, -v);
      this._sync();
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  // ── capture ────────────────────────────────────────────────────────────────
  // Renders the current view at w x h. Returns PNG blobs + the camera.
  async capture(w, h, { depthCurve = "inverse" } = {}) {
    await this._init;
    if (!this.ready) throw new Error("the scene is not loaded yet");
    const THREE = this.THREE;
    const r = this.renderer;
    const prevSize = r.getSize(new THREE.Vector2());
    const prevRatio = r.getPixelRatio();
    const prevAspect = this.camera.aspect;
    const look = this.look;
    const sel = this.selected;
    this.selected = -1; // no selection glow in the picture
    this._highlight();
    try {
      r.setPixelRatio(1);
      r.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();

      const grab = () => {
        const cv = document.createElement("canvas");
        cv.width = w; cv.height = h;
        cv.getContext("2d").drawImage(r.domElement, 0, 0, w, h);
        return cv;
      };
      const figs = this.figures.length > 0;
      // 1) colour, with the current look (depth / normal previews give colour here):
      //    the clean scene, then the same view with the mannequins.
      this.look = look === "clay" ? "clay" : "color";
      const restore = this._setVisible({ figures: false });
      this._render(true);
      restore();
      const color = grab();
      this._render(true);
      const composite = figs ? grab() : color;

      // 2) one float pass -> depth, normal, mask (mannequins included).
      const rt = this._floatRT(w, h);
      this._renderData(rt);
      const buf = this._readFloat(rt);
      const passes = this._buildPasses(buf, w, h, depthCurve);
      // 3) figures: who owns each pixel, each full silhouette, keypoints.
      let fig = null;
      if (figs) fig = this._figurePasses(rt, w, h);
      // 4) regions: which object owns each pixel (MOSS-Image's colour region map).
      const regions = this._regionPass(rt, w, h);
      rt.dispose();
      const toBlob = (cv) => new Promise((res, rej) =>
        cv.toBlob((b) => (b ? res(b) : rej(new Error("could not encode PNG"))), "image/png"));
      const list = [color, passes.depth, passes.normal, passes.mask, composite, regions];
      if (fig) list.push(fig.pose, fig.figmap);
      const blobs = await Promise.all(list.map(toBlob));
      const out = { image: blobs[0], depth: blobs[1], normal: blobs[2], mask: blobs[3], composite: blobs[4],
        regions: blobs[5] };
      if (fig) { out.pose = blobs[6]; out.figmap = blobs[7]; }
      const camera = this.cameraInfo(w, h, passes.near, passes.far);
      camera.figures = fig ? fig.info : [];
      return { ...out, camera };
    } finally {
      this.look = look;
      this.selected = sel;
      this._highlight();
      r.setPixelRatio(prevRatio);
      r.setSize(prevSize.x, prevSize.y, false);
      this.camera.aspect = prevAspect;
      this.camera.updateProjectionMatrix();
      this.draw();
    }
  }

  // Figure passes at the capture size (rt is a w x h float target to reuse).
  // figmap PNG: R = full-silhouette bits (bit i = figure i), G = id of the figure
  // you actually see there (0 = none), so a mannequin behind a table has its legs
  // in R but not in G.
  _figurePasses(rt, w, h) {
    const THREE = this.THREE;
    const n = w * h;
    const figs = this.figures;
    // Scene-only depth, to test which joints something in the scene hides.
    this._renderData(rt, { figures: false });
    const sceneBuf = this._readFloat(rt);
    // Which figure is visible at each pixel.
    this._renderData(rt, { ids: true });
    const idBuf = this._readFloat(rt);
    const full = new Uint8Array(n);
    const vis = new Uint8Array(n);
    const stats = figs.map(() => ({ full: 0, vis: 0, fb: [w, h, -1, -1], vb: [w, h, -1, -1] }));
    const grow = (b, x, y) => { if (x < b[0]) b[0] = x; if (y < b[1]) b[1] = y; if (x > b[2]) b[2] = x; if (y > b[3]) b[3] = y; };
    for (let i = 0; i < figs.length; i++) {
      this._renderData(rt, { ids: true, model: false, figures: false, only: figs[i] });
      const one = this._readFloat(rt);
      const bit = 1 << i;
      const st = stats[i];
      for (let y = 0; y < h; y++) {
        const src = (h - 1 - y) * w;
        for (let x = 0; x < w; x++) {
          if (one[(src + x) * 4] > 0.5) { full[y * w + x] |= bit; st.full++; grow(st.fb, x, y); }
        }
      }
    }
    for (let y = 0; y < h; y++) {
      const src = (h - 1 - y) * w;
      for (let x = 0; x < w; x++) {
        const id = Math.round(idBuf[(src + x) * 4]);
        if (id > 0 && id <= figs.length) { vis[y * w + x] = id; stats[id - 1].vis++; grow(stats[id - 1].vb, x, y); }
      }
    }
    const mk = () => { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; };
    const fm = mk();
    const fctx = fm.getContext("2d");
    const img = fctx.createImageData(w, h);
    for (let i = 0; i < n; i++) {
      img.data[i * 4] = full[i]; img.data[i * 4 + 1] = vis[i]; img.data[i * 4 + 3] = 255;
    }
    fctx.putImageData(img, 0, 0);

    // Keypoints: project, and test against the scene-only depth.
    const people = [];
    const info = [];
    const camInv = this.camera.quaternion.clone().invert();
    const fy = h / (2 * Math.tan((this.camera.fov * DEG) / 2));
    figs.forEach((f, i) => {
      const kps = f.keypoints();
      const pts = COCO.map((name) => {
        const wp = kps[name];
        const pc = wp.clone().project(this.camera);
        const x = (pc.x + 1) / 2 * w, y = (1 - pc.y) / 2 * h;
        const depth = -wp.clone().applyMatrix4(this.camera.matrixWorldInverse).z;
        const inFrame = pc.z > -1 && pc.z < 1 && x >= 0 && x < w && y >= 0 && y < h && depth > 0;
        let visible = inFrame;
        if (inFrame) {
          const s = sceneBuf[((h - 1 - Math.floor(y)) * w + Math.floor(x)) * 4];
          if (s > 0 && s < depth - (0.04 + 0.01 * depth)) visible = false;
        }
        return { x, y, visible, inFrame, name };
      });
      people.push(pts);
      const byName = Object.fromEntries(pts.map((p) => [p.name, p]));
      const hidden = Object.entries(PARTS)
        .filter(([_part, names]) => names.filter((k) => !byName[k].visible).length * 2 > names.length)
        .map(([part]) => part);
      const outOfFrame = Object.entries(PARTS)
        .filter(([_part, names]) => names.filter((k) => !byName[k].inFrame).length * 2 > names.length)
        .map(([part]) => part);
      // Head box from the head sphere.
      const hs = f.headSphere();
      const hc = hs.center.clone().project(this.camera);
      const hd = -hs.center.clone().applyMatrix4(this.camera.matrixWorldInverse).z;
      const rpx = hd > 0 ? hs.radius * fy / hd : 0;
      const hx = (hc.x + 1) / 2 * w, hy = (1 - hc.y) / 2 * h;
      // Facing, relative to the camera (0 = towards it, +90 = towards frame right).
      const fc = f.facing().applyQuaternion(camInv);
      const yawCam = Math.atan2(fc.x, fc.z) * 180 / Math.PI;
      const st = stats[i];
      const box = (b) => (b[2] < 0 ? null : [b[0], b[1], b[2] + 1, b[3] + 1]);
      const s = f.spec;
      info.push({
        index: i + 1, label: `Figure ${i + 1}`, color: PALETTE[s.color].name,
        kind: f.kind, look: f.kind === "daz" ? s.look : "color", height_cm: s.height_cm, sex: s.sex || null, build: s.build || null,
        pose: s.pose || "daz", pose_label: f.kind === "daz" ? f.poseLabel : f.promptLabel,
        posture: f.kind === "daz" ? null : f.posture(), name: f.label || null,
        distance_m: +f.joints.hips.getWorldPosition(new THREE.Vector3()).distanceTo(this.camera.position).toFixed(2),
        visible_fraction: st.full ? +(st.vis / st.full).toFixed(3) : 0,
        bbox_visible: box(st.vb), bbox_full: box(st.fb),
        head_box: rpx > 0 ? [Math.round(hx - rpx), Math.round(hy - rpx), Math.round(hx + rpx), Math.round(hy + rpx)] : null,
        face_visible: ["nose", "r_eye", "l_eye"].some((k) => byName[k].visible) && Math.abs(yawCam) < 110,
        hidden_parts: hidden, out_of_frame: outOfFrame,
        facing: describeFacing(yawCam, f), facing_deg: Math.round(yawCam),
        keypoints: pts.map((p) => [+p.x.toFixed(1), +p.y.toFixed(1), p.visible ? 1 : 0]),
      });
    });
    const pc = mk();
    drawOpenPose(pc.getContext("2d"), w, h, people);
    return { pose: pc, figmap: fm, info };
  }

  // Regions PNG: the id of the object you see at each pixel as 24-bit RGB (R low byte),
  // 0 = no surface. One id per scene object and one per figure (all its meshes). A
  // multi-material object, which the GLB loader splits into a group of meshes, counts once.
  // Ids are only labels: the Studio turns them into one flat colour per region.
  _regionPass(rt, w, h) {
    const THREE = this.THREE;
    const owners = new Map(); // mesh -> what it belongs to
    const ownerOf = (o) => {
      const p = o.parent;
      const split = p && p.isGroup && p.children.length > 1 && p.children.every((c) => c.isMesh && !c.children.length);
      return split ? p : o;
    };
    this.model?.traverse((o) => { if (o.isMesh) owners.set(o, ownerOf(o)); });
    for (const f of this.figures) f.root.traverse((o) => { if (o.isMesh) owners.set(o, f); });
    const ids = new Map();
    for (const ow of owners.values()) if (!ids.has(ow)) ids.set(ow, ids.size + 1);
    const made = [];
    const swapped = [];
    const hidden = [];
    const regionMat = (m, id) => {
      const cut = hasCutout(m);
      const d = new THREE.ShaderMaterial({
        vertexShader: DATA_VERT, fragmentShader: DATA_FRAG,
        uniforms: {
          map: { value: cut ? m.map : null }, useMap: { value: cut ? 1 : 0 }, alphaCut: { value: m.alphaTest || 0.5 },
          idMode: { value: 1 }, figId: { value: id },
        },
        side: THREE.DoubleSide,
      });
      d.visible = !isGhost(m);
      made.push(d);
      return d;
    };
    // Everything else (gizmos, helpers, lines) stays out of the id picture.
    this.scene.traverse((o) => {
      if (owners.has(o)) {
        const id = ids.get(owners.get(o));
        swapped.push([o, o.material]);
        o.material = Array.isArray(o.material) ? o.material.map((m) => regionMat(m, id)) : regionMat(o.material, id);
      } else if ((o.isMesh || o.isLine || o.isPoints || o.isSprite) && o.visible) {
        hidden.push(o);
        o.visible = false;
      }
    });
    const r = this.renderer;
    const bg = this.scene.background;
    const env = this.scene.environment;
    const prevClear = r.getClearColor(new THREE.Color());
    const prevAlpha = r.getClearAlpha();
    const vis = this._setVisible({ figures: true, model: true });
    let buf;
    try {
      this.scene.background = null;
      this.scene.environment = null;
      r.setRenderTarget(rt);
      r.setClearColor(0x000000, 0);
      r.clear();
      r.render(this.scene, this.camera);
      buf = this._readFloat(rt);
    } finally {
      r.setRenderTarget(null);
      r.setClearColor(prevClear, prevAlpha);
      this.scene.background = bg;
      this.scene.environment = env;
      vis();
      for (const [o, m] of swapped) o.material = m;
      for (const o of hidden) o.visible = true;
      for (const d of made) d.dispose();
    }
    const cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      const src = (h - 1 - y) * w; // GL rows are bottom-up
      for (let x = 0; x < w; x++) {
        const id = Math.max(0, Math.round(buf[(src + x) * 4]));
        const o = (y * w + x) * 4;
        img.data[o] = id & 255; img.data[o + 1] = (id >> 8) & 255; img.data[o + 2] = (id >> 16) & 255;
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return cv;
  }

  _buildPasses(buf, w, h, depthCurve) {
    const n = w * h;
    const ds = [];
    const stride = Math.max(1, Math.floor(n / 200000));
    for (let i = 0; i < n; i += stride) { const d = buf[i * 4]; if (d > 0) ds.push(d); }
    const [near, far] = depthRange(ds) || [1, 10];
    const mk = () => { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; };
    const dc = mk(), nc = mk(), mc = mk();
    const di = dc.getContext("2d").createImageData(w, h);
    const ni = nc.getContext("2d").createImageData(w, h);
    const mi = mc.getContext("2d").createImageData(w, h);
    const linear = depthCurve === "linear";
    const span = Math.max(1e-6, far - near);
    const invFar = 1 / far;
    const invSpan = Math.max(1e-9, 1 / near - 1 / far);
    for (let y = 0; y < h; y++) {
      const src = (h - 1 - y) * w; // GL rows are bottom-up
      for (let x = 0; x < w; x++) {
        const s = (src + x) * 4;
        const o = (y * w + x) * 4;
        const d = buf[s];
        let dv = 0, nr = 0, ng = 0, nb = 0, m = 0;
        if (d > 0) {
          // Disparity, min-max normalised per picture like MiDaS / Depth Anything:
          // the nearest surface is white, the farthest black.
          dv = linear ? clamp(1 - (d - near) / span, 0, 1) : clamp((1 / d - invFar) / invSpan, 0, 1);
          nr = buf[s + 1] * 0.5 + 0.5; ng = buf[s + 2] * 0.5 + 0.5; nb = buf[s + 3] * 0.5 + 0.5;
          m = 1;
        } else {
          // Background: no surface, "facing the camera" in the normal map like a ControlNet
          // preprocessor's empty sky.
          nr = 0.5; ng = 0.5; nb = 1;
        }
        const dvb = Math.round(dv * 255);
        di.data[o] = di.data[o + 1] = di.data[o + 2] = dvb; di.data[o + 3] = 255;
        ni.data[o] = Math.round(nr * 255); ni.data[o + 1] = Math.round(ng * 255);
        ni.data[o + 2] = Math.round(nb * 255); ni.data[o + 3] = 255;
        mi.data[o] = mi.data[o + 1] = mi.data[o + 2] = m * 255; mi.data[o + 3] = 255;
      }
    }
    dc.getContext("2d").putImageData(di, 0, 0);
    nc.getContext("2d").putImageData(ni, 0, 0);
    mc.getContext("2d").putImageData(mi, 0, 0);
    return { depth: dc, normal: nc, mask: mc, near, far };
  }

  cameraInfo(w, h, near, far) {
    const v = this.getView();
    const fovV = this.camera.fov;
    const aspect = w / h;
    const fovH = 2 * Math.atan(Math.tan((fovV * DEG) / 2) * aspect) / DEG;
    // Focal length on a 36 x 24 mm sensor, horizontal fit (what Blender / Daz show).
    const lens = 18 / Math.tan((fovH * DEG) / 2);
    return {
      position: v.pos, quaternion: v.quat, fov_vertical: +fovV.toFixed(3), fov_horizontal: +fovH.toFixed(3),
      focal_length_mm_36: +lens.toFixed(2), width: w, height: h, up: "Y", units: "m",
      depth_near_m: near != null ? +near.toFixed(4) : null, depth_far_m: far != null ? +far.toFixed(4) : null,
      scene: this.key,
    };
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    this._ro?.disconnect();
    this._prevRT?.dispose();
    this._smallRT?.dispose();
    for (const m of this._matCache.values()) m.dispose();
    this._matCache.clear();
    for (const f of this.figures) f.dispose();
    this.figures = [];
    this.labels?.remove();
    // Not tc.dispose(): it unhooks pointer events from a canvas it was never given.
    this._tc?.detach();
    this._tc?.getHelper().dispose?.();
    this._handleGeo?.dispose();
    this._handleMat?.dispose();
    this._handleSel?.dispose();
    this.renderer?.dispose();
    this.renderer?.forceContextLoss?.();
    this.canvas.remove();
  }
}
