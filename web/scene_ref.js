// SceneRef - the node face and the Run hook for "Scene Reference Viewer".
//
// Face: Browse / Open / Reconvert, a status line while Blender or Daz converts
// the scene, the 3D view (framed to the output aspect), camera + saved shots,
// look / light / exposure. Everything the user sets lives in
// node.properties.sceneRef so it is saved with the workflow.
//
// Run: app.graphToPrompt is wrapped; for every SceneRef node in the prompt the
// view (or every saved shot) is rendered at width x height, the four PNGs are
// uploaded to temp/scene_ref, and their names go into the hidden input.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
// Bump the query whenever scene_ref_viewer.mjs changes: browsers cache it hard.
import { Viewer, poseThumbnail } from "./scene_ref_viewer.mjs?v=13";
import { POSES, PALETTE, BUILDS, JOINT_LABELS, COCO, drawOpenPose } from "./scene_ref_figures.mjs?v=6";

const CLASS = "SceneRefViewer";
const HIDDEN = "scene_ref_capture";
const PROP = "sceneRef";
const MIN_UI_H = 460;

// ── styles ───────────────────────────────────────────────────────────────────
const CSS = `
.sr-root{display:flex;flex-direction:column;gap:5px;width:100%;height:100%;box-sizing:border-box;
  font:12px/1.3 system-ui,sans-serif;color:var(--input-text,#ddd);padding:2px 0}
.sr-row{display:flex;gap:4px;align-items:center;flex-wrap:nowrap;min-height:24px}
.sr-row>*{flex:0 0 auto}
.sr-grow{flex:1 1 auto!important;min-width:0}
.sr-btn{background:var(--comfy-input-bg,#222);color:inherit;border:1px solid var(--border-color,#444);
  border-radius:5px;padding:3px 8px;cursor:pointer;font:inherit;white-space:nowrap}
.sr-btn:hover{border-color:#7aa7ff}
.sr-btn.on{background:#2f5bb7;border-color:#4d7ce0;color:#fff}
.sr-btn:disabled{opacity:.45;cursor:default}
.sr-sel,.sr-num{background:var(--comfy-input-bg,#222);color:inherit;border:1px solid var(--border-color,#444);
  border-radius:5px;padding:2px 4px;font:inherit;min-width:0}
.sr-status{font-size:11px;opacity:.85;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sr-status.err{color:#ff8a80;opacity:1;white-space:normal}
.sr-vp{position:relative;flex:1 1 auto;min-height:120px;background:#16181b;border-radius:6px;overflow:hidden}
.sr-canvas{position:absolute;outline:none;box-shadow:0 0 0 1px #4d7ce0}
.sr-canvas:focus{box-shadow:0 0 0 2px #7aa7ff}
.sr-overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;
  padding:12px;pointer-events:none;color:#aaa;font-size:12px;white-space:pre-line}
.sr-hint{position:absolute;left:6px;bottom:4px;font-size:10px;color:#9aa;pointer-events:none;
  text-shadow:0 1px 2px #000;white-space:nowrap;overflow:hidden;max-width:calc(100% - 12px)}
.sr-flash{position:absolute;right:6px;top:4px;font-size:11px;color:#fff;background:#0009;padding:2px 6px;
  border-radius:4px;pointer-events:none;transition:opacity .3s}
.sr-range{width:80px}
.sr-lbl{opacity:.7;font-size:11px}
.sr-modal{position:fixed;inset:0;background:#0008;z-index:10000;display:flex;align-items:center;justify-content:center}
.sr-dlg{background:#1e2024;color:#ddd;border:1px solid #444;border-radius:8px;width:min(720px,92vw);height:min(560px,86vh);
  display:flex;flex-direction:column;font:13px system-ui,sans-serif;box-shadow:0 10px 40px #000a}
.sr-dlg header{padding:10px 12px;border-bottom:1px solid #333;display:flex;gap:6px;align-items:center}
.sr-dlg header input{flex:1;background:#111;color:#ddd;border:1px solid #444;border-radius:5px;padding:4px 6px;font:inherit}
.sr-list{flex:1;overflow:auto;padding:4px 0}
.sr-item{padding:4px 14px;cursor:pointer;display:flex;gap:8px;align-items:center}
.sr-item:hover{background:#2a3140}
.sr-item .sz{margin-left:auto;opacity:.6;font-size:11px}
.sr-dlg footer{padding:8px 12px;border-top:1px solid #333;font-size:11px;opacity:.7}
.sr-labels{position:absolute;inset:0;pointer-events:none;overflow:hidden}
.sr-tag{position:absolute;transform:translate(-50%,-100%);font:bold 11px system-ui,sans-serif;color:#fff;
  border-radius:9px;padding:0 6px;box-shadow:0 1px 3px #000a;text-shadow:0 1px 1px #0008}
.sr-tag.sel{outline:2px solid #fff}
.sr-num{width:52px}
.sr-hide{display:none!important}
.sr-dlg.wide{width:min(980px,96vw);height:min(720px,92vh)}
.sr-tabs{display:flex;gap:4px;padding:6px 12px 0}
.sr-tabs .sr-btn{border-bottom-left-radius:0;border-bottom-right-radius:0}
.sr-bar{display:flex;gap:6px;align-items:center;padding:6px 12px;border-bottom:1px solid #333;flex-wrap:wrap}
.sr-bar input[type=text],.sr-bar input:not([type]){flex:1;min-width:120px;background:#111;color:#ddd;border:1px solid #444;border-radius:5px;padding:4px 6px;font:inherit}
.sr-grid{flex:1;overflow:auto;padding:8px 10px;display:grid;grid-template-columns:repeat(auto-fill,minmax(118px,1fr));gap:8px;align-content:start;grid-auto-rows:max-content}
.sr-card{background:#262a31;border:1px solid #363b44;border-radius:6px;cursor:pointer;display:flex;flex-direction:column;overflow:hidden;min-height:0}
.sr-card:hover{border-color:#7aa7ff}
.sr-card.dir{justify-content:center;align-items:center;min-height:60px;font-size:12px;padding:6px;text-align:center}
.sr-card .th{flex:0 0 auto;height:118px;background:#1a1c20 center/contain no-repeat;display:flex;align-items:center;justify-content:center;color:#666;font-size:11px}
.sr-card .nm{font-size:11px;padding:3px 5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sr-card .tg{font-size:9px;opacity:.6;padding:0 5px 3px}
.sr-drop{border:2px dashed #555;border-radius:8px;margin:10px;padding:18px;text-align:center;color:#aaa}
.sr-drop.on{border-color:#7aa7ff;color:#fff}
`;
function injectCss() {
  if (document.getElementById("scene-ref-css")) return;
  const s = document.createElement("style");
  s.id = "scene-ref-css";
  s.textContent = CSS;
  document.head.appendChild(s);
}

// ── helpers ──────────────────────────────────────────────────────────────────
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
};
const widget = (node, name) => node.widgets?.find((w) => w.name === name);
const fmtMB = (b) => (b > 1e9 ? (b / 1e9).toFixed(2) + " GB" : (b / 1e6).toFixed(1) + " MB");
const fmtTime = (s) => (s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`);
const fmtInt = (n) => Math.round(n).toLocaleString("en-US");

function props(node) {
  node.properties = node.properties || {};
  const p = node.properties[PROP] || {};
  return {
    view: p.view || null,
    shots: Array.isArray(p.shots) ? p.shots : [],
    look: p.look || "color",
    light: p.light || "studio",
    exposure: typeof p.exposure === "number" ? p.exposure : 1,
    allShots: !!p.allShots,
    depthCurve: p.depthCurve === "linear" ? "linear" : "inverse",
    figures: Array.isArray(p.figures) ? p.figures : [],
  };
}
function setProps(node, patch) {
  node.properties = node.properties || {};
  node.properties[PROP] = { ...props(node), ...patch };
}

async function postJSON(route, body) {
  const res = await api.fetchApi(route, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}
async function getJSON(route) {
  const res = await api.fetchApi(route);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// ── file browser ─────────────────────────────────────────────────────────────
function browse(startPath) {
  return new Promise((resolve) => {
    const modal = el("div", "sr-modal");
    const dlg = el("div", "sr-dlg");
    const head = el("header");
    const up = el("button", "sr-btn", "↑ Up");
    const input = el("input");
    const go = el("button", "sr-btn", "Go");
    const cancel = el("button", "sr-btn", "Cancel");
    head.append(up, input, go, cancel);
    const list = el("div", "sr-list");
    const foot = el("footer", null, "Blender (.blend), Daz Studio (.duf) and .glb files. Double-click a folder to open it, click a file to pick it.");
    dlg.append(head, list, foot);
    modal.append(dlg);
    let parent = "";
    const close = (v) => { modal.remove(); document.removeEventListener("keydown", onKey, true); resolve(v); };
    const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(null); } };
    document.addEventListener("keydown", onKey, true);
    modal.addEventListener("pointerdown", (e) => { if (e.target === modal) close(null); });
    cancel.onclick = () => close(null);
    const open = async (path) => {
      list.textContent = "Loading…";
      try {
        const d = await getJSON("/scene_ref/browse?path=" + encodeURIComponent(path || ""));
        input.value = d.path || "";
        parent = d.parent ?? "";
        up.disabled = !d.path;
        list.textContent = "";
        if (d.path) try { localStorage.setItem("sceneRef.lastDir", d.path); } catch (_e) { /* private mode */ }
        for (const name of d.dirs) {
          const it = el("div", "sr-item", "📁 " + name);
          const full = d.path ? d.path.replace(/[\\/]+$/, "") + (d.path.includes("/") && !d.path.includes("\\") ? "/" : "\\") + name : name;
          it.onclick = () => open(full);
          list.append(it);
        }
        for (const f of d.files) {
          const icon = f.name.toLowerCase().endsWith(".duf") ? "🧍" : f.name.toLowerCase().endsWith(".blend") ? "🟧" : "🧊";
          const it = el("div", "sr-item", `${icon} ${f.name}`);
          it.append(el("span", "sz", fmtMB(f.size)));
          const sep = d.path.includes("/") && !d.path.includes("\\") ? "/" : "\\";
          it.onclick = () => close(d.path.replace(/[\\/]+$/, "") + sep + f.name);
          list.append(it);
        }
        if (!d.dirs.length && !d.files.length) list.append(el("div", "sr-item", "(no folders or scene files here)"));
      } catch (e) {
        list.textContent = "⚠ " + e.message;
      }
    };
    up.onclick = () => open(parent);
    go.onclick = () => open(input.value.trim());
    input.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") open(input.value.trim()); });
    document.body.append(modal);
    let start = startPath ? startPath.replace(/[\\/][^\\/]*$/, "") : "";
    if (!start) try { start = localStorage.getItem("sceneRef.lastDir") || ""; } catch (_e) { /* private mode */ }
    open(start);
  });
}

// ── pose browser ─────────────────────────────────────────────────────────────
// Library (CMU motion capture, bundled), My poses (models/poses), Files (Daz pose presets,
// OpenPose JSON, BVH anywhere on disk) and OpenPose (drop or paste keypoints). Clicking a
// pose puts it on the selected mannequin; the dialog stays open to try others.
const _thumbs = new Map(); // pose id -> data URL (mannequin previews drawn here)
let _thumbQueue = Promise.resolve();
function mannequinThumb(id, getPose) {
  if (_thumbs.has(id)) return Promise.resolve(_thumbs.get(id));
  const job = _thumbQueue.then(async () => {
    if (_thumbs.has(id)) return _thumbs.get(id);
    const pose = await getPose();
    const url = await poseThumbnail(pose.poses ? pose.poses[0] : pose);
    _thumbs.set(id, url);
    return url;
  });
  _thumbQueue = job.catch(() => {});
  return job;
}

// Stick figure of OpenPose keypoints, as the ControlNet preprocessors draw them.
function openPoseThumb(p) {
  const cv = document.createElement("canvas");
  cv.width = cv.height = 120;
  const kps = p.keypoints || [];
  const pts = kps.filter((k) => k[2] > 0);
  if (!pts.length) return null;
  const xs = pts.map((k) => k[0]), ys = pts.map((k) => k[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const s = 100 / Math.max(x1 - x0, y1 - y0, 1e-6);
  const names = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"];
  const byName = {};
  const body25 = kps.length === 25;
  const order = body25 ? ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "pelvis",
    "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"] : names;
  order.forEach((n, i) => { byName[n] = kps[i]; });
  const one = COCO.map((n) => {
    const k = byName[n];
    return k && k[2] > 0 ? { x: 10 + (k[0] - x0) * s + (100 - (x1 - x0) * s) / 2, y: 10 + (k[1] - y0) * s, visible: true, inFrame: true, name: n }
      : { x: 0, y: 0, visible: false, inFrame: false, name: n };
  });
  drawOpenPose(cv.getContext("2d"), 120, 120, [one]);
  return cv.toDataURL("image/png");
}

function poseBrowser(node, startTab = "library") {
  const v = node._sr.viewer;
  const modal = el("div", "sr-modal");
  const dlg = el("div", "sr-dlg wide");
  const head = el("header");
  const title = el("div", "sr-grow", "Poses");
  title.style.fontWeight = "600";
  const target = el("span", "sr-lbl");
  const done = el("button", "sr-btn", "Done");
  head.append(title, target, done);
  const tabs = el("div", "sr-tabs");
  const tabBtns = {};
  for (const [k, label] of [["library", "Library"], ["mine", "My poses"], ["files", "Files (Daz, OpenPose, BVH)"], ["openpose", "OpenPose JSON"]]) {
    tabBtns[k] = el("button", "sr-btn", label);
    tabBtns[k].onclick = () => show(k);
    tabs.append(tabBtns[k]);
  }
  const bar = el("div", "sr-bar");
  const grid = el("div", "sr-grid");
  const foot = el("footer");
  dlg.append(head, tabs, bar, grid, foot);
  modal.append(dlg);
  const close = () => { modal.remove(); document.removeEventListener("keydown", onKey, true); obs.disconnect(); };
  const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
  document.addEventListener("keydown", onKey, true);
  // Typing in the dialog must not reach ComfyUI's shortcuts (after the field's own handler).
  modal.addEventListener("keydown", (e) => e.stopPropagation());
  modal.addEventListener("pointerdown", (e) => { if (e.target === modal) close(); });
  done.onclick = close;
  const sayFoot = (t, err) => { foot.textContent = t; foot.style.color = err ? "#ff8a80" : ""; foot.style.opacity = err ? 1 : ""; };
  const fig = () => v.figures[v.selected];
  const updateTarget = () => {
    const f = fig();
    target.textContent = !f ? "select a mannequin in the view first" : f.kind === "daz" ? "the selected figure is a Daz figure (posed in Daz)"
      : `on Figure ${v.selected + 1} · ${PALETTE[f.spec.color].name}`;
  };
  updateTarget();

  // Cards appear as they scroll into view; their previews are drawn then.
  const obs = new IntersectionObserver((ents) => {
    for (const en of ents) if (en.isIntersecting) { obs.unobserve(en.target); en.target._load?.(); }
  }, { root: grid, rootMargin: "200px" });
  const card = (name, tag, { thumbUrl, makeThumb, onPick }) => {
    const c = el("div", "sr-card");
    const th = el("div", "th", "…");
    c.append(th, el("div", "nm", name));
    if (tag) c.append(el("div", "tg", tag));
    c.title = name;
    const setImg = (u) => { th.textContent = ""; th.style.backgroundImage = `url("${u}")`; };
    c._load = () => {
      if (thumbUrl) {
        const img = new Image();
        img.onload = () => setImg(thumbUrl);
        img.onerror = () => (makeThumb ? makeThumb().then(setImg, () => (th.textContent = "?")) : (th.textContent = "—"));
        img.src = thumbUrl;
      } else if (makeThumb) makeThumb().then(setImg, (e) => { th.textContent = "⚠"; c.title = name + "\n" + e.message; });
    };
    obs.observe(c);
    c.onclick = () => onPick(c);
    return c;
  };
  const apply = (pose) => {
    const f = fig();
    if (!f) { sayFoot("Select a mannequin in the view first (click it), then pick a pose.", true); return; }
    if (f.kind === "daz") { sayFoot("Daz figures keep the pose they have in Daz: pick a mannequin.", true); return; }
    v.applyPose(v.selected, pose);
    refreshFigures(node);
    sayFoot(`"${pose.name}" on Figure ${v.selected + 1}${pose.only ? " (only " + pose.only.length + " joints: a partial preset)" : ""}. Fine-tune with the Bones gizmo (key 4).`);
  };
  const cache = new Map();
  const fetchPose = (id) => {
    if (!cache.has(id)) {
      const p = getJSON("/scene_ref/poses/get?id=" + encodeURIComponent(id));
      cache.set(id, p);
      p.catch(() => cache.delete(id));
    }
    return cache.get(id);
  };
  // A file with several poses (BVH frames, a group in OpenPose): show them to choose.
  const pickFrom = async (id, name) => {
    let d;
    try { d = await fetchPose(id); } catch (e) { sayFoot(e.message, true); return; }
    if (!d.poses) { apply(d); return; }
    grid.textContent = "";
    const back = el("div", "sr-card dir", "← back");
    back.onclick = () => show(cur);
    grid.append(back);
    d.poses.forEach((p, i) => grid.append(card(p.name, null, {
      makeThumb: () => mannequinThumb(id + "#" + i, async () => p), onPick: () => apply(p),
    })));
    sayFoot(`${name}: ${d.poses.length} poses in this file`);
  };

  let index = null;
  const loadIndex = async () => {
    index ??= getJSON("/scene_ref/poses" + (node._sr.path ? "?near=" + encodeURIComponent(node._sr.path) : ""));
    return index;
  };
  let cur = startTab;
  const search = el("input");
  search.placeholder = "search";
  search.addEventListener("input", () => render());
  const cat = el("select", "sr-sel");

  const render = async () => {
    grid.textContent = "";
    const q = search.value.trim().toLowerCase();
    let d;
    try { d = await loadIndex(); } catch (e) { sayFoot(e.message, true); return; }
    if (cur === "library") {
      const items = d.library.filter((p) => (!cat.value || p.category === cat.value) && (!q || p.name.toLowerCase().includes(q)));
      for (const p of items) {
        grid.append(card(p.name, p.category, {
          makeThumb: () => mannequinThumb(p.id, async () => p), onPick: () => apply(p),
        }));
      }
      sayFoot(`${items.length} poses · from the CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu)`);
    } else if (cur === "mine") {
      const items = d.mine.filter((p) => !q || p.name.toLowerCase().includes(q));
      for (const p of items) {
        grid.append(card(p.name, p.kind === "srpose" ? (p.folder || "") : p.kind, {
          thumbUrl: p.thumb ? api.apiURL("/scene_ref/poses/thumb?id=" + encodeURIComponent(p.id)) : null,
          makeThumb: () => mannequinThumb(p.id, () => fetchPose(p.id)), onPick: () => pickFrom(p.id, p.name),
        }));
      }
      if (!items.length) grid.append(el("div", "sr-lbl", "No saved poses yet. Pose a mannequin, then save it above."));
      sayFoot(`Saved in ${d.user_dir} (also put Daz .duf, OpenPose .json or .bvh files there)`);
    }
  };

  // Files: one folder at a time (a Daz library holds hundreds of thousands of files).
  let folderPath = "";
  let folderReq = 0;
  const openFolder = async (path) => {
    const req = ++folderReq; // a slow listing must not replace a later one
    grid.textContent = "Loading…";
    let d;
    try { d = await getJSON("/scene_ref/poses/folder?path=" + encodeURIComponent(path || "")); } catch (e) { if (req === folderReq) grid.textContent = "⚠ " + e.message; return; }
    if (req !== folderReq || cur !== "files") return;
    folderPath = d.path || "";
    pathIn.value = folderPath;
    if (d.path) try { localStorage.setItem("sceneRef.poseDir", d.path); } catch (_e) { /* private mode */ }
    grid.textContent = "";
    const sep = folderPath.includes("/") && !folderPath.includes("\\") ? "/" : "\\";
    const join = (n) => (folderPath ? folderPath.replace(/[\\/]+$/, "") + sep + n : n);
    const q = search.value.trim().toLowerCase();
    if (d.parent !== null && d.path) {
      const up = el("div", "sr-card dir", "↑ up");
      up.onclick = () => openFolder(d.parent);
      grid.append(up);
    }
    for (const n of d.dirs) {
      if (q && !n.toLowerCase().includes(q)) continue;
      const c = el("div", "sr-card dir", "📁 " + n);
      c.onclick = () => openFolder(join(n));
      grid.append(c);
    }
    const items = d.poses.filter((p) => !q || p.name.toLowerCase().includes(q));
    for (const p of items) {
      grid.append(card(p.name, p.kind, {
        thumbUrl: p.thumb ? api.apiURL("/scene_ref/poses/thumb?id=" + encodeURIComponent(p.id)) : null,
        makeThumb: () => mannequinThumb(p.id, () => fetchPose(p.id)), onPick: () => pickFrom(p.id, p.name),
      }));
    }
    sayFoot(`${d.dirs.length} folders · ${d.poses.length} pose files. Daz presets are read with the figure's own skeleton from the content library; partial presets (one leg, the arms) change only those joints.`);
  };
  const pathIn = el("input");
  pathIn.placeholder = "folder";
  pathIn.addEventListener("keydown", (e) => { if (e.key === "Enter") openFolder(pathIn.value.trim()); });

  // OpenPose: drop / pick / paste JSON.
  const openposeTab = () => {
    grid.textContent = "";
    const drop = el("div", "sr-drop", "Drop OpenPose .json files here (pose_keypoints_2d: OpenPose, ControlNet preprocessors, openpose editors), or paste the JSON below.");
    const file = el("input");
    file.type = "file";
    file.accept = ".json,application/json";
    file.multiple = true;
    const ta = el("textarea");
    Object.assign(ta.style, { width: "100%", height: "90px", background: "#111", color: "#ddd", border: "1px solid #444", borderRadius: "5px", font: "11px monospace" });
    const go = el("button", "sr-btn", "Read pasted JSON");
    const wrap = el("div");
    wrap.style.gridColumn = "1 / -1";
    wrap.append(drop, file, ta, go);
    grid.append(wrap);
    const show2 = (d) => {
      const list = d.poses || [d];
      list.forEach((p) => {
        const op = openPoseThumb(p);
        grid.append(card(p.name, "OpenPose", {
          thumbUrl: op, makeThumb: () => poseThumbnail(p), onPick: () => apply(p),
        }));
        grid.append(card(p.name + " (3D)", "as a mannequin", {
          makeThumb: () => poseThumbnail(p), onPick: () => apply(p),
        }));
      });
      sayFoot("2D keypoints lifted to 3D from the bone lengths: which way a bone points towards the camera is a guess (knees forward, hands in front). Fine-tune with the Bones gizmo.");
    };
    const parse = async (name, text) => {
      try { show2(await postJSON("/scene_ref/poses/parse", { name, data: JSON.parse(text) })); } catch (e) { sayFoot(`${name}: ${e.message}`, true); }
    };
    const readFiles = async (files) => { for (const f of files) await parse(f.name.replace(/\.json$/i, ""), await f.text()); };
    file.onchange = () => readFiles(file.files);
    go.onclick = () => parse("Pasted pose", ta.value);
    drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("on"); };
    drop.ondragleave = () => drop.classList.remove("on");
    drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("on"); readFiles(e.dataTransfer.files); };
    sayFoot("");
  };

  // Save the selected mannequin's pose (My poses tab).
  const saveName = el("input");
  saveName.placeholder = "name for the selected figure's pose";
  const saveBtn = el("button", "sr-btn", "💾 Save pose");
  saveBtn.onclick = async () => {
    const p = v.poseOfFigure(v.selected);
    if (!p) { sayFoot("Select a mannequin to save its pose.", true); return; }
    const name = saveName.value.trim() || p.name;
    try {
      const png = await poseThumbnail({ bones: p.bones, anchor: p.anchor }, { size: 160 });
      const r = await postJSON("/scene_ref/poses/save", { name, bones: p.bones, anchor: p.anchor, png });
      const d = await loadIndex();
      d.mine = r.mine;
      _thumbs.delete(r.id);
      sayFoot(`saved "${name}"`);
      render();
    } catch (e) { sayFoot(e.message, true); }
  };

  const show = (k) => {
    cur = k;
    for (const [n, b] of Object.entries(tabBtns)) b.classList.toggle("on", n === k);
    bar.textContent = "";
    search.value = "";
    if (k === "library") {
      bar.append(search, cat);
      loadIndex().then((d) => {
        if (cat.options.length) return;
        cat.append(Object.assign(el("option", null, "All kinds"), { value: "" }));
        for (const c of [...new Set(d.library.map((p) => p.category))]) cat.append(Object.assign(el("option", null, c), { value: c }));
      });
      cat.onchange = () => render();
      render();
    } else if (k === "mine") {
      saveName.value = fig()?.kind === "mannequin" ? fig().poseLabel.replace(/ \(adjusted\)$/, "") : "";
      bar.append(saveName, saveBtn, search);
      render();
    } else if (k === "files") {
      const up = el("button", "sr-btn", "Go");
      up.onclick = () => openFolder(pathIn.value.trim());
      search.oninput = () => openFolder(folderPath);
      bar.append(pathIn, up, search);
      loadIndex().then((d) => {
        let start = "";
        try { start = localStorage.getItem("sceneRef.poseDir") || ""; } catch (_e) { /* private mode */ }
        openFolder(folderPath || start || d.folder || "");
      }, () => openFolder(""));
    } else {
      openposeTab();
    }
    if (k !== "files") search.oninput = () => render();
  };
  document.body.append(modal);
  show(startTab);
}

// ── the node face ────────────────────────────────────────────────────────────
function buildFace(node) {
  injectCss();
  const root = el("div", "sr-root");
  for (const t of ["pointerdown", "wheel", "keydown", "keyup", "contextmenu", "dblclick"]) {
    root.addEventListener(t, (e) => { if (e.target.closest(".sr-vp") || e.target.tagName === "SELECT" || e.target.tagName === "INPUT") e.stopPropagation(); });
  }

  const r1 = el("div", "sr-row");
  const bBrowse = el("button", "sr-btn", "📂 Browse…");
  const bOpen = el("button", "sr-btn", "Open");
  const bRe = el("button", "sr-btn", "⟳");
  bRe.title = "Convert again (ignores the cache)";
  const bCancel = el("button", "sr-btn", "Cancel");
  bCancel.style.display = "none";
  const status = el("div", "sr-status sr-grow", "Pick a .blend or .duf scene.");
  r1.append(bBrowse, bOpen, bRe, bCancel, status);

  const vp = el("div", "sr-vp");
  const overlay = el("div", "sr-overlay", "No scene loaded.\nBrowse… to a Blender or Daz Studio scene.");
  const hint = el("div", "sr-hint",
    "L-drag orbit · R-drag look · hold R + WASD/QE fly (Shift fast) · wheel dolly · Shift/M-drag pan · dbl-click pivot · F frame · R reset · figures: 1 move 2 rotate 3 scale 4 bones 0 off");
  const flash = el("div", "sr-flash");
  flash.style.opacity = 0;
  vp.append(overlay, hint, flash);

  const r2 = el("div", "sr-row");
  const camSel = el("select", "sr-sel sr-grow");
  camSel.title = "Scene cameras and your saved shots";
  const bSave = el("button", "sr-btn", "＋ Shot");
  bSave.title = "Save the current view as a named shot";
  const bDel = el("button", "sr-btn", "🗑");
  bDel.title = "Delete the selected saved shot";
  const lensLbl = el("span", "sr-lbl", "Lens");
  const lens = el("input", "sr-range");
  Object.assign(lens, { type: "range", min: 10, max: 200, step: 1 });
  const lensVal = el("span", "sr-lbl", "35mm");
  r2.append(camSel, bSave, bDel, lensLbl, lens, lensVal);

  const r3 = el("div", "sr-row");
  const looks = {};
  for (const k of ["color", "clay", "depth", "normal"]) {
    looks[k] = el("button", "sr-btn", k[0].toUpperCase() + k.slice(1));
    r3.append(looks[k]);
  }
  const light = el("select", "sr-sel");
  for (const [v, t] of [["studio", "Studio light"], ["scene", "Scene lights"], ["flat", "Flat"]]) {
    const o = el("option", null, t); o.value = v; light.append(o);
  }
  const exp = el("input", "sr-range");
  Object.assign(exp, { type: "range", min: -3, max: 3, step: 0.05 });
  exp.title = "Exposure";
  const allShots = el("label", "sr-lbl");
  const allCb = el("input");
  allCb.type = "checkbox";
  allShots.append(allCb, document.createTextNode(" all shots"));
  allShots.title = "On Run, render every saved shot as one batch";
  r3.append(light, exp, allShots);

  // Figures: mannequins placed in the scene (stay put when the camera moves).
  const r4 = el("div", "sr-row");
  const bFig = el("button", "sr-btn", "＋ Figure");
  bFig.title = "Add a mannequin on the surface in the middle of the view. Drag it to move it over floors, seats and beds.";
  const figSel = el("select", "sr-sel");
  figSel.title = "Selected figure (click one in the view to select it)";
  const pose = el("select", "sr-sel sr-grow");
  for (const [k, p] of Object.entries(POSES)) { const o = el("option", null, p.label); o.value = k; pose.append(o); }
  pose.title = "Pose preset; describe the exact pose in the prompt";
  const bFigDel = el("button", "sr-btn", "🗑");
  bFigDel.title = "Remove the selected figure (or press Delete)";
  r4.append(bFig, figSel, pose, bFigDel);

  const r5 = el("div", "sr-row");
  const sex = el("select", "sr-sel");
  for (const [v, t] of [["female", "Female"], ["male", "Male"]]) { const o = el("option", null, t); o.value = v; sex.append(o); }
  const build = el("select", "sr-sel");
  for (const k of Object.keys(BUILDS)) { const o = el("option", null, k[0].toUpperCase() + k.slice(1)); o.value = k; build.append(o); }
  const height = el("input", "sr-num");
  Object.assign(height, { type: "number", min: 60, max: 230, step: 1 });
  height.title = "Height in cm: the model copies this size relative to the furniture";
  const slider = (label, min, max, title) => {
    const l = el("span", "sr-lbl", label);
    const s = el("input", "sr-range");
    Object.assign(s, { type: "range", min, max, step: 1 });
    s.title = title;
    s.style.width = "62px";
    return [l, s];
  };
  const [turnL, turn] = slider("Turn", -180, 180, "Which way the figure faces ([ and ] keys)");
  const [headL, head] = slider("Head", -80, 80, "Head turn");
  const [leanL, lean] = slider("Lean", -30, 45, "Lean forward / back");
  const dazLook = el("select", "sr-sel");
  for (const [v, t] of [["color", "Colour tag"], ["textured", "Textured"]]) { const o = el("option", null, t); o.value = v; dazLook.append(o); }
  dazLook.title = "Daz figure: solid colour tag (the model replaces it with your character) or its Daz textures";
  r5.append(sex, build, height, el("span", "sr-lbl", "cm"), turnL, turn, headL, head, leanL, lean, dazLook);

  // Gizmos on the selected mannequin, the pose library, and the bones tools.
  const r6 = el("div", "sr-row");
  const gizmos = {};
  for (const [k, label, title] of [
    ["move", "✥ Move", "Move along the axes (key 1). Ctrl snaps to 5 cm."],
    ["rotate", "⟳ Rotate", "Turn and tilt the whole figure (key 2): lying, leaning, upside down. Ctrl snaps to 15°."],
    ["scale", "⤢ Scale", "Height (key 3): drag any handle"],
    ["bones", "🦴 Bones", "Pose by hand (key 4): click a yellow joint, turn it with the rings"],
  ]) {
    gizmos[k] = el("button", "sr-btn", label);
    gizmos[k].title = title;
    gizmos[k].onclick = () => viewer.setGizmo(viewer.gizmo === k ? "off" : k);
    r6.append(gizmos[k]);
  }
  const boneLbl = el("span", "sr-lbl sr-grow", "");
  const bBoneReset = el("button", "sr-btn", "↺ Joint");
  bBoneReset.title = "Back to the preset's angle for this joint";
  const bBonesReset = el("button", "sr-btn", "↺ All");
  bBonesReset.title = "Drop every hand-set joint (back to the preset)";
  const bMirror = el("button", "sr-btn", "⇋ Mirror");
  bMirror.title = "Swap left and right";
  const bPoses = el("button", "sr-btn", "📚 Poses…");
  bPoses.title = "Pose library, your saved poses, Daz pose presets, OpenPose and BVH files";
  r6.append(boneLbl, bBoneReset, bBonesReset, bMirror, bPoses);

  root.append(r1, vp, r2, r3, r4, r5, r6);

  const els = { root, status, overlay, flash, camSel, lens, lensVal, looks, light, exp, allCb, bOpen, bRe, bCancel, bBrowse, bSave, bDel, vp,
    bFig, figSel, pose, bFigDel, sex, build, height, turn, head, lean, r5, dazLook, r6, gizmos, boneLbl, bBoneReset, bBonesReset, bMirror, bPoses };
  node._sr = { els, viewer: null, key: null, path: null, poll: 0 };

  const say = (text, err = false) => { status.textContent = text; status.classList.toggle("err", err); status.title = text; };
  node._sr.say = say;
  let flashT = 0;
  node._sr.flash = (t) => { flash.textContent = t; flash.style.opacity = 1; clearTimeout(flashT); flashT = setTimeout(() => (flash.style.opacity = 0), 1400); };

  const viewer = new Viewer(vp, {
    onChange: (v) => { setProps(node, { view: v }); syncLens(node); },
    onStatus: (t) => node._sr.flash(t),
    onFigures: (specs) => { setProps(node, { figures: specs }); refreshFigures(node); },
    onSelect: () => refreshFigures(node),
    onGizmo: () => refreshFigures(node),
  });
  node._sr.viewer = viewer;
  bBoneReset.onclick = () => viewer.resetBones(false);
  bBonesReset.onclick = () => viewer.resetBones(true);
  bMirror.onclick = () => viewer.mirrorPose();
  bPoses.onclick = () => {
    if (!viewer.ready) { node._sr.flash("open a scene first"); return; }
    poseBrowser(node);
  };

  bFig.onclick = () => {
    if (!viewer.ready) { node._sr.flash("open a scene first"); return; }
    const i = viewer.addFigure();
    if (i >= 0) node._sr.flash(`Figure ${i + 1} added · drag it into place`);
  };
  figSel.onchange = () => viewer.select(+figSel.value);
  bFigDel.onclick = () => { if (viewer.selected >= 0) viewer.removeFigure(viewer.selected); };
  const edit = (patch) => { if (viewer.selected >= 0) viewer.updateFigure(viewer.selected, patch); };
  pose.onchange = () => { if (pose.value !== "custom") edit({ pose: pose.value, bones: {} }); };
  sex.onchange = () => edit({ sex: sex.value });
  build.onchange = () => edit({ build: build.value });
  height.onchange = () => edit({ height_cm: +height.value });
  height.addEventListener("keydown", (e) => e.stopPropagation());
  turn.oninput = () => edit({ yaw: +turn.value });
  head.oninput = () => edit({ head_turn: +head.value });
  lean.oninput = () => edit({ lean: +lean.value });
  dazLook.onchange = () => edit({ look: dazLook.value });

  bBrowse.onclick = async () => {
    const pw = widget(node, "scene_path");
    const picked = await browse(pw?.value || "");
    if (picked && pw) {
      pw.value = picked;
      pw.callback?.(picked);
      node.setDirtyCanvas?.(true, true);
      openScene(node, false);
    }
  };
  bOpen.onclick = () => openScene(node, false);
  bRe.onclick = () => openScene(node, true);
  bCancel.onclick = async () => {
    if (node._sr.key) await postJSON("/scene_ref/cancel/" + node._sr.key).catch(() => {});
  };

  camSel.onchange = () => {
    const v = camSel.value;
    const cam = cameraByValue(node, v);
    if (cam) viewer.setView(cam);
  };
  bSave.onclick = () => {
    const p = props(node);
    const name = `Shot ${p.shots.length + 1} · ${node._sr.els.lensVal.textContent}`;
    const shots = [...p.shots, { name: name.slice(0, 60), ...viewer.getView() }];
    setProps(node, { shots });
    fillCameras(node, "shot:" + (shots.length - 1));
    node._sr.flash(`saved "${name}"`);
  };
  bDel.onclick = () => {
    const v = camSel.value;
    if (!v.startsWith("shot:")) { node._sr.flash("pick a saved shot to delete"); return; }
    const i = +v.slice(5);
    const shots = props(node).shots.filter((_s, j) => j !== i);
    setProps(node, { shots });
    fillCameras(node);
  };
  lens.oninput = () => {
    const mm = +lens.value;
    // Horizontal 36 mm sensor, like Blender / Daz; converted to three.js vertical FOV.
    const a = viewer.aspect || 1;
    const fovH = 2 * Math.atan(18 / mm);
    const fovV = 2 * Math.atan(Math.tan(fovH / 2) / a) * 180 / Math.PI;
    viewer.setFov(fovV);
    lensVal.textContent = mm + "mm";
  };
  for (const [k, b] of Object.entries(looks)) {
    b.onclick = () => { setProps(node, { look: k }); viewer.setLook(k); refreshControls(node); };
  }
  light.onchange = () => { setProps(node, { light: light.value }); viewer.setLight(light.value); };
  exp.oninput = () => { const e = Math.pow(2, +exp.value); setProps(node, { exposure: e }); viewer.setExposure(e); };
  allCb.onchange = () => setProps(node, { allShots: allCb.checked });

  return root;
}

function cameraByValue(node, v) {
  if (v.startsWith("cam:")) return node._sr.viewer.sceneCameras?.[+v.slice(4)];
  if (v.startsWith("shot:")) return props(node).shots[+v.slice(5)];
  return null;
}

function fillCameras(node, select) {
  const { camSel } = node._sr.els;
  const keep = select ?? camSel.value;
  camSel.textContent = "";
  const free = el("option", null, "— free view —");
  free.value = "";
  camSel.append(free);
  const cams = node._sr.viewer.sceneCameras || [];
  if (cams.length) {
    const g = el("optgroup");
    g.label = "Scene cameras";
    cams.forEach((c, i) => { const o = el("option", null, c.name); o.value = "cam:" + i; g.append(o); });
    camSel.append(g);
  }
  const shots = props(node).shots;
  if (shots.length) {
    const g = el("optgroup");
    g.label = "Saved shots";
    shots.forEach((s, i) => { const o = el("option", null, s.name); o.value = "shot:" + i; g.append(o); });
    camSel.append(g);
  }
  camSel.value = [...camSel.options].some((o) => o.value === keep) ? keep : "";
}

function syncLens(node) {
  const v = node._sr.viewer;
  if (!v.camera) return;
  const a = v.aspect || 1;
  const fovH = 2 * Math.atan(Math.tan((v.camera.fov * Math.PI) / 360) * a);
  const mm = Math.round(18 / Math.tan(fovH / 2));
  node._sr.els.lens.value = mm;
  node._sr.els.lensVal.textContent = mm + "mm";
}

// Figure controls follow the selected figure.
function refreshFigures(node) {
  const e = node._sr.els;
  const v = node._sr.viewer;
  const figs = v.figures;
  e.figSel.textContent = "";
  figs.forEach((f, i) => {
    const o = el("option", null, `${i + 1} · ${PALETTE[f.spec.color].name}${f.kind === "daz" ? " · " + f.label : ""}`);
    o.value = i;
    e.figSel.append(o);
  });
  const has = figs.length > 0;
  for (const x of [e.figSel, e.pose, e.bFigDel]) x.classList.toggle("sr-hide", !has);
  e.r5.classList.toggle("sr-hide", !has);
  e.r6.classList.toggle("sr-hide", !has);
  const f = figs[v.selected] || null;
  for (const [k, b] of Object.entries(e.gizmos)) { b.classList.toggle("on", v.gizmo === k); b.disabled = !f || f.kind === "daz"; }
  const bones = v.gizmo === "bones" && f && f.kind !== "daz";
  for (const x of [e.bBoneReset, e.bBonesReset]) x.classList.toggle("sr-hide", !bones);
  e.bBoneReset.disabled = !v.bone;
  e.bMirror.disabled = !f || f.kind === "daz";
  e.boneLbl.textContent = bones ? (v.bone ? JOINT_LABELS[v.bone] : "click a yellow joint") : "";
  for (const x of [e.pose, e.sex, e.build, e.height, e.turn, e.head, e.lean, e.bFigDel]) x.disabled = !f;
  const daz = f?.kind === "daz";
  for (const x of [e.pose, e.sex, e.build, e.height, e.turn, e.head, e.lean, e.bFigDel]) if (daz) x.disabled = true;
  e.dazLook.classList.toggle("sr-hide", !daz);
  if (!f) return;
  e.figSel.value = v.selected;
  const s = f.spec;
  if (daz) { e.dazLook.value = s.look; e.height.value = s.height_cm; return; }
  // A pose from the library shows under its own name.
  let custom = [...e.pose.options].find((o) => o.value === "custom");
  if (s.pose === "custom") {
    if (!custom) { custom = el("option"); custom.value = "custom"; e.pose.prepend(custom); }
    custom.textContent = "★ " + f.poseLabel;
  } else custom?.remove();
  e.pose.value = s.pose;
  e.pose.title = (s.pose === "custom" ? f.poseLabel + " (from the pose library)" : "Pose preset") +
    (Object.keys(s.bones || {}).length && s.pose !== "custom" ? " · adjusted with the Bones gizmo" : "") + "; describe the exact pose in the prompt";
  e.sex.value = s.sex;
  e.build.value = s.build;
  if (document.activeElement !== e.height) e.height.value = s.height_cm;
  e.turn.value = ((s.yaw + 540) % 360) - 180;
  e.head.value = s.head_turn;
  e.lean.value = s.lean;
}

function refreshControls(node) {
  const p = props(node);
  const e = node._sr.els;
  for (const [k, b] of Object.entries(e.looks)) b.classList.toggle("on", k === p.look);
  e.light.value = p.light;
  e.exp.value = Math.log2(p.exposure);
  e.allCb.checked = p.allShots;
}

// A wired width / height: the widget keeps its old value (the real one only exists when the
// graph runs), so framing to it made 4:3 captures that were stretched to e.g. 9:16. Read the
// source node instead: a widget of the same name, a primitive's value, or ComfyUI's
// Resolution Selector (its formula); else the size the last run reported.
function resolutionSelector(src, name) {
  const m = String(widget(src, "aspect_ratio")?.value || "").match(/(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)/);
  if (!m) return null;
  const [rw, rh] = [+m[1], +m[2]];
  const mult = +widget(src, "multiple")?.value || 8;
  const scale = Math.sqrt((+widget(src, "megapixels")?.value || 1) * 1024 * 1024 / (rw * rh));
  return Math.round((name === "height" ? rh : rw) * scale / mult) * mult;
}

// SceneRef Resolution: the same formula as scene_ref/outputs.py resolution().
function sceneRefResolution(src) {
  const m = String(widget(src, "aspect_ratio")?.value || "").match(/(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)/);
  if (!m) return null;
  const [rw, rh] = [+m[1], +m[2]];
  const mult = +widget(src, "multiple_of")?.value || 16;
  const s = Math.sqrt((+widget(src, "megapixels")?.value || 1) * 1024 * 1024 / (rw * rh));
  return [Math.max(mult, Math.round(rw * s / mult) * mult), Math.max(mult, Math.round(rh * s / mult) * mult)];
}

function linkedValue(node, name) {
  const input = node.inputs?.find((i) => i.name === name || i.widget?.name === name);
  if (input?.link == null || !node.graph) return null;
  const g = node.graph;
  const link = g.getLink?.(input.link) ?? g.links?.get?.(input.link) ?? g.links?.[input.link];
  const src = link && g.getNodeById(link.origin_id);
  if (!src) return null;
  const out = src.outputs?.[link.origin_slot]?.name;
  if ((src.comfyClass || src.type) === "ResolutionSelector") return resolutionSelector(src, out);
  if ((src.comfyClass || src.type) === "SceneRefResolution") return sceneRefResolution(src)?.[out === "height" ? 1 : 0] ?? null;
  const w = widget(src, out) || (src.widgets?.length === 1 ? src.widgets[0] : widget(src, "value"));
  const v = +w?.value;
  return Number.isFinite(v) && v > 0 ? v : (node.properties?.srRunSize?.[name === "height" ? 1 : 0] || null);
}

function outputSize(node) {
  const w = linkedValue(node, "width") || +widget(node, "width")?.value || 1024;
  const h = linkedValue(node, "height") || +widget(node, "height")?.value || 1024;
  return [Math.max(64, Math.min(4096, w)), Math.max(64, Math.min(4096, h))];
}

// ── open / convert ───────────────────────────────────────────────────────────
async function openScene(node, force) {
  const sr = node._sr;
  const path = String(widget(node, "scene_path")?.value || "").trim();
  if (!path) { sr.say("Pick a .blend or .duf scene first."); return; }
  clearInterval(sr.poll);
  sr.path = path;
  sr.els.overlay.style.display = "";
  sr.els.overlay.textContent = "Opening…";
  try {
    let st = await postJSON("/scene_ref/open", { path, force });
    sr.key = st.key;
    while (st.state === "queued" || st.state === "running") {
      showJob(node, st);
      await new Promise((r) => setTimeout(r, 1000));
      if (sr.path !== path) return; // the user picked something else meanwhile
      st = await getJSON("/scene_ref/status/" + st.key);
    }
    sr.els.bCancel.style.display = "none";
    if (st.state !== "done") {
      sr.say(`${st.state}: ${st.message || ""}`, true);
      sr.els.overlay.textContent = "Conversion " + st.state + ".\n" + (st.log || []).slice(-6).join("\n");
      return;
    }
    await loadIntoViewer(node, st.key);
  } catch (e) {
    sr.els.bCancel.style.display = "none";
    sr.say(e.message, true);
    sr.els.overlay.textContent = "⚠ " + e.message;
  }
}

function showJob(node, st) {
  const sr = node._sr;
  const who = st.phase.startsWith("daz") ? "Daz Studio" : st.phase.startsWith("blender") ? "Blender" : "";
  sr.els.bCancel.style.display = "";
  sr.say(`Converting${who ? " with " + who : ""} · ${st.phase} · ${fmtTime(st.elapsed)}`);
  sr.els.overlay.textContent =
    `Converting the scene (only the first time; the result is cached).\n${who ? who + " runs minimized in the background.\n" : ""}\n` +
    (st.log || []).slice(-5).join("\n");
}

async function loadIntoViewer(node, key) {
  const sr = node._sr;
  const t0 = performance.now();
  sr.say("Loading the 3D scene into the browser…");
  sr.els.overlay.textContent = "Downloading the converted scene…";
  const v = sr.viewer;
  const p = props(node);
  if (p.view && sr.restoreView !== false) v._pendingView = p.view;
  sr.restoreView = true;
  await v.load(key, (loaded, total) => {
    sr.els.overlay.textContent = `Downloading the converted scene… ${fmtMB(loaded)}${total ? " / " + fmtMB(total) : ""}`;
  });
  v.setLook(p.look);
  v.setLight(p.light);
  v.setExposure(p.exposure);
  sr.els.overlay.style.display = "none";
  fillCameras(node);
  syncLens(node);
  const s = v.stats();
  const cams = (v.sceneCameras || []).length;
  const people = (v.dazMeshes || []).length;
  sr.say(`${fmtInt(s.tris)} triangles · ${s.materials} materials · ${cams} camera${cams === 1 ? "" : "s"}` +
    (people ? ` · ${people} Daz figure${people === 1 ? "" : "s"}` : "") +
    ` · loaded in ${fmtTime((performance.now() - t0) / 1000)}`);
}

// ── registration ─────────────────────────────────────────────────────────────
app.registerExtension({
  name: "SceneRef.Viewer",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === "SceneRefResolution") {
      // The size it gives, under the widgets.
      const draw = nodeType.prototype.onDrawForeground;
      nodeType.prototype.onDrawForeground = function (ctx) {
        const r = draw?.apply(this, arguments);
        const wh = sceneRefResolution(this);
        if (wh && !this.flags?.collapsed) {
          ctx.save();
          ctx.font = "13px system-ui, sans-serif";
          ctx.fillStyle = "#9ab";
          ctx.textAlign = "center";
          ctx.fillText(`${wh[0]} × ${wh[1]}  ·  ${(wh[0] * wh[1] / 1e6).toFixed(2)} MP`, this.size[0] / 2, this.size[1] - 8);
          ctx.restore();
        }
        return r;
      };
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = created?.apply(this, arguments);
        this.size = [Math.max(this.size[0], 260), this.size[1] + 22];
        return r;
      };
      return;
    }
    if (nodeData?.name !== CLASS) return;

    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const node = this;
      const root = buildFace(node);
      const w = node.addDOMWidget("scene_ref_ui", "scene_ref_ui", root, {
        serialize: false,
        getMinHeight: () => MIN_UI_H,
      });
      w.serialize = false;
      node.size = [Math.max(node.size?.[0] || 0, 600), Math.max(node.size?.[1] || 0, 760)];
      refreshControls(node);
      refreshFigures(node);
      // Aspect follows width / height, however they change (typing, undo, a wire).
      // For the Studio panel: the current view on demand (its 'Write the description' button)
      // and the size the viewer renders at (its picture-size line).
      node._srCapture = () => captureNode(node);
      node._srOutputSize = () => outputSize(node);
      node._sr.sizeTimer = setInterval(() => {
        const [ow, oh] = outputSize(node);
        node._sr.viewer.setAspect(ow / oh);
      }, 300);
      const pw = widget(node, "scene_path");
      if (pw) {
        const cb = pw.callback;
        pw.callback = function (v) {
          const res = cb?.apply(this, arguments);
          if (String(v || "").trim() && String(v).trim() !== node._sr.path) openScene(node, false);
          return res;
        };
      }
      return r;
    };

    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = configure?.apply(this, arguments);
      const node = this;
      queueMicrotask(() => {
        if (!node._sr) return;
        refreshControls(node);
        node._sr.viewer.setFigures(props(node).figures);
        refreshFigures(node);
        const [ow, oh] = outputSize(node);
        node._sr.viewer.setAspect(ow / oh);
        if (node.graph && String(widget(node, "scene_path")?.value || "").trim()) openScene(node, false);
      });
      return r;
    };

    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      const r = executed?.apply(this, arguments);
      const size = msg?.sr_size;
      if (Array.isArray(size) && size.length === 2) {
        this.properties = this.properties || {};
        this.properties.srRunSize = size.map(Number);
      }
      return r;
    };

    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      clearInterval(this._sr?.sizeTimer);
      if (this._sr) this._sr.path = null;
      this._sr?.viewer?.dispose();
      return removed?.apply(this, arguments);
    };
  },
});

// ── Run: render, upload, inject ──────────────────────────────────────────────
const _uploaded = new Map(); // render key -> names (same view = same files = ComfyUI cache hit)
api.addEventListener("reconnected", () => _uploaded.clear());

function hash(str) {
  let h1 = 0x811c9dc5, h2 = 0x5bd1e995;
  for (let i = 0; i < str.length; i++) {
    const c = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 16777619);
    h2 = Math.imul(h2 ^ c, 0x5bd1e995) ^ (h2 >>> 15);
  }
  return (h1 >>> 0).toString(16).padStart(8, "0") + (h2 >>> 0).toString(16).padStart(8, "0");
}

async function upload(blob, filename) {
  const body = new FormData();
  body.append("image", blob, filename);
  body.append("type", "temp");
  body.append("subfolder", "scene_ref");
  body.append("overwrite", "true");
  const res = await api.fetchApi("/upload/image", { method: "POST", body });
  if (res.status !== 200) throw new Error(`upload failed (${res.status})`);
  const d = await res.json();
  return `${d.subfolder || "scene_ref"}/${d.name || filename} [temp]`;
}

async function captureNode(node) {
  const sr = node._sr;
  if (!sr?.viewer?.ready || !sr.key) return {};
  const v = sr.viewer;
  const p = props(node);
  const [w, h] = outputSize(node);
  const views = p.allShots && p.shots.length ? p.shots : [v.getView()];
  const current = v.getView();
  const shots = [];
  try {
    for (const view of views) {
      const key = hash(JSON.stringify([3, sr.key, w, h, view.pos, view.quat, view.fov, v.look === "clay", p.light, p.exposure,
        p.depthCurve, v.figureSpecs()]));
      let names = _uploaded.get(key);
      if (!names) {
        if (view !== views[0] || p.allShots) v.setView(view);
        const shot = await v.capture(w, h, { depthCurve: p.depthCurve });
        names = { camera: shot.camera };
        for (const pass of ["image", "depth", "normal", "mask", "composite", "pose", "figmap"]) {
          if (shot[pass]) names[pass] = await upload(shot[pass], `sr_${key}_${pass}.png`);
        }
        _uploaded.set(key, names);
        while (_uploaded.size > 128) _uploaded.delete(_uploaded.keys().next().value);
      }
      shots.push({ ...names, camera: { ...names.camera, name: view.name || null } });
    }
  } finally {
    if (p.allShots) v.setView(current);
  }
  return { key: sr.key, shots };
}

function findNode(id) {
  const s = String(id);
  const visit = (graph, prefix) => {
    for (const n of graph?._nodes || graph?.nodes || []) {
      if (!n) continue;
      if ((n.comfyClass === CLASS || n.type === CLASS) && (prefix + n.id === s || String(n.id) === s.split(":").pop())) return n;
      if (n.subgraph) {
        const f = visit(n.subgraph, prefix + n.id + ":");
        if (f) return f;
      }
    }
    return null;
  };
  return visit(app.graph, "");
}

const _g2p = app.graphToPrompt;
app.graphToPrompt = async function (...args) {
  const result = await _g2p.apply(app, args);
  try {
    const out = result?.output;
    for (const id in out || {}) {
      const entry = out[id];
      if (entry?.class_type !== CLASS) continue;
      const node = findNode(id);
      entry.inputs = entry.inputs || {};
      let state = {};
      try {
        state = node ? await captureNode(node) : {};
      } catch (e) {
        console.error("[SceneRef] capture failed", e);
        node?._sr?.say("Capture failed: " + e.message, true);
      }
      entry.inputs[HIDDEN] = JSON.stringify(state);
    }
  } catch (e) {
    console.error("[SceneRef] graphToPrompt hook failed", e);
  }
  return result;
};
