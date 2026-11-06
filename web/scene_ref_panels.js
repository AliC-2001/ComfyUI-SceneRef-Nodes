// SceneRef - faces for the LoRA Stack and the Studio control panel.
//
// LoRA Stack: the list is edited here and stored as JSON in the (hidden)
// loras_json widget: [{name, strength, on, trigger}]. LoRAs are picked from ComfyUI's
// own filterable list menu.
//
// Studio: image or movie at the top; only the settings of what is being made are shown. The
// prompt box is the prompt as the model gets it; 'Write the prompt' fills it before Run.
// Prompt Expander and Character Maker: the text they wrote is shown on the node after a run.
//
// SceneRef Scene / Scene Maker: the place's pictures (one main view, other angles), edited
// like the Character Maker's. The Studio's 'Write the prompt' asks a Scene node for them.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CSS = `
.srl-root{display:flex;flex-direction:column;gap:4px;width:100%;box-sizing:border-box;
  font:12px/1.3 system-ui,sans-serif;color:var(--input-text,#ddd)}
.srl-row{display:flex;gap:4px;align-items:center;min-height:24px}
.srl-row>*{flex:0 0 auto}
.srl-grow{flex:1 1 auto!important;min-width:0}
.srl-btn{background:var(--comfy-input-bg,#222);color:inherit;border:1px solid var(--border-color,#444);
  border-radius:5px;padding:2px 8px;cursor:pointer;font:inherit;white-space:nowrap}
.srl-btn:hover{border-color:#7aa7ff}
.srl-in{background:var(--comfy-input-bg,#222);color:inherit;border:1px solid var(--border-color,#444);
  border-radius:5px;padding:2px 4px;font:inherit;min-width:0}
.srl-num{width:52px}
.srl-off{opacity:.45}
.srl-empty{opacity:.6;padding:4px 2px}
.srl-text{width:100%;height:100%;min-height:80px;box-sizing:border-box;resize:none;font-size:11px;opacity:.9}
.srl-prompt{flex:0 0 auto;width:100%;box-sizing:border-box;resize:vertical;min-height:80px;font-size:11px;line-height:1.35}
`;
let styled = false;
function style() {
  if (styled) return;
  styled = true;
  const s = document.createElement("style");
  s.textContent = CSS;
  document.head.append(s);
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

// Hide a widget but keep it serialized.
function hideWidget(w, hidden = true) {
  if (!w) return;
  if (!w._srOrig) w._srOrig = { computeSize: w.computeSize, type: w.type };
  w.hidden = hidden;
  w.computeSize = hidden ? () => [0, -4] : w._srOrig.computeSize;
  if (w.element) w.element.style.display = hidden ? "none" : "";
}

function fitHeight(node) {
  const min = node.computeSize();
  node.setSize([node.size[0], min[1]]);
  node.setDirtyCanvas(true, true);
}

// ── LoRA Stack ───────────────────────────────────────────────────────────────
let loraList = null;
async function loras(force = false) {
  if (!loraList || force) {
    try {
      const r = await api.fetchApi("/scene_ref/loras");
      loraList = r.ok ? await r.json() : [];
    } catch (_e) {
      loraList = [];
    }
  }
  return loraList;
}

function readStack(node) {
  try {
    const v = JSON.parse(widget(node, "loras_json")?.value || "[]");
    return Array.isArray(v) ? v : [];
  } catch (_e) {
    return [];
  }
}

function writeStack(node, items) {
  const w = widget(node, "loras_json");
  if (w) w.value = JSON.stringify(items);
  node.setDirtyCanvas(true, true);
}

// ComfyUI's own list menu, the one its model dropdowns open: with more than four entries it
// gets the "Filter list" box at the top (arrows + Enter work too).
function pickFromList(event, values, onPick) {
  const LG = window.LiteGraph;
  if (!LG?.ContextMenu || !values.length) return;
  new LG.ContextMenu(values, {
    event,
    className: "dark",
    scale: Math.max(1, app.canvas?.ds?.scale || 1),
    callback: (v) => onPick(typeof v === "string" ? v : v?.content ?? v),
  });
}

function buildLoraFace(node) {
  style();
  const root = el("div", "srl-root");
  const bar = el("div", "srl-row");
  const add = el("button", "srl-btn", "＋ LoRA");
  add.title = "Add a LoRA: pick it from the list (type in the box at its top to filter, e.g. flux9 or a name).";
  const reload = el("button", "srl-btn", "⟳");
  reload.title = "Re-read the LoRA folders (after adding files) and their trigger words";
  const hint = el("span", null, "the list filters as you type");
  hint.style.opacity = ".6";
  bar.append(add, reload, hint);
  const list = el("div");
  root.append(bar, list);

  const render = async () => {
    const all = await loras();
    list.innerHTML = "";
    const items = readStack(node);
    if (!items.length) list.append(el("div", "srl-empty", "No LoRAs yet: press ＋ LoRA and pick one."));
    items.forEach((it, i) => {
      const row = el("div", "srl-row" + (it.on === false ? " srl-off" : ""));
      const on = el("input");
      on.type = "checkbox";
      on.checked = it.on !== false;
      on.title = "On / off (off = kept in the list, not used)";
      const name = el("button", "srl-btn srl-grow", it.name);
      name.style.cssText = "text-align:left;overflow:hidden;text-overflow:ellipsis";
      name.title = `${it.name}
Click to pick another LoRA (the list has a filter box at the top).`;
      const str = el("input", "srl-in srl-num");
      str.type = "number";
      str.step = "0.05";
      str.value = it.strength ?? 1;
      str.title = "Strength: 1 = as trained; 0.5-0.8 tones it down";
      const trig = el("input", "srl-in");
      trig.style.width = "34%";
      trig.placeholder = "trigger words";
      const known = all.find((x) => x.name === it.name)?.triggers || [];
      // Rows saved before the words were known get them now; typed words are kept.
      trig.value = it.trigger || known.join(", ");
      trig.title = known.length
        ? `Words that switch this LoRA on, added to the prompt. Found with the file: ${known.join(", ")}`
        : "Words that switch this LoRA on, added to the prompt. None found with this file (no Civitai / LoRA Manager / rgthree notes, nothing in the file): type them from the LoRA's page.";
      const del = el("button", "srl-btn", "✕");
      del.title = "Remove this LoRA from the list";
      const save = (patch = {}) => {
        const cur = readStack(node);
        cur[i] = { name: it.name, strength: parseFloat(str.value) || 0, on: on.checked, trigger: trig.value, ...patch };
        writeStack(node, cur);
        row.classList.toggle("srl-off", !on.checked);
      };
      on.onchange = () => save();
      str.onchange = () => save();
      trig.onchange = () => save();
      name.onclick = (e) => pickFromList(e, all.map((x) => x.name), (v) => {
        // The words belong to the file: a new file brings its own (or none), never the old ones.
        const info = all.find((x) => x.name === v);
        save({ name: v, trigger: (info?.triggers || []).join(", ") });
        render();
      });
      del.onclick = () => {
        const cur = readStack(node);
        cur.splice(i, 1);
        writeStack(node, cur);
        render();
      };
      row.append(on, name, str, trig, del);
      list.append(row);
    });
    requestAnimationFrame(() => fitHeight(node));
  };

  add.onclick = async (e) => {
    const all = await loras();
    pickFromList(e, all.map((x) => x.name), (v) => {
      const info = all.find((x) => x.name === v);
      const cur = readStack(node);
      cur.push({ name: v, strength: 1, on: true, trigger: (info?.triggers || []).join(", ") });
      writeStack(node, cur);
      render();
    });
  };
  reload.onclick = async () => {
    await loras(true);
    render();
  };
  node._srlRender = render;
  return root;
}

// ── text shown after a run (the Studio's prompt, the expander's text) ────────
function addTextView(nodeType, title) {
  const executed = nodeType.prototype.onExecuted;
  nodeType.prototype.onExecuted = function (msg) {
    const r = executed?.apply(this, arguments);
    const text = (msg?.text || []).join("\n\n");
    if (!text) return r;
    if (!this._srText) {
      style();
      const box = el("textarea", "srl-in srl-text");
      box.readOnly = true;
      box.title = title;
      const w = this.addDOMWidget("scene_ref_text", "scene_ref_text", box, { serialize: false, getMinHeight: () => 90 });
      w.serialize = false;
      this._srText = box;
    }
    this._srText.value = text;
    requestAnimationFrame(() => fitHeight(this));
    return r;
  };
}

// ── Studio ───────────────────────────────────────────────────────────────────
// Everything the separate Character / LoRA Stack / Prompt Expander nodes do, in
// one panel: image or movie at the top, the action and the prompt box with its 'Write the
// prompt' button, the cast list (cast_json), the LoRA list (loras_json), and an Advanced
// toggle for sampling and models. Only the settings of what is being made are shown.
const KLEIN = "FLUX.2 Klein 9B";
const ONLY = {
  [KLEIN]: ["klein_model", "klein_text_encoder", "klein_vae", "ref_megapixels"],
  "FLUX.3 Klein": ["klein3_upsampler", "klein3_model", "klein3_text_encoder", "klein_vae"],
  "Qwen-Image 2.1 + ControlNet": ["qwen_model", "qwen_text_encoder", "qwen_vae", "qwen_controlnet", "control_strength", "control_mode"],
  "Krea2 + identity edit": ["krea2_model", "krea2_text_encoder", "krea2_vae", "krea2_identity_lora", "ref_boost"],
  "MOSS-Image + scene control": ["moss_model", "moss_text_encoder", "moss_vae", "moss_edit_lora", "moss_control_lora",
    "moss_control_strength", "moss_control_end", "moss_regions", "moss_layout"],
};
const FACE = ["face_pass", "face_steps", "face_denoise", "face_sampler", "face_scheduler"];
const IMAGE_ONLY = ["backend", "cfg", "image_resolution", "image_aspect", ...FACE, ...Object.values(ONLY).flat()];
const MOVIE_ONLY = ["video_seconds", "video_resolution", "aspect_ratio", "h3_ref_size", "h3_model", "h3_text_encoder",
  "h3_vae", "h3_audio_vae"];
// The model's usual steps / cfg / sampler / scheduler per backend (and movie mode), sent by
// the server with the backend list.
let BACKEND_DEFAULTS = {};
const SAMPLING = ["steps", "cfg", "sampler_name", "scheduler"];
// Unused settings, kept only so older workflows load: the removed H3 picture paths, and the
// style / expand switch from before the prompt box, the built-in H3 turbo LoRA ('expanded' is shown as the prompt box).
const LEGACY = ["filename_prefix", "identity_pass", "identity_strength", "h3_strength", "h3_resolution", "video", "video_motion",
  "style", "expand", "h3_turbo_lora"];
const ALWAYS_HIDDEN = ["cast_json", "loras_json", "expanded", ...LEGACY];
const MODELS = ["klein_model", "klein_text_encoder", "klein_vae", "klein3_model", "klein3_text_encoder", "qwen_model", "qwen_text_encoder", "qwen_vae",
  "qwen_controlnet", "krea2_model", "krea2_text_encoder", "krea2_vae", "krea2_identity_lora", "moss_model",
  "moss_text_encoder", "moss_vae", "moss_edit_lora", "moss_control_lora", "h3_model", "h3_text_encoder", "h3_vae",
  "h3_audio_vae"];
const ADVANCED = [...SAMPLING, "klein3_upsampler", "control_strength", "control_mode", "ref_megapixels", "ref_boost",
  "moss_control_strength", "moss_control_end", ...FACE,
  "h3_ref_size", ...MODELS];
// The order the widgets are shown in. Saved workflows keep the server's order (see
// arrange / canonical below), so this can change without breaking them.
const SHOWN_ORDER = ["create", "backend", "action", "expander_model", "room_details", "moss_layout", "scene_ref_expand", "seed",
  "image_resolution", "image_aspect", "moss_regions", "video_seconds", "video_resolution", "aspect_ratio", "scene_ref_size",
  "clear_cache_after", "scene_ref_studio", ...ADVANCED];
// Clearer names on the node (the inputs keep theirs).
const LABELS = { h3_vae: "h3_video_vae", h3_audio_vae: "h3_audio_vae", image_resolution: "picture_size",
  image_aspect: "picture_aspect", aspect_ratio: "movie_aspect", video_resolution: "movie_size",
  video_seconds: "movie_seconds", klein3_upsampler: "prompt_upsampler" };
const IMAGE_MP = { "as framed": 0, "0.5 MP": 0.5, "1 MP": 1, "1.5 MP": 1.5, "2 MP": 2, "3 MP": 3, "4 MP": 4 };
const VIDEO_MP = { "0.4 MP (fast)": 0.4, "0.6 MP": 0.6, "0.9 MP (720p)": 0.92, "1.1 MP": 1.1, "1.5 MP": 1.5, "2.1 MP (1080p)": 2.07 };
const COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "pink"];
const NONE = "(pictures only)";
const PIC_DIR = "scene_ref_pictures";
const VOICE_DIR = "scene_ref_voices";

const CAST_CSS = `
.srs-sec{display:flex;align-items:center;gap:6px;margin-top:6px;font-weight:600;opacity:.85}
.srs-sec::after{content:"";flex:1;border-top:1px solid var(--border-color,#444)}
.srs-card{border:1px solid var(--border-color,#444);border-radius:6px;padding:4px;display:flex;flex-direction:column;gap:3px}
.srs-desc{width:100%;box-sizing:border-box;resize:vertical;min-height:30px;height:30px}
.srs-pics{display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.srs-pic{position:relative;width:46px;height:46px;border-radius:4px;overflow:hidden;background:#111}
.srs-pic img{width:100%;height:100%;object-fit:cover}
.srs-pic span{position:absolute;left:0;bottom:0;right:0;font-size:9px;background:#000a;text-align:center}
.srs-pic b{position:absolute;right:2px;top:-1px;cursor:pointer;font-size:13px;text-shadow:0 0 2px #000}
.srs-dot{width:10px;height:10px;border-radius:50%}
`;
let castStyled = false;

let sourceList = null;
async function sources() {
  if (!sourceList) {
    try {
      const r = await api.fetchApi("/object_info/SceneRefCharacter");
      sourceList = (await r.json())?.SceneRefCharacter?.input?.required?.source?.[0] || [NONE];
    } catch (_e) {
      sourceList = [NONE];
    }
  }
  return sourceList;
}

function readCast(node) {
  try {
    const v = JSON.parse(widget(node, "cast_json")?.value || "[]");
    return Array.isArray(v) ? v : [];
  } catch (_e) {
    return [];
  }
}

function writeCast(node, rows) {
  const w = widget(node, "cast_json");
  if (w) w.value = JSON.stringify(rows);
  node.setDirtyCanvas(true, true);
}

// ComfyUI's /upload/image stores any file (its audio loader uploads through it too).
async function uploadPicture(file, subfolder = PIC_DIR) {
  const fd = new FormData();
  fd.append("image", file, file.name);
  fd.append("subfolder", subfolder);
  fd.append("type", "input");
  const r = await api.fetchApi("/upload/image", { method: "POST", body: fd });
  if (!r.ok) throw new Error(`upload failed (${r.status})`);
  const j = await r.json();
  return j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
}

function pictureURL(rel) {
  const i = rel.lastIndexOf("/");
  const q = new URLSearchParams({ filename: rel.slice(i + 1), subfolder: i < 0 ? "" : rel.slice(0, i), type: "input" });
  return api.apiURL(`/view?${q}`);
}

function option(select, value, text) {
  const o = el("option", null, text ?? value);
  o.value = value;
  select.append(o);
}

function buildCast(node) {
  if (!castStyled) {
    castStyled = true;
    const s = document.createElement("style");
    s.textContent = CAST_CSS;
    document.head.append(s);
  }
  const wrap = el("div");
  wrap.style.cssText = "display:flex;flex-direction:column;gap:4px";
  const list = el("div");
  list.style.cssText = "display:flex;flex-direction:column;gap:4px";
  const bar = el("div", "srl-row");
  const add = el("button", "srl-btn", "＋ character");
  add.title = "Add a person: a .char / RefMod, or ＋ picture for photos. Optionally tie them to a mannequin (colour); without one the action text says where they are.";
  const hint = el("span", null, "who is in it (a mannequin colour is optional)");
  hint.style.opacity = ".6";
  bar.append(add, hint);
  wrap.append(list, bar);

  const update = (i, patch, rerender = false) => {
    const rows = readCast(node);
    rows[i] = { ...rows[i], ...patch };
    writeCast(node, rows);
    if (rerender) render();
  };

  const render = async () => {
    const srcs = await sources();
    const rows = readCast(node);
    list.innerHTML = "";
    if (!rows.length) list.append(el("div", "srl-empty", "Nobody yet: press ＋ character for each person the picture should recognise."));
    rows.forEach((r, i) => {
      const card = el("div", "srs-card" + (r.on === false ? " srl-off" : ""));
      const top = el("div", "srl-row");
      const on = el("input");
      on.type = "checkbox";
      on.checked = r.on !== false;
      on.title = "On / off";
      on.onchange = () => {
        update(i, { on: on.checked });
        card.classList.toggle("srl-off", !on.checked);
      };
      const dot = el("span", "srs-dot");
      const paint = (f) => (dot.style.background = f ? COLORS[(f - 1) % 8] : "transparent");
      paint(r.figure);
      const fig = el("select", "srl-in");
      fig.title = "Optional: the mannequin this character plays. 'no figure' = placed by the action text (scenes without mannequins)";
      for (let f = 0; f <= 8; f++) option(fig, String(f), f ? `${f} · ${COLORS[f - 1]}` : "no figure");
      fig.value = String(r.figure ?? 0);
      fig.onchange = () => {
        update(i, { figure: +fig.value });
        paint(+fig.value);
      };
      const src = el("select", "srl-in srl-grow");
      src.title = "A .char file (models/characters), a RefMod (models/refmods), or pictures only";
      for (const v of new Set([...srcs, r.source || NONE])) option(src, v);
      src.value = r.source || NONE;
      const name = el("input", "srl-in");
      name.style.width = "22%";
      name.placeholder = "name (from file)";
      name.title = "The name used in the prompt and in 'action'. Empty = the name stored in the file. Write the action with this name.";
      name.value = r.name || "";
      name.onchange = () => update(i, { name: name.value });
      const max = el("input", "srl-in srl-num");
      max.type = "number";
      max.min = "1";
      max.max = "8";
      max.value = r.max_refs ?? 3;
      max.title = "At most this many reference pictures (faces first)";
      max.onchange = () => update(i, { max_refs: Math.max(1, Math.min(8, +max.value || 3)) });
      const del = el("button", "srl-btn", "✕");
      del.title = "Remove this character";
      del.onclick = () => {
        const rs = readCast(node);
        rs.splice(i, 1);
        writeCast(node, rs);
        render();
      };
      top.append(on, dot, fig, src, name, max, del);

      const desc = el("textarea", "srl-in srs-desc");
      desc.placeholder = "looks, clothes, age: added to the prompt (the pictures show the face)";
      desc.title = "Anything the pictures don't show: clothes for this scene, age, hair colour if it changed. Added to the prompt for this person. Leave the face to the pictures.";
      desc.value = r.description || "";
      desc.onchange = () => update(i, { description: desc.value });

      const pics = el("div", "srs-pics");
      (r.pictures || []).forEach((p, k) => {
        const t = el("div", "srs-pic");
        const img = el("img");
        img.src = pictureURL(p.file);
        img.title = p.file;
        const x = el("b", null, "×");
        x.title = "Remove picture";
        x.onclick = () => {
          const ps = [...(readCast(node)[i].pictures || [])];
          ps.splice(k, 1);
          update(i, { pictures: ps }, true);
        };
        t.append(img, el("span", null, p.role || "face"), x);
        pics.append(t);
      });
      const role = el("select", "srl-in");
      role.title = "What the next picture you add shows: face (close-ups, most important), body (full figure, for build) or outfit (clothes to wear)";
      for (const v of ["face", "body", "outfit"]) option(role, v);
      const addPic = el("button", "srl-btn", "＋ picture");
      addPic.title = `Pick ${"face / body / outfit"} pictures from your computer (copied into ComfyUI/input/${PIC_DIR})`;
      const file = el("input");
      file.type = "file";
      file.accept = "image/*";
      file.multiple = true;
      file.style.display = "none";
      addPic.onclick = () => file.click();
      file.onchange = async () => {
        const added = [];
        addPic.disabled = true;
        try {
          for (const f of file.files) added.push({ file: await uploadPicture(f), role: role.value });
        } catch (e) {
          hint.textContent = `Picture upload failed: ${e.message}`;
        }
        addPic.disabled = false;
        update(i, { pictures: [...(readCast(node)[i].pictures || []), ...added] }, true);
      };
      pics.append(role, addPic, file);
      // The text stored in a .char / RefMod stays out of the prompt unless asked for: a stored
      // a stored "curly hair" wins over photos that show straight hair.
      const ft = el("label", "srl-row");
      ft.style.cssText = "gap:4px;opacity:.8;font-size:11px";
      const ftBox = el("input");
      ftBox.type = "checkbox";
      ftBox.checked = !!r.file_text;
      const ftSay = el("span", null, "also use the text stored in the file");
      // On: the stored text is put in the box above, to read and edit. Off: the box is emptied.
      const fillFileText = async () => {
        if (!ftBox.checked) {
          desc.value = "";
          update(i, { file_text: false, description: "" });
          return;
        }
        const source = src.value;
        if (source === NONE) {
          update(i, { file_text: true });
          return;
        }
        ftSay.textContent = "reading the file…";
        try {
          const res = await api.fetchApi(`/scene_ref/char_text?source=${encodeURIComponent(source)}`);
          const j = await res.json().catch(() => ({}));
          if (!res.ok) throw new Error(j.error || `failed (${res.status})`);
          desc.value = j.text || "";
          update(i, { file_text: true, description: desc.value });
          ftSay.textContent = j.text ? "also use the text stored in the file" : "also use the text stored in the file (it has none)";
        } catch (e) {
          update(i, { file_text: true });
          ftSay.textContent = `also use the text stored in the file (⚠ ${e.message})`;
        }
      };
      ftBox.onchange = fillFileText;
      src.onchange = () => {
        update(i, { source: src.value });
        if (ftBox.checked) fillFileText();
      };
      ft.append(ftBox, ftSay);
      ft.title = "On: the text stored in the .char / RefMod is put in the box above (edit it there). Off: the box is emptied and the pictures alone say what this person looks like. A stored description that disagrees with them (age, hair) wins over the pictures.";

      // Voice (MiniMax H3 movies) and a check of what the pictures give.
      const vrow = el("div", "srl-row");
      vrow.style.cssText = "gap:4px;font-size:11px";
      const vsel = el("select", "srl-in");
      vsel.title = "The voice H3 gives this person when they speak in a movie: the one stored in the .char / RefMod, none, or a recording you add";
      option(vsel, "", "voice from the file");
      option(vsel, "off", "no voice");
      if (r.voice && r.voice !== "off") option(vsel, r.voice, `🎙 ${r.voice.split("/").pop()}`);
      vsel.value = r.voice || "";
      vsel.onchange = () => update(i, { voice: vsel.value });
      const addVoice = el("button", "srl-btn", "＋ voice");
      addVoice.title = `A recording of this person speaking: 5 to 30 s of clear speech, no music (wav, mp3, flac, m4a, ogg; copied into ComfyUI/input/${VOICE_DIR}). Only used when they speak in the prompt.`;
      const vfile = el("input");
      vfile.type = "file";
      vfile.accept = "audio/*,video/mp4,video/webm";
      vfile.style.display = "none";
      addVoice.onclick = () => vfile.click();
      vfile.onchange = async () => {
        if (!vfile.files[0]) return;
        addVoice.disabled = true;
        try {
          update(i, { voice: await uploadPicture(vfile.files[0], VOICE_DIR) }, true);
        } catch (e) {
          hint.textContent = `Voice upload failed: ${e.message}`;
          addVoice.disabled = false;
        }
      };
      const check = el("button", "srl-btn", "🔍 check");
      check.title = "Look at the pictures the way the Run will: which ones are used, face crops, blurred or missing faces, a picture of someone else, the same photo twice, and the voice";
      const report = el("div");
      report.style.cssText = "white-space:pre-wrap;font-size:11px;opacity:.85;display:none";
      check.onclick = async () => {
        check.disabled = true;
        report.style.display = "";
        report.textContent = "looking at the pictures…";
        try {
          const res = await api.fetchApi("/scene_ref/char_check", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ row: readCast(node)[i] }),
          });
          const j = await res.json().catch(() => ({}));
          if (!res.ok) throw new Error(j.error || `failed (${res.status})`);
          report.textContent = j.report;
        } catch (e) {
          report.textContent = `⚠ ${e.message}`;
        }
        check.disabled = false;
        requestAnimationFrame(() => fitHeight(node));
      };
      vrow.append(vsel, addVoice, vfile, check);
      card.append(top, desc, ft, pics, vrow, report);
      list.append(card);
    });
    requestAnimationFrame(() => fitHeight(node));
  };

  add.onclick = () => {
    // No figure by default: mannequins are optional, the text can place people too.
    const rows = readCast(node);
    rows.push({ on: true, figure: 0, source: NONE, name: "", description: "", max_refs: 3, pictures: [] });
    writeCast(node, rows);
    render();
  };
  node._srCastRender = render;
  return wrap;
}

// Sampling follows the backend (or movie mode): the model's usual steps, cfg, sampler and
// scheduler are filled in when it changes (and over the old "0 = usual" placeholder), so the
// values shown are the ones used. Any can be changed afterwards.
function applyBackendDefaults(node, force, keys = SAMPLING) {
  const movie = widget(node, "create")?.value === "movie";
  const d = BACKEND_DEFAULTS[movie ? "movie" : widget(node, "backend")?.value];
  if (!d) return;
  for (const k of keys) {
    const w = widget(node, k);
    if (w && d[k] != null && (force || ((k === "steps" || k === "cfg") && !Number(w.value)))) w.value = d[k];
  }
  node.setDirtyCanvas(true, true);
}

// Widget order. Saved workflows store widget values by position in the server's order, so
// the node keeps that order on disk (canonical) and only shows its own (arrange).
function canonical(node) {
  const have = new Set(node.widgets || []);
  const out = (node._srCanon || []).filter((w) => have.has(w));
  for (const w of node.widgets || []) if (!out.includes(w)) out.push(w);
  return out;
}

function arrange(node) {
  const canon = canonical(node);
  const placed = new Set();
  const out = [];
  const put = (w) => {
    if (!w || placed.has(w)) return;
    placed.add(w);
    out.push(w);
    // seed's "control after generate" (and any other linked widget) stays right after it
    for (const l of w.linkedWidgets || []) put(l);
  };
  for (const name of SHOWN_ORDER) put(canon.find((w) => w.name === name));
  for (const w of canon) put(w);
  node.widgets.splice(0, node.widgets.length, ...out);
}

function serializeValue(v) {
  return typeof v === "object" && v ? JSON.parse(JSON.stringify(v)) : v ?? null;
}

// The node wired to an input (by name, or the input itself).
function upstream(node, name) {
  const input = typeof name === "string" ? node.inputs?.find((i) => i.name === name) : name;
  const g = node.graph;
  if (input?.link == null || !g) return null;
  const link = g.getLink?.(input.link) ?? g.links?.get?.(input.link) ?? g.links?.[input.link];
  return link ? g.getNodeById(link.origin_id) : null;
}

// The upstream viewer or SceneRef Scene (wired to 'shot'), through Reroutes, bypassed nodes and
// ComfyUI's If/Else Switch (the branch its toggle picks, also a toggle wired in from a Boolean node).
function viewerOf(node) {
  let n = upstream(node, "shot");
  const seen = new Set();
  while (n && !seen.has(n.id) && !n._srCapture && !n._srSceneRequest) {
    seen.add(n.id);
    let input;
    if (n.type === "ComfySwitchNode") {
      const wired = upstream(n, "switch");
      const toggle = wired ? wired.widgets?.find((w) => typeof w.value === "boolean") : widget(n, "switch");
      input = toggle ? n.inputs?.find((i) => i.name === (toggle.value ? "on_true" : "on_false"))
        : n.inputs?.find((i) => i.name !== "switch" && i.link != null);
    } else if (n.mode === 4 || n.type === "Reroute") {
      input = n.inputs?.find((i) => i.link != null);
    } else {
      break;
    }
    n = input ? upstream(n, input) : null;
  }
  return n;
}

// What size the picture / clip comes out at, as the server works it out (studio.py
// aspect_crop + h3_size).
function outSize(node, cam) {
  const movie = widget(node, "create")?.value === "movie";
  const ratio = String(widget(node, movie ? "aspect_ratio" : "image_aspect")?.value || "as framed");
  const m = ratio.match(/(\d+):(\d+)/);
  let [w, h] = cam;
  if (m) {
    const r = +m[1] / +m[2];
    if (w / h > r) w = Math.floor(h * r / 16) * 16;
    else h = Math.floor(w / r / 16) * 16;
  }
  const mp = movie ? VIDEO_MP[widget(node, "video_resolution")?.value] || 0.4 : IMAGE_MP[widget(node, "image_resolution")?.value] || 0;
  const grid = movie ? 32 : 16;
  if (!mp) return [Math.floor(w / 16) * 16, Math.floor(h / 16) * 16];
  const s = Math.sqrt(mp * 1e6 / (w * h));
  return [Math.max(grid, Math.round(w * s / grid) * grid), Math.max(grid, Math.round(h * s / grid) * grid)];
}

function refreshStudio(node) {
  const backend = widget(node, "backend")?.value;
  const movie = widget(node, "create")?.value === "movie";
  const adv = !!node.properties?.srAdvanced;
  for (const w of node.widgets || []) {
    if (!w.name) continue;
    if (LABELS[w.name]) w.label = LABELS[w.name];
    let hide = ALWAYS_HIDDEN.includes(w.name) || (ADVANCED.includes(w.name) && !adv);
    if (movie ? IMAGE_ONLY.includes(w.name) : MOVIE_ONLY.includes(w.name)) hide = true;
    // A backend's own settings (klein_vae is both Kleins'): hidden unless the backend shown lists it.
    if (!movie && Object.values(ONLY).some((names) => names.includes(w.name)) && !(ONLY[backend] || []).includes(w.name)) hide = true;
    hideWidget(w, hide);
  }
  if (node._srAdvBtn) node._srAdvBtn.textContent = `${adv ? "▾" : "▸"} Advanced: sampling, models${movie ? "" : ", face pass"}`;
  if (node._srExpandBtn) {
    node._srExpandBtn.textContent = `✎ Write the ${movie ? "movie" : "picture"} prompt`;
    node._srExpandBtn.title = !movie && backend === "FLUX.3 Klein"
      ? "BFL's FLUX.3 prompt upsampler (qwen3.5_9b_flux3_prompt_upsampler, else the expander model) looks at the scene (the viewer's current shot, or a SceneRef Scene's pictures) and the cast pictures, reads the action and writes the full scene caption FLUX.3 Klein is trained on, into the box below. It replaces what is in the box (↶ brings the previous text back). Edit it, then Run: the box is used exactly as written."
      : `The vision model reads the action and looks at the scene (the viewer's current shot, or a SceneRef Scene's main view) and writes the whole prompt for ${movie ? "MiniMax H3 (its reference-mode sections)" : backend}, following that model's prompt guide, into the box below. It replaces what is in the box (↶ brings the previous text back). Edit it, then Run: the box is used exactly as written.`;
  }
  node._srPromptSync?.();
  node._srSizeLine?.();
  requestAnimationFrame(() => fitHeight(node));
}

// The prompt box (the hidden 'expanded' widget holds its text) and the button that has the
// vision model write it. The box keeps the height it is dragged to (node property).
function buildExpandRow(node) {
  style();
  const root = el("div", "srl-root");
  const row = el("div", "srl-row");
  const btn = el("button", "srl-btn");
  const undo = el("button", "srl-btn", "↶");
  undo.title = "Bring back the text that was in the box before it was written";
  undo.style.display = "none";
  const say = el("span", "srl-grow");
  say.style.cssText = "opacity:.7;font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
  row.append(btn, undo, say);
  // Who the action / prompt can name, and their @numbers (characters.resolve_refs).
  const who = el("div");
  who.style.cssText = "opacity:.8;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-height:15px;margin:2px 0";
  who.title = "Write a name, or its @number, in the action or the prompt: @1, @2 ... become the names (at ✎ and at Run).";
  node._srWhoLine = () => {
    let rows;
    try {
      rows = [...readCast(node).map((r) => ({ ...r, slot: "Cast list" })), ...wiredCast(node)];
    } catch (e) {
      node._srWhoKey = null;
      who.textContent = `⚠ ${e.message}`;
      return;
    }
    const key = JSON.stringify(rows);
    if (key === node._srWhoKey) return;
    node._srWhoKey = key;
    const none = "Nobody in the cast yet.";
    if (!rows.length) {
      who.textContent = none;
      return;
    }
    api.fetchApi("/scene_ref/cast_names", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ cast_json: key }) })
      .then((r) => r.json())
      .then((j) => {
        if (node._srWhoKey !== key) return;
        const kept = rows.map((r, i) => [r, j.names?.[i]]).filter(([, n]) => n);
        who.textContent = kept.length ? `In the prompt: ${kept.map(([r, n], i) => `@${i + 1} ${n} (${r.slot})`).join(" · ")}` : none;
      })
      .catch(() => (node._srWhoKey = null));
  };
  const box = el("textarea", "srl-in srl-prompt");
  box.placeholder = "The prompt, exactly as the model gets it. Write it here, or write the action above and press ✎ to have the vision model write it.";
  box.title = "The prompt, used exactly as written: nothing is added at Run (LoRA trigger words included: ✎ writes them in). Drag the corner to make the box bigger.";
  box.style.height = `${node.properties?.srPromptH || 180}px`;
  root.append(row, who, box);
  const store = (text) => {
    const w = widget(node, "expanded");
    if (w) w.value = text;
    node.setDirtyCanvas(true, true);
  };
  box.addEventListener("input", () => store(box.value));
  // Keyboard shortcuts of the canvas (delete node, copy...) must not fire while typing here.
  box.addEventListener("keydown", (e) => e.stopPropagation());
  node._srPromptSync = () => {
    const v = widget(node, "expanded")?.value ?? "";
    if (document.activeElement !== box && box.value !== v) box.value = v;
  };
  new ResizeObserver(() => {
    const h = Math.round(box.offsetHeight);
    if (!h || h === node.properties?.srPromptH) return;
    node.properties = node.properties || {};
    node.properties.srPromptH = h;
    requestAnimationFrame(() => fitHeight(node));
  }).observe(box);
  node._srExpandBtn = btn;
  undo.onclick = () => {
    if (node._srPrevPrompt == null) return;
    box.value = node._srPrevPrompt;
    store(box.value);
    node._srPrevPrompt = null;
    undo.style.display = "none";
    say.textContent = "previous text back";
  };
  btn.onclick = async () => {
    const viewer = viewerOf(node);
    if (!viewer?._srCapture && !viewer?._srSceneRequest) {
      say.textContent = "Wire the 'shot' of a Scene Reference Viewer or a SceneRef Scene first.";
      return;
    }
    btn.disabled = true;
    const t0 = performance.now();
    const tick = setInterval(() => (say.textContent = `writing… ${Math.round((performance.now() - t0) / 1000)} s (the model loads first)`), 500);
    try {
      let scene;
      if (viewer._srSceneRequest) {
        scene = { scene: viewer._srSceneRequest() };
      } else {
        const capture = await viewer._srCapture();
        if (!capture?.shots?.length) throw new Error("the viewer has no scene loaded");
        const [width, height] = viewer._srOutputSize();
        scene = { capture, width, height };
      }
      const value = (k, d) => widget(node, k)?.value ?? d;
      const r = await api.fetchApi("/scene_ref/expand", {
        // No time limit: the frontend's default (60 s) is shorter than a first model load.
        timeoutMs: null,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...scene,
          action: value("action", ""),
          backend: value("backend", KLEIN),
          movie: value("create") === "movie",
          cast_json: JSON.stringify([...readCast(node), ...wiredCast(node)]),
          loras_json: value("loras_json", "[]"),
          room_details: !!value("room_details", true),
          expander_model: value("expander_model", "auto"),
          control_strength: value("control_strength", 0.75),
          image_aspect: value("image_aspect", "as framed"),
          aspect_ratio: value("aspect_ratio", "as framed"),
          video_seconds: value("video_seconds", 5),
          klein3_upsampler: value("klein3_upsampler", "auto"),
          image_resolution: value("image_resolution", "as framed"),
          moss_layout: !!value("moss_layout", false),
        }),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.error || `failed (${r.status})`);
      if (box.value.trim()) {
        node._srPrevPrompt = box.value;
        undo.style.display = "";
      }
      box.value = j.text || "";
      store(box.value);
      say.textContent = `written in ${Math.round((performance.now() - t0) / 1000)} s: edit it below, then Run`;
    } catch (e) {
      say.textContent = `⚠ ${e.message}`;
    } finally {
      clearInterval(tick);
      btn.disabled = false;
      node.setDirtyCanvas(true, true);
    }
  };
  node._srPromptRoot = root;
  return root;
}

// The size it will come out at, and a button that makes it the camera's own.
function buildSizeRow(node) {
  style();
  const row = el("div", "srl-root srl-row");
  row.style.flexDirection = "row";
  const line = el("span", "srl-grow");
  line.style.cssText = "opacity:.8;font-size:11px";
  const match = el("button", "srl-btn", "⤢ Match camera");
  match.title = "Make the picture / movie the camera's own shape (the viewer's width x height); pictures also its size.";
  row.append(line, match);
  const update = () => {
    const up = viewerOf(node);
    const cam = up?._srOutputSize?.();
    const movie = widget(node, "create")?.value === "movie";
    if (!cam) {
      line.textContent = up?._srSceneRequest ? "Size: shown once the scene has its main view (or after a Run)."
        : "Wire the viewer's 'shot' to see the size.";
      return;
    }
    const [w, h] = outSize(node, cam);
    line.textContent = `${up._srSceneRequest ? "main view" : "camera"} ${cam[0]} × ${cam[1]}  →  ${movie ? "movie" : "picture"} ${w} × ${h}`;
  };
  match.onclick = () => {
    const movie = widget(node, "create")?.value === "movie";
    for (const k of movie ? ["aspect_ratio"] : ["image_aspect", "image_resolution"]) {
      const w = widget(node, k);
      if (w) w.value = "as framed";
    }
    update();
    node.setDirtyCanvas(true, true);
  };
  node._srSizeLine = update;
  return row;
}

// 'auto' model widgets show the file the server would pick, so it is visible which one runs.
// They are still saved and queued as 'auto' (unless another file was picked since), so a shared
// workflow or picture carries no local file names and loads on other installs.
let picks = null;
const autoValue = (w) => (w._srAutoPick && w.value === w._srAutoPick ? "auto" : w.value);
async function showPicks(node) {
  if (!picks) {
    try {
      const r = await api.fetchApi("/scene_ref/studio_picks");
      picks = r.ok ? await r.json() : {};
    } catch (_e) {
      picks = {};
    }
  }
  for (const [name, file] of Object.entries(picks)) {
    const w = widget(node, name);
    if (!w || !file) continue;
    const values = w.options?.values;
    if (w.value === "auto" && Array.isArray(values) && values.includes(file)) {
      w.value = file;
      w._srAutoPick = file;
      w.serializeValue = () => autoValue(w);
    }
  }
  node.setDirtyCanvas(true, true);
}

// Workflows from before image / movie: the H3 backend made pictures (now Klein does), the
// "also make a video clip" switch becomes movie mode.
// v3: the sampler / scheduler widgets are new; a workflow from before gets the ones its
// backend (or movie mode) used to have built in.
function migrateStudio(node) {
  node.properties = node.properties || {};
  const v = node.properties.srStudioV || 1;
  if (v >= 3) return;
  node.properties.srStudioV = 3;
  if (v < 2) {
    const b = widget(node, "backend");
    if (b && !b.options?.values?.includes(b.value)) b.value = KLEIN;
    if (widget(node, "video")?.value === true && widget(node, "create")) widget(node, "create").value = "movie";
  }
  applyBackendDefaults(node, true, ["sampler_name", "scheduler"]);
}

function buildStudioPanel(node) {
  style();
  const root = el("div", "srl-root");
  const inner = el("div");
  inner.style.cssText = "display:flex;flex-direction:column;gap:4px";
  const sec = (t) => el("div", "srs-sec", t);
  const adv = el("button", "srl-btn");
  adv.style.alignSelf = "flex-start";
  adv.onclick = () => {
    node.properties = node.properties || {};
    node.properties.srAdvanced = !node.properties.srAdvanced;
    refreshStudio(node);
  };
  node._srAdvBtn = adv;
  inner.append(sec("Cast"), buildCast(node), sec("LoRAs"), buildLoraFace(node), adv);
  root.append(inner);
  node._srPanelInner = inner;
  return root;
}

// ── Character Maker, Scene, Scene Maker ──────────────────────────────────────
// The pictures for a new .char (or a scene): picked from the computer (into ComfyUI/input),
// each with its role; stored in the hidden pictures_json widget.
const CHAR_PICS = {
  roles: [["face", "face"], ["body", "body"], ["outfit", "outfit"]],
  hint: "Faces first: a few clear photos from different angles work best; add a full body photo for the build.",
  roleTip: "What this picture shows: face, body (full figure) or outfit",
};
// One main view (the camera); the first picture added becomes it unless a .scene file has one.
const SCENE_PICS = {
  roles: [["main", "main view"], ["angle", "other angle"]],
  single: "main",
  hint: "The main view is the camera: the result keeps its angle, framing and furniture. Other angles of the same place go to movies (and to pictures with angles_in_pictures on). Nobody is placed: say in the action who is where.",
  roleTip: "main view = the camera the result keeps (only one); other angle = the same place from elsewhere",
};
function buildMakerPanel(node, opts = CHAR_PICS) {
  style();
  if (!castStyled) {
    castStyled = true;
    const s = document.createElement("style");
    s.textContent = CAST_CSS;
    document.head.append(s);
  }
  const read = () => {
    try {
      const v = JSON.parse(widget(node, "pictures_json")?.value || "[]");
      return Array.isArray(v) ? v : [];
    } catch (_e) {
      return [];
    }
  };
  const write = (rows) => {
    const w = widget(node, "pictures_json");
    if (w) w.value = JSON.stringify(rows);
    node.setDirtyCanvas(true, true);
  };
  const root = el("div", "srl-root");
  const inner = el("div");
  inner.style.cssText = "display:flex;flex-direction:column;gap:4px";
  const pics = el("div", "srs-pics");
  const hint = el("div", null, opts.hint);
  hint.style.cssText = "opacity:.6;font-size:11px";
  inner.append(el("div", "srs-sec", "Pictures"), pics, hint);
  root.append(inner);
  const render = () => {
    pics.innerHTML = "";
    read().forEach((p, k) => {
      const t = el("div", "srs-pic");
      t.style.cssText = "width:72px;height:72px";
      const img = el("img");
      img.src = pictureURL(p.file);
      img.title = p.file;
      const roleSel = el("select");
      roleSel.style.cssText = "position:absolute;left:0;right:0;bottom:0;font-size:10px;background:#000a;color:#fff;border:0";
      for (const [v, text] of opts.roles) option(roleSel, v, text);
      roleSel.value = p.role || opts.roles[0][0];
      roleSel.title = opts.roleTip;
      roleSel.onchange = () => {
        let rows = read();
        if (opts.single && roleSel.value === opts.single) rows = rows.map((r) => (r.role === opts.single ? { ...r, role: opts.roles[1][0] } : r));
        rows[k] = { ...rows[k], role: roleSel.value };
        write(rows);
        if (opts.single) render();
      };
      const x = el("b", null, "×");
      x.title = "Remove picture";
      x.onclick = () => {
        const rows = read();
        rows.splice(k, 1);
        write(rows);
        render();
      };
      t.append(img, roleSel, x);
      pics.append(t);
    });
    const role = el("select", "srl-in");
    role.title = `The next pictures you add: ${opts.roleTip}`;
    for (const [v, text] of opts.roles) option(role, v, text);
    if (opts.single) {
      const src = widget(node, "source")?.value;
      const hasMain = read().some((r) => r.role === opts.single) || (src && src !== NONE);
      role.value = hasMain ? opts.roles[1][0] : opts.single;
    }
    const add = el("button", "srl-btn", "＋ picture");
    add.title = `Pick pictures from your computer (copied into ComfyUI/input/${PIC_DIR})`;
    const file = el("input");
    file.type = "file";
    file.accept = "image/*";
    file.multiple = true;
    file.style.display = "none";
    add.onclick = () => file.click();
    file.onchange = async () => {
      const added = [];
      add.disabled = true;
      try {
        for (const f of file.files) added.push({ file: await uploadPicture(f), role: role.value });
      } catch (e) {
        hint.textContent = `Picture upload failed: ${e.message}`;
      }
      let rows = [...read(), ...added];
      if (opts.single && added.some((r) => r.role === opts.single)) {
        // Only one main view: the first one just added.
        const first = rows.findIndex((r, i) => i >= rows.length - added.length && r.role === opts.single);
        rows = rows.map((r, i) => (r.role === opts.single && i !== first ? { ...r, role: opts.roles[1][0] } : r));
      }
      write(rows);
      render();
    };
    pics.append(role, add, file);
    requestAnimationFrame(() => fitHeight(node));
  };
  node._srMakerRender = render;
  node._srPanelInner = inner;
  return root;
}

// The input file of a Load Image node wired to `name`: undefined = nothing wired, null = wired
// to something whose picture exists only at Run.
function linkedFile(node, name, key = "image") {
  if (!node.inputs?.find((i) => i.name === name && i.link != null)) return undefined;
  const v = (() => {
    const src = upstream(node, name);
    return src ? widget(src, key)?.value : null;
  })();
  if (typeof v !== "string" || !v || (/\]$/.test(v) && !v.endsWith(" [input]"))) return null;
  return v.replace(/ \[input\]$/, "");
}

function readPictures(node) {
  try {
    const v = JSON.parse(widget(node, "pictures_json")?.value || "[]");
    return Array.isArray(v) ? v.filter((r) => r?.file) : [];
  } catch (_e) {
    return [];
  }
}

// The main view's file, in the order scenes.gather picks it; null = not known before Run
// (a .scene file's own main view, or a picture made by another node).
function sceneMainFile(node) {
  const wired = linkedFile(node, "main_view");
  if (wired !== undefined) return wired;
  const rows = readPictures(node);
  const main = rows.find((r) => r.role === "main");
  if (main) return main.file;
  const src = widget(node, "source")?.value;
  if (src && src !== NONE) return null;
  return rows[0]?.file || sceneAngleFiles(node).find((f) => f) || null;
}

// The files of the Load Image nodes wired to other_angles, other_angles_2 ... (null where a
// picture comes from another node).
function sceneAngleFiles(node) {
  return grownInputs(node, "other_angles").map((i) => linkedFile(node, i.name)).filter((f) => f !== undefined);
}

// scenes.output_size: shrunk (never enlarged) to `mp`, sides on a 16 grid.
function sceneSize(w, h, mp) {
  const s = Math.min(1, Math.sqrt(mp * 1e6 / (w * h)));
  return [Math.max(16, Math.floor(Math.floor(w * s) / 16) * 16), Math.max(16, Math.floor(Math.floor(h * s) / 16) * 16)];
}

// What the Studio asks a SceneRef Scene for: its size, and its settings for 'Write the prompt'.
function setupScene(node) {
  node._srOutputSize = () => {
    const file = sceneMainFile(node);
    const mp = +(widget(node, "max_megapixels")?.value || 1);
    if (!file) return node._srRunSize || null;
    const key = `${file}|${mp}`;
    if (node._srSizeKey !== key) {
      node._srSizeKey = key;
      node._srSize = null;
      const img = new Image();
      img.onload = () => {
        if (node._srSizeKey === key) node._srSize = sceneSize(img.naturalWidth, img.naturalHeight, mp);
      };
      img.src = pictureURL(file);
    }
    return node._srSize || null;
  };
  node._srSceneRequest = () => {
    const main = linkedFile(node, "main_view");
    const angles = sceneAngleFiles(node);
    if (main === null || angles.includes(null)) {
      throw new Error("✎ reads the scene's pictures from the Scene node's own list or from Load Image nodes; a picture made by another node exists only at Run. Add it on the Scene node with ＋ picture.");
    }
    const value = (k, d) => widget(node, k)?.value ?? d;
    return {
      source: value("source", NONE), pictures_json: value("pictures_json", "[]"),
      description: value("description", ""), use_file_text: !!value("use_file_text", true),
      max_megapixels: value("max_megapixels", 1), angles_in_pictures: !!value("angles_in_pictures", false),
      main_file: main || "", angle_files: angles,
    };
  };
}

// Inputs that grow (characters.grow_inputs): the node declares `base`, base_2 ... base_N;
// it shows every wired one plus one empty one while there is room. Python reads them by
// name, so the order on the node does not matter.
function grownInputs(node, base) {
  const num = (i) => (i.name === base ? 1 : +i.name.slice(base.length + 1));
  return (node.inputs || []).filter((i) => isGrown(i.name, base)).sort((a, b) => num(a) - num(b));
}

function isGrown(name, base) {
  return name === base || (name.startsWith(`${base}_`) && /^\d+$/.test(name.slice(base.length + 1)));
}

function growInputs(nodeType, nodeData, base) {
  const optional = nodeData.input?.optional || {};
  const type = optional[base]?.[0];
  const names = Object.keys(optional).filter((k) => isGrown(k, base));
  if (!type || names.length < 2) return;
  const tidy = (node) => {
    const empty = grownInputs(node, base).filter((i) => i.link == null);
    for (const i of empty.slice(1).reverse()) node.removeInput(node.inputs.indexOf(i));
    const used = new Set(grownInputs(node, base).map((i) => i.name));
    const free = names.find((n) => !used.has(n));
    if (!empty.length && free) node.addInput(free, type);
    node.setDirtyCanvas?.(true, true);
  };
  const later = (node) => setTimeout(() => node.graph && tidy(node), 0);
  const created = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const r = created?.apply(this, arguments);
    tidy(this);
    return r;
  };
  const configure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const r = configure?.apply(this, arguments);
    later(this);
    return r;
  };
  const changed = nodeType.prototype.onConnectionsChange;
  nodeType.prototype.onConnectionsChange = function (side) {
    const r = changed?.apply(this, arguments);
    if (side === 1) later(this);  // LiteGraph.INPUT
    return r;
  };
}

// The Character nodes wired into the Studio's cast inputs, as rows of the panel's cast list,
// in the order the Run adds them (each chain from its first node), so 'Write the prompt'
// knows everyone the Run will get.
function wiredCast(node) {
  const rows = [];
  const seen = new Set();
  const walk = (n) => {
    if (!n || seen.has(n.id)) return;
    seen.add(n.id);
    if (n.mode === 2) return;  // muted
    if (n.mode === 4 || n.type === "Reroute") {  // bypassed: the cast passes through
      const input = n.inputs?.find((i) => i.type === "SCENEREF_CAST" || i.type === "*");
      return input ? walk(upstream(n, input)) : undefined;
    }
    if (n.type !== "SceneRefCharacter") {
      throw new Error(`✎ reads the cast from SceneRef Character nodes; what ${n.title || n.type} gives is only known at Run.`);
    }
    walk(upstream(n, "cast"));
    const pictures = [];
    for (const role of ["face", "body", "outfit"]) {
      const file = linkedFile(n, role);
      if (file === null) {
        throw new Error(`✎ reads a character's ${role} pictures from Load Image nodes; the one wired into ${n.title || "a Character node"} is made by another node and exists only at Run.`);
      }
      if (file) pictures.push({ file, role });
    }
    const value = (k, d) => widget(n, k)?.value ?? d;
    rows.push({
      source: value("source", NONE), figure: value("figure", 0), name: value("name", ""),
      description: value("description", ""), max_refs: value("max_refs", 3), refmod_vae: value("refmod_vae", ""),
      file_text: !!value("use_file_text", false), pictures, voice: linkedFile(n, "voice", "audio") || "",
    });
  };
  for (const input of grownInputs(node, "cast")) {
    const start = rows.length;
    walk(upstream(node, input));
    for (const r of rows.slice(start)) r.slot = input.name;
  }
  return rows;
}

function hookCallback(node, name, fn) {
  const w = widget(node, name);
  if (!w) return;
  const cb = w.callback;
  w.callback = (...a) => {
    const res = cb?.apply(w, a);
    fn();
    return res;
  };
}

app.registerExtension({
  name: "SceneRef.Panels",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === "SceneRefLoraStack") {
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = created?.apply(this, arguments);
        hideWidget(widget(this, "loras_json"));
        const w = this.addDOMWidget("scene_ref_loras", "scene_ref_loras", buildLoraFace(this), {
          serialize: false,
          getMinHeight: () => 34 + 28 * Math.max(1, readStack(this).length),
        });
        w.serialize = false;
        this.size = [Math.max(this.size?.[0] || 0, 480), this.size?.[1] || 120];
        this._srlRender();
        return r;
      };
      const configure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const r = configure?.apply(this, arguments);
        queueMicrotask(() => this._srlRender?.());
        return r;
      };
    }
    if (nodeData?.name === "SceneRefPromptExpander") addTextView(nodeType, "What the expander wrote");
    if (nodeData?.name === "SceneRefCharacter") addTextView(nodeType, "The pictures used, and what looks wrong");
    const scene = nodeData?.name === "SceneRefScene" || nodeData?.name === "SceneRefSceneMaker";
    if (nodeData?.name === "SceneRefScene") {
      addTextView(nodeType, "The scene as the model gets it");
      const executed = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (msg) {
        const r = executed?.apply(this, arguments);
        if (Array.isArray(msg?.sr_size)) this._srRunSize = msg.sr_size;
        return r;
      };
    }
    if (nodeData?.name === "SceneRefCharacterMaker" || nodeData?.name === "SceneRefRefModMaker" || scene) {
      if (nodeData.name !== "SceneRefScene") addTextView(nodeType, "What was saved");
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = created?.apply(this, arguments);
        hideWidget(widget(this, "pictures_json"));
        if (nodeData.name === "SceneRefScene") setupScene(this);
        if (scene) hookCallback(this, "source", () => this._srMakerRender?.());
        const w = this.addDOMWidget("scene_ref_maker", "scene_ref_maker", buildMakerPanel(this, scene ? SCENE_PICS : CHAR_PICS), {
          serialize: false,
          getMinHeight: () => Math.max(110, (this._srPanelInner?.offsetHeight || 100) + 8),
        });
        w.serialize = false;
        this.size = [Math.max(this.size?.[0] || 0, 460), this.size?.[1] || 400];
        this._srMakerRender();
        return r;
      };
      const configure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const r = configure?.apply(this, arguments);
        queueMicrotask(() => {
          hideWidget(widget(this, "pictures_json"));
          this._srMakerRender?.();
        });
        return r;
      };
    }
    if (nodeData?.name === "SceneRefStudio") growInputs(nodeType, nodeData, "cast");
    if (scene) growInputs(nodeType, nodeData, "other_angles");
    if (nodeData?.name === "SceneRefStudio") {
      BACKEND_DEFAULTS = nodeData.input?.required?.backend?.[1]?.sr_defaults || {};
      // Shown order on the node, server order on disk (see arrange): configure reads values by
      // position, so it runs on the server's order; serialize writes them in it.
      const configureFull = nodeType.prototype.configure;
      nodeType.prototype.configure = function () {
        if (this._srCanon) this.widgets.splice(0, this.widgets.length, ...canonical(this));
        try {
          return configureFull.apply(this, arguments);
        } finally {
          if (this._srCanon) arrange(this);
        }
      };
      const serialize = nodeType.prototype.onSerialize;
      nodeType.prototype.onSerialize = function (o) {
        const r = serialize?.apply(this, arguments);
        // A node made in this version is saved as such, so loading it never migrates it.
        this.properties = this.properties || {};
        if (!this.properties.srStudioV) this.properties.srStudioV = 3;
        if (o) o.properties = { ...(o.properties || {}), srStudioV: this.properties.srStudioV };
        if (this._srCanon && o?.widgets_values) {
          o.widgets_values = canonical(this).filter((w) => w.serialize !== false).map((w) => serializeValue(autoValue(w)));
        }
        if (o?.widgets_values_named) {
          for (const w of this.widgets || []) if (w.name in o.widgets_values_named) o.widgets_values_named[w.name] = autoValue(w);
        }
        return r;
      };
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = created?.apply(this, arguments);
        this._srCanon = [...this.widgets];
        const w = this.addDOMWidget("scene_ref_studio", "scene_ref_studio", buildStudioPanel(this), {
          serialize: false,
          getMinHeight: () => Math.max(120, (this._srPanelInner?.offsetHeight || 200) + 8),
        });
        w.serialize = false;
        // The box sets its own height (dragged, kept in srPromptH); the row is that plus the button and names lines.
        const promptHeight = () => (this.properties?.srPromptH || 180) + 55;
        for (const [name, build, height] of [["scene_ref_expand", buildExpandRow, promptHeight],
          ["scene_ref_size", buildSizeRow, () => 28]]) {
          const x = this.addDOMWidget(name, name, build(this), { serialize: false, getMinHeight: height, getMaxHeight: height });
          x.serialize = false;
        }
        arrange(this);
        const refresh = () => refreshStudio(this);
        hookCallback(this, "backend", () => {
          applyBackendDefaults(this, true);
          refresh();
        });
        hookCallback(this, "create", () => {
          applyBackendDefaults(this, true);
          refresh();
        });
        for (const k of ["image_resolution", "image_aspect", "video_resolution", "aspect_ratio"]) {
          hookCallback(this, k, () => this._srSizeLine?.());
        }
        this._srSizeTimer = setInterval(() => {
          this._srSizeLine?.();
          this._srWhoLine?.();
        }, 1000);
        this.size = [Math.max(this.size?.[0] || 0, 540), this.size?.[1] || 600];
        this._srCastRender();
        this._srlRender();
        refresh();
        showPicks(this);
        return r;
      };
      const configure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const r = configure?.apply(this, arguments);
        queueMicrotask(() => {
          migrateStudio(this);
          applyBackendDefaults(this, false);
          this._srCastRender?.();
          this._srlRender?.();
          refreshStudio(this);
          showPicks(this);
        });
        return r;
      };
      const removed = nodeType.prototype.onRemoved;
      nodeType.prototype.onRemoved = function () {
        clearInterval(this._srSizeTimer);
        return removed?.apply(this, arguments);
      };
    }
    if (nodeData?.name === "SceneRefOutput") {
      // manual: the result stays a preview until 'Save to output' copies it there.
      const executed = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (msg) {
        const r = executed?.apply(this, arguments);
        this._srShown = (msg?.images || []).filter((f) => f.type === "temp");
        const auto = !!msg?.sr_saved?.[0];
        if (this._srSaveBtn) {
          this._srSaveBtn.disabled = auto || !this._srShown.length;
          this._srSaveSay.textContent = auto ? "saved to output" : this._srShown.length ? "preview only: not saved yet" : "";
        }
        return r;
      };
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = created?.apply(this, arguments);
        style();
        const row = el("div", "srl-root srl-row");
        row.style.flexDirection = "row";
        const btn = el("button", "srl-btn", "💾 Save to output");
        btn.title = "Copy the result shown here into ComfyUI/output under filename_prefix (manual mode).";
        btn.disabled = true;
        const say = el("span", "srl-grow");
        say.style.cssText = "opacity:.7;font-size:11px";
        row.append(btn, say);
        btn.onclick = async () => {
          btn.disabled = true;
          try {
            const res = await api.fetchApi("/scene_ref/keep", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ files: this._srShown || [], prefix: widget(this, "filename_prefix")?.value || "" }),
            });
            const j = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(j.error || `failed (${res.status})`);
            say.textContent = `saved: ${j.saved.join(", ")}`;
          } catch (e) {
            say.textContent = `⚠ ${e.message}`;
            btn.disabled = false;
          }
        };
        this._srSaveBtn = btn;
        this._srSaveSay = say;
        const w = this.addDOMWidget("scene_ref_save", "scene_ref_save", row, { serialize: false, getMinHeight: () => 28 });
        w.serialize = false;
        const sync = () => {
          const manual = widget(this, "save_mode")?.value === "manual";
          row.style.display = manual ? "" : "none";
        };
        hookCallback(this, "save_mode", sync);
        sync();
        this.size = [Math.max(this.size?.[0] || 0, 360), Math.max(this.size?.[1] || 0, 320)];
        return r;
      };
    }
  },
});

