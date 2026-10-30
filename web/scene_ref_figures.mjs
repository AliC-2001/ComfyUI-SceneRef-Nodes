// SceneRef - placeable mannequins ("figures").
//
// A figure is an artist's mannequin built from real body proportions (fractions
// of the height, from standard anthropometric tables), posed through joint
// angles. It lives in the scene's world space, so changing the camera or the
// saved shot never moves it. Its joints give exact OpenPose keypoints and the
// head box for the face pass; its colour tells the image model which figure is
// which ("the red mannequin").
//
// Local frame of a figure: +Y up, +Z forward (where the face looks), +X the
// figure's own left. Limbs hang along -Y from their joint. Joint angles are in
// degrees, Euler XYZ; with that, for a hanging limb -X swings it forward, and for
// the spine / neck +X leans forward.

export const PALETTE = [
  { name: "red", hex: 0xd8453a },
  { name: "blue", hex: 0x3a6fd8 },
  { name: "green", hex: 0x3aa85a },
  { name: "yellow", hex: 0xe0b62e },
  { name: "purple", hex: 0x9a4fd0 },
  { name: "orange", hex: 0xe07a2e },
  { name: "cyan", hex: 0x2ec0c8 },
  { name: "pink", hex: 0xe05aa0 },
];
export const MAX_FIGURES = PALETTE.length; // the figure map packs one bit per figure in a byte

// anchor: which part of the body sits on the clicked surface.
//   "lowest" - the lowest point of the posed body (feet, knees, back, side)
//   "seat"   - under the pelvis (chairs, sofas, beds edges)
export const POSES = {
  stand: { label: "Stand", anchor: "lowest", joints: {
    l_shoulder: [0, 0, 6], r_shoulder: [0, 0, -6], l_elbow: [-8, 0, 0], r_elbow: [-8, 0, 0] } },
  walk: { label: "Walk", anchor: "lowest", joints: {
    l_hip: [-22, 0, 0], r_hip: [14, 0, 0], l_knee: [8, 0, 0], r_knee: [28, 0, 0], r_ankle: [-10, 0, 0],
    l_shoulder: [16, 0, 6], r_shoulder: [-18, 0, -6], l_elbow: [-12, 0, 0], r_elbow: [-22, 0, 0] } },
  sit: { label: "Sit (chair, sofa)", anchor: "seat", joints: {
    l_hip: [-90, 0, 3], r_hip: [-90, 0, -3], l_knee: [88, 0, 0], r_knee: [88, 0, 0],
    spine: [4, 0, 0], l_shoulder: [-18, 0, 8], r_shoulder: [-18, 0, -8], l_elbow: [-55, 0, 0], r_elbow: [-55, 0, 0] } },
  sit_back: { label: "Sit, leaning back", anchor: "seat", joints: {
    l_hip: [-72, 0, 6], r_hip: [-72, 0, -6], l_knee: [70, 0, 0], r_knee: [76, 0, 0],
    spine: [-14, 0, 0], chest: [-6, 0, 0], neck: [10, 0, 0],
    l_shoulder: [8, 0, 22], r_shoulder: [8, 0, -22], l_elbow: [-30, 0, 0], r_elbow: [-30, 0, 0] } },
  sit_floor: { label: "Sit on the floor", anchor: "lowest", joints: {
    l_hip: [-105, 0, 8], r_hip: [-105, 0, -8], l_knee: [115, 0, 0], r_knee: [115, 0, 0],
    spine: [14, 0, 0], l_shoulder: [-48, 0, 6], r_shoulder: [-48, 0, -6], l_elbow: [-38, 0, 0], r_elbow: [-38, 0, 0] } },
  lie_back: { label: "Lie on the back", anchor: "lowest", joints: {
    hips: [-90, 0, 0], l_shoulder: [0, 0, 10], r_shoulder: [0, 0, -10], l_hip: [0, 0, 4], r_hip: [0, 0, -4],
    l_ankle: [30, 0, 0], r_ankle: [30, 0, 0] } },
  lie_side: { label: "Lie on the side (sleep)", anchor: "lowest", joints: {
    hips: [0, 0, 90], l_hip: [-38, 0, 0], r_hip: [-28, 0, 0], l_knee: [70, 0, 0], r_knee: [55, 0, 0],
    spine: [10, 0, 0], neck: [10, 0, 0], l_shoulder: [-70, 0, 0], r_shoulder: [-80, 0, 0], l_elbow: [-60, 0, 0], r_elbow: [-80, 0, 0] } },
  kneel: { label: "Kneel", anchor: "lowest", joints: {
    l_knee: [90, 0, 0], r_knee: [90, 0, 0], l_ankle: [40, 0, 0], r_ankle: [40, 0, 0],
    l_shoulder: [-10, 0, 8], r_shoulder: [-10, 0, -8], l_elbow: [-30, 0, 0], r_elbow: [-30, 0, 0] } },
  crouch: { label: "Crouch", anchor: "lowest", joints: {
    l_hip: [-115, 0, 12], r_hip: [-115, 0, -12], l_knee: [135, 0, 0], r_knee: [135, 0, 0], l_ankle: [-20, 0, 0], r_ankle: [-20, 0, 0],
    spine: [28, 0, 0], neck: [-20, 0, 0], l_shoulder: [-40, 0, 10], r_shoulder: [-40, 0, -10], l_elbow: [-40, 0, 0], r_elbow: [-40, 0, 0] } },
  reach: { label: "Reach forward", anchor: "lowest", joints: {
    spine: [8, 0, 0], l_shoulder: [-75, 0, 8], r_shoulder: [-80, 0, -8], l_elbow: [-12, 0, 0], r_elbow: [-10, 0, 0] } },
  hold: { label: "Hold / pour (hands front)", anchor: "lowest", joints: {
    spine: [4, 0, 0], l_shoulder: [-30, 0, 10], r_shoulder: [-45, 0, -12], l_elbow: [-85, 0, 0], r_elbow: [-70, 0, 0] } },
  arm_up: { label: "Raise an arm", anchor: "lowest", joints: {
    r_shoulder: [0, 0, -160], r_elbow: [-15, 0, 0], l_shoulder: [0, 0, 6], l_elbow: [-8, 0, 0] } },
  lean: { label: "Lean on (counter, table)", anchor: "lowest", joints: {
    spine: [22, 0, 0], chest: [8, 0, 0], neck: [-18, 0, 0], l_hip: [-6, 0, 3], r_hip: [4, 0, -3],
    l_shoulder: [-55, 0, 14], r_shoulder: [-55, 0, -14], l_elbow: [-35, 0, 0], r_elbow: [-35, 0, 0] } },
};

// The preset behind a spec; for pose "custom" (from the pose library) its stored label and anchor.
export function poseOf(spec) {
  if (spec?.pose === "custom") {
    return { label: spec.pose_label || "Custom pose", anchor: spec.anchor === "seat" ? "seat" : "lowest", joints: {} };
  }
  return POSES[spec?.pose] || POSES.stand;
}

// Joints that can be turned by hand, in the order the bone list shows them.
export const JOINTS = ["hips", "spine", "chest", "neck", "head", "l_shoulder", "l_elbow", "l_wrist",
  "r_shoulder", "r_elbow", "r_wrist", "l_hip", "l_knee", "l_ankle", "r_hip", "r_knee", "r_ankle"];
export const JOINT_LABELS = {
  hips: "Hips", spine: "Waist", chest: "Chest", neck: "Neck", head: "Head",
  l_shoulder: "Left upper arm", l_elbow: "Left forearm", l_wrist: "Left hand",
  r_shoulder: "Right upper arm", r_elbow: "Right forearm", r_wrist: "Right hand",
  l_hip: "Left thigh", l_knee: "Left shin", l_ankle: "Left foot",
  r_hip: "Right thigh", r_knee: "Right shin", r_ankle: "Right foot",
};
// Left joint <-> right joint, for mirroring.
export const MIRROR = Object.fromEntries(JOINTS.map((j) => [j, j.startsWith("l_") ? "r_" + j.slice(2) : j.startsWith("r_") ? "l_" + j.slice(2) : j]));

export const BUILDS = { slim: 0.86, average: 1, curvy: 1.12, heavy: 1.28 };
export const SEXES = { female: "female", male: "male" };

// Proportions as fractions of the height (adult, standing).
function proportions(sex, build) {
  const f = sex === "female";
  const b = BUILDS[build] || 1;
  const curvy = build === "curvy";
  return {
    hipY: 0.53, ankleY: 0.039, thigh: 0.245, shin: 0.246, foot: 0.152,
    spine: 0.08, chest: 0.21, neck: 0.045, headH: 0.13,
    upperArm: f ? 0.182 : 0.188, forearm: f ? 0.142 : 0.148, hand: 0.105,
    shoulderW: (f ? 0.232 : 0.258) * (0.94 + 0.06 * b),
    hipW: (f ? 0.205 : 0.188) * b * (curvy ? 1.08 : 1),
    rThigh: 0.052 * b * (curvy ? 1.08 : 1), rShin: 0.034 * b, rUpper: 0.03 * b, rFore: 0.024 * b,
    torsoD: 0.105 * b, chestD: (f ? 0.115 : 0.12) * b * (curvy ? 1.06 : 1),
    waistW: (f ? 0.15 : 0.165) * b,
  };
}

export class Figure {
  // spec: { id, color (palette index), height_cm, sex, build, pose, at:[x,y,z], yaw, pitch, roll,
  //         head_turn, lean, bones: {joint: [x, y, z] degrees} (hand-set joints, in place of the
  //         preset's), and for pose "custom" (from the pose library): pose_label, anchor }
  constructor(THREE, spec) {
    this.THREE = THREE;
    this.root = new THREE.Group();
    this.root.userData.srFigure = true;
    this.spec = {};
    this.set(spec);
  }

  set(patch) {
    const old = this.spec;
    const s = { ...old, ...patch };
    s.height_cm = Math.min(230, Math.max(60, +s.height_cm || 170));
    s.sex = s.sex === "male" ? "male" : "female";
    s.build = BUILDS[s.build] ? s.build : "average";
    s.pose = POSES[s.pose] || s.pose === "custom" ? s.pose : "stand";
    s.yaw = +s.yaw || 0;
    s.pitch = +s.pitch || 0;
    s.roll = +s.roll || 0;
    // Another preset picked: its own joints, not the hand-set ones of the last pose.
    if (patch.pose !== undefined && patch.pose !== old.pose && patch.bones === undefined) s.bones = {};
    const bones = {};
    for (const [k, v] of Object.entries(s.bones || {})) {
      if (JOINTS.includes(k) && Array.isArray(v) && v.length === 3) bones[k] = v.map((x) => Math.round((+x || 0) * 10) / 10);
    }
    s.bones = bones;
    if (s.pose === "custom") {
      s.anchor = s.anchor === "seat" ? "seat" : "lowest";
      s.pose_label = String(s.pose_label || "Custom pose").slice(0, 80);
    } else {
      delete s.anchor;
      delete s.pose_label;
    }
    s.head_turn = +s.head_turn || 0;
    s.lean = +s.lean || 0;
    s.color = ((+s.color || 0) % PALETTE.length + PALETTE.length) % PALETTE.length;
    s.at = Array.isArray(s.at) && s.at.length === 3 ? s.at.map(Number) : [0, 0, 0];
    this.spec = s;
    const rebuild = !this.joints || old.height_cm !== s.height_cm || old.sex !== s.sex || old.build !== s.build;
    if (rebuild) this._build();
    else if (old.color !== s.color) this.material.color.setHex(PALETTE[s.color].hex);
    this._pose();
    this.place();
    return this;
  }

  get colorName() { return PALETTE[this.spec.color].name; }

  get kind() { return "mannequin"; }

  get poseLabel() {
    const p = poseOf(this.spec);
    return this.spec.pose !== "custom" && Object.keys(this.spec.bones).length ? p.label + " (adjusted)" : p.label;
  }

  get anchor() { return poseOf(this.spec).anchor; }

  // What the body is doing, from where its joints are (any pose, any tilt).
  posture() {
    // Heights as fractions of the figure's height (standing: hips 0.53, knees 0.28).
    const T = this.THREE;
    const H = this.H;
    const floor = new T.Box3().setFromObject(this.body).min.y;
    const w = (n) => this.joints[n].getWorldPosition(new T.Vector3());
    const hips = w("hips");
    const hip = (hips.y - floor) / H;
    const knee = (Math.min(w("l_knee").y, w("r_knee").y) - floor) / H;
    const torso = w("neck").sub(hips).normalize();
    if (Math.abs(torso.y) < 0.5 && hip < 0.3) return "lying down";
    if (this.anchor === "seat") return "sitting";
    // Both feet out in front of the hips: sitting on the ground, not crouching (in a lunge one
    // foot is behind).
    const fwd = new T.Vector3(0, 0, 1).applyQuaternion(this.joints.hips.getWorldQuaternion(new T.Quaternion())).setY(0).normalize();
    const out = (n) => w(n).sub(hips).dot(fwd) / H;
    const feet = Math.min(out("l_ankle"), out("r_ankle"));
    if (knee < 0.12) return hip < 0.2 ? "sitting on the floor" : "kneeling";
    if (hip < 0.3) return feet > 0.17 ? "sitting on the floor" : "crouching";
    return hip < 0.42 ? "crouching" : "standing";
  }

  // The pose in words for the prompt. Library names are kept ("standing, waving"); file
  // names that read like codes ("MP G8F 02", "Pose458") give way to the posture.
  get promptLabel() {
    const s = this.spec;
    if (s.pose !== "custom") return poseOf(s).label;
    const posture = this.posture();
    const label = String(s.pose_label || "").replace(/\s+\d+$/, "").trim();
    if (!label || /\d|_|\bG\d|\bpose\b/i.test(label)) return posture;
    const first = posture.split(" ")[0].replace(/ing$/, "").replace(/^ly$/, "l[iy]");
    return new RegExp("\\b" + first, "i").test(label) ? label : `${posture}, ${label.toLowerCase()}`;
  }

  _build() {
    const THREE = this.THREE;
    const s = this.spec;
    const H = s.height_cm / 100;
    const P = proportions(s.sex, s.build);
    this.P = P;
    this.H = H;
    for (const c of [...this.root.children]) this.root.remove(c);
    this._disposeGeo();
    this.material?.dispose();
    this.material = new THREE.MeshStandardMaterial({ color: PALETTE[s.color].hex, roughness: 0.55, metalness: 0.0 });
    this.material.userData.srFigure = true;
    this.faceMat?.dispose();
    this.faceMat = new THREE.MeshStandardMaterial({ color: 0xf2efe8, roughness: 0.5 });
    this.faceMat.userData.srFigure = true;
    this.material.userData.srFigureOwner = this;
    this.faceMat.userData.srFigureOwner = this;
    this._geos = [];

    const J = {};
    const joint = (name, parent, x, y, z) => {
      const o = new THREE.Object3D();
      o.name = name;
      o.position.set(x * H, y * H, z * H);
      (parent || this.body).add(o);
      J[name] = o;
      return o;
    };
    const mesh = (parent, geo, x, y, z, mat = this.material) => {
      this._geos.push(geo);
      const m = new THREE.Mesh(geo, mat);
      m.position.set(x * H, y * H, z * H);
      m.userData.srFigure = true;
      parent.add(m);
      return m;
    };
    // A limb from its joint down to the next joint, as a capsule.
    const limb = (parent, len, r) => mesh(parent, new THREE.CapsuleGeometry(r * H, Math.max(0.001, (len - 2 * r) * H), 6, 12), 0, -len / 2, 0);
    const ellipsoid = (parent, rx, ry, rz, x, y, z, mat) => {
      const m = mesh(parent, new THREE.SphereGeometry(1, 20, 14), x, y, z, mat);
      m.scale.set(rx * H, ry * H, rz * H);
      return m;
    };
    const ball = (parent, r) => mesh(parent, new THREE.SphereGeometry(r * H, 12, 8), 0, 0, 0);

    this.body = new THREE.Group();
    this.root.add(this.body);
    // Torso chain.
    const hips = joint("hips", this.body, 0, P.hipY, 0);
    ellipsoid(hips, P.hipW * 0.55, 0.075, P.torsoD * 0.55, 0, -0.01, -0.005);
    const spine = joint("spine", hips, 0, P.spine, 0);
    ellipsoid(spine, P.waistW * 0.5, 0.07, P.torsoD * 0.5, 0, 0.0, 0);
    const chest = joint("chest", spine, 0, 0.06, 0);
    ellipsoid(chest, P.shoulderW * 0.42, P.chest * 0.55, P.chestD * 0.5, 0, P.chest * 0.42, 0.0);
    if (s.sex === "female") {
      for (const sx of [1, -1]) ellipsoid(chest, 0.032 * (BUILDS[s.build]), 0.03, 0.03 * BUILDS[s.build], sx * 0.045, P.chest * 0.42, P.chestD * 0.42);
    }
    const neckY = 0.818 - P.hipY - P.spine - 0.06;
    const neck = joint("neck", chest, 0, neckY, 0);
    mesh(neck, new THREE.CapsuleGeometry(0.03 * H, 0.03 * H, 4, 10), 0, 0.02, 0);
    const head = joint("head", neck, 0, P.neck, 0);
    ellipsoid(head, 0.052, P.headH * 0.5, 0.06, 0, P.headH * 0.5, 0.004);
    // Face plate and nose: which way the head looks is obvious from any angle.
    const face = ellipsoid(head, 0.036, 0.045, 0.012, 0, P.headH * 0.47, 0.052, this.faceMat);
    face.userData.srFace = true;
    const nose = mesh(head, new THREE.ConeGeometry(0.009 * H, 0.025 * H, 8), 0, P.headH * 0.45, 0.068, this.faceMat);
    nose.rotation.x = Math.PI / 2;
    // Arms.
    for (const [side, sx] of [["l", 1], ["r", -1]]) {
      const sh = joint(side + "_shoulder", chest, sx * (P.shoulderW / 2 - 0.022), neckY - 0.022, 0);
      ball(sh, P.rUpper * 1.15);
      limb(sh, P.upperArm, P.rUpper);
      const el = joint(side + "_elbow", sh, 0, -P.upperArm, 0);
      limb(el, P.forearm, P.rFore);
      const wr = joint(side + "_wrist", el, 0, -P.forearm, 0);
      const hand = ellipsoid(wr, 0.022, P.hand * 0.45, 0.011, 0, -P.hand * 0.42, 0.004);
      hand.userData.srHand = true;
    }
    // Legs.
    for (const [side, sx] of [["l", 1], ["r", -1]]) {
      const hp = joint(side + "_hip", hips, sx * P.hipW * 0.24, 0, 0);
      limb(hp, P.thigh, P.rThigh);
      const kn = joint(side + "_knee", hp, 0, -P.thigh, 0);
      limb(kn, P.shin, P.rShin);
      const an = joint(side + "_ankle", kn, 0, -P.shin, 0);
      const foot = new THREE.BoxGeometry(0.055 * H, P.ankleY * H, P.foot * H);
      mesh(an, foot, 0, -P.ankleY / 2, P.foot * 0.32);
    }
    // Keypoint markers (OpenPose / COCO-18), right = the figure's own right (-X).
    const kp = (name, parent, x, y, z) => {
      const o = new THREE.Object3D();
      o.name = "kp_" + name;
      o.position.set(x * H, y * H, z * H);
      parent.add(o);
      return o;
    };
    const hy = P.headH * 0.55;
    this.kp = {
      nose: kp("nose", head, 0, P.headH * 0.45, 0.075),
      neck: kp("neck", neck, 0, 0, 0),
      r_shoulder: kp("r_shoulder", J.r_shoulder, 0, 0, 0), r_elbow: kp("r_elbow", J.r_elbow, 0, 0, 0), r_wrist: kp("r_wrist", J.r_wrist, 0, 0, 0),
      l_shoulder: kp("l_shoulder", J.l_shoulder, 0, 0, 0), l_elbow: kp("l_elbow", J.l_elbow, 0, 0, 0), l_wrist: kp("l_wrist", J.l_wrist, 0, 0, 0),
      r_hip: kp("r_hip", J.r_hip, 0, 0, 0), r_knee: kp("r_knee", J.r_knee, 0, 0, 0), r_ankle: kp("r_ankle", J.r_ankle, 0, 0, 0),
      l_hip: kp("l_hip", J.l_hip, 0, 0, 0), l_knee: kp("l_knee", J.l_knee, 0, 0, 0), l_ankle: kp("l_ankle", J.l_ankle, 0, 0, 0),
      r_eye: kp("r_eye", head, -0.02, hy, 0.058), l_eye: kp("l_eye", head, 0.02, hy, 0.058),
      r_ear: kp("r_ear", head, -0.052, hy * 0.92, 0.0), l_ear: kp("l_ear", head, 0.052, hy * 0.92, 0.0),
    };
    this.headCenter = kp("head_center", head, 0, P.headH * 0.5, 0.004);
    this.joints = J;
  }

  _pose() {
    const THREE = this.THREE;
    const s = this.spec;
    const pose = poseOf(s).joints;
    for (const [name, o] of Object.entries(this.joints)) {
      const a = s.bones[name] || pose[name] || [0, 0, 0];
      o.rotation.set(a[0] * Math.PI / 180, a[1] * Math.PI / 180, a[2] * Math.PI / 180, "XYZ");
    }
    // User tweaks on top of the preset.
    this.joints.neck.rotation.y += (s.head_turn * 0.4) * Math.PI / 180;
    this.joints.head.rotation.y += (s.head_turn * 0.6) * Math.PI / 180;
    this.joints.spine.rotation.x += (s.lean * 0.6) * Math.PI / 180;
    this.joints.chest.rotation.x += (s.lean * 0.4) * Math.PI / 180;
    this.body.position.set(0, 0, 0);
    this.root.rotation.set(s.pitch * Math.PI / 180, s.yaw * Math.PI / 180, s.roll * Math.PI / 180, "YXZ");
    this.root.position.set(0, 0, 0);
    this.root.updateMatrixWorld(true);
  }

  // What the head-turn and lean sliders add to a joint; hand-set angles are stored without it.
  tweakOf(name) {
    const s = this.spec;
    const d = { neck: [0, s.head_turn * 0.4, 0], head: [0, s.head_turn * 0.6, 0], spine: [s.lean * 0.6, 0, 0], chest: [s.lean * 0.4, 0, 0] };
    return d[name] || [0, 0, 0];
  }

  // A joint's angles as shown now (degrees, Euler XYZ, without the slider tweaks).
  jointAngles(name) {
    const e = new this.THREE.Euler().setFromQuaternion(this.joints[name].quaternion, "XYZ");
    const t = this.tweakOf(name);
    return [e.x, e.y, e.z].map((r, i) => Math.round((r * 180 / Math.PI - t[i]) * 10) / 10);
  }

  allAngles() {
    const out = {};
    for (const name of JOINTS) out[name] = this.jointAngles(name);
    return out;
  }

  // Joint angles (for spec.bones) that give this mannequin the pose of a set of points:
  // figure-local, +Y up, +Z forward, +X the figure's own left, any scale and origin
  // (scene_ref/poses.py makes them from Daz presets, OpenPose, BVH and the library). Only
  // directions are used, so the mannequin keeps its own proportions. Joints are aimed from
  // the hips outwards; elbows and knees bend on their hinge only.
  fitPose(points) {
    const THREE = this.THREE;
    const V = (k) => (Array.isArray(points[k]) ? new THREE.Vector3().fromArray(points[k]) : null);
    const J = this.joints;
    const keepQ = this.root.quaternion.clone();
    this.root.quaternion.identity();
    for (const o of Object.values(J)) o.rotation.set(0, 0, 0);
    this.root.updateMatrixWorld(true);
    const wq = (o) => o.getWorldQuaternion(new THREE.Quaternion());
    const setWorld = (name, q) => {
      const o = J[name];
      o.quaternion.copy(wq(o.parent).invert().multiply(q));
      o.updateMatrixWorld(true);
    };
    // The rotation whose Y axis is `y` and whose X axis is as close to `x` as it can be.
    const frame = (x, y) => {
      const Y = y.clone().normalize();
      let X = x.clone().addScaledVector(Y, -x.dot(Y));
      if (X.lengthSq() < 1e-8) X = new THREE.Vector3(1, 0, 0).addScaledVector(Y, -Y.x);
      X.normalize();
      const Z = new THREE.Vector3().crossVectors(X, Y);
      return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(X, Y, Z));
    };
    // The same with the X axis exact (a hip or shoulder line), Y as close to `y` as it can be.
    const frameX = (x, y) => {
      const X = x.clone().normalize();
      const Y = y.clone().addScaledVector(X, -y.dot(X));
      return frame(X, Y.lengthSq() > 1e-8 ? Y : new THREE.Vector3(0, 1, 0));
    };
    const axis = (q, x, y, z) => new THREE.Vector3(x, y, z).applyQuaternion(q);
    const mid = (a, b) => (a && b ? a.clone().add(b).multiplyScalar(0.5) : a || b);

    const lh = V("l_hip"), rh = V("r_hip"), ls = V("l_shoulder"), rs = V("r_shoulder");
    const pelvis = V("pelvis") || mid(lh, rh);
    const neck = V("neck") || mid(ls, rs);
    const waist = V("spine");
    // Torso: the pelvis from the hip line and the lower back, the chest from the shoulder
    // line; the twist and bend between them is shared by the waist and the chest.
    if (pelvis && neck) {
      const side = lh && rh ? lh.clone().sub(rh) : ls && rs ? ls.clone().sub(rs) : new THREE.Vector3(1, 0, 0);
      const qh = frameX(side, (waist || neck).clone().sub(pelvis));
      setWorld("hips", qh);
      const qc = frameX(ls && rs ? ls.clone().sub(rs) : side, neck.clone().sub(waist || pelvis));
      J.spine.quaternion.copy(new THREE.Quaternion().slerp(qh.clone().invert().multiply(qc), 0.5));
      J.spine.updateMatrixWorld(true);
      setWorld("chest", qc);
    }
    // Neck towards the head; the head up its crown, facing the nose.
    const head = V("head"), top = V("head_top"), nose = V("nose");
    if (neck && (head || top)) {
      setWorld("neck", frame(axis(wq(J.chest), 1, 0, 0), (head || top).clone().sub(neck)));
      const hup = top && head ? top.clone().sub(head) : (head || top).clone().sub(neck);
      const face = nose || mid(V("l_eye"), V("r_eye"));
      let hx = axis(wq(J.neck), 1, 0, 0);
      if (face && head) hx = new THREE.Vector3().crossVectors(hup.clone().normalize(), face.clone().sub(head)); // up x forward = left
      setWorld("head", frame(hx, hup));
    }
    // Limbs: the upper bone aimed at the middle joint, its hinge across the bend. Elbows
    // fold to the front of the upper arm (-X angle), knees to the back of the thigh (+X).
    const limb = (s, a0, b0, c0, d0, arm) => {
      const a = V(s + a0), b = V(s + b0), c = V(s + c0), d = V(s + d0);
      if (!a || !b) return;
      const u = b.clone().sub(a).normalize();
      const bend = c ? c.clone().sub(b).normalize() : null;
      const perp = bend ? bend.clone().addScaledVector(u, -bend.dot(u)) : new THREE.Vector3();
      let z, angle = 0;
      if (perp.length() > 0.06) {
        z = perp.clone().normalize().multiplyScalar(arm ? 1 : -1);
        angle = Math.atan2(perp.length(), bend.dot(u)) * (arm ? -1 : 1);
      } else {
        // Straight: keep the hinge facing the body's front.
        const body = wq(arm ? J.chest : J.hips);
        for (const f of [axis(body, 0, 0, 1), axis(body, 0, -1, 0)]) {
          z = f.addScaledVector(u, -f.dot(u));
          if (z.lengthSq() > 1e-4) break;
        }
        z.normalize();
      }
      const y = u.clone().negate();
      const upper = s + (arm ? "_shoulder" : "_hip"), hinge = s + (arm ? "_elbow" : "_knee"), end = s + (arm ? "_wrist" : "_ankle");
      setWorld(upper, frame(new THREE.Vector3().crossVectors(y, z), y));
      J[hinge].rotation.set(angle, 0, 0);
      J[hinge].updateMatrixWorld(true);
      // Hand / foot towards the fingers / toes, the least turn from its rest direction.
      if (c && d) {
        const local = d.clone().sub(c).applyQuaternion(wq(J[hinge]).invert()).normalize();
        const rest = arm ? new THREE.Vector3(0, -1, 0) : new THREE.Vector3(0, -0.4, 1).normalize();
        J[end].quaternion.setFromUnitVectors(rest, local);
        J[end].updateMatrixWorld(true);
      }
    };
    for (const s of ["l", "r"]) {
      limb(s, "_shoulder", "_elbow", "_wrist", "_hand", true);
      limb(s, "_hip", "_knee", "_ankle", "_toe", false);
    }
    const out = {};
    for (const name of JOINTS) {
      const e = new THREE.Euler().setFromQuaternion(J[name].quaternion, "XYZ");
      out[name] = [e.x, e.y, e.z].map((r) => Math.round(r * 1800 / Math.PI) / 10);
    }
    this.root.quaternion.copy(keepQ);
    this._pose();
    this.place();
    return out;
  }

  // Put the posed body's anchor on the stored surface point `at`.
  place() {
    const THREE = this.THREE;
    const s = this.spec;
    this.root.position.set(0, 0, 0);
    this.root.updateMatrixWorld(true);
    let ax, ay, az;
    if (poseOf(s).anchor === "seat") {
      const p = this.joints.hips.getWorldPosition(new THREE.Vector3());
      ax = p.x; az = p.z; ay = p.y - 0.06 * this.H; // under the buttocks
    } else {
      const box = new THREE.Box3().setFromObject(this.body);
      const c = box.getCenter(new THREE.Vector3());
      ax = c.x; az = c.z; ay = box.min.y;
    }
    this.root.position.set(s.at[0] - ax, s.at[1] - ay, s.at[2] - az);
    this.root.updateMatrixWorld(true);
  }

  meshes() {
    const out = [];
    this.root.traverse((o) => { if (o.isMesh) out.push(o); });
    return out;
  }

  keypoints() {
    const THREE = this.THREE;
    const out = {};
    for (const [k, o] of Object.entries(this.kp)) out[k] = o.getWorldPosition(new THREE.Vector3());
    return out;
  }

  // World-space head sphere (centre, radius) for the face box.
  headSphere() {
    return { center: this.headCenter.getWorldPosition(new this.THREE.Vector3()), radius: this.P.headH * 0.62 * this.H };
  }

  // Unit vector the body faces (local +Z of the chest), world space.
  facing() {
    const THREE = this.THREE;
    const q = this.joints.chest.getWorldQuaternion(new THREE.Quaternion());
    return new THREE.Vector3(0, 0, 1).applyQuaternion(q);
  }

  _disposeGeo() {
    for (const g of this._geos || []) g.dispose();
    this._geos = [];
  }

  dispose() {
    this._disposeGeo();
    this.material?.dispose();
    this.faceMat?.dispose();
    this.root.parent?.remove(this.root);
  }
}

// ── Daz figures ───────────────────────────────────────────────────────────────
// A Genesis character posed in Daz Studio, exported as its own mesh (SRFIG_<n>)
// with its joint positions. Same interface as Figure, but it stays where Daz put
// it: only its colour tag and look (colour / textured) change.
export class DazFigure {
  // mesh: the SRFIG_<n> Object3D from the converted scene; data: info.figures[n]
  constructor(THREE, mesh, data, spec, paletteMat) {
    this.THREE = THREE;
    this.kind = "daz";
    this.root = new THREE.Group();
    this.root.userData.srFigure = true;
    mesh.updateMatrixWorld(true);
    const m = mesh.matrixWorld.clone();
    mesh.parent?.remove(mesh);
    mesh.matrix.copy(m);
    mesh.matrix.decompose(mesh.position, mesh.quaternion, mesh.scale);
    this.root.add(mesh);
    this.mesh = mesh;
    this.data = data || {};
    this.label = this.data.label || mesh.name;
    this._paletteMat = paletteMat;
    this.orig = new Map();
    mesh.traverse((o) => { if (o.isMesh) this.orig.set(o, o.material); });
    const box = new THREE.Box3().setFromObject(mesh);
    this.H = Math.max(0.3, box.max.y - box.min.y);
    // Joint markers in world space (the scene root has no transform).
    const J = this.data.joints || {};
    const v = (k) => (Array.isArray(J[k]) ? new THREE.Vector3().fromArray(J[k]) : null);
    const fwd = new THREE.Vector3().fromArray(J.forward || [0, 0, 1]).setY(0).normalize();
    if (!fwd.lengthSq()) fwd.set(0, 0, 1);
    this.fwd = fwd;
    const up = new THREE.Vector3(0, 1, 0);
    const le = v("l_eye"), re = v("r_eye"), head = v("head");
    const eyes = le && re ? le.clone().add(re).multiplyScalar(0.5) : (head ? head.clone().add(new THREE.Vector3(0, 0.09, 0)) : null);
    const pts = {};
    for (const k of ["neck", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
      "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]) pts[k] = v(k);
    if (eyes) {
      const side = le && re ? le.clone().sub(re) : new THREE.Vector3().crossVectors(up, fwd).multiplyScalar(0.06);
      pts.l_eye = le || eyes.clone().addScaledVector(side, 0.5);
      pts.r_eye = re || eyes.clone().addScaledVector(side, -0.5);
      pts.nose = eyes.clone().addScaledVector(fwd, 0.045).addScaledVector(up, -0.035);
      pts.l_ear = pts.l_eye.clone().addScaledVector(side, 0.9).addScaledVector(fwd, -0.07);
      pts.r_ear = pts.r_eye.clone().addScaledVector(side, -0.9).addScaledVector(fwd, -0.07);
      this._head = { center: eyes.clone().addScaledVector(fwd, -0.03).addScaledVector(up, 0.02), radius: 0.13 };
    } else {
      this._head = { center: box.getCenter(new THREE.Vector3()).setY(box.max.y - 0.1), radius: 0.13 };
    }
    this._kp = pts;
    // Standing height from the bone chain, so a seated or crouching figure still
    // reports its real height (the box would include hair and raised arms).
    const seg = (a, b) => (a && b ? a.distanceTo(b) : 0);
    const leg = (s) => seg(v(s + "_ankle"), v(s + "_knee")) + seg(v(s + "_knee"), v(s + "_hip"));
    const hipMid = v("l_hip") && v("r_hip") ? v("l_hip").add(v("r_hip")).multiplyScalar(0.5) : null;
    const chain = Math.max(leg("l"), leg("r")) + seg(hipMid, v("neck")) + seg(v("neck"), eyes);
    if (chain > 0.8) this.H = chain + 0.115 + 0.075; // eyes to crown, ankle to sole
    const hips = new THREE.Object3D();
    hips.position.copy(v("pelvis") || box.getCenter(new THREE.Vector3()));
    this.root.add(hips);
    this.joints = { hips };
    this.spec = {};
    this.set({ kind: "daz", mesh: mesh.name, color: 0, look: "color", ...spec });
  }

  get poseLabel() { return "posed in Daz Studio"; }

  set(patch) {
    // Only the tag and look belong to a Daz figure; the pose is Daz's.
    const m = { ...this.spec, ...patch };
    const s = { kind: "daz", mesh: this.mesh.name, color: m.color, look: m.look };
    s.color = ((+s.color || 0) % PALETTE.length + PALETTE.length) % PALETTE.length;
    s.look = s.look === "textured" ? "textured" : "color";
    s.height_cm = Math.round(this.H * 100);
    this.spec = s;
    this.mesh.traverse((o) => {
      if (!o.isMesh) return;
      const orig = this.orig.get(o);
      o.material = s.look === "textured" ? this._tag(orig) : this._paletteMat(orig, PALETTE[s.color].hex, this);
    });
    // The highlight / data passes look at this one.
    this.material = this._paletteMat(this.orig.values().next().value, PALETTE[s.color].hex, this);
    return this;
  }

  _tag(m) {
    const tag = (x) => { x.userData.srFigure = true; x.userData.srFigureOwner = this; return x; };
    return Array.isArray(m) ? m.map(tag) : tag(m);
  }

  place() {}

  meshes() {
    const out = [];
    this.root.traverse((o) => { if (o.isMesh) out.push(o); });
    return out;
  }

  keypoints() {
    const THREE = this.THREE;
    const out = {};
    for (const k of COCO) out[k] = this._kp[k] ? this._kp[k].clone() : new THREE.Vector3(0, -1e6, 0);
    return out;
  }

  headSphere() { return { center: this._head.center.clone(), radius: this._head.radius }; }

  facing() { return this.fwd.clone(); }

  dispose() {
    this.root.parent?.remove(this.root);
  }
}

// ── OpenPose (COCO-18) drawing ────────────────────────────────────────────────
export const COCO = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
  "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"];
// Limb pairs and colours exactly as the OpenPose ControlNet preprocessors draw them.
const LIMBS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
  [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]];
const COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]];

// people: [[{x, y, visible} x 18], ...] in pixels.
export function drawOpenPose(ctx, w, h, people) {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  const stick = Math.max(2, Math.round(Math.min(w, h) / 160));
  for (const kps of people) {
    LIMBS.forEach(([a, b], i) => {
      const p = kps[a], q = kps[b];
      if (!p?.visible || !q?.visible) return;
      const [r, g, bl] = COLORS[i];
      ctx.strokeStyle = `rgba(${r},${g},${bl},0.6)`;
      ctx.lineWidth = stick * 2;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(q.x, q.y);
      ctx.stroke();
    });
    kps.forEach((p, i) => {
      if (!p?.visible) return;
      const [r, g, b] = COLORS[i];
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, stick, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

// Body parts for "what is hidden" summaries.
export const PARTS = {
  head: ["nose", "r_eye", "l_eye", "r_ear", "l_ear"],
  torso: ["neck", "r_shoulder", "l_shoulder", "r_hip", "l_hip"],
  "left arm": ["l_shoulder", "l_elbow", "l_wrist"],
  "right arm": ["r_shoulder", "r_elbow", "r_wrist"],
  "left leg": ["l_hip", "l_knee", "l_ankle"],
  "right leg": ["r_hip", "r_knee", "r_ankle"],
};
