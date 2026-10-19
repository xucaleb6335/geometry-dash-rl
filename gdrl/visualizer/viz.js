/* GD-RL network visualizer. Replays a gdrl.tools.nn_trace trace: the game scene,
   the policy network's inputs, hidden activations and decision, frame by frame.
   Vanilla JS + canvas, no dependencies. Data: /api/traces + /api/trace?name= (the
   dashboard server) or an embedded <script id="trace-data"> (standalone export). */
"use strict";
(() => {
  /* ================================================================ helpers */
  const $ = (s, r = document) => r.querySelector(s);
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const lerp = (a, b, t) => a + (b - a) * t;
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  const fmtInt = v => (v == null || !isFinite(v) ? "-" : Math.round(v).toLocaleString("en-US"));
  const fmtS = (v, d = 2) => (v == null || !isFinite(v) ? "-" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d));
  const fmtPct = p => (p >= 0.9995 ? ">99.9%" : p <= 0.0005 ? "<0.1%" : (p * 100).toFixed(p > 0.995 || p < 0.005 ? 1 : 0) + "%");
  const fmtTime = s => (s < 60 ? s.toFixed(1) + " s" : Math.floor(s / 60) + ":" + (s % 60).toFixed(1).padStart(4, "0"));
  const reduceMotion = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ================================================================ constants */
  const MODE_NAMES = ["Cube", "Ship", "Ball", "UFO", "Wave", "Robot", "Spider", "Swing"];
  // GD's own portal colours, stepped so the five modes these levels use stay
  // apart under colour-vision deficiency (checked all-pairs on the dark panel).
  const MODE_DARK = ["#9bff6a", "#c44dff", "#ff5c7a", "#ffa630", "#2fc6ff", "#e9e9ff", "#f58cff", "#ffe14d"];
  const MODE_LIGHT = ["#2f9e1f", "#9b2fd6", "#d6284a", "#c96a00", "#0088c8", "#6b6f99", "#b03ab8", "#a88600"];
  const SPEEDS = [[251.16, "0.5×", "#ffa630"], [311.58, "1×", "#2fc6ff"], [387.42, "2×", "#9bff6a"], [468.0, "3×", "#ff4fd8"], [576.0, "4×", "#ff5c7a"]];
  const K = { SOLID: 1, HAZARD: 2, ORB: 3, PAD: 4, PORTAL: 5, OTHER: 255 };
  const ST = { BLOCK: 1, SLOPE: 2, SPIKE: 3, SAW: 4 };
  const FL = { DEAD: 1, COMPLETED: 2, ON_GROUND: 4, UPSIDE_DOWN: 8, HOLDING: 16, MINI: 32, DUAL: 64 };
  const KIND_COLOR = { solid: "#7f93d8", hazard: "#ff3b5c", orb: "#ffd400", pad: "#ff4fd8", portal: "#2fc6ff", other: "#9a9bc4" };
  const KIND_KEY = { solid: "#6a7bb8", hazard: "#d62f4f", orb: "#c9a400", pad: "#c63aa8", portal: "#2b8fc4", other: "#999" }; // the same, on white
  const ORB_COLOR = { 10: "#ffd400", 11: "#ff4fd8", 12: "#2f8bff", 13: "#3dff7a", 14: "#30303a", 15: "#ff3b5c", 16: "#3dff7a", 17: "#ff4fd8", 18: "#b04dff", 19: "#00e5ff", 20: "#cfcfe8" };
  const PAD_COLOR = { 30: "#ffd400", 31: "#ff4fd8", 32: "#2f8bff", 33: "#ff3b5c", 34: "#b04dff" };
  const DPS = 60; // decisions per second at 1x (240 ticks/s, 4 ticks per decision)
  const SPEED_STEPS = [0.25, 0.5, 1, 2, 4, 8];

  /* ================================================================ state */
  const S = {
    T: null, name: null, list: [], embedded: false,
    f: 0, ff: 0, playing: false, speed: 1, dragging: false,
    edgeMode: "path", tg: { slots: true, terrain: true, fov: true }, zoom: 1,
    hover: null, pin: null, tlHover: null, sceneHover: null,
    dirty: true, lastDraw: 0,
  };
  const PREF_KEY = "gdrl-viz";
  try {
    const p = JSON.parse(localStorage.getItem(PREF_KEY) || "{}");
    if (SPEED_STEPS.includes(p.speed)) S.speed = p.speed;
    if (p.edgeMode === "top" || p.edgeMode === "path") S.edgeMode = p.edgeMode;
    if (p.tg) Object.assign(S.tg, p.tg);
    if (p.zoom > 0.4 && p.zoom < 3) S.zoom = p.zoom;
    S.lastTrace = p.trace;
  } catch (e) { /* storage blocked: defaults */ }
  // One-view overrides from the URL (for links and recordings), e.g.
  // ?trace=sm&overlays=all&zoom=1&edges=path&speed=1&autoplay=2
  const Q = new URLSearchParams(location.search);
  if (Q.has("overlays")) { const v = Q.get("overlays"); for (const k of Object.keys(S.tg)) S.tg[k] = v === "all" || v.split(",").includes(k); }
  if (+Q.get("zoom") > 0.4 && +Q.get("zoom") < 3) S.zoom = +Q.get("zoom");
  if (Q.get("edges") === "path" || Q.get("edges") === "top") S.edgeMode = Q.get("edges");
  if (SPEED_STEPS.includes(+Q.get("speed"))) S.speed = +Q.get("speed");
  const AUTOPLAY = Q.has("autoplay") ? Math.max(0, +Q.get("autoplay") || 0) : null; // seconds after load
  const NO_SAVE = ["overlays", "zoom", "edges", "speed", "autoplay"].some(k => Q.has(k));
  if (Q.get("bare") === "1") document.documentElement.classList.add("bare"); // no article below (embeds, recordings)
  const savePrefs = () => {
    if (NO_SAVE) return; // a URL with overrides never rewrites the viewer's saved settings
    try { localStorage.setItem(PREF_KEY, JSON.stringify({ speed: S.speed, edgeMode: S.edgeMode, tg: S.tg, zoom: S.zoom, trace: S.name })); } catch (e) { }
  };

  /* ================================================================ theme */
  let TH = {};
  function hexRgb(h) {
    h = h.trim();
    if (h.startsWith("rgb")) return h.replace(/[^\d,]/g, "").split(",").slice(0, 3).map(Number);
    if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3];
    const n = parseInt(h.slice(1, 7), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  // Colours come from viz.css: TensorFlow Playground's scale (orange -1, grey 0, blue +1).
  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    const v = n => cs.getPropertyValue(n).trim();
    TH = {
      text: v("--text"), ink: v("--ink"), label: v("--label"), dim: v("--dim"), line: v("--line"), line2: v("--line2"),
      soft: v("--soft"), head: v("--head"), neg: hexRgb(v("--neg")), mid: hexRgb(v("--mid")), pos: hexRgb(v("--pos")),
      val: v("--val"), font: getComputedStyle(document.body).fontFamily,
    };
    TH.modes = MODE_LIGHT;      // marks on the light page (run chart)
    TH.sceneModes = MODE_DARK;  // the player and portals in the dark game panel
    // Linear in RGB between the three stops, as the Playground's d3.scale.linear does.
    TH.lut = [];
    for (let i = 0; i <= 256; i++) {
      const x = i / 128 - 1, t2 = Math.abs(x), pole = x < 0 ? TH.neg : TH.pos;
      TH.lut.push(`rgb(${Math.round(lerp(TH.mid[0], pole[0], t2))},${Math.round(lerp(TH.mid[1], pole[1], t2))},${Math.round(lerp(TH.mid[2], pole[2], t2))})`);
    }
    S.dirty = true;
  }
  const divColor = v => TH.lut[Math.round((clamp(v, -1, 1) + 1) * 128)];

  /* ================================================================ data */
  function b64Bytes(b64) {
    const bin = atob(b64), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  const TYPED = { f4: Float32Array, i1: Int8Array, u1: Uint8Array, i2: Int16Array, u2: Uint16Array, i4: Int32Array, u4: Uint32Array };
  function unpack(p) {
    const bytes = b64Bytes(p.b64), C = TYPED[p.dtype];
    const a = new C(bytes.buffer, 0, bytes.byteLength / C.BYTES_PER_ELEMENT);
    if (!p.scale) return a;
    const f = new Float32Array(a.length), s = 1 / p.scale;
    for (let i = 0; i < a.length; i++) f[i] = a[i] * s;
    return f;
  }
  async function gunzipJSON(bytes) {
    if (typeof DecompressionStream === "undefined")
      throw new Error("This browser cannot unpack the embedded trace (no DecompressionStream). Open it in a current Chrome, Edge, Firefox or Safari.");
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text());
  }

  /* Derived arrays for one trace. */
  function prepare(raw) {
    if (raw.format !== "gdrl-nn-trace") throw new Error("not a gdrl-nn-trace file");
    const F = raw.frames, E = raw.encoder, N = raw.meta.decisions, D = E.size;
    const T = { raw, meta: raw.meta, enc: E, feats: raw.features, groups: raw.groups, N, D, events: raw.events || [] };
    T.obs = unpack(F.obs); T.H = F.hidden.map(unpack); T.logits = unpack(F.logits); T.value = unpack(F.value);
    T.act = unpack(F.action);
    for (const k of ["x", "y", "vy", "rot", "xs", "mode", "flags", "size", "p2x", "p2y", "p2mode", "p2flags", "tick", "slots"]) T[k] = unpack(F[k]);
    T.sizes = raw.network.sizes;
    T.layers = raw.network.layers.map((l, i) => ({ W: unpack(l.W), b: unpack(l.b), nIn: T.sizes[i], nOut: T.sizes[i + 1] }));
    T.nh = T.H.length;
    T.p = new Float32Array(N);
    for (let f = 0; f < N; f++) T.p[f] = sigmoid(T.logits[2 * f + 1] - T.logits[2 * f]);
    T.time = new Float32Array(N);
    for (let f = 0; f < N; f++) T.time[f] = (T.tick[f] - T.tick[0]) / (raw.meta.tick_rate || 240);
    T.dur = T.time[N - 1] + 1 / DPS;
    let vmin = Infinity, vmax = -Infinity;
    for (let f = 0; f < N; f++) { vmin = Math.min(vmin, T.value[f]); vmax = Math.max(vmax, T.value[f]); }
    T.vmin = Math.min(vmin, -0.05); T.vmax = Math.max(vmax, 0.05); T.vabs = Math.max(Math.abs(T.vmin), Math.abs(T.vmax));

    // Objects and the per-frame view (the bridge's report), as CSR rows.
    const O = raw.objects;
    T.O = { n: O.count, names: { kind: O.kind_names, sub: O.subtype_names } };
    for (const k of ["x", "y", "w", "h", "kind", "subtype", "oflags", "first", "last", "activated", "orient", "gap"]) T.O[k] = unpack(O[k]);
    const ap = unpack(raw.view.add_ptr), ad = unpack(raw.view.add), dp = unpack(raw.view.del_ptr), dd = unpack(raw.view.del);
    const ptr = new Int32Array(N + 1), rows = [];
    let cur = new Set();
    for (let f = 0; f < N; f++) {
      for (let i = dp[f]; i < dp[f + 1]; i++) cur.delete(dd[i]);
      for (let i = ap[f]; i < ap[f + 1]; i++) cur.add(ad[i]);
      ptr[f] = rows.length;
      for (const v of cur) rows.push(v);
    }
    ptr[N] = rows.length;
    T.viewPtr = ptr; T.view = Int32Array.from(rows);
    T.inView = new Uint8Array(T.O.n);
    // Stable objects (in view for 4+ decisions) can be drawn as context outside their report.
    const st = [];
    let maxW = 0;
    for (let i = 0; i < T.O.n; i++) if (T.O.last[i] - T.O.first[i] >= 3) { st.push(i); maxW = Math.max(maxW, T.O.w[i]); }
    st.sort((a, b) => (T.O.x[a] - T.O.w[a] / 2) - (T.O.x[b] - T.O.w[b] / 2));
    T.stable = Int32Array.from(st); T.stableLeft = Float32Array.from(st.map(i => T.O.x[i] - T.O.w[i] / 2)); T.stableMaxW = maxW;

    // Presses and events (portal markers where the mode, gravity, size or dual changed).
    T.presses = [];
    for (let f = 0; f < N; f++) if (T.act[f] && (f === 0 || !T.act[f - 1])) T.presses.push(f);
    T.marks = [];
    for (const e of T.events) {
      if (e.f >= N) continue;
      const x = T.x[e.f], y = T.y[e.f];
      if (e.type === "mode") T.marks.push({ f: e.f, x, y, label: MODE_NAMES[raw.modes.indexOf(e.to)] || e.to, mode: raw.modes.indexOf(e.to), kind: "mode" });
      else if (e.type === "gravity") T.marks.push({ f: e.f, x, y, label: e.flipped ? "gravity ↑" : "gravity ↓", kind: "gravity", flipped: e.flipped });
      else if (e.type === "size") T.marks.push({ f: e.f, x, y, label: e.mini ? "mini" : "normal size", kind: "size", mini: e.mini });
      else if (e.type === "dual") T.marks.push({ f: e.f, x, y, label: e.on ? "dual" : "solo", kind: "dual" });
    }
    T.eventFrames = [...new Set(T.events.filter(e => e.type !== "gravity").map(e => e.f))].sort((a, b) => a - b);

    // Camera: zero-phase smoothed player height, kept so the ground shows when low.
    const cy = new Float32Array(N);
    let c = T.y[0];
    for (let f = 0; f < N; f++) { c += (T.y[f] - c) * 0.07; cy[f] = c; }
    for (let f = N - 1; f >= 0; f--) { c += (cy[f] - c) * 0.07; cy[f] = c; }
    T.camY = cy;

    // Neuron display order: by effective vote on (jump - release), linearised.
    const L = T.layers, last = L[L.length - 1];
    let vote = new Float64Array(last.nIn);
    for (let i = 0; i < last.nIn; i++) vote[i] = last.W[last.nIn + i] - last.W[i];
    T.vote = [];
    T.vote[T.nh - 1] = vote;
    for (let l = T.nh - 2; l >= 0; l--) {
      const Wn = L[l + 1], v2 = new Float64Array(Wn.nIn);
      for (let j = 0; j < Wn.nOut; j++) { const vj = T.vote[l + 1][j], row = j * Wn.nIn; for (let i = 0; i < Wn.nIn; i++) v2[i] += Wn.W[row + i] * vj; }
      T.vote[l] = v2;
    }
    T.order = T.vote.map(v => Array.from(v.keys()).sort((a, b) => v[a] - v[b]));
    T.rank = T.order.map(o => { const r = new Int32Array(o.length); o.forEach((n, k) => { r[n] = k; }); return r; });

    // Knock-out groups: player, each object slot (with its timing bumps), terrain, position.
    const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
    T.ko = [{ id: "player", label: "Player", idx: range(0, E.player_features) }];
    const nc = E.rbf_centers.length;
    for (let s = 0; s < E.k; s++) {
      const idx = range(E.player_features + s * E.slot_size, E.player_features + (s + 1) * E.slot_size);
      if (s < E.near_slots) idx.push(...range(E.rbf_start + s * nc, E.rbf_start + (s + 1) * nc));
      T.ko.push({ id: "o" + (s + 1), label: "Object " + (s + 1), slot: s, idx });
    }
    if (E.profile) T.ko.push({ id: "terrain", label: "Terrain", idx: range(E.profile_start, E.profile_start + 3 * E.profile_dx.length) });
    if (E.position) T.ko.push({ id: "position", label: "Position", idx: range(E.position_start, E.size) });
    for (const g of T.ko) { g.mask = new Uint8Array(D); for (const i of g.idx) g.mask[i] = 1; }
    T.featGroupLabel = T.feats.map(fe => { const g = T.groups.find(gg => gg.grid.some(r => r.includes(T.feats.indexOf(fe)))); return g ? g.label : ""; });
    return T;
  }
  const obsAt = (T, f) => T.obs.subarray(f * T.D, (f + 1) * T.D);
  const hAt = (T, l, f) => { const n = T.sizes[l + 1]; return T.H[l].subarray(f * n, (f + 1) * n); };
  const flagAt = (T, f, b) => (T.flags[f] & b) !== 0;
  function speedInfo(xs) {
    let best = SPEEDS[1], d = Infinity;
    for (const s of SPEEDS) if (Math.abs(s[0] - xs) < d) { d = Math.abs(s[0] - xs); best = s; }
    return best;
  }
  function slotKind(T, f, s) {
    const E = T.enc, base = f * T.D + E.player_features + s * E.slot_size;
    let best = -1, bv = 0.5;
    for (let k = 0; k < E.kinds.length; k++) if (T.obs[base + 4 + k] > bv) { bv = T.obs[base + 4 + k]; best = k; }
    return best < 0 ? null : E.kinds[best];
  }
  function slotDesc(T, f, s) {
    const E = T.enc, base = f * T.D + E.player_features + s * E.slot_size, kind = slotKind(T, f, s);
    if (!kind) return null;
    const dx = T.obs[base] * E.max_dx, w = T.obs[base + 2] * E.max_dx, dy = T.obs[base + 1] * E.max_dy;
    const left = dx - w / 2;
    return { kind, dx, dy, left, text: `${kind} · ${left >= 0 ? Math.round(left) + " u ahead" : "under the player"}${Math.abs(dy) > 20 ? `, ${Math.round(Math.abs(dy))} u ${dy > 0 ? "up" : "down"}` : ""}` };
  }

  /* ================================================================ network math */
  // Logit gap (jump - release) for the observation x with the features in mask set to 0.
  const BUF = { z: [] };
  function logitGap(T, x, nz, mask) {
    const L = T.layers;
    let a = null;
    for (let l = 0; l < L.length; l++) {
      const { W, b, nIn, nOut } = L[l];
      const z = BUF.z[l] && BUF.z[l].length === nOut ? BUF.z[l] : (BUF.z[l] = new Float64Array(nOut));
      for (let j = 0; j < nOut; j++) {
        let s = b[j]; const row = j * nIn;
        if (l === 0) { for (let q = 0; q < nz.length; q++) { const i = nz[q]; if (!mask || !mask[i]) s += W[row + i] * x[i]; } }
        else for (let i = 0; i < nIn; i++) s += W[row + i] * a[i];
        z[j] = l < L.length - 1 ? Math.tanh(s) : s;
      }
      a = z;
    }
    return a[1] - a[0];
  }
  const KO = { key: null, res: null };
  function knockouts(T, f) {
    if (KO.key === T.name + ":" + f) return KO.res;
    const x = obsAt(T, f), nz = [];
    for (let i = 0; i < x.length; i++) if (x[i] !== 0) nz.push(i);
    const base = sigmoid(logitGap(T, x, nz, null));
    const rows = T.ko.map(g => {
      const p = sigmoid(logitGap(T, x, nz, g.mask));
      const empty = g.slot != null && !slotKind(T, f, g.slot);
      return { g, p, dp: base - p, empty };
    });
    KO.key = T.name + ":" + f; KO.res = { base, rows };
    return KO.res;
  }
  // d(jump - release) / d(activation) for every hidden layer and the input, at frame f.
  const GR = { key: null, res: null };
  function gradients(T, f) {
    if (GR.key === T.name + ":" + f) return GR.res;
    const L = T.layers, nh = T.nh, gh = [];
    const last = L[L.length - 1];
    let g = new Float64Array(last.nIn);
    for (let i = 0; i < last.nIn; i++) g[i] = last.W[last.nIn + i] - last.W[i];
    gh[nh - 1] = g;
    for (let l = nh - 1; l >= 0; l--) {
      const h = hAt(T, l, f), Wl = L[l], gz = new Float64Array(Wl.nOut);
      for (let j = 0; j < Wl.nOut; j++) gz[j] = gh[l][j] * (1 - h[j] * h[j]);
      const gp = new Float64Array(Wl.nIn);
      for (let j = 0; j < Wl.nOut; j++) { const v = gz[j]; if (!v) continue; const row = j * Wl.nIn; for (let i = 0; i < Wl.nIn; i++) gp[i] += Wl.W[row + i] * v; }
      if (l > 0) gh[l - 1] = gp; else GR.gx = gp;
    }
    GR.key = T.name + ":" + f; GR.res = { gh, gx: GR.gx };
    return GR.res;
  }
  // Activation of layer l-1 (l = 0 means the input) at frame f.
  const srcAct = (T, l, f) => (l === 0 ? obsAt(T, f) : hAt(T, l - 1, f));

  /* Connections to draw: {l, i (source), j (target; for the output layer 0 = release, 1 = jump), c (signed), s (0..1)}.
     Layer l connects activation l-1 (input when l = 0) to layer l; l = nh is the output. */
  function topK(k) {
    const items = [];
    let minIdx = -1, minV = -Infinity;
    return {
      push(v, item) {
        if (items.length < k) { items.push([v, item]); if (items.length === k) recompute(); return; }
        if (v <= minV) return;
        items[minIdx] = [v, item]; recompute();
      },
      list: () => items.slice().sort((a, b) => b[0] - a[0]),
    };
    function recompute() { minV = Infinity; for (let q = 0; q < items.length; q++) if (items[q][0] < minV) { minV = items[q][0]; minIdx = q; } }
  }
  const EC = { key: null, edges: null };
  function selectEdges(T, f) {
    const focus = S.hover && S.hover.net ? S.hover : S.pin;
    const fk = focus ? `${focus.layer}:${focus.i}` : "";
    const key = `${T.name}:${f}:${S.edgeMode}:${fk}`;
    if (EC.key === key) return EC.edges;
    const L = T.layers, nh = T.nh, edges = [];
    const out = L[nh];
    const hl = hAt(T, nh - 1, f);
    const cOut = new Float64Array(out.nIn);
    for (let i = 0; i < out.nIn; i++) cOut[i] = (out.W[out.nIn + i] - out.W[i]) * hl[i];
    const pushLayer = (l, list) => {
      if (!list.length) return;
      const m = list[0][0] || 1;
      for (const [v, e] of list) edges.push({ ...e, l, s: clamp(v / m, 0, 1) });
    };
    if (focus) {
      // Edges into and out of one neuron / input / output.
      const { layer, i } = focus; // layer: -1 input, 0..nh-1 hidden, nh output
      if (layer >= 0 && layer < nh) {
        const W = L[layer], a = srcAct(T, layer, f), row = i * W.nIn, tk = topK(14);
        for (let q = 0; q < W.nIn; q++) { const c = W.W[row + q] * a[q]; if (c) tk.push(Math.abs(c), { i: q, j: i, c }); }
        pushLayer(layer, tk.list());
        const ho = hAt(T, layer, f)[i];
        if (layer === nh - 1) pushLayer(nh, [[Math.abs(cOut[i]), { i, j: cOut[i] >= 0 ? 1 : 0, c: cOut[i] }]]);
        else {
          const Wn = L[layer + 1], tk2 = topK(14);
          for (let j = 0; j < Wn.nOut; j++) { const c = Wn.W[j * Wn.nIn + i] * ho; if (c) tk2.push(Math.abs(c), { i, j, c }); }
          pushLayer(layer + 1, tk2.list());
        }
      } else if (layer === -1) {
        const W = L[0], x = obsAt(T, f)[i], tk = topK(16);
        if (x) for (let j = 0; j < W.nOut; j++) { const c = W.W[j * W.nIn + i] * x; tk.push(Math.abs(c), { i, j, c }); }
        pushLayer(0, tk.list());
      } else if (layer === nh) {
        const tk = topK(14);
        for (let q = 0; q < out.nIn; q++) { const c = out.W[i * out.nIn + q] * hl[q]; if (c) tk.push(Math.abs(c), { i: q, j: i, c }); }
        pushLayer(nh, tk.list());
      }
    } else if (S.edgeMode === "top") {
      const tko = topK(10);
      for (let i = 0; i < out.nIn; i++) tko.push(Math.abs(cOut[i]), { i, j: cOut[i] >= 0 ? 1 : 0, c: cOut[i] });
      pushLayer(nh, tko.list());
      for (let l = nh - 1; l >= 0; l--) {
        const W = L[l], a = srcAct(T, l, f), tk = topK(l === 0 ? 44 : 40);
        for (let q = 0; q < W.nIn; q++) {
          const aq = a[q]; if (!aq) continue;
          for (let j = 0; j < W.nOut; j++) { const c = W.W[j * W.nIn + q] * aq; const v = Math.abs(c); if (v > 0.02) tk.push(v, { i: q, j, c }); }
        }
        pushLayer(l, tk.list());
      }
    } else {
      // Decision path: strongest output contributions traced back layer by layer.
      const tko = topK(8);
      for (let i = 0; i < out.nIn; i++) tko.push(Math.abs(cOut[i]), { i, j: cOut[i] >= 0 ? 1 : 0, c: cOut[i] });
      const lo = tko.list();
      pushLayer(nh, lo);
      let imp = new Map(); // neuron of the current layer -> importance
      const m0 = lo.length ? lo[0][0] : 1;
      for (const [v, e] of lo) imp.set(e.i, v / m0);
      for (let l = nh - 1; l >= 0; l--) {
        const W = L[l], a = srcAct(T, l, f), tk = topK(l === 0 ? 40 : 34);
        for (const [j, w] of imp) {
          const row = j * W.nIn;
          for (let q = 0; q < W.nIn; q++) { const aq = a[q]; if (!aq) continue; const c = W.W[row + q] * aq; const v = Math.abs(c) * w; if (v > 1e-4) tk.push(v, { i: q, j, c }); }
        }
        const lst = tk.list();
        pushLayer(l, lst);
        const nimp = new Map(), m = lst.length ? lst[0][0] : 1;
        for (const [v, e] of lst) nimp.set(e.i, (nimp.get(e.i) || 0) + v / m);
        imp = nimp;
      }
    }
    EC.key = key; EC.edges = edges;
    return edges;
  }

  /* ================================================================ canvases */
  const CV = {};
  const MIN_W = { net: 660, tl: 360 }; // below this the panel scrolls sideways instead of squeezing
  function fitCanvas(id) {
    const cv = $("#" + id);
    if (!cv) return null;
    const wrap = cv.parentElement;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(MIN_W[id] || 20, Math.floor(wrap.clientWidth)), h = Math.max(20, Math.floor(wrap.clientHeight));
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    cv.style.width = w + "px"; cv.style.height = h + "px";
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return (CV[id] = { cv, ctx, w, h, dpr });
  }
  function offscreen(w, h, dpr) {
    const cv = document.createElement("canvas"); cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { cv, ctx };
  }
  const STATS = { layouts: 0, frames: 0, drawMs: 0 };
  function layoutAll() {
    if (!S.T) return;
    STATS.layouts++;
    if (!fitCanvas("scene") || !fitCanvas("net") || !fitCanvas("tl")) return;
    layoutNet(); renderTimelineStatic();
    EC.key = null; S.dirty = true;
  }

  /* ================================================================ scene */
  const SC = { camX: 0, camY: 0, s: 1, ox: 0, oy: 0 };
  function sceneCamera(T, ff) {
    const { w, h } = CV.scene, f0 = Math.floor(ff), f1 = Math.min(T.N - 1, f0 + 1), t = ff - f0;
    const px = lerp(T.x[f0], T.x[f1], t), py = lerp(T.y[f0], T.y[f1], t);
    const want = 820 / S.zoom; // world units across the view
    const s = Math.min(w / want, h / (want * 0.78));
    const halfH = h / 2 / s;
    let cy = lerp(T.camY[f0], T.camY[f1], t);
    cy = Math.max(cy, halfH - 75); // keep the ground near the bottom when the player is low
    Object.assign(SC, { s, camX: px, camY: cy, ox: w * 0.22, oy: h / 2, px, py, t, f0, f1 });
  }
  const wx = x => SC.ox + (x - SC.camX) * SC.s;
  const wy = y => SC.oy - (y - SC.camY) * SC.s;

  function drawScene(ff) {
    const T = S.T, C = CV.scene; if (!T || !C) return;
    const ctx = C.ctx, { w, h } = C;
    sceneCamera(T, ff);
    const f = SC.f0, s = SC.s;
    const x0 = SC.camX - SC.ox / s - 40, x1 = SC.camX + (w - SC.ox) / s + 40;
    // background (flat)
    ctx.fillStyle = "#14182e"; ctx.fillRect(0, 0, w, h);
    // block grid + x ruler
    if (30 * s >= 7) {
      ctx.strokeStyle = "rgba(255,255,255,0.045)"; ctx.lineWidth = 1; ctx.beginPath();
      for (let gx = Math.floor(x0 / 30) * 30; gx <= x1; gx += 30) { const X = Math.round(wx(gx)) + 0.5; ctx.moveTo(X, 0); ctx.lineTo(X, h); }
      const yTop = SC.camY + SC.oy / s, yBot = SC.camY - (h - SC.oy) / s;
      for (let gy = Math.floor(yBot / 30) * 30; gy <= yTop; gy += 30) { const Y = Math.round(wy(gy)) + 0.5; ctx.moveTo(0, Y); ctx.lineTo(w, Y); }
      ctx.stroke();
    }
    ctx.font = "10px " + TH.font;
    ctx.fillStyle = "rgba(255,255,255,0.4)"; ctx.textAlign = "left";
    for (let gx = Math.ceil(x0 / 300) * 300; gx <= x1; gx += 300) { const X = wx(gx); ctx.fillRect(X, h - 6, 1, 6); ctx.fillText("x " + fmtInt(gx), X + 3, h - 3); }
    // ground (flat band, one line)
    const gy = wy(0);
    if (gy < h) {
      ctx.fillStyle = "#1d2244"; ctx.fillRect(0, gy, w, h - gy);
      ctx.fillStyle = "#8fa3e0"; ctx.fillRect(0, gy - 1, w, 1.5);
    }
    // what the bridge reported this frame
    const O = T.O, v0 = T.viewPtr[f], v1 = T.viewPtr[f + 1];
    for (let q = v0; q < v1; q++) T.inView[T.view[q]] = 1;
    // context: stable objects known from nearby frames but not in this report
    const ghosts = [];
    const G = 300, lo = bsearch(T.stableLeft, x0 - T.stableMaxW);
    for (let q = lo; q < T.stable.length && T.stableLeft[q] <= x1; q++) {
      const i = T.stable[q];
      if (T.inView[i] || O.x[i] + O.w[i] / 2 < x0) continue;
      if (O.first[i] - G > f || O.last[i] + G < f) continue;
      ghosts.push(i);
    }
    const vis = [];
    for (let q = v0; q < v1; q++) { const i = T.view[q]; if (O.x[i] + O.w[i] / 2 >= x0 && O.x[i] - O.w[i] / 2 <= x1) vis.push(i); }
    drawObjects(ctx, T, ghosts, f, true);
    drawObjects(ctx, T, vis, f, false);
    for (let q = v0; q < v1; q++) T.inView[T.view[q]] = 0;
    drawMarks(ctx, T, x0, x1, f);
    // encoder overlays
    if (S.tg.fov) drawFov(ctx, T, w, h);
    if (S.tg.terrain && T.enc.profile) drawProfile(ctx, T, f);
    if (S.tg.slots) drawSlots(ctx, T, f);
    // the player
    drawTrail(ctx, T, ff);
    const mode = T.mode[f], fl = T.flags[f];
    const size = T.size[f] > 0 ? T.size[f] : 30;
    drawPlayer(ctx, wx(SC.px), wy(SC.py), size * s, mode, T.rot[f], T.vy[f], T.xs[f], (fl & FL.UPSIDE_DOWN) !== 0, TH.sceneModes[mode] || "#fff", T.act[f] === 1);
    if (fl & FL.DUAL) {
      const m2 = T.p2mode[f];
      drawPlayer(ctx, wx(T.p2x[f]), wy(T.p2y[f]), (T.p2flags[f] & FL.MINI ? 18 : 30) * s, m2, T.rot[f], 0, T.xs[f], (T.p2flags[f] & FL.UPSIDE_DOWN) !== 0, TH.sceneModes[m2] || "#fff", T.act[f] === 1, true);
    }
    SC.vis = vis; SC.ghosts = ghosts;
  }
  function bsearch(arr, v) { let a = 0, b = arr.length; while (a < b) { const m = (a + b) >> 1; if (arr[m] < v) a = m + 1; else b = m; } return a; }

  function hazardTri(T, i) {
    const O = T.O, x = O.x[i], y = O.y[i], w = O.w[i], h = O.h[i], o = O.orient[i], gap = O.gap[i];
    const side = o < 2 ? h : w, across = o < 2 ? w : h;
    let L, B, base;
    if (gap >= 0) { L = clamp(2 * (side / 2 + gap), 6, 36); B = clamp(across * 5, 8, 30); }
    else { L = clamp(side * 1.6, 6, 32); B = clamp(Math.max(across, 6) * 1.8, 8, 30); }
    // base line and apex in world coordinates
    if (o === 0) { base = gap >= 0 ? y - h / 2 - gap : y - L / 2; return [[x - B / 2, base], [x + B / 2, base], [x, base + L]]; }
    if (o === 1) { base = gap >= 0 ? y + h / 2 + gap : y + L / 2; return [[x - B / 2, base], [x + B / 2, base], [x, base - L]]; }
    if (o === 2) { base = gap >= 0 ? x + w / 2 + gap : x + L / 2; return [[base, y - B / 2], [base, y + B / 2], [base - L, y]]; }
    base = gap >= 0 ? x - w / 2 - gap : x - L / 2; return [[base, y - B / 2], [base, y + B / 2], [base + L, y]];
  }
  function drawObjects(ctx, T, list, f, ghost) {
    if (!list.length) return;
    const O = T.O, s = SC.s;
    ctx.save();
    ctx.globalAlpha = ghost ? 0.2 : 1;
    // solids: outline pass then fill pass, so merged / overlapping blocks read as one shape
    const sol = list.filter(i => O.kind[i] === K.SOLID);
    if (sol.length) {
      ctx.fillStyle = ghost ? "#4b5aa0" : "#7f93d8";
      ctx.beginPath();
      for (const i of sol) ctx.rect(wx(O.x[i] - O.w[i] / 2) - 1, wy(O.y[i] + O.h[i] / 2) - 1, O.w[i] * s + 2, O.h[i] * s + 2);
      ctx.fill();
      ctx.fillStyle = "#242a52";
      ctx.beginPath();
      for (const i of sol) ctx.rect(wx(O.x[i] - O.w[i] / 2), wy(O.y[i] + O.h[i] / 2), O.w[i] * s, O.h[i] * s);
      ctx.fill();
      // slopes: a diagonal so they do not read as plain blocks
      ctx.strokeStyle = "rgba(140,160,255,0.55)"; ctx.lineWidth = 1; ctx.beginPath();
      for (const i of sol) if (O.subtype[i] === ST.SLOPE) { const l = wx(O.x[i] - O.w[i] / 2), t = wy(O.y[i] + O.h[i] / 2); ctx.moveTo(l, t + O.h[i] * s); ctx.lineTo(l + O.w[i] * s, t); }
      ctx.stroke();
    }
    for (const i of list) {
      const k = O.kind[i], st = O.subtype[i];
      if (k === K.SOLID) continue;
      const X = wx(O.x[i]), Y = wy(O.y[i]), W = Math.max(2, O.w[i] * s), H = Math.max(2, O.h[i] * s);
      if (k === K.HAZARD) {
        const big = O.w[i] > 34 || O.h[i] > 34, ar = O.w[i] / Math.max(1, O.h[i]);
        if (st === ST.SAW || (big && ar > 0.7 && ar < 1.4)) { // saw-like: a disc
          const r = Math.min(W, H) / 2;
          ctx.fillStyle = "rgba(255,59,92,0.30)"; ctx.strokeStyle = "#ff3b5c"; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(X, Y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
          ctx.beginPath(); ctx.arc(X, Y, r * 0.35, 0, Math.PI * 2); ctx.stroke();
          continue;
        }
        if (big) { // a long hazard (beam, spike wall): its hitbox, hatched
          ctx.save();
          ctx.beginPath(); ctx.rect(X - W / 2, Y - H / 2, W, H); ctx.clip();
          ctx.fillStyle = "rgba(255,59,92,0.22)"; ctx.fillRect(X - W / 2, Y - H / 2, W, H);
          ctx.strokeStyle = "rgba(255,59,92,0.55)"; ctx.lineWidth = 1; ctx.beginPath();
          for (let d = -H; d < W; d += 7) { ctx.moveTo(X - W / 2 + d, Y + H / 2); ctx.lineTo(X - W / 2 + d + H, Y - H / 2); }
          ctx.stroke(); ctx.restore();
          ctx.strokeStyle = "#ff3b5c"; ctx.lineWidth = 1.4; ctx.strokeRect(X - W / 2, Y - H / 2, W, H);
          continue;
        }
        const tri = hazardTri(T, i);
        ctx.beginPath(); ctx.moveTo(wx(tri[0][0]), wy(tri[0][1])); ctx.lineTo(wx(tri[1][0]), wy(tri[1][1])); ctx.lineTo(wx(tri[2][0]), wy(tri[2][1])); ctx.closePath();
        ctx.fillStyle = ghost ? "#7a2238" : "#3a0d1e"; ctx.fill();
        ctx.strokeStyle = "#ff3b5c"; ctx.lineWidth = 1.4; ctx.stroke();
        if (!ghost) { ctx.fillStyle = "rgba(255,59,92,0.55)"; ctx.fillRect(X - W / 2, Y - H / 2, W, H); }
      } else if (k === K.ORB) {
        const r = Math.max(5, Math.min(W, H) / 2), c = ORB_COLOR[st] || "#ffd400", used = O.activated[i] >= 0 && O.activated[i] <= f;
        ctx.globalAlpha = (ghost ? 0.2 : 1) * (used ? 0.35 : 1);
        ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X, Y, r, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = c; ctx.beginPath(); ctx.arc(X, Y, r * 0.55, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = ghost ? 0.2 : 1;
      } else if (k === K.PAD) {
        const c = PAD_COLOR[st] || "#ffd400";
        ctx.fillStyle = c; ctx.beginPath(); ctx.ellipse(X, Y + H / 2, Math.max(W / 2, 6), Math.max(H, 4), 0, Math.PI, 0); ctx.fill();
      } else if (k === K.PORTAL) {
        drawPortalObject(ctx, X, Y, W, H, st);
      } else {
        ctx.strokeStyle = "rgba(160,160,200,0.6)"; ctx.lineWidth = 1; ctx.strokeRect(X - W / 2, Y - H / 2, W, H);
      }
    }
    ctx.restore();
  }
  function drawPortalObject(ctx, X, Y, W, H, st) {
    if (st >= 60 && st <= 64) { // speed portal: chevrons
      const sp = SPEEDS[st - 60], n = st === 60 ? 1 : st - 60;
      const hh = Math.max(H, 26) / 2, cw = Math.max(5, Math.min(10, W));
      ctx.strokeStyle = sp[2]; ctx.lineWidth = 2.4; ctx.lineJoin = "round";
      for (let q = 0; q < Math.max(1, n); q++) { const ox = X + (q - (n - 1) / 2) * cw * 1.1; ctx.beginPath(); ctx.moveTo(ox - cw / 2, Y - hh * 0.6); ctx.lineTo(ox + cw / 2, Y); ctx.lineTo(ox - cw / 2, Y + hh * 0.6); ctx.stroke(); }
      return;
    }
    let c = "#2fc6ff";
    if (st >= 40 && st <= 47) c = TH.sceneModes[st - 40];
    else if (st === 50) c = "#2f8bff"; else if (st === 51) c = "#ffd400"; else if (st === 52) c = "#3dff7a";
    else if (st === 53) c = "#ff4fd8"; else if (st === 54) c = "#3dff7a"; else if (st === 55) c = "#ffa630";
    ctx.strokeStyle = c; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.ellipse(X, Y, Math.max(W / 2, 5), Math.max(H / 2, 20), 0, 0, Math.PI * 2); ctx.stroke();
  }
  function drawMarks(ctx, T, x0, x1, f) {
    const s = SC.s;
    ctx.save();
    ctx.font = "600 10px " + TH.font; ctx.textAlign = "center";
    for (const m of T.marks) {
      if (m.x < x0 || m.x > x1) continue;
      let c = "#cfd3f0";
      if (m.kind === "mode") c = TH.sceneModes[m.mode] || c; else if (m.kind === "gravity") c = m.flipped ? "#ffd400" : "#2f8bff";
      else if (m.kind === "size") c = m.mini ? "#ff4fd8" : "#3dff7a"; else if (m.kind === "dual") c = "#ffa630";
      const X = wx(m.x - 4), Y = wy(m.y), hh = Math.max(34, 90 * s) / 2, ww = Math.max(5, 12 * s) / 2;
      const passed = m.f <= f;
      ctx.globalAlpha = passed ? 0.5 : 0.9;
      ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.ellipse(X, Y, ww, hh, 0, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
      if (m.kind === "mode" || m.kind === "dual" || m.kind === "size") { ctx.fillStyle = c; ctx.fillText(m.label, X, Y - hh - 5); }
    }
    ctx.restore();
  }
  function drawFov(ctx, T, w, h) {
    const E = T.enc, l = wx(SC.px - E.block / 2), r = wx(SC.px + E.max_dx), t = wy(SC.py + E.max_dy), b = wy(SC.py - E.max_dy);
    ctx.save();
    ctx.fillStyle = "rgba(4,4,14,0.34)";
    ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.rect(l, t, r - l, b - t); ctx.fill("evenodd");
    ctx.strokeStyle = "rgba(0,229,255,0.75)"; ctx.lineWidth = 1.5;
    const c = 12; ctx.beginPath();
    for (const [X, Y, dx, dy] of [[l, t, 1, 1], [r, t, -1, 1], [l, b, 1, -1], [r, b, -1, -1]]) { ctx.moveTo(X + dx * c, Y); ctx.lineTo(X, Y); ctx.lineTo(X, Y + dy * c); }
    ctx.stroke();
    ctx.fillStyle = "rgba(0,229,255,0.8)"; ctx.font = "600 10px " + TH.font; ctx.textAlign = "left";
    ctx.fillText(`ENCODER VIEW · ${E.max_dx} u ahead, ±${E.max_dy} u`, Math.max(l + 6, 6), Math.max(t + 13, 13));
    ctx.restore();
  }
  function drawProfile(ctx, T, f) {
    const E = T.enc, P = E.profile_dx.length, o = f * T.D + E.profile_start, s = SC.s;
    const flipped = flagAt(T, f, FL.UPSIDE_DOWN), py = T.y[f], feet = py - E.block / 2;
    const toWorld = yy => (flipped ? 2 * py - yy : yy);
    const colW = Math.max(3, 15 * s * 0.45);
    ctx.save();
    for (let j = 0; j < P; j++) {
      const sup = T.obs[o + j] * E.max_dy, haz = T.obs[o + P + j] > 0.5, cei = T.obs[o + 2 * P + j] * E.max_dy;
      const X = wx(SC.px + E.profile_dx[j]);
      const ys = wy(toWorld(feet + sup)), yc = wy(toWorld(feet + cei));
      const top = Math.min(ys, yc), bot = Math.max(ys, yc);
      const open = cei >= E.max_dy - 1;
      const g = ctx.createLinearGradient(0, flipped ? bot : top, 0, flipped ? top : bot);
      const col = haz ? "255,59,92" : "0,229,255";
      g.addColorStop(0, `rgba(${col},${open ? 0 : 0.16})`); g.addColorStop(1, `rgba(${col},0.16)`);
      ctx.fillStyle = g; ctx.fillRect(X - colW / 2, top, colW, bot - top);
      ctx.fillStyle = haz ? "#ff3b5c" : "rgba(0,229,255,0.9)";
      ctx.fillRect(X - colW / 2 - 1, ys - 1, colW + 2, 2.5);
      if (!open) { ctx.fillStyle = "rgba(0,229,255,0.55)"; ctx.fillRect(X - colW / 2, yc - 1, colW, 2); }
    }
    ctx.restore();
  }
  function drawSlots(ctx, T, f) {
    const E = T.enc, O = T.O, s = SC.s;
    ctx.save();
    ctx.font = "700 10px " + TH.font; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const badges = [];
    for (let q = 0; q < E.k; q++) {
      const i = T.slots[f * E.k + q];
      if (i < 0) continue;
      const l = wx(O.x[i] - O.w[i] / 2), t = wy(O.y[i] + O.h[i] / 2), W = Math.max(4, O.w[i] * s), H = Math.max(4, O.h[i] * s);
      const hot = S.hover && S.hover.slot === q;
      ctx.strokeStyle = hot ? "#ffffff" : "rgba(255,255,255,0.85)"; ctx.lineWidth = hot ? 2.5 : 1.4;
      ctx.strokeRect(l - 3, t - 3, W + 6, H + 6);
      let bx = l - 3, by = t - 11;
      for (const b of badges) if (Math.abs(b[0] - bx) < 15 && Math.abs(b[1] - by) < 15) by -= 15;
      badges.push([bx, by]);
      ctx.fillStyle = q < E.near_slots ? "#ffffff" : "#c9cbe8";
      ctx.beginPath(); ctx.arc(bx, by, 7, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#0b0b1a"; ctx.fillText(String(q + 1), bx, by + 0.5);
    }
    // the nearest obstacle's left-edge distance: what the timing code encodes
    const i0 = T.slots[f * E.k];
    if (i0 >= 0) {
      const left = O.x[i0] - O.w[i0] / 2, X0 = wx(SC.px), X1 = wx(left), Y = wy(SC.py);
      if (X1 > X0 + 6) {
        ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(X0, Y); ctx.lineTo(X1, Y); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = "rgba(255,255,255,0.85)"; ctx.font = "600 10px " + TH.font;
        ctx.fillText(Math.round(left - SC.px) + " u", (X0 + X1) / 2, Y - 8);
      }
    }
    ctx.restore();
  }
  function drawTrail(ctx, T, ff) {
    const f = Math.floor(ff), n = 42, a = Math.max(0, f - n);
    if (f - a < 2) return;
    ctx.save(); ctx.lineCap = "round"; ctx.lineJoin = "round";
    const wave = T.mode[f] === 4;
    for (let q = a + 1; q <= f; q++) {
      const t = (q - a) / (f - a);
      ctx.strokeStyle = rgba(hexRgb(TH.sceneModes[T.mode[q]] || "#ffffff"), (wave ? 0.85 : 0.5) * t);
      ctx.lineWidth = (wave ? 4.5 : 2.2) * (0.35 + 0.65 * t);
      ctx.beginPath(); ctx.moveTo(wx(T.x[q - 1]), wy(T.y[q - 1])); ctx.lineTo(wx(q === f ? SC.px : T.x[q]), wy(q === f ? SC.py : T.y[q])); ctx.stroke();
    }
    ctx.restore();
  }
  function drawPlayer(ctx, X, Y, size, mode, rot, vy, xs, flipped, color, held, second) {
    const s = Math.max(9, size);
    ctx.save();
    ctx.translate(X, Y);
    if (second) ctx.globalAlpha = 0.75;
    const dark = "#0b0b1a";
    ctx.lineWidth = Math.max(1.5, s * 0.08); ctx.strokeStyle = dark; ctx.fillStyle = color;
    const tilt = clamp(-Math.atan2(vy, Math.max(200, xs)), -0.9, 0.9);
    if (mode === 0 || mode === 5 || mode === 6) { // cube, robot, spider
      ctx.rotate((rot * Math.PI) / 180);
      roundRect(ctx, -s / 2, -s / 2, s, s, s * 0.14); ctx.fill(); ctx.stroke();
      ctx.fillStyle = dark; roundRect(ctx, -s * 0.22, -s * 0.22, s * 0.44, s * 0.44, s * 0.06); ctx.fill();
      ctx.fillStyle = color; ctx.fillRect(-s * 0.1, -s * 0.1, s * 0.2, s * 0.2);
      if (mode !== 0) { ctx.strokeStyle = color; ctx.lineWidth = s * 0.1; ctx.beginPath(); ctx.moveTo(-s * 0.3, s * 0.5); ctx.lineTo(-s * 0.45, s * 0.8); ctx.moveTo(s * 0.3, s * 0.5); ctx.lineTo(s * 0.45, s * 0.8); ctx.stroke(); }
    } else if (mode === 1 || mode === 7) { // ship, swing
      if (flipped) ctx.scale(1, -1);
      ctx.rotate(flipped ? -tilt : tilt);
      ctx.beginPath(); ctx.moveTo(-s * 0.85, s * 0.05); ctx.lineTo(s * 0.85, s * 0.18); ctx.lineTo(s * 0.55, s * 0.45); ctx.lineTo(-s * 0.7, s * 0.45); ctx.closePath(); ctx.fill(); ctx.stroke();
      roundRect(ctx, -s * 0.32, -s * 0.42, s * 0.5, s * 0.5, s * 0.08); ctx.fill(); ctx.stroke();
    } else if (mode === 2) { // ball
      ctx.rotate((rot * Math.PI) / 180);
      ctx.beginPath(); ctx.arc(0, 0, s / 2, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = dark; ctx.lineWidth = s * 0.1; ctx.beginPath(); ctx.moveTo(-s * 0.35, 0); ctx.lineTo(s * 0.35, 0); ctx.moveTo(0, -s * 0.35); ctx.lineTo(0, s * 0.35); ctx.stroke();
    } else if (mode === 3) { // UFO
      if (flipped) ctx.scale(1, -1);
      ctx.beginPath(); ctx.arc(0, -s * 0.05, s * 0.32, Math.PI, 0); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.ellipse(0, s * 0.12, s * 0.62, s * 0.24, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    } else if (mode === 4) { // wave: a dart along its velocity
      const ang = -Math.atan2(vy, Math.max(100, xs));
      ctx.rotate(ang);
      const ws = Math.max(14, s * 1.5);
      ctx.beginPath(); ctx.moveTo(ws * 0.6, 0); ctx.lineTo(-ws * 0.45, -ws * 0.42); ctx.lineTo(-ws * 0.2, 0); ctx.lineTo(-ws * 0.45, ws * 0.42); ctx.closePath(); ctx.fill(); ctx.stroke();
    }
    ctx.restore();
    if (!second && held) { // button held: a thin ring
      ctx.save(); ctx.strokeStyle = "rgba(255,255,255,0.85)"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(X, Y, s * 0.95, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
    }
  }
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r); ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h); ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r); ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
  }
  function updateHud() {
    const T = S.T, f = S.f;
    const m = T.mode[f], sp = speedInfo(T.xs[f]), fl = T.flags[f];
    const chips = [`<span class="chip"><i style="background:${TH.sceneModes[m]}"></i>${MODE_NAMES[m] || "?"}</span>`, `<span class="chip">${sp[1]} speed</span>`];
    if (fl & FL.UPSIDE_DOWN) chips.push(`<span class="chip">gravity flipped</span>`);
    if (fl & FL.MINI) chips.push(`<span class="chip">mini</span>`);
    if (fl & FL.DUAL) chips.push(`<span class="chip">dual</span>`);
    if (T.act[f]) chips.push(`<span class="chip held">button held</span>`);
    const html = chips.join("");
    if (S._hud !== html) { $("#hud").innerHTML = html; S._hud = html; }
    const ban = $("#banner"), end = f === T.N - 1;
    if (end && (T.meta.completed || T.meta.dead)) {
      ban.hidden = false; ban.className = "banner" + (T.meta.completed ? "" : " dead");
      ban.textContent = T.meta.completed ? "Level complete" : "Crashed at " + (100 * T.meta.progress).toFixed(1) + "%";
    } else ban.hidden = true;
  }

  /* ================================================================ network layout */
  // Drawn the way TensorFlow Playground draws its network: square nodes with a
  // dark border, curved links whose colour is the sign and width the size,
  // uppercase column headings (FEATURES, HIDDEN LAYERS, OUTPUT).
  const NG = {};
  function layoutNet() {
    const T = S.T, C = CV.net; if (!T || !C) return;
    const { w, h } = C, pad = 4, top = 56, bot = 22;
    const inW = clamp(w * 0.35, 200, 330), outW = clamp(w * 0.2, 150, 210);
    const midX0 = pad + inW, midX1 = w - pad - outW, midW = midX1 - midX0;
    const availH = h - top - bot;
    // hidden layers: pick the grid shape that gives the roomiest nodes
    const n1 = T.sizes[1], hcW = midW * 0.2;
    let best = null;
    for (const cols of [2, 4, 8, 16]) {
      if (n1 % cols) continue;
      const rows = n1 / cols, sp = Math.min(hcW / cols, availH / rows);
      if (!best || sp > best.sp) best = { cols, rows, sp };
    }
    const sp = Math.min(best.sp, 22), half = clamp(sp * 0.36, 2.2, 8);
    const gridW = best.cols * sp, gridH = best.rows * sp;
    const gap = (midW - T.nh * gridW) / (T.nh + 1);
    NG.hid = [];
    for (let l = 0; l < T.nh; l++) {
      const gx = midX0 + gap * (l + 1) + gridW * l, gy = top + (availH - gridH) / 2;
      const n = T.sizes[l + 1], pos = new Float32Array(n * 2);
      for (let i = 0; i < n; i++) { const k = T.rank[l][i], cx = k % best.cols, cy = Math.floor(k / best.cols); pos[2 * i] = gx + (cx + 0.5) * sp; pos[2 * i + 1] = gy + (cy + 0.5) * sp; }
      NG.hid.push({ x: gx, y: gy, w: gridW, h: gridH, pos, r: half, n });
    }
    // outputs: two squares, Release above Jump
    const ox = midX1 + 30, cyMid = top + availH * 0.42;
    NG.out = [{ x: ox, y: cyMid - 30, r: 13, label: "Release" }, { x: ox, y: cyMid + 30, r: 13, label: "Jump" }];
    NG.value = { x: ox - 13, y: cyMid + 96, w: w - pad - (ox - 13) };
    NG.cols = { inX: pad, inW, out: ox, midX0, midX1 };
    // features: groups stacked; each is a header, column labels, a grid of cells, an axis
    const labW = 42, cellMax = 15, HEAD = 15, COLL = 11, AXIS = 12, GAP = 9;
    const items = T.groups.map(g => {
      const cols = Math.max(...g.grid.map(r2 => r2.length));
      return { g, cols, rows: g.grid.length, c: Math.min(cellMax, (inW - labW - 4) / cols), top: g.col_labels ? COLL : 0, bot: g.axis ? AXIS : 0 };
    });
    const avail = h - top - bot + 8;
    const fixed = items.reduce((a, it) => a + HEAD + it.top + it.bot + GAP, 0) - GAP;
    const cellsH = items.reduce((a, it) => a + it.rows * it.c, 0);
    const scale = clamp((avail - fixed) / cellsH, 0.3, 1);
    for (const it of items) it.c *= scale;
    const total = fixed + items.reduce((a, it) => a + it.rows * it.c, 0);
    let y = top - 8 + Math.max(0, (avail - total) / 2);
    NG.cells = new Float32Array(T.D * 3); // x, y, size
    NG.port = new Float32Array(T.D); // where a feature's links leave its group: the grid's right edge
    NG.groups = [];
    for (const it of items) {
      const g = it.g, c = it.c, gx = pad + labW;
      const rec = { g, x: pad, y, labW, c, cols: it.cols, rows: it.rows };
      y += HEAD; rec.colLabY = y; y += it.top; rec.gridY = y;
      g.grid.forEach((row, ri) => row.forEach((fi, ci) => { NG.cells[3 * fi] = gx + ci * c; NG.cells[3 * fi + 1] = y + ri * c; NG.cells[3 * fi + 2] = c; NG.port[fi] = gx + it.cols * c + 2; }));
      y += it.rows * c; rec.bottom = y; y += it.bot + GAP;
      NG.groups.push(rec);
    }
    renderNetStatic();
  }
  function renderNetStatic() {
    const T = S.T, C = CV.net, { w, h, dpr } = C;
    const off = offscreen(w, h, dpr), ctx = off.ctx;
    const font = TH.font;
    // column headings, as in the Playground
    ctx.fillStyle = TH.text; ctx.font = `400 15px ${font}`; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.fillText("FEATURES", NG.cols.inX, 16);
    const hx = (NG.hid[0].x + NG.hid[T.nh - 1].x + NG.hid[T.nh - 1].w) / 2;
    ctx.textAlign = "center"; ctx.fillText(`${T.nh} HIDDEN LAYERS`, hx, 16);
    ctx.textAlign = "left"; ctx.fillText("OUTPUT", NG.out[0].x - NG.out[0].r, 16);
    ctx.font = `300 12px ${font}`; ctx.fillStyle = TH.label;
    ctx.textAlign = "left"; ctx.fillText(`${T.D} numbers from the encoder`, NG.cols.inX, 33);
    NG.hid.forEach(hd => { ctx.textAlign = "center"; ctx.fillText(`${hd.n} neurons, tanh`, hd.x + hd.w / 2, 33); });
    ctx.textAlign = "left"; ctx.fillText("softmax", NG.out[0].x - NG.out[0].r, 33);
    ctx.font = `300 10.5px ${font}`; ctx.fillStyle = TH.dim;
    NG.hid.forEach(hd => {
      ctx.textAlign = "center";
      ctx.fillText("↑ toward Release", hd.x + hd.w / 2, hd.y - 6);
      ctx.fillText("toward Jump ↓", hd.x + hd.w / 2, hd.y + hd.h + 14);
    });
    // feature groups
    for (const G of NG.groups) {
      const g = G.g, gx = G.x + G.labW;
      ctx.textAlign = "left"; ctx.fillStyle = TH.text; ctx.font = `400 11px ${font}`;
      ctx.fillText(g.label.toUpperCase(), G.x, G.y + 10);
      const tw = ctx.measureText(g.label.toUpperCase()).width;
      ctx.fillStyle = TH.label; ctx.font = `300 11px ${font}`;
      const room = NG.cols.inX + NG.cols.inW - (G.x + tw + 6);
      let sub = g.sub || "";
      if (ctx.measureText(sub).width > room) { // shorten to fit, with an ellipsis
        while (sub && ctx.measureText(sub + "…").width > room) sub = sub.slice(0, -1);
        sub = sub.replace(/[\s,]+$/, "") + "…";
      }
      ctx.fillText(sub, G.x + tw + 6, G.y + 10);
      if (g.col_labels) { // the kind columns of the object slots get colour keys
        ctx.font = `300 ${G.c < 13 ? 8 : 9}px ${font}`; ctx.textAlign = "center";
        const KC = { sol: "solid", haz: "hazard", orb: "orb", pad: "pad", por: "portal", oth: "other" };
        g.col_labels[0].forEach((lab, ci) => {
          const fi = g.grid[0][ci];
          if (fi == null) return;
          const cx = NG.cells[3 * fi] + G.c / 2, ly = G.colLabY + 8;
          if (g.id === "slots" && KC[lab]) { ctx.fillStyle = KIND_KEY[KC[lab]]; ctx.fillRect(cx - 3, ly - 6, 6, 6); }
          else { ctx.fillStyle = TH.label; ctx.fillText(lab, cx, ly); }
        });
      }
      ctx.font = `300 10px ${font}`; ctx.textAlign = "right"; ctx.fillStyle = TH.label;
      if (g.row_labels) g.row_labels.forEach((lab, ri) => { ctx.fillText(lab, gx - 5, G.gridY + ri * G.c + G.c / 2 + 3.5); });
      else if (g.id === "player") { ctx.fillText("now", gx - 5, G.gridY + G.c / 2 + 3.5); }
      if (g.axis) { // under the grid (timing, terrain, position)
        ctx.fillStyle = TH.dim; ctx.font = `300 9.5px ${font}`;
        const yy = G.bottom + 10, xL = gx, xR = gx + G.cols * G.c;
        ctx.textAlign = "left"; ctx.fillText(fmtAxis(g.axis[0], g), xL, yy);
        ctx.textAlign = "right"; ctx.fillText(fmtAxis(g.axis[1], g) + " " + (g.axis_unit || ""), xR, yy);
      }
    }
    // outputs and value labels
    ctx.textAlign = "left";
    NG.out.forEach(o => { ctx.fillStyle = TH.text; ctx.font = `300 13px ${font}`; ctx.fillText(o.label, o.x + o.r + 10, o.y - 3); });
    ctx.font = `300 12px ${font}`; ctx.fillStyle = TH.label;
    ctx.fillText("Value V(s)", NG.value.x, NG.value.y - 10);
    ctx.font = `300 11px ${font}`; ctx.fillStyle = TH.dim;
    ctx.fillText("separate critic network", NG.value.x, NG.value.y + 40);
    NG.static = off.cv;
  }
  function fmtAxis(v, g) {
    if (g.axis_log) return v >= 1000 ? (v / 1000).toFixed(0) + "k" : String(Math.round(v));
    return (v > 0 && g.id === "timing" ? "+" : "") + Math.round(v);
  }

  /* ================================================================ network draw */
  function bez(x0, y0, x1, y1) { const dx = (x1 - x0) * 0.5; return [x0, y0, x0 + dx, y0, x1 - dx, y1, x1, y1]; }
  function nodePos(l, i) { // l: -1 feature, 0..nh-1 hidden, nh output
    if (l === -1) { const c = NG.cells[3 * i + 2]; return [NG.port[i], NG.cells[3 * i + 1] + c / 2]; }
    if (l < S.T.nh) { const hd = NG.hid[l]; return [hd.pos[2 * i], hd.pos[2 * i + 1]]; }
    return [NG.out[i].x, NG.out[i].y];
  }
  function drawNet() {
    const T = S.T, C = CV.net; if (!T || !C || !NG.static) return;
    const ctx = C.ctx, { w, h } = C, f = S.f, nh = T.nh;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(NG.static, 0, 0, w, h);
    // links: colour = sign of input x weight (weak ones fade toward grey), width = its size
    const edges = selectEdges(T, f);
    ctx.save();
    ctx.lineCap = "butt";
    // while playing, links are dashed and the dashes move toward the output with the
    // decision counter (the Playground moves its link dashes with the epoch counter)
    const flowing = S.playing && !reduceMotion;
    if (flowing) { ctx.setLineDash([9, 3]); ctx.lineDashOffset = -S.ff * 1.5; }
    const order = edges.slice().sort((a, b) => a.s - b.s); // strong links on top
    for (const e of order) {
      const [x0, y0] = nodePos(e.l - 1, e.i);
      const [x1, y1] = e.l === nh ? [NG.out[e.j].x - NG.out[e.j].r, NG.out[e.j].y] : nodePos(e.l, e.j);
      const b = bez(x0 + (e.l === 0 ? 1 : NG.hid[0].r), y0, x1 - (e.l === nh ? 0 : NG.hid[0].r), y1);
      ctx.strokeStyle = divColor((e.c >= 0 ? 1 : -1) * (0.25 + 0.75 * e.s));
      ctx.lineWidth = 0.75 + 3.5 * e.s;
      ctx.globalAlpha = 0.55 + 0.45 * e.s;
      ctx.beginPath(); ctx.moveTo(b[0], b[1]); ctx.bezierCurveTo(b[2], b[3], b[4], b[5], b[6], b[7]); ctx.stroke();
    }
    ctx.restore();
    // feature cells (cells that feed a drawn link get an outline in its colour)
    const x = obsAt(T, f);
    for (let i = 0; i < T.D; i++) {
      const c = NG.cells[3 * i + 2];
      ctx.fillStyle = divColor(x[i]);
      ctx.fillRect(NG.cells[3 * i] + 0.5, NG.cells[3 * i + 1] + 0.5, c - 1, c - 1);
    }
    ctx.lineWidth = 1.5;
    for (const e of edges) {
      if (e.l !== 0 || e.s < 0.12) continue;
      const c = NG.cells[3 * e.i + 2];
      ctx.strokeStyle = divColor((e.c >= 0 ? 1 : -1) * (0.4 + 0.6 * e.s));
      ctx.strokeRect(NG.cells[3 * e.i] + 0.75, NG.cells[3 * e.i + 1] + 0.75, c - 1.5, c - 1.5);
    }
    // hidden neurons: squares with a dark border
    for (let l = 0; l < nh; l++) {
      const hd = NG.hid[l], a = hAt(T, l, f), r = hd.r;
      for (let i = 0; i < hd.n; i++) {
        ctx.fillStyle = divColor(a[i]);
        ctx.fillRect(hd.pos[2 * i] - r, hd.pos[2 * i + 1] - r, 2 * r, 2 * r);
      }
      ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.lineWidth = 1; ctx.beginPath();
      for (let i = 0; i < hd.n; i++) ctx.rect(hd.pos[2 * i] - r + 0.5, hd.pos[2 * i + 1] - r + 0.5, 2 * r - 1, 2 * r - 1);
      ctx.stroke();
    }
    // outputs: Release (orange side) and Jump (blue side), shaded by their probability
    const p = T.p[f], act = T.act[f];
    NG.out.forEach((o, k) => {
      const pk = k ? p : 1 - p;
      ctx.fillStyle = divColor(k ? pk : -pk);
      ctx.fillRect(o.x - o.r, o.y - o.r, 2 * o.r, 2 * o.r);
      ctx.lineWidth = act === k ? 2.5 : 1; ctx.strokeStyle = act === k ? "#000" : "rgba(0,0,0,0.55)";
      ctx.strokeRect(o.x - o.r, o.y - o.r, 2 * o.r, 2 * o.r);
      // label above, then a bar with the probability at its right end
      const bx = o.x + o.r + 10, by = o.y + 4, room = Math.max(60, CV.net.w - 6 - bx), bw = room - 46;
      ctx.fillStyle = rgba(TH.mid, 1); ctx.fillRect(bx, by, bw, 4);
      ctx.fillStyle = divColor(k ? 1 : -1); ctx.fillRect(bx, by, bw * pk, 4);
      ctx.fillStyle = TH.ink; ctx.font = `400 12px ${TH.font}`; ctx.textAlign = "right";
      ctx.fillText(fmtPct(pk), bx + room, by + 5);
      ctx.textAlign = "left";
    });
    // value: a number, as the Playground shows its loss
    const V = T.value[f], vb = NG.value, mid = vb.x + vb.w / 2, halfW = vb.w / 2;
    ctx.fillStyle = TH.ink; ctx.font = `300 22px ${TH.font}`; ctx.textAlign = "left";
    ctx.fillText(fmtS(V), vb.x, vb.y + 14);
    ctx.fillStyle = rgba(TH.mid, 1); ctx.fillRect(vb.x, vb.y + 22, vb.w, 3);
    const vw = clamp(V / T.vabs, -1, 1) * halfW;
    ctx.fillStyle = TH.val; ctx.fillRect(Math.min(mid, mid + vw), vb.y + 22, Math.abs(vw), 3);
    ctx.fillStyle = TH.label; ctx.fillRect(mid, vb.y + 19, 1, 9);
    // hover (grey) / pinned (black) outlines
    for (const sel of [S.pin, S.hover && S.hover.net ? S.hover : null]) {
      if (!sel) continue;
      ctx.strokeStyle = sel === S.pin ? "#000" : "#666"; ctx.lineWidth = 2;
      if (sel.layer === -1) { const c = NG.cells[3 * sel.i + 2]; ctx.strokeRect(NG.cells[3 * sel.i] - 1, NG.cells[3 * sel.i + 1] - 1, c + 1, c + 1); }
      else if (sel.layer < nh) { const hd = NG.hid[sel.layer], r = hd.r + 2; ctx.strokeRect(hd.pos[2 * sel.i] - r, hd.pos[2 * sel.i + 1] - r, 2 * r, 2 * r); }
      else { const o = NG.out[sel.i], r = o.r + 3; ctx.strokeRect(o.x - r, o.y - r, 2 * r, 2 * r); }
    }
    // the slot hovered in the game panel: outline its feature row
    if (S.hover && S.hover.slot != null) {
      const g = T.groups.find(gg => gg.id === "slots");
      if (g) { const row = g.grid[S.hover.slot], a = row[0], b = row[row.length - 1], c = NG.cells[3 * a + 2]; ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5; ctx.strokeRect(NG.cells[3 * a] - 2, NG.cells[3 * a + 1] - 2, NG.cells[3 * b] - NG.cells[3 * a] + c + 3, c + 3); }
    }
  }
  function netHit(mx, my) {
    const T = S.T; if (!T || !NG.hid) return null;
    for (let i = 0; i < T.D; i++) { const x = NG.cells[3 * i], y = NG.cells[3 * i + 1], c = NG.cells[3 * i + 2]; if (mx >= x && mx < x + c && my >= y && my < y + c) return { net: true, layer: -1, i }; }
    for (let l = 0; l < T.nh; l++) {
      const hd = NG.hid[l];
      if (mx < hd.x - 6 || mx > hd.x + hd.w + 6 || my < hd.y - 6 || my > hd.y + hd.h + 6) continue;
      let best = -1, bd = Infinity;
      for (let i = 0; i < hd.n; i++) { const d = (hd.pos[2 * i] - mx) ** 2 + (hd.pos[2 * i + 1] - my) ** 2; if (d < bd) { bd = d; best = i; } }
      if (bd <= (hd.r + 4) ** 2) return { net: true, layer: l, i: best };
    }
    for (let k = 0; k < 2; k++) { const o = NG.out[k]; if (Math.abs(o.x - mx) <= o.r + 6 && Math.abs(o.y - my) <= o.r + 6) return { net: true, layer: T.nh, i: k }; }
    return null;
  }

  /* ================================================================ tooltips */
  const tip = $("#tip");
  function showTip(html, ev) {
    tip.innerHTML = html; tip.style.opacity = 1;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = ev.clientX + 16, y = ev.clientY + 14;
    if (x + w > innerWidth - 8) x = ev.clientX - w - 14;
    if (y + h > innerHeight - 8) y = Math.max(8, innerHeight - h - 8);
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  const hideTip = () => { tip.style.opacity = 0; };
  const row = (color, label, val) => `<div class="r"><i style="background:${color}"></i><span>${esc(label)}</span><b>${esc(val)}</b></div>`;
  function featValue(T, i, v) {
    const fe = T.feats[i];
    if (fe.kind === "bool" || fe.kind === "onehot") return v > 0.5 ? "yes" : "no";
    if (fe.kind === "bump") return v.toFixed(2) + " (bump height)";
    if (fe.kind === "wave") return v.toFixed(3);
    if (fe.scale) { const u = v * fe.scale; return `${v.toFixed(3)} → ${Math.abs(u) >= 100 ? Math.round(u) : u.toFixed(1)} ${fe.unit || ""}${Math.abs(v) >= 0.999 ? " (clipped)" : ""}`; }
    return v.toFixed(3);
  }
  function featName(T, i) { return T.feats[i] ? T.feats[i].label : "input " + i; }
  function neuronName(T, l, i) { return l === -1 ? featName(T, i) : l < T.nh ? `H${l + 1} neuron ${i}` : ["Release", "Jump"][i]; }
  function netTip(hit, ev) {
    const T = S.T, f = S.f, nh = T.nh, G = gradients(T, f);
    let html = "";
    if (hit.layer === -1) {
      const v = obsAt(T, f)[hit.i], fe = T.feats[hit.i];
      const gi = G.gx[hit.i] * v;
      html = `<div class="t">${esc(fe.label)}</div><div class="s">${esc(T.featGroupLabel[hit.i])} · input ${hit.i} · <span class="mono">${esc(fe.id)}</span></div><hr>`;
      html += row(divColor(v), "value", featValue(T, hit.i, v));
      html += row(gi >= 0 ? rgba(TH.pos, 1) : rgba(TH.neg, 1), "gradient × input", fmtS(gi, 3));
      html += `<div class="s" style="margin-top:4px">Positive means this input, as it is now, nudges the decision toward Jump.</div>`;
    } else if (hit.layer < nh) {
      const l = hit.layer, a = hAt(T, l, f)[hit.i], W = T.layers[l], src = srcAct(T, l, f), row0 = hit.i * W.nIn;
      const contrib = [];
      for (let q = 0; q < W.nIn; q++) { const c = W.W[row0 + q] * src[q]; if (c) contrib.push([c, q]); }
      contrib.sort((p, q) => Math.abs(q[0]) - Math.abs(p[0]));
      const z = W.b[hit.i] + contrib.reduce((s2, c) => s2 + c[0], 0);
      const effect = G.gh[l][hit.i] * a;
      html = `<div class="t">Hidden layer ${l + 1} · neuron ${hit.i}</div><div class="s">tanh(${fmtS(z)}) = <b style="color:var(--text)">${fmtS(a)}</b> · effect on the decision ${fmtS(effect, 3)} (${effect >= 0 ? "toward Jump" : "toward Release"})</div><hr><div class="s">Largest inputs × weights now:</div>`;
      for (const [c, q] of contrib.slice(0, 6)) html += row(c >= 0 ? rgba(TH.pos, 1) : rgba(TH.neg, 1), neuronName(T, l - 1, q), fmtS(c));
      html += `<div class="s" style="margin-top:4px">Click to pin it and plot it on the timeline.</div>`;
    } else {
      const k = hit.i, out = T.layers[nh], hl = hAt(T, nh - 1, f), contrib = [];
      for (let q = 0; q < out.nIn; q++) contrib.push([out.W[k * out.nIn + q] * hl[q], q]);
      contrib.sort((p, q) => Math.abs(q[0]) - Math.abs(p[0]));
      const lg = T.logits[2 * f + k], pk = k ? T.p[f] : 1 - T.p[f];
      html = `<div class="t">${k ? "Jump" : "Release"} output</div><div class="s">logit ${fmtS(lg)} · softmax probability <b style="color:var(--text)">${fmtPct(pk)}</b></div><hr><div class="s">Largest hidden-2 contributions:</div>`;
      for (const [c, q] of contrib.slice(0, 6)) html += row(c >= 0 ? rgba(TH.pos, 1) : rgba(TH.neg, 1), `H${nh} neuron ${q}`, fmtS(c));
    }
    showTip(html, ev);
  }

  /* ================================================================ side panel */
  function buildWhy() {
    const T = S.T;
    $("#why").innerHTML = T.ko.map((g, q) => `<div class="wrow" data-q="${q}"><div class="wl"><span class="kd"></span>${esc(g.label.replace("Object ", "Obj "))}</div><div class="wd"></div><div class="wb"><i></i></div><div class="wv">-</div></div>`).join("");
    S.whyRows = [...$("#why").children].map(el => ({ el, kd: el.querySelector(".kd"), small: el.querySelector(".wd"), bar: el.querySelector(".wb i"), val: el.querySelector(".wv") }));
  }
  function updateSide() {
    const T = S.T, f = S.f, p = T.p[f], act = T.act[f];
    $("#d-index").textContent = `#${fmtInt(f + 1)} of ${fmtInt(T.N)} · ${fmtTime(T.time[f])}`;
    $("#d-name").textContent = act ? "Jump" : "Release";
    $("#d-sw").style.background = divColor(act ? 1 : -1);
    $("#d-asub").textContent = act ? "button held for the next 4 ticks" : "button up for the next 4 ticks";
    $("#d-pfill").style.width = (p * 100).toFixed(2) + "%";
    $("#d-p").textContent = "p(jump) " + fmtPct(p);
    $("#d-logit").textContent = fmtS(T.logits[2 * f + 1] - T.logits[2 * f]);
    const V = T.value[f];
    $("#d-value").textContent = fmtS(V);
    // knock-outs
    const K = knockouts(T, f);
    let topQ = -1, topV = 0;
    K.rows.forEach((r, q) => { const v = act ? r.dp : -r.dp; if (!r.empty && v > topV) { topV = v; topQ = q; } });
    K.rows.forEach((r, q) => {
      const R = S.whyRows[q]; if (!R) return;
      const g = r.g, half2 = clamp(r.dp, -1, 1) * 50;
      R.bar.style.left = (half2 >= 0 ? 50 : 50 + half2) + "%"; R.bar.style.width = Math.max(Math.abs(half2), r.dp ? 0.6 : 0) + "%";
      R.bar.style.background = divColor(r.dp >= 0 ? 1 : -1);
      R.val.textContent = Math.abs(r.dp) < 0.005 ? "0" : fmtS(r.dp, 2);
      R.el.classList.toggle("top", q === topQ && topV > 0.05);
      R.el.classList.toggle("empty", r.empty);
      if (g.slot != null) {
        const d = slotDesc(T, f, g.slot);
        R.small.textContent = d ? d.text : "empty slot";
        R.kd.style.background = d ? KIND_KEY[d.kind] || "#999" : "transparent";
      } else { R.small.textContent = g.id === "player" ? `${MODE_NAMES[T.mode[f]]}, ${flagAt(T, f, FL.ON_GROUND) ? "on the ground" : "in the air"}` : g.id === "terrain" ? `${T.enc.profile_dx.length} columns ahead` : "sin/cos of x"; R.kd.style.background = "transparent"; }
    });
    // one-sentence reading
    let st = "";
    if (topQ >= 0 && topV > 0.15) {
      const r = K.rows[topQ], d = r.g.slot != null ? slotDesc(T, f, r.g.slot) : null;
      const what = d ? `<b>${esc(r.g.label.toLowerCase())}</b> (${esc(d.text)})` : `<b>${esc(r.g.label.toLowerCase())}</b>`;
      st = act ? `The jump hangs on ${what}: hide it and p(jump) falls from ${fmtPct(K.base)} to ${fmtPct(r.p)}.`
        : `Holding off is driven by ${what}: hide it and p(jump) rises from ${fmtPct(K.base)} to ${fmtPct(r.p)}.`;
    } else {
      st = `No single input group decides this one: p(jump) is ${fmtPct(K.base)} and stays on the same side whichever group is hidden, so the decision comes from several signals together.`;
    }
    $("#story").innerHTML = st;
    // transport readout
    const x = T.x[f];
    $("#readout").innerHTML = `x <b>${fmtInt(x)}</b> of ${fmtInt(T.meta.level_length)} · ${MODE_NAMES[T.mode[f]]} · ${speedInfo(T.xs[f])[1]} speed`;
    $("#c-decision").textContent = `${fmtInt(f + 1)} / ${fmtInt(T.N)}`;
    $("#c-time").textContent = fmtTime(T.time[f]);
    $("#c-progress").textContent = `${(100 * x / (T.meta.level_length || 1)).toFixed(1)}%`;
    updateHud();
  }

  /* ================================================================ timeline */
  const TL = {};
  function tlLayout() {
    const { w, h } = CV.tl;
    const L = { left: 76, right: 10, w, h };
    const lanes = S.pin ? [["ev", 16], ["mode", 9], ["grav", 3], ["gap", 5], ["p", 36], ["gap", 5], ["v", 26], ["gap", 5], ["pin", 22], ["axis", 15]]
      : [["ev", 18], ["mode", 10], ["grav", 3], ["gap", 7], ["p", 46], ["gap", 7], ["v", 34], ["axis", 16]];
    const fixed = lanes.reduce((a, l) => a + l[1], 0), extra = Math.max(0, h - 6 - fixed);
    let y = 4;
    for (const [k, hh0] of lanes) { const hh = hh0 + (k === "p" ? extra * 0.6 : k === "v" ? extra * 0.4 : 0); L[k] = [y, hh]; y += hh; }
    L.iw = w - L.left - L.right;
    return L;
  }
  const tx = f => TL.L.left + (S.T.N > 1 ? f / (S.T.N - 1) : 0) * TL.L.iw;
  function renderTimelineStatic() {
    const T = S.T, C = CV.tl; if (!T || !C) return;
    const { w, h, dpr } = C, L = (TL.L = tlLayout());
    const off = offscreen(w, h, dpr), ctx = off.ctx, N = T.N;
    const font = TH.font;
    ctx.textAlign = "right";
    const lab = (k, txt, sub) => { const [y, hh] = L[k]; ctx.fillStyle = TH.text; ctx.font = `300 12px ${font}`; ctx.fillText(txt, L.left - 10, y + hh / 2 + (sub ? -1 : 4)); if (sub) { ctx.fillStyle = TH.label; ctx.font = `300 10px ${font}`; ctx.fillText(sub, L.left - 10, y + hh / 2 + 11); } };
    lab("mode", "mode"); lab("p", "p(jump)", "held ▬"); lab("v", "value V(s)");
    if (S.pin) lab("pin", pinLabel(T, S.pin).slice(0, 14));
    // mode band + gravity strip + held strip (runs of equal values)
    const runs = (arr, fn) => { let a = 0; for (let f = 1; f <= N; f++) if (f === N || arr[f] !== arr[a]) { fn(a, f, arr[a]); a = f; } };
    const [my, mh] = L.mode;
    runs(T.mode, (a, b, m) => { ctx.fillStyle = TH.modes[m] || "#999"; ctx.fillRect(tx(a), my, Math.max(1, tx(b) - tx(a) - 1), mh); });
    const grav = new Uint8Array(N); for (let f = 0; f < N; f++) grav[f] = (T.flags[f] & FL.UPSIDE_DOWN) ? 1 : 0;
    const [gy, gh] = L.grav;
    runs(grav, (a, b, v) => { if (v) { ctx.fillStyle = "#bbb"; ctx.fillRect(tx(a), gy, Math.max(1, tx(b) - tx(a)), gh); } });
    // p(jump): per-pixel envelope, area from 0
    const [py, ph] = L.p, pyb = py + ph;
    ctx.strokeStyle = TH.soft; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L.left, py + 0.5); ctx.lineTo(w - L.right, py + 0.5); ctx.stroke();
    ctx.strokeStyle = TH.line2; ctx.beginPath(); ctx.moveTo(L.left, pyb - 0.5); ctx.lineTo(w - L.right, pyb - 0.5); ctx.stroke();
    ctx.fillStyle = TH.label; ctx.font = `300 10px ${font}`; ctx.textAlign = "right"; ctx.fillText("1", L.left - 2, py + 8); ctx.fillText("0", L.left - 2, pyb - 1);
    const cols = Math.max(1, Math.floor(L.iw));
    const env = (arr, fn) => { for (let c = 0; c < cols; c++) { const a = Math.floor(c / cols * N), b = Math.max(a + 1, Math.floor((c + 1) / cols * N)); let lo = Infinity, hi = -Infinity, s = 0; for (let f = a; f < b && f < N; f++) { const v = arr[f]; if (v < lo) lo = v; if (v > hi) hi = v; s += v; } fn(c, lo, hi, s / (b - a)); } };
    ctx.fillStyle = rgba(TH.pos, 0.22);
    env(T.p, (c, lo, hi) => { const y0 = pyb - hi * (ph - 4); ctx.fillRect(L.left + c, y0, 1, pyb - y0); });
    ctx.fillStyle = rgba(TH.pos, 1);
    env(T.p, (c, lo, hi, m) => { const y0 = pyb - hi * (ph - 4); ctx.fillRect(L.left + c, y0, 1, 1.2); });
    // 0.5 threshold
    ctx.strokeStyle = TH.soft; ctx.beginPath(); ctx.moveTo(L.left, pyb - 0.5 * (ph - 4) + 0.5); ctx.lineTo(w - L.right, pyb - 0.5 * (ph - 4) + 0.5); ctx.stroke();
    // held strip under the lane
    runs(T.act, (a, b, v) => { if (v) { ctx.fillStyle = rgba(TH.pos, 1); ctx.fillRect(tx(a), pyb + 1, Math.max(1, tx(b) - tx(a)), 3); } });
    // value lane
    const [vy, vh] = L.v, vz = vy + vh / 2, vs = (vh / 2 - 2) / T.vabs;
    ctx.strokeStyle = TH.line2; ctx.beginPath(); ctx.moveTo(L.left, vz + 0.5); ctx.lineTo(w - L.right, vz + 0.5); ctx.stroke();
    ctx.fillStyle = TH.val;
    env(T.value, (c, lo, hi) => { const a = vz - hi * vs, b = vz - lo * vs; ctx.fillRect(L.left + c, a, 1, Math.max(1.2, b - a)); });
    ctx.fillStyle = TH.label; ctx.font = `300 10px ${font}`; ctx.textAlign = "right"; ctx.fillText(fmtS(T.vabs, 1), L.left - 2, vy + 8); ctx.fillText("0", L.left - 2, vz + 3.5);
    // pinned neuron / input lane
    if (S.pin) {
      const [qy, qh] = L.pin, qz = qy + qh / 2, arr = pinSeries(T, S.pin);
      ctx.strokeStyle = TH.line2; ctx.beginPath(); ctx.moveTo(L.left, qz + 0.5); ctx.lineTo(w - L.right, qz + 0.5); ctx.stroke();
      env(arr, (c, lo, hi, m) => { ctx.fillStyle = divColor(m); const a = qz - hi * (qh / 2 - 1), b = qz - lo * (qh / 2 - 1); ctx.fillRect(L.left + c, a, 1, Math.max(1.2, b - a)); });
    }
    // events lane: mode changes (labelled), speed, size, dual; gravity flips as ticks
    const [ey, eh] = L.ev;
    let lastLabelX = -1e9;
    for (const e of T.events) {
      const X = tx(Math.min(e.f, N - 1));
      if (e.type === "gravity") { ctx.fillStyle = "#c8c8c8"; ctx.fillRect(X, ey + eh - 5, 1, 5); continue; }
      if (e.type === "mode") {
        const m = T.raw.modes.indexOf(e.to), c = TH.modes[m] || "#999";
        ctx.fillStyle = c; ctx.beginPath(); ctx.moveTo(X, ey + eh - 1); ctx.lineTo(X - 4, ey + eh - 8); ctx.lineTo(X + 4, ey + eh - 8); ctx.closePath(); ctx.fill();
        if (X - lastLabelX > 34) { ctx.fillStyle = TH.text; ctx.font = `300 10px ${font}`; ctx.textAlign = "left"; ctx.fillText(MODE_NAMES[m] || e.to, X + 3, ey + 8); lastLabelX = X; }
      } else if (e.type === "speed") {
        ctx.strokeStyle = "#999"; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(X - 2, ey + eh - 9); ctx.lineTo(X + 1, ey + eh - 5); ctx.lineTo(X - 2, ey + eh - 1); ctx.stroke();
      } else if (e.type === "size") { ctx.fillStyle = "#999"; ctx.beginPath(); ctx.arc(X, ey + eh - 4, 2, 0, Math.PI * 2); ctx.fill(); }
      else if (e.type === "dual") { ctx.strokeStyle = "#777"; ctx.lineWidth = 1; ctx.strokeRect(X - 2.5, ey + eh - 7.5, 5, 5); }
    }
    // time axis
    const [ay] = L.axis;
    ctx.font = `300 11px ${font}`; ctx.textAlign = "center";
    const dur = T.dur, step = niceStep(dur / Math.max(2, L.iw / 90));
    for (let t = 0; t <= dur + 1e-6; t += step) {
      const f = timeToFrame(T, t), X = tx(f);
      ctx.fillStyle = TH.line2; ctx.fillRect(X, ay, 1, 3); ctx.fillStyle = TH.label;
      ctx.fillText(t >= 60 ? `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, "0")}` : `${Math.round(t)} s`, X, ay + 12);
    }
    TL.static = off.cv;
  }
  function niceStep(raw) { const p = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / p; return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * p; }
  function timeToFrame(T, t) { let a = 0, b = T.N - 1; while (a < b) { const m = (a + b) >> 1; if (T.time[m] < t) a = m + 1; else b = m; } return a; }
  function pinSeries(T, pin) {
    const N = T.N, out = new Float32Array(N);
    if (pin.layer === -1) for (let f = 0; f < N; f++) out[f] = T.obs[f * T.D + pin.i];
    else if (pin.layer < T.nh) { const n = T.sizes[pin.layer + 1]; for (let f = 0; f < N; f++) out[f] = T.H[pin.layer][f * n + pin.i]; }
    else for (let f = 0; f < N; f++) out[f] = pin.i ? T.p[f] * 2 - 1 : 1 - 2 * T.p[f];
    return out;
  }
  function pinLabel(T, pin) { return pin.layer === -1 ? (T.feats[pin.i] ? T.feats[pin.i].id : "input") : pin.layer < T.nh ? `H${pin.layer + 1} #${pin.i}` : ["Release", "Jump"][pin.i]; }
  function drawTimeline() {
    const T = S.T, C = CV.tl; if (!T || !C || !TL.static) return;
    const ctx = C.ctx, { w, h } = C, L = TL.L;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(TL.static, 0, 0, w, h);
    // hover crosshair
    if (S.tlHover != null) {
      const X = tx(S.tlHover);
      ctx.fillStyle = "#999"; ctx.fillRect(X, L.mode[0], 1, L.axis[0] - L.mode[0]);
    }
    // playhead
    const X = tx(S.ff);
    ctx.fillStyle = TH.head; ctx.fillRect(X - 0.75, L.ev[0], 1.5, L.axis[0] - L.ev[0]);
    ctx.beginPath(); ctx.moveTo(X - 5, L.ev[0]); ctx.lineTo(X + 5, L.ev[0]); ctx.lineTo(X, L.ev[0] + 6); ctx.closePath(); ctx.fill();
    // current values beside the playhead
    const f = S.f, [py, ph] = L.p, pyb = py + ph;
    ctx.fillStyle = rgba(TH.pos, 1); ctx.beginPath(); ctx.arc(X, pyb - T.p[f] * (ph - 4), 3, 0, Math.PI * 2); ctx.fill();
    const [vy, vh] = L.v, vz = vy + vh / 2;
    ctx.fillStyle = TH.val; ctx.beginPath(); ctx.arc(X, vz - T.value[f] * (vh / 2 - 2) / T.vabs, 3, 0, Math.PI * 2); ctx.fill();
  }
  function tlFrameAt(clientX) {
    const r = CV.tl.cv.getBoundingClientRect(), L = TL.L;
    return clamp(Math.round((clientX - r.left - L.left) / L.iw * (S.T.N - 1)), 0, S.T.N - 1);
  }
  function tlTip(f, ev) {
    const T = S.T, x = T.x[f], ev2 = T.events.filter(e => e.f === f && e.type !== "gravity").map(e => e.type === "mode" ? `→ ${MODE_NAMES[T.raw.modes.indexOf(e.to)]}` : e.type === "speed" ? `speed ${speedInfo(e.speed)[1]}` : e.type === "size" ? (e.mini ? "mini" : "normal size") : e.type === "dual" ? (e.on ? "dual" : "solo") : "").filter(Boolean);
    let html = `<div class="t">${fmtTime(T.time[f])} · decision ${fmtInt(f + 1)}</div><div class="s">x ${fmtInt(x)} · ${(100 * x / (T.meta.level_length || 1)).toFixed(1)}% · ${MODE_NAMES[T.mode[f]]}${flagAt(T, f, FL.UPSIDE_DOWN) ? " · gravity flipped" : ""}</div><hr>`;
    html += row(rgba(TH.pos, 1), "p(jump)", fmtPct(T.p[f]));
    html += row(TH.val, "value V(s)", fmtS(T.value[f]));
    html += row(T.act[f] ? rgba(TH.pos, 1) : rgba(TH.neg, 1), "action", T.act[f] ? "jump (held)" : "release");
    if (S.pin) { const v = pinSeries(T, S.pin)[f]; html += row(divColor(v), pinLabel(T, S.pin), fmtS(v)); }
    if (ev2.length) html += `<div class="s" style="margin-top:3px">${esc(ev2.join(" · "))}</div>`;
    showTip(html, ev);
  }

  /* ================================================================ scene hover */
  function sceneHit(mx, my) {
    const T = S.T; if (!T || !SC.vis) return null;
    const O = T.O, s = SC.s;
    const test = list => {
      for (let q = list.length - 1; q >= 0; q--) {
        const i = list[q], l = wx(O.x[i] - O.w[i] / 2) - 4, t = wy(O.y[i] + O.h[i] / 2) - 4, W = O.w[i] * s + 8, H = O.h[i] * s + 8;
        if (mx >= l && mx <= l + W && my >= t && my <= t + H) return i;
      }
      return -1;
    };
    let i = test(SC.vis), ghost = false;
    if (i < 0) { i = test(SC.ghosts); ghost = i >= 0; }
    if (i < 0) return null;
    const f = S.f, k = T.enc.k;
    let slot = null;
    for (let q = 0; q < k; q++) if (T.slots[f * k + q] === i) { slot = q; break; }
    return { obj: i, ghost, slot };
  }
  function sceneTip(hit, ev) {
    const T = S.T, O = T.O, i = hit.obj, f = S.f;
    const kind = T.O.names.kind[O.kind[i]] || "object", sub = T.O.names.sub[O.subtype[i]] || "";
    const dx = O.x[i] - O.w[i] / 2 - T.x[f], dy = O.y[i] - T.y[f];
    let html = `<div class="t">${esc(kind)}${sub && sub !== "unknown" && sub !== kind ? " · " + esc(sub.replace(/_/g, " ")) : ""}</div>`;
    html += `<div class="s">${hit.slot != null ? `in encoder slot <b style="color:var(--text)">${hit.slot + 1}</b>` : hit.ghost ? "not in this frame's report (seen in other frames)" : "reported by the bridge, not in a slot"}</div><hr>`;
    html += row("#7f93d8", "left edge vs player", `${fmtS(dx, 0)} u`);
    html += row("#7f93d8", "centre height vs player", `${fmtS(dy, 0)} u`);
    html += row("#7f93d8", "hitbox", `${O.w[i].toFixed(1)} × ${O.h[i].toFixed(1)} u`);
    html += row("#7f93d8", "position", `x ${fmtInt(O.x[i])}, y ${fmtInt(O.y[i])}`);
    if (O.oflags[i] & 4) html += `<div class="s">hidden in the editor (still collides)</div>`;
    showTip(html, ev);
  }

  /* ================================================================ playback */
  function setFrame(f, keepPlaying) {
    const T = S.T; if (!T) return;
    S.ff = clamp(f, 0, T.N - 1); S.f = Math.floor(S.ff);
    if (!keepPlaying) setPlaying(false);
    S.dirty = true; scheduleHash();
  }
  function setPlaying(on) {
    const T = S.T;
    if (on && T && S.f >= T.N - 1) { S.ff = 0; S.f = 0; }
    S.playing = !!on;
    $("#b-play").setAttribute("aria-label", on ? "Pause" : "Play");
    $("#i-play").innerHTML = on ? '<path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/>' : '<path d="M8 5l11 7-11 7z"/>';
    if (!on) scheduleHash();
  }
  function setSpeed(v) {
    S.speed = v; savePrefs();
    $("#speed").value = String(v);
    S.dirty = true;
  }
  let hashT = 0;
  function scheduleHash() {
    clearTimeout(hashT);
    hashT = setTimeout(() => {
      if (!S.T || S.playing) return;
      const qs = new URLSearchParams(location.search);
      if (!S.embedded) qs.set("trace", S.name);
      qs.delete("autoplay"); // a reload should not start playing again
      const q = qs.toString() ? "?" + qs.toString() : "";
      try { history.replaceState(null, "", `${location.pathname}${q}#f=${S.f + 1}`); } catch (e) { }
    }, 350);
  }
  function jumpTo(list, dir) {
    const f = S.f;
    if (dir > 0) { const n = list.find(v => v > f); if (n != null) setFrame(n); }
    else { let p = null; for (const v of list) { if (v < f) p = v; else break; } if (p != null) setFrame(p); }
  }
  let lastT = 0;
  function loop(t) {
    requestAnimationFrame(loop);
    const T = S.T; if (!T) return;
    const dt = lastT ? Math.min(0.25, (t - lastT) / 1000) : 0; lastT = t;
    if (S.playing && !S.dragging) {
      S.ff += dt * DPS * S.speed;
      if (S.ff >= T.N - 1) { S.ff = T.N - 1; setPlaying(false); }
      S.dirty = true;
    }
    const f = Math.floor(S.ff), changed = f !== S.f || S.dirty;
    S.f = f;
    // idle: redraw at ~30 fps for the edge animation only
    if (!changed && !S.playing && t - S.lastDraw < 33) return;
    S.lastDraw = t;
    const t0 = performance.now();
    drawScene(S.ff);
    drawNet();
    drawTimeline();
    // the text panel at ~15 Hz while playing (60 Hz numbers are unreadable anyway), at once otherwise
    if (changed && (!S.playing || t - (S.lastSide || 0) > 66)) { updateSide(); S.lastSide = t; S.sideStale = false; }
    else if (changed) S.sideStale = true;
    if (!S.playing && S.sideStale) { updateSide(); S.sideStale = false; }
    S.dirty = false;
    STATS.frames++; STATS.drawMs += (performance.now() - t0 - STATS.drawMs) * 0.05;
  }

  /* ================================================================ header / meta */
  function fillHeader() {
    const T = S.T, m = T.meta;
    const diff = m.difficulty ? String(m.difficulty).replace(/_/g, " ") : "";
    $("#lvl-title").innerHTML = `${esc(m.title || S.name)}<span class="tag">${m.completed ? "completed" : `died at ${(100 * m.progress).toFixed(1)}%`}</span>`;
    $("#lvl-sub").textContent = [diff, m.level, `${fmtInt(m.decisions)} decisions`, `${m.seconds} s of level time`, `${m.presses} presses`, `greedy policy ${String(m.checkpoint || "").split(/[\\/]/).pop()}`].filter(Boolean).join(" · ");
    document.title = `${m.title || S.name} · GD-RL network`;
    const sz = T.sizes;
    $("#net-sub").textContent = `${sz.join(" → ")}, tanh · hover for details`;
    const E = T.enc, parts = [`own state (${E.player_features})`, `the ${E.k} nearest ${E.focus === "obstacles" ? "obstacles" : "objects"} (${E.k * E.slot_size})`];
    if (E.near_slots) parts.push(`a timing code for the ${E.near_slots} nearest left edges (${E.near_slots * E.rbf_centers.length})`);
    if (E.profile) parts.push(`the terrain ahead (${3 * E.profile_dx.length})`);
    if (E.position) parts.push(`the position in the level (${2 * E.position_periods.length})`);
    $("#ex-encoder").textContent = `The encoder turns that into ${E.size} numbers in [-1, 1], all measured from the player: ${parts.join(", ")}. The same situation encodes the same way anywhere in the level${E.position ? ", except for the position inputs, which let the policy learn a level's timing" : ""}.`;
    $("#ex-net").textContent = `${sz.length - 2} hidden layers (${sz.slice(1, -1).join(" and ")} neurons). Each neuron adds up its inputs times learned weights plus a bias and squashes the sum with tanh into [-1, 1]: orange is positive, blue negative. The lines show the largest inputs times weights, flowing toward the output.`;
    const meta = [["recorded", m.recorded_at], ["level", m.level], ["checkpoint", m.checkpoint], ["config", m.config], ["start", m.start_pos === -2 ? "level start" : m.start_pos], ["objects", fmtInt(T.O.n)], ["held", (100 * (m.held_fraction || 0)).toFixed(1) + "% of decisions"], ["greedy check", m.greedy_agreement === m.decisions ? "recomputed actions match all decisions" : `${m.greedy_agreement}/${m.decisions} match`]];
    $("#meta").innerHTML = meta.filter(r => r[1] != null && r[1] !== "").map(([k, v]) => `<span><b>${esc(k)}</b> ${esc(v)}</span>`).join("");
    const tr = m.training || {};
    $("#ex-caveat").textContent = tr.behaviour_cloning
      ? "This policy is PPO plus behaviour cloning: routes found by a checkpoint search were cloned into it and held in place by a cloning anchor during PPO. On this level it reproduces a learned route; it is not a general Geometry Dash player, and with random action noise it would not finish."
      : `This policy was trained with PPO only${tr.curriculum ? ", with a start-position curriculum" : ""}. It learned this level; it is not a general Geometry Dash player.`;
    $("#scene-sub").textContent = `${MODE_NAMES[T.mode[0]]} start · ${fmtInt(m.level_length)} u long`;
  }

  /* ================================================================ loading */
  function showLoading(msg, sub) {
    let el = $(".loading");
    if (!el) { el = document.createElement("div"); el.className = "loading"; document.body.appendChild(el); }
    el.innerHTML = `<div style="text-align:center"><b>${esc(msg)}</b>${esc(sub || "")}<div class="bar"><i></i></div></div>`;
    el.hidden = false;
  }
  function showMessage(msg, sub) {
    let el = $(".loading");
    if (!el) { el = document.createElement("div"); el.className = "loading"; document.body.appendChild(el); }
    el.innerHTML = `<div style="text-align:center;max-width:560px;padding:20px"><b>${esc(msg)}</b><div>${sub || ""}</div></div>`;
    el.hidden = false;
  }
  const hideLoading = () => { const el = $(".loading"); if (el) el.remove(); };
  function setTrace(raw, name) {
    const T = prepare(raw), first = !S.T;
    T.name = name;
    S.T = T; S.name = name; S.pin = null; S.hover = null; S.tlHover = null;
    KO.key = GR.key = EC.key = null;
    const m = first && /f=(\d+)/.exec(location.hash); // a deep link applies to the first run shown
    S.ff = S.f = m ? clamp(+m[1] - 1, 0, T.N - 1) : 0;
    if (!first) scheduleHash();
    fillHeader(); buildWhy(); layoutAll(); updateSide();
    savePrefs();
    hideLoading();
  }
  async function openTrace(name) {
    showLoading(`Loading ${name}`, "");
    const r = await fetch("/api/trace?name=" + encodeURIComponent(name), { cache: "no-store" });
    if (!r.ok) throw new Error(`/api/trace?name=${name}: ${r.status}`);
    setPlaying(false);
    setTrace(await r.json(), name);
  }

  /* ================================================================ events */
  function bindUI() {
    $("#btn-help").addEventListener("click", () => { $("#help").hidden = false; });
    $("#help-x").addEventListener("click", () => { $("#help").hidden = true; });
    $("#help").addEventListener("click", e => { if (e.target.id === "help") $("#help").hidden = true; });
    $("#b-play").addEventListener("click", () => setPlaying(!S.playing));
    $("#b-back").addEventListener("click", () => setFrame(S.f - 1));
    $("#b-fwd").addEventListener("click", () => setFrame(S.f + 1));
    $("#b-start").addEventListener("click", () => setFrame(0));
    $("#b-end").addEventListener("click", () => setFrame(S.T.N - 1));
    $("#b-prevpress").addEventListener("click", () => jumpTo(S.T.presses, -1));
    $("#b-nextpress").addEventListener("click", () => jumpTo(S.T.presses, 1));
    $("#b-nextevent").addEventListener("click", () => jumpTo(S.T.eventFrames, 1));
    $("#speed").addEventListener("change", e => setSpeed(+e.target.value));
    $("#edge-mode").addEventListener("change", e => setEdgeMode(e.target.value));
    document.querySelectorAll("input[data-tg]").forEach(b => {
      b.checked = !!S.tg[b.dataset.tg];
      b.addEventListener("change", () => { S.tg[b.dataset.tg] = b.checked; savePrefs(); S.dirty = true; });
    });
    $("#trace-pick").addEventListener("change", e => { openTrace(e.target.value).catch(err => showMessage("Could not load the trace", esc(err.message))); });
    // network hover / pin
    const net = $("#net");
    net.addEventListener("mousemove", e => {
      const r = net.getBoundingClientRect(), hit = netHit(e.clientX - r.left, e.clientY - r.top);
      const same = hit && S.hover && S.hover.net && S.hover.layer === hit.layer && S.hover.i === hit.i;
      if (!same) { S.hover = hit; S.dirty = true; }
      net.style.cursor = hit ? "pointer" : "default";
      if (hit) netTip(hit, e); else hideTip();
    });
    net.addEventListener("mouseleave", () => { S.hover = null; S.dirty = true; hideTip(); });
    net.addEventListener("click", e => {
      const r = net.getBoundingClientRect(), hit = netHit(e.clientX - r.left, e.clientY - r.top);
      if (!hit) { if (S.pin) { S.pin = null; renderTimelineStatic(); S.dirty = true; } return; }
      S.pin = S.pin && S.pin.layer === hit.layer && S.pin.i === hit.i ? null : { layer: hit.layer, i: hit.i };
      EC.key = null; renderTimelineStatic(); S.dirty = true;
    });
    // scene hover + zoom
    const sc = $("#scene");
    sc.addEventListener("mousemove", e => {
      const r = sc.getBoundingClientRect(), hit = sceneHit(e.clientX - r.left, e.clientY - r.top);
      const slot = hit ? hit.slot : null;
      if ((S.hover && S.hover.slot) !== slot || (!!S.hover) !== (!!hit)) { S.hover = hit ? { slot } : null; S.dirty = true; }
      if (hit) sceneTip(hit, e); else hideTip();
    });
    sc.addEventListener("mouseleave", () => { S.hover = null; S.dirty = true; hideTip(); });
    sc.addEventListener("wheel", e => { e.preventDefault(); S.zoom = clamp(S.zoom * Math.pow(1.0015, -e.deltaY), 0.5, 2.6); savePrefs(); S.dirty = true; }, { passive: false });
    // timeline scrub
    const tl = $("#tl");
    const scrub = e => { setFrame(tlFrameAt(e.clientX)); };
    tl.addEventListener("pointerdown", e => { S.dragging = true; tl.setPointerCapture(e.pointerId); scrub(e); });
    tl.addEventListener("pointermove", e => {
      if (S.dragging) scrub(e);
      const f = tlFrameAt(e.clientX); S.tlHover = f; S.dirty = true; tlTip(f, e);
    });
    tl.addEventListener("pointerup", e => { S.dragging = false; try { tl.releasePointerCapture(e.pointerId); } catch (x) { } });
    tl.addEventListener("pointerleave", () => { S.tlHover = null; S.dirty = true; hideTip(); });
    // keyboard
    document.addEventListener("keydown", e => {
      if (!S.T || e.target.closest && e.target.closest("select,input,textarea")) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key;
      if (k === " ") { e.preventDefault(); setPlaying(!S.playing); }
      else if (k === "ArrowRight") { e.preventDefault(); setFrame(S.f + (e.shiftKey ? DPS : 1)); }
      else if (k === "ArrowLeft") { e.preventDefault(); setFrame(S.f - (e.shiftKey ? DPS : 1)); }
      else if (k === "Home") { e.preventDefault(); setFrame(0); }
      else if (k === "End") { e.preventDefault(); setFrame(S.T.N - 1); }
      else if (k === "]" || k === "[") { const i = SPEED_STEPS.indexOf(S.speed); setSpeed(SPEED_STEPS[clamp(i + (k === "]" ? 1 : -1), 0, SPEED_STEPS.length - 1)]); }
      else if (k === "j" || k === "J") jumpTo(S.T.presses, -1);
      else if (k === "k" || k === "K") jumpTo(S.T.presses, 1);
      else if (k === "n" || k === "N") jumpTo(S.T.eventFrames, 1);
      else if (k === "p" || k === "P") jumpTo(S.T.eventFrames, -1);
      else if (k === "e" || k === "E") setEdgeMode(S.edgeMode === "path" ? "top" : "path");
      else if (k === "?") { $("#help").hidden = !$("#help").hidden; }
      else if (k === "Escape") { $("#help").hidden = true; if (S.pin) { S.pin = null; renderTimelineStatic(); S.dirty = true; } }
    });
    // Space already toggled playback on keydown: stop a focused button from also "clicking" on keyup.
    document.addEventListener("keyup", e => {
      if (e.key === " " && e.target.closest && e.target.closest("button") && !e.target.closest(".modal")) e.preventDefault();
    });
    // resizing
    const sizes = new Map();
    const ro = new ResizeObserver(entries => {
      let changed = false;
      for (const e of entries) {
        const k = e.target.id, v = `${e.target.clientWidth}x${e.target.clientHeight}`;
        if (sizes.get(k) !== v) { sizes.set(k, v); changed = true; }
      }
      if (changed) layoutAll();
    });
    ["#scene-wrap", "#net-wrap", "#tl-wrap"].forEach(s => ro.observe($(s)));
    document.addEventListener("visibilitychange", () => { lastT = 0; });
    window.addEventListener("hashchange", () => {
      const m = /f=(\d+)/.exec(location.hash);
      if (m && S.T && +m[1] - 1 !== S.f) setFrame(+m[1] - 1);
    });
  }
  function setEdgeMode(v) {
    S.edgeMode = v; savePrefs(); EC.key = null; S.dirty = true;
    $("#edge-mode").value = v;
  }

  /* ================================================================ boot */
  async function boot() {
    readTheme();
    bindUI();
    setSpeed(S.speed); setEdgeMode(S.edgeMode);
    const emb = document.getElementById("trace-data");
    try {
      if (emb) {
        S.embedded = true;
        $("#home").removeAttribute("href"); $("#home").classList.add("static"); $("#home").title = "";
        showLoading("Unpacking the trace", "");
        const raw = await gunzipJSON(b64Bytes(emb.textContent.trim()));
        setTrace(raw, emb.dataset.name || raw.meta.name);
      } else {
        showLoading("Loading traces", "");
        const r = await fetch("/api/traces", { cache: "no-store" });
        if (!r.ok) throw new Error("/api/traces: " + r.status);
        S.list = await r.json();
        if (!S.list.length) {
          showMessage("No traces yet", "Record one with <code>python -m gdrl.tools.nn_trace record rk_beat</code> (plays the policy in GD), or on the mock game with <code>python -m gdrl.tools.nn_trace mock --config ... --checkpoint ...</code>. Traces land in <code>results/viz/</code>.");
          return;
        }
        const want = new URLSearchParams(location.search).get("trace") || S.lastTrace;
        const name = S.list.some(t => t.name === want) ? want : S.list[0].name;
        const pick = $("#trace-pick");
        pick.innerHTML = S.list.map(t => `<option value="${esc(t.name)}" ${t.name === name ? "selected" : ""}>${esc(t.title)} · ${t.completed ? "completed" : `died ${(100 * (t.progress || 0)).toFixed(0)}%`} · ${fmtInt(t.decisions)} decisions</option>`).join("");
        $("#pick-wrap").hidden = S.list.length < 2;
        await openTrace(name);
      }
    } catch (err) {
      console.error(err);
      showMessage("Could not load the trace", esc(err.message || String(err)));
      return;
    }
    requestAnimationFrame(loop);
    if (AUTOPLAY != null) setTimeout(() => { S.ff = S.f = 0; setPlaying(true); document.title = "\u25b6 " + document.title; }, AUTOPLAY * 1000);
  }
  // For debugging in the console: the state, and a way to seek.
  window.gdrlViz = {
    state: S, stats: STATS, seek: f => setFrame(f - 1), play: on => setPlaying(on !== false),
    // ms per rendered frame over n consecutive decisions (scene + network + timeline + side panel)
    bench(n = 300, from = 0) {
      const T = S.T, parts = { scene: 0, net: 0, timeline: 0, side: 0 }, t0 = performance.now();
      for (let q = 0; q < n; q++) {
        S.ff = S.f = (from + q) % T.N;
        let t = performance.now(); drawScene(S.ff); parts.scene += performance.now() - t;
        t = performance.now(); drawNet(); parts.net += performance.now() - t;
        t = performance.now(); drawTimeline(); parts.timeline += performance.now() - t;
        t = performance.now(); updateSide(); parts.side += performance.now() - t;
      }
      const per = (performance.now() - t0) / n;
      Object.keys(parts).forEach(k => { parts[k] = +(parts[k] / n).toFixed(3); });
      return { msPerFrame: +per.toFixed(3), ...parts };
    },
  };
  boot();
})();
