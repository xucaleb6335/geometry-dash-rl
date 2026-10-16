/* GD-RL mission control: vanilla JS, no dependencies. Data: /api/state, /api/run?id=  (or window.__SNAPSHOT__). */
"use strict";
const SNAP = window.__SNAPSHOT__;
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ------------------------------------------------------------------ constants */
const TIERS = {
  mock: { label: "Mock", color: "#4dd0e1", face: "robot" },
  mock_heldout: { label: "Mock held-out", color: "#7aa2ff", face: "robot" },
  mock_demon: { label: "Mock demon", color: "#c77dff", face: "robotdemon" },
  easy: { label: "Easy", color: "#5ad1ff", face: "smile" },
  normal: { label: "Normal", color: "#6bff6b", face: "smile" },
  hard: { label: "Hard", color: "#ffc233", face: "flat" },
  harder: { label: "Harder", color: "#ff8a3d", face: "frown" },
  insane: { label: "Insane", color: "#ff4fd8", face: "angry" },
  easy_demon: { label: "Easy demon", color: "#b6ff4d", face: "demon" },
  medium_demon: { label: "Medium demon", color: "#ffd23d", face: "demon" },
  hard_demon: { label: "Hard demon", color: "#ff7a33", face: "demon" },
  insane_demon: { label: "Insane demon", color: "#ff3b7a", face: "demon" },
  extreme_demon: { label: "Extreme demon", color: "#b02bff", face: "xdemon" },
  toy: { label: "Sanity check", color: "#9a9bc4", face: "robot" },
  unknown: { label: "Unknown", color: "#9a9bc4", face: "flat" },
};
const tier = d => TIERS[d] || TIERS.unknown;
const PALETTE = ["#00e5ff", "#ff2bd6", "#7dff4f", "#ffd400", "#ff8a1f", "#9b5cff", "#4d8dff", "#ff3b5c", "#2fe0b0", "#f58cff", "#c4ff3d", "#8fb4ff"];
const NAV = [
  ["overview", "Level board", "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z"],
  ["curves", "Training curves", "M3 20l5-8 4 4 5-9 4 6M3 4v16h18"],
  ["deaths", "Death map", "M12 3a7 7 0 00-4 12.7V19h8v-3.3A7 7 0 0012 3zM9.5 11h.01M14.5 11h.01M10 19v2M14 19v2"],
  ["curriculum", "Curriculum", "M4 20V10M10 20V4M16 20v-8M22 20H2"],
  ["search", "Search / genetic", "M11 4a7 7 0 100 14 7 7 0 000-14zM21 21l-5-5"],
  ["compare", "Agent comparison", "M4 6h16M4 12h10M4 18h6"],
  ["ledger", "Beat ledger", "M12 3l2.6 5.6 6.1.7-4.5 4.2 1.2 6L12 16.4 6.6 19.5l1.2-6L3.3 9.3l6.1-.7z"],
  ["live", "Live feed", "M3 12h4l3-8 4 16 3-8h4"],
];
const VIEWS = NAV.map(n => n[0]);

/* ------------------------------------------------------------------ state */
const S = {
  state: null, version: null, view: "overview", detail: {}, detailVer: {}, lastFetchOk: 0, err: null,
  f: { q: "", diff: "", status: "", agent: "", run: "", from: "", to: "" },
  ui: { smooth: 0.6, xaxis: "step", log: false, group: true, sel: [], more: false, heatNorm: "row", heatRun: "", curRun: "",
        curStatus: "all", searchRun: "", cmpMode: "greedy", cmpMetric: "completion", collapsed: {}, theme: "dark", hideUntrained: false },
  seen: new Set(), firstFeed: true,
};
try { Object.assign(S.ui, JSON.parse(localStorage.getItem("gdrl-ui") || "{}")); } catch (e) { }
const saveUi = () => { try { localStorage.setItem("gdrl-ui", JSON.stringify(S.ui)); } catch (e) { } };

/* ------------------------------------------------------------------ formatting */
const fmtPct = (v, d = 1) => v == null ? "-" : (v * 100).toFixed(d) + "%";
function fmtNum(v) {
  if (v == null || !isFinite(v)) return "-";
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 1 : 2) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(1) + "k";
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  if (a === 0) return "0";
  if (a >= 0.001) return v.toFixed(4);
  return v.toExponential(1);
}
function fmtDur(s) {
  if (s == null || !isFinite(s)) return "-";
  if (s < 90) return Math.round(s) + "s";
  if (s < 5400) return Math.round(s / 60) + "m";
  const h = s / 3600;
  if (h < 48) return h.toFixed(1) + "h";
  return (h / 24).toFixed(1) + "d";
}
function fmtAgo(t) {
  if (!t) return "-";
  const d = Date.now() / 1000 - t;
  if (d < 0) return "now";
  if (d < 60) return Math.round(d) + "s ago";
  if (d < 3600) return Math.round(d / 60) + "m ago";
  if (d < 86400) return (d / 3600).toFixed(1) + "h ago";
  return Math.round(d / 86400) + "d ago";
}
const fmtClock = t => t ? new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "-";
const fmtDate = t => t ? new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "-";
const runLabel = r => `${r.base}${r.seed != null ? " s" + r.seed : ""} · ${fmtDate(r.created).replace(/,? /, " ")}`;
const hashStr = s => { let h = 7; for (const c of s) h = (h * 31 + c.charCodeAt(0)) >>> 0; return h; };
const runColor = r => PALETTE[hashStr(r.id) % PALETTE.length];

/* ------------------------------------------------------------------ faces */
function face(diff, big) {
  const t = tier(diff), c = t.color, k = t.face;
  const id = "g" + Math.random().toString(36).slice(2, 7);
  let s = `<svg class="face" viewBox="0 0 48 48" style="--c:${c}" aria-hidden="true"><defs><radialGradient id="${id}" cx=".35" cy=".3" r=".9"><stop offset="0" stop-color="#fff" stop-opacity=".55"/><stop offset=".45" stop-color="${c}"/><stop offset="1" stop-color="${c}" stop-opacity=".62"/></radialGradient></defs>`;
  const horns = (w = 1) => `<path d="M9 17 L7 3 L19 11Z M39 17 L41 3 L29 11Z" fill="${c}" stroke="#0a0a18" stroke-width="1.3" stroke-linejoin="round"/>`;
  if (k === "robot" || k === "robotdemon") {
    if (k === "robotdemon") s += horns();
    s += `<rect x="6" y="11" width="36" height="31" rx="8" fill="url(#${id})" stroke="#0a0a18" stroke-width="1.5"/><path d="M24 11V6" stroke="${c}" stroke-width="2"/><circle cx="24" cy="5" r="2.2" fill="#fff"/>`;
    s += `<rect x="12" y="20" width="8" height="7" rx="2" fill="#0a0a18"/><rect x="28" y="20" width="8" height="7" rx="2" fill="#0a0a18"/><rect x="14" y="22" width="3" height="3" fill="#7dff4f"/><rect x="30" y="22" width="3" height="3" fill="#7dff4f"/><path d="M14 34h20M19 34v3M24 34v3M29 34v3" stroke="#0a0a18" stroke-width="1.8" stroke-linecap="round"/>`;
  } else {
    if (k === "demon") s += horns();
    if (k === "xdemon") s += `<path d="M7 18 L3 1 L19 10Z M41 18 L45 1 L29 10Z M20 9 L24 0 L28 9Z" fill="${c}" stroke="#0a0a18" stroke-width="1.3" stroke-linejoin="round"/>`;
    s += `<circle cx="24" cy="27" r="18" fill="url(#${id})" stroke="#0a0a18" stroke-width="1.5"/>`;
    if (k === "xdemon") s += `<circle cx="17" cy="24" r="4.2" fill="#ff3b5c"/><circle cx="31" cy="24" r="4.2" fill="#ff3b5c"/><circle cx="17" cy="24" r="1.6" fill="#fff"/><circle cx="31" cy="24" r="1.6" fill="#fff"/><path d="M12 19l9 3M36 19l-9 3" stroke="#0a0a18" stroke-width="2.4" stroke-linecap="round"/>`;
    else s += `<ellipse cx="17" cy="24" rx="4" ry="4.6" fill="#fff"/><ellipse cx="31" cy="24" rx="4" ry="4.6" fill="#fff"/><circle cx="17.6" cy="25" r="2" fill="#0a0a18"/><circle cx="30.4" cy="25" r="2" fill="#0a0a18"/>`;
    if (k === "smile") s += `<path d="M14 33q10 9 20 0" stroke="#0a0a18" stroke-width="2.4" fill="none" stroke-linecap="round"/>`;
    if (k === "flat") s += `<path d="M16 35h16" stroke="#0a0a18" stroke-width="2.4" stroke-linecap="round"/>`;
    if (k === "frown") s += `<path d="M15 37q9-7 18 0" stroke="#0a0a18" stroke-width="2.4" fill="none" stroke-linecap="round"/><path d="M12 18l8 3M36 18l-8 3" stroke="#0a0a18" stroke-width="2" stroke-linecap="round"/>`;
    if (k === "angry") s += `<path d="M10 17l10 5M38 17l-10 5" stroke="#0a0a18" stroke-width="2.6" stroke-linecap="round"/><path d="M13 35l3-3 3 3 3-3 3 3 3-3 3 3 3-3" stroke="#0a0a18" stroke-width="2" fill="none" stroke-linejoin="round" stroke-linecap="round"/>`;
    if (k === "demon" || k === "xdemon") s += `<path d="M13 34q11 8 22 0" stroke="#0a0a18" stroke-width="2.4" fill="#0a0a18" stroke-linecap="round"/><path d="M17 35l2 5 2-4.2M27 35.8l2 4.2 2-5" fill="#fff" stroke="#fff" stroke-width="1" stroke-linejoin="round"/>`;
  }
  return s + "</svg>";
}

/* ------------------------------------------------------------------ data access */
async function getJSON(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(path + " -> " + r.status);
  return r.json();
}
async function loadState() {
  if (SNAP) { if (!S.state) { S.state = SNAP.state; S.version = SNAP.state.version; return true; } return false; }
  try {
    const d = await getJSON("/api/state?v=" + (S.version ?? ""));
    S.lastFetchOk = Date.now(); S.err = null;
    if (d.unchanged) return false;
    S.state = d; S.version = d.version;
    return true;
  } catch (e) { S.err = e.message; return false; }
}
async function loadDetail(id) {
  if (SNAP) { S.detail[id] = SNAP.details[id] || null; return; }
  const r = S.state.runs.find(x => x.id === id);
  const ver = S.version;
  if (S.detail[id] && (S.detailVer[id] === ver || (r && r.status !== "running" && S.detail[id]))) return;
  try {
    S.detail[id] = await getJSON("/api/run?id=" + encodeURIComponent(id));
    S.detailVer[id] = ver;
  } catch (e) { S.detail[id] = S.detail[id] || null; }
}
const runs = () => S.state ? S.state.runs : [];
const runById = id => runs().find(r => r.id === id);

/* ------------------------------------------------------------------ filters */
const ts = s => s ? new Date(s + "T00:00:00").getTime() / 1000 : null;
function dateOk(t) {
  const a = ts(S.f.from), b = S.f.to ? ts(S.f.to) + 86400 : null;
  if (t == null) return !(a || b);
  return (!a || t >= a) && (!b || t < b);
}
const dateOn = () => !!(S.f.from || S.f.to);
function levelPass(l) {
  const f = S.f;
  if (f.q) { const q = f.q.toLowerCase(); if (!(`${l.name} ${l.key} ${l.group} ${(l.agents || []).join(" ")}`.toLowerCase().includes(q))) return false; }
  if (f.diff && l.difficulty !== f.diff) return false;
  if (f.status && l.status !== f.status) return false;
  if (f.agent && !(l.agents || []).includes(f.agent)) return false;
  if (f.run && !(l.runs || []).includes(f.run)) return false;
  if (dateOn()) { const t = l.beat ? l.beat.ts : l.last_update; if (!dateOk(t)) return false; }
  if (S.ui.hideUntrained && l.status === "untrained") return false;
  return true;
}
function runPass(r) {
  const f = S.f;
  if (f.q) { const q = f.q.toLowerCase(); if (!(`${r.id} ${r.level_label} ${r.agent} ${r.levels.join(" ")}`.toLowerCase().includes(q))) return false; }
  if (f.run && r.id !== f.run) return false;
  if (f.agent && r.agent !== f.agent) return false;
  if (f.status && ["running", "finished", "crashed", "stopped"].includes(f.status) && r.status !== f.status) return false;
  if (f.diff) { const ok = r.levels.some(k => { const l = S.state.levels.find(x => x.key === k); return l && l.difficulty === f.diff; }); if (!ok) return false; }
  if (!dateOk(r.created)) return false;
  return true;
}
function ledgerPass(b) {
  const f = S.f;
  if (f.q && !(`${b.level} ${b.agent} ${b.notes}`.toLowerCase().includes(f.q.toLowerCase()))) return false;
  if (f.agent && b.agent !== f.agent) return false;
  if (f.diff) { const ok = b.keys.some(k => { const l = S.state.levels.find(x => x.key === k); return l && l.difficulty === f.diff; }); if (!ok) return false; }
  if (!dateOk(b.ts)) return false;
  return true;
}

/* ------------------------------------------------------------------ chrome */
function renderNav() {
  $("#nav").innerHTML = NAV.map(([id, label, d]) => `<a data-v="${id}" class="${S.view === id ? "on" : ""}" href="#${id}"><svg viewBox="0 0 24 24"><path d="${d}"/></svg><span>${label}</span></a>`).join("");
}
function renderLivebox() {
  const st = S.state; if (!st) return;
  const n = st.totals.running;
  const stale = !SNAP && !document.hidden && Date.now() - S.lastFetchOk > 15000;
  $("#livebox").innerHTML = SNAP
    ? `<div><span class="pulse off"></span><span class="lt">Snapshot</span></div><div class="lt">${fmtDate(SNAP.exported)}</div>`
    : `<div><span class="pulse ${n && !stale ? "" : "off"}"></span><span class="lt">${stale ? "Disconnected" : n ? n + " run" + (n > 1 ? "s" : "") + " live" : "Idle"}</span></div><div class="lt">updated ${fmtClock(st.generated)}</div>`;
}
function renderFilters() {
  const st = S.state; if (!st) return;
  const el = $("#filters");
  const active = document.activeElement;
  if (active && el.contains(active) && (active.tagName === "INPUT" && active.type === "search")) return; // do not disturb typing
  const diffs = [...new Set(st.levels.map(l => l.difficulty))];
  const agents = [...new Set([...st.runs.map(r => r.agent), ...st.levels.flatMap(l => l.agents || [])])].sort();
  const opt = (v, t, cur) => `<option value="${esc(v)}" ${cur === v ? "selected" : ""}>${esc(t)}</option>`;
  const h = `
    <input class="f search" type="search" id="f-q" placeholder="Search level, run, agent" value="${esc(S.f.q)}" aria-label="Search">
    <select class="f" id="f-diff" aria-label="Difficulty">${opt("", "All difficulties", S.f.diff)}${diffs.map(d => opt(d, tier(d).label, S.f.diff)).join("")}</select>
    <select class="f" id="f-status" aria-label="Status">${opt("", "Any status", S.f.status)}${["untrained", "training", "beaten", "robust-beaten", "running", "finished", "crashed", "stopped"].map(d => opt(d, d, S.f.status)).join("")}</select>
    <select class="f" id="f-agent" aria-label="Agent">${opt("", "All agents", S.f.agent)}${agents.map(d => opt(d, d, S.f.agent)).join("")}</select>
    <select class="f" id="f-run" aria-label="Run">${opt("", "All runs", S.f.run)}${st.runs.slice().reverse().map(r => opt(r.id, runLabel(r), S.f.run)).join("")}</select>
    <input class="f" type="date" id="f-from" value="${S.f.from}" aria-label="From date" title="From date">
    <input class="f" type="date" id="f-to" value="${S.f.to}" aria-label="To date" title="To date">
    <button class="btn" id="f-reset" title="Clear filters">Reset</button>
    <span class="spacer"></span>
    <button class="btn" id="f-theme" title="Toggle theme">${S.ui.theme === "light" ? "Dark" : "Light"}</button>`;
  if (el.dataset.h !== h) { el.innerHTML = h; el.dataset.h = h; }
}
function bindFilters() {
  const top = $("#top");
  top.addEventListener("input", e => {
    const id = e.target.id;
    if (id === "f-q") { S.f.q = e.target.value; debounceRender(); }
  });
  top.addEventListener("change", e => {
    const m = { "f-diff": "diff", "f-status": "status", "f-agent": "agent", "f-run": "run", "f-from": "from", "f-to": "to" }[e.target.id];
    if (m) { S.f[m] = e.target.value; $("#filters").dataset.h = ""; render(); }
  });
  top.addEventListener("click", e => {
    if (e.target.id === "f-reset") { S.f = { q: "", diff: "", status: "", agent: "", run: "", from: "", to: "" }; $("#filters").dataset.h = ""; render(); }
    if (e.target.id === "f-theme") { S.ui.theme = S.ui.theme === "light" ? "dark" : "light"; applyTheme(); saveUi(); $("#filters").dataset.h = ""; render(); }
  });
}
let dbt = null;
function debounceRender() { clearTimeout(dbt); dbt = setTimeout(() => render(true), 160); }
function applyTheme() { document.documentElement.dataset.theme = S.ui.theme; }

/* ------------------------------------------------------------------ router */
function route() {
  const h = (location.hash || "#overview").slice(1).split("?")[0];
  S.view = VIEWS.includes(h) ? h : "overview";
}
function go(v, extra) { if (extra) Object.assign(S.ui, extra); saveUi(); closeDrawer(); location.hash = "#" + v; }

/* ------------------------------------------------------------------ views: overview */
function statTile(k, v, s) { return `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s || "&nbsp;"}</div></div>`; }
function levelCard(l) {
  const t = tier(l.difficulty);
  const pct = l.best_progress || 0;
  const done = l.status === "beaten" || l.status === "robust-beaten";
  const agents = (l.agents || []).slice(0, 6).map(a => `<span class="chip">${esc(a)}</span>`).join("") + ((l.agents || []).length > 6 ? `<span class="chip">+${l.agents.length - 6}</span>` : "");
  const beat = l.beat ? `<div class="beat">Beaten ${esc(l.beat.when || fmtDate(l.beat.ts))} by ${esc(l.beat.agent || "?")}${l.beat.attempts != null ? ` in ${fmtNum(l.beat.attempts)} attempts` : ""}</div>` : "";
  return `<div class="card ${l.status} fade" style="--c:${t.color}" data-key="${esc(l.key)}" tabindex="0" role="button" aria-label="${esc(l.name)}">
    <div class="hd">${face(l.difficulty)}<div style="min-width:0"><div class="dl">${t.label}</div><div class="nm">${esc(l.name)}</div></div>
      <span class="spacer"></span>${l.live ? '<span class="pulse" title="training now"></span>' : ""}</div>
    <div class="bar ${done ? "done" : ""}"><i style="width:${(pct * 100).toFixed(1)}%"></i></div>
    <div class="barrow"><span>best progress</span><b>${fmtPct(pct, pct >= 1 ? 0 : 1)}</b></div>
    <div style="margin-top:8px"><span class="chip ${l.status}">${l.status}</span></div>
    <div class="kv"><div>attempts<b>${fmtNum(l.attempts)}</b></div><div>env steps<b>${fmtNum(l.steps)}</b></div><div>wall time<b>${fmtDur(l.wall_s)}</b></div></div>
    ${agents ? `<div class="agents">${agents}</div>` : ""}${beat}</div>`;
}
function viewOverview() {
  const st = S.state, T = st.totals;
  const lv = st.levels.filter(levelPass);
  const groups = st.groups.filter(g => lv.some(l => l.group === g));
  const tr = st.levels.filter(l => l.status !== "untrained").length;
  let h = `<div class="pageh"><div><h1>Level board</h1><p>Every level the agents are working toward, from mock tiers to extreme demons.</p></div><span class="spacer"></span>
    <label class="note"><input type="checkbox" id="hide-un" ${S.ui.hideUntrained ? "checked" : ""}> hide untrained</label></div>
    <div class="grid stats">
      ${statTile("Levels beaten", `${T.beaten}<span style="font-size:15px;color:var(--mute)"> / ${T.levels}</span>`, `${tr} with any training`)}
      ${statTile("Runs live", T.running, `${T.runs} runs recorded`)}
      ${statTile("Env steps", fmtNum(T.steps), "across all runs")}
      ${statTile("Wall time", fmtDur(T.wall_s), "estimated from steps/s")}
    </div>`;
  if (!lv.length) h += `<div class="empty" style="margin-top:20px"><b>No levels match these filters</b>Reset the filters to see the whole board.</div>`;
  for (const g of groups) {
    const items = lv.filter(l => l.group === g);
    const beaten = items.filter(l => l.status.endsWith("beaten")).length;
    const allUn = items.every(l => l.status === "untrained");
    const col = S.ui.collapsed[g] ?? allUn;
    h += `<div class="group"><h2 class="ghead" data-g="${esc(g)}">${col ? "&#9656;" : "&#9662;"} ${esc(g)} <span class="n">${beaten}/${items.length} beaten</span></h2>`;
    if (!col) h += `<div class="grid cards">${items.map(levelCard).join("")}</div>`;
    h += "</div>";
  }
  return h;
}
function openDrawer(key) {
  const st = S.state, l = st.levels.find(x => x.key === key); if (!l) return;
  const rs = (l.runs || []).map(runById).filter(Boolean).sort((a, b) => b.created - a.created);
  const t = tier(l.difficulty);
  const best = Object.entries(l.best_by || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${fmtPct(v)}</td></tr>`).join("");
  $("#drawer").innerHTML = `<button class="x" id="dx" aria-label="Close">&times;</button>
    <div class="hd" style="display:flex;gap:14px;align-items:center;--c:${t.color}">${face(l.difficulty)}<div><div class="dl" style="color:${t.color};font-size:11px;letter-spacing:.1em;text-transform:uppercase;font-weight:700">${t.label} &middot; ${esc(l.group)}</div><div style="font-size:22px;font-weight:800">${esc(l.name)}</div><span class="chip ${l.status}">${l.status}</span></div></div>
    <div class="kv" style="grid-template-columns:repeat(2,1fr);margin:16px 0"><div>best progress<b>${fmtPct(l.best_progress)}</b></div><div>attempts<b>${fmtNum(l.attempts)}</b></div><div>env steps<b>${fmtNum(l.steps)}</b></div><div>wall time<b>${fmtDur(l.wall_s)}</b></div></div>
    ${l.length ? `<div class="note">Level length ${fmtNum(l.length)} units</div>` : ""}
    <h2>Best progress by source</h2>${best ? `<table><tbody>${best}</tbody></table>` : '<div class="note">No data yet.</div>'}
    <h2>Beats</h2>${(l.beats || []).length ? (l.beats).map(b => `<div class="runcard" style="margin-bottom:8px"><b>${esc(b.agent || "?")}${b.partial ? " (any level in a range)" : ""}</b><span class="note">${esc(b.when || "")} &middot; attempts ${fmtNum(b.attempts)} &middot; generations ${fmtNum(b.generations)} &middot; wall ${b.wall_hours != null ? b.wall_hours + "h" : "-"}</span></div>`).join("") : '<div class="note">Not beaten yet.</div>'}
    <h2>Runs (${rs.length})</h2>${rs.map(r => `<div class="runcard" style="margin-bottom:8px"><div class="t">${esc(r.id)}</div><div style="display:flex;gap:6px;flex-wrap:wrap"><span class="chip ${r.status}">${r.status}</span><span class="chip">${esc(r.agent)}</span><span class="chip">${fmtNum(r.steps)} steps</span><span class="chip">best ${fmtPct(r.best_progress)}</span></div></div>`).join("") || '<div class="note">No runs.</div>'}
    <div style="display:flex;gap:8px;margin-top:16px;flex-wrap:wrap"><button class="btn" data-go="curves" data-run="${esc(rs[0]?.id || "")}">Curves</button><button class="btn" data-go="deaths" data-run="${esc(rs[0]?.id || "")}">Death map</button><button class="btn" data-go="ledger">Ledger</button></div>`;
  $("#drawer").classList.add("open");
}
function closeDrawer() { $("#drawer").classList.remove("open"); }

/* ------------------------------------------------------------------ charts */
const CHARTS = {};
function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const step0 = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(step0))), norm = step0 / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
function ema(y, a) {
  if (!a) return y;
  let last = 0, n = 0;
  return y.map(v => { if (v == null || !isFinite(v)) return v; last = last * a + (1 - a) * v; n++; return last / (1 - Math.pow(a, n)); });
}
function bsearch(xs, x) {
  let lo = 0, hi = xs.length - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (xs[m] < x) lo = m + 1; else hi = m; }
  if (lo > 0 && Math.abs(xs[lo - 1] - x) <= Math.abs(xs[lo] - x)) lo--;
  return lo;
}
function interp(x, y, xq) {
  const out = new Array(xq.length); let j = 0;
  for (let i = 0; i < xq.length; i++) {
    const q = xq[i];
    while (j < x.length - 2 && x[j + 1] < q) j++;
    if (q < x[0] || q > x[x.length - 1] || x.length < 2) { out[i] = null; continue; }
    const x0 = x[j], x1 = x[j + 1], f = x1 === x0 ? 0 : (q - x0) / (x1 - x0);
    out[i] = y[j] + (y[j + 1] - y[j]) * f;
  }
  return out;
}
function xUnit(kind, series) {
  if (kind !== "wall") return { div: 1, suffix: kind === "step" ? "" : "" };
  let mx = 0; series.forEach(s => s.x.forEach(v => { if (v > mx) mx = v; }));
  return mx > 7200 ? { div: 3600, suffix: " h" } : mx > 180 ? { div: 60, suffix: " min" } : { div: 1, suffix: " s" };
}
function chartSVG(o) {
  const W = 560, H = o.h || 250, m = { l: 50, r: 12, t: 10, b: 28 };
  const series = o.series.filter(s => s.x && s.x.length);
  if (!series.length) return `<div class="empty" style="padding:30px 10px">No data for this chart yet.</div>`;
  const U = xUnit(o.xKind, series);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  const L = o.logY;
  const ty = v => v;
  for (const s of series) {
    s.xs = s.x.map(v => v / U.div);
    for (let i = 0; i < s.xs.length; i++) {
      const xv = s.xs[i]; if (xv < x0) x0 = xv; if (xv > x1) x1 = xv;
      for (const arr of [s.y, s.lo, s.hi]) {
        if (!arr) continue; const v = arr[i];
        if (v == null || !isFinite(v) || (L && v <= 0)) continue;
        const w = L ? Math.log10(v) : v; if (w < y0) y0 = w; if (w > y1) y1 = w;
      }
    }
  }
  if (!isFinite(y0)) return `<div class="empty" style="padding:30px 10px">No plottable values (log scale needs positive data).</div>`;
  if (o.ymin != null && !L) y0 = Math.min(y0, o.ymin) === y0 && o.ymin != null ? o.ymin : y0;
  if (o.ymin != null && !L) y0 = o.ymin;
  if (o.ymax != null && !L) y1 = o.ymax;
  if (y1 === y0) { y1 += 1; y0 -= 1; }
  const padY = (y1 - y0) * (o.ymax != null ? 0.02 : 0.06); if (o.ymin == null || L) y0 -= padY; y1 += padY;
  if (x1 === x0) x1 = x0 + 1;
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const sx = v => m.l + (v - x0) / (x1 - x0) * iw;
  const sy = v => m.t + ih - ((L ? Math.log10(v) : v) - y0) / (y1 - y0) * ih;
  let g = "";
  const yt = L ? (() => { const a = []; for (let p = Math.ceil(y0); p <= Math.floor(y1); p++) a.push(Math.pow(10, p)); return a.length ? a : [Math.pow(10, Math.round((y0 + y1) / 2))]; })() : niceTicks(y0, y1, 4);
  for (const t of yt) {
    const yy = sy(t); if (yy < m.t - 1 || yy > m.t + ih + 1) continue;
    g += `<line class="gl" x1="${m.l}" x2="${W - m.r}" y1="${yy}" y2="${yy}"/><text x="${m.l - 6}" y="${yy + 3.5}" text-anchor="end">${o.yFmt ? o.yFmt(t) : fmtNum(t)}</text>`;
  }
  for (const t of niceTicks(x0, x1, 6)) {
    const xx = sx(t); g += `<line class="gl" x1="${xx}" x2="${xx}" y1="${m.t}" y2="${m.t + ih}" opacity=".45"/><text x="${xx}" y="${H - 8}" text-anchor="middle">${o.xKind === "step" ? fmtNum(t * U.div) : (+t.toFixed(2)) + U.suffix}</text>`;
  }
  let paths = "", bands = "";
  for (const s of series) {
    if (s.lo && s.hi) {
      let up = [], dn = [];
      for (let i = 0; i < s.xs.length; i++) { const a = s.hi[i], b = s.lo[i]; if (a == null || b == null || (L && (a <= 0 || b <= 0))) continue; up.push(`${sx(s.xs[i]).toFixed(1)},${sy(a).toFixed(1)}`); dn.push(`${sx(s.xs[i]).toFixed(1)},${sy(b).toFixed(1)}`); }
      if (up.length > 1) bands += `<polygon points="${up.join(" ")} ${dn.reverse().join(" ")}" fill="${s.color}" opacity=".16"/>`;
    }
    let d = "", pen = false;
    for (let i = 0; i < s.xs.length; i++) {
      const v = s.y[i];
      if (v == null || !isFinite(v) || (L && v <= 0)) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${sx(s.xs[i]).toFixed(1)},${sy(v).toFixed(1)}`; pen = true;
    }
    if (d) paths += `<path class="ln" d="${d}" stroke="${s.color}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} ${s.faint ? 'opacity=".55"' : ""}/>`;
  }
  const cid = o.id;
  CHARTS[cid] = { series, sx, sy, x0, x1, m, W, H, iw, ih, U, o };
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title)}">${g}<line class="ax" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${m.t + ih}"/><line class="ax" x1="${m.l}" x2="${W - m.r}" y1="${m.t + ih}" y2="${m.t + ih}"/>${bands}${paths}<line class="cur" y1="${m.t}" y2="${m.t + ih}" x1="0" x2="0"/></svg>`;
}
function chartPanel(o) {
  const body = chartSVG(o);
  const lg = o.series.filter(s => s.x && s.x.length).slice(0, 14).map(s => `<span><i style="background:${s.color};${s.dash ? "opacity:.6" : ""}"></i>${esc(s.name)}</span>`).join("");
  return `<div class="panel chart" data-cid="${esc(o.id)}"><h3>${esc(o.title)} ${o.sub ? `<small>${esc(o.sub)}</small>` : ""}</h3>${body}<div class="legend">${lg}</div></div>`;
}
const tip = $("#tip");
function showTip(html, ev) {
  tip.innerHTML = html; tip.style.opacity = 1;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + 16, y = ev.clientY + 12;
  if (x + w > innerWidth - 8) x = ev.clientX - w - 16;
  if (y + h > innerHeight - 8) y = innerHeight - h - 8;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.style.opacity = 0; };
function chartHover(ev) {
  const el = ev.target.closest && ev.target.closest(".chart[data-cid]");
  if (!el) { hideTip(); return; }
  const C = CHARTS[el.dataset.cid]; const svg = el.querySelector("svg");
  if (!C || !svg) return;
  const r = svg.getBoundingClientRect();
  const px = (ev.clientX - r.left) / r.width * C.W;
  if (px < C.m.l || px > C.W - C.m.r) { hideTip(); svg.querySelector(".cur").style.opacity = 0; return; }
  const xv = C.x0 + (px - C.m.l) / C.iw * (C.x1 - C.x0);
  const cur = svg.querySelector(".cur"); cur.setAttribute("x1", px); cur.setAttribute("x2", px); cur.style.opacity = 1;
  let rows = "", head = "";
  for (const s of C.series) {
    const i = bsearch(s.xs, xv); const v = s.y[i]; if (v == null) continue;
    if (!head) head = `<div style="color:var(--mute);margin-bottom:4px">${C.o.xKind === "step" ? "step " + fmtNum(s.x[i]) : C.o.xKind === "update" ? "update " + (+s.x[i].toFixed(1)) : C.o.xKind === "gen" ? "gen " + s.x[i] : fmtDur(s.x[i])}</div>`;
    rows += `<div class="r"><i style="background:${s.color}"></i>${esc(s.name)}<b>${C.o.yFmt ? C.o.yFmt(v) : fmtNum(v)}</b></div>`;
  }
  if (rows) showTip(head + rows, ev); else hideTip();
}

/* ------------------------------------------------------------------ views: curves */
const SPECS = [
  { id: "return", title: "Return", keys: [["episode/return_mean", "train"], ["eval/return", "eval (greedy)"]] },
  { id: "progress", title: "Progress", keys: [["episode/progress_mean", "mean"], ["eval/progress", "eval greedy from start"], ["@max:episode/progress_mean", "max so far"]], pct: true, ymin: 0, ymax: 1 },
  { id: "completion", title: "Completion rate", keys: [["episode/completed_mean", "train"], ["eval/completed", "eval greedy"]], pct: true, ymin: 0, ymax: 1 },
  { id: "entropy", title: "Policy entropy", keys: [["train/entropy", ""]] },
  { id: "kl", title: "Approx KL", keys: [["train/approx_kl", ""]], logable: true },
  { id: "vloss", title: "Value loss", keys: [["train/value_loss", ""]], logable: true },
  { id: "ev", title: "Explained variance", keys: [["train/explained_variance", ""]] },
  { id: "sps", title: "Steps / second", keys: [["perf/steps_per_second", ""]] },
  { id: "ploss", title: "Policy loss", keys: [["train/policy_loss", ""]], more: true },
  { id: "clip", title: "Clip fraction", keys: [["train/clipfrac", ""]], more: true },
  { id: "lr", title: "Learning rate", keys: [["train/learning_rate", ""]], more: true },
  { id: "len", title: "Episode length", keys: [["episode/length_mean", ""]], more: true },
  { id: "vmean", title: "Value mean", keys: [["train/value_mean", ""]], more: true },
];
const xKey = () => ({ step: "x", update: "u", wall: "w" }[S.ui.xaxis]);
function groupOf(r) { return r.family + "|" + r.levels.join(","); }
function defaultSelection() {
  const all = runs().filter(r => r.has.metrics && runPass(r));
  if (!all.length) return [];
  const live = all.filter(r => r.status === "running");
  const pool = live.length ? live : [all.slice().sort((a, b) => b.created - a.created)[0]];
  const fams = new Set(pool.map(groupOf));
  return all.filter(r => fams.has(groupOf(r))).map(r => r.id);
}
function buildSeries(spec, sel, details) {
  const out = [], xk = xKey(), a = S.ui.smooth;
  const groups = {};
  sel.forEach(id => { const r = runById(id); if (r) (groups[groupOf(r)] ||= []).push(r); });
  for (const [gk, rs] of Object.entries(groups)) {
    spec.keys.forEach(([key, vname], vi) => {
      const lines = [];
      for (const r of rs) {
        const d = details[r.id]; if (!d || !d.curves) continue;
        let cu;
        if (key.startsWith("@max:")) { const c = d.curves[key.slice(5)]; if (!c) continue; let m = -Infinity; cu = { ...c, y: c.y.map(v => (m = Math.max(m, v))) }; }
        else cu = d.curves[key];
        if (!cu || !cu[xk]) continue;
        lines.push({ r, x: cu[xk], y: key.startsWith("@max:") ? cu.y : ema(cu.y, a) });
      }
      if (!lines.length) return;
      const dash = vi === 0 ? null : vi === 1 ? "6 4" : "2 3";
      const suffix = vname ? " · " + vname : "";
      const useGroup = S.ui.group && lines.length > 1 && rs.length > 1 && new Set(rs.map(r => r.seed)).size > 1;
      if (useGroup) {
        const lo = Math.max(...lines.map(l => l.x[0])), hi = Math.min(...lines.map(l => l.x[l.x.length - 1]));
        if (hi > lo) {
          const grid = Array.from({ length: 160 }, (_, i) => lo + (hi - lo) * i / 159);
          const ys = lines.map(l => interp(l.x, l.y, grid));
          const mean = grid.map((_, i) => { const v = ys.map(y => y[i]).filter(z => z != null); return v.length ? v.reduce((p, q) => p + q, 0) / v.length : null; });
          const sd = grid.map((_, i) => { const v = ys.map(y => y[i]).filter(z => z != null); if (v.length < 2) return 0; const m = v.reduce((p, q) => p + q, 0) / v.length; return Math.sqrt(v.reduce((p, q) => p + (q - m) ** 2, 0) / v.length); });
          const col = PALETTE[hashStr(gk) % PALETTE.length];
          out.push({ name: `${rs[0].base} (mean of ${lines.length}, ±sd)${suffix}`, color: col, dash, x: grid, y: mean, lo: mean.map((m, i) => m == null ? null : m - sd[i]), hi: mean.map((m, i) => m == null ? null : m + sd[i]) });
          return;
        }
      }
      for (const l of lines) out.push({ name: runLabel(l.r) + suffix, color: runColor(l.r), dash, x: l.x, y: l.y, faint: vi === 2 });
    });
  }
  return out;
}
function viewCurves() {
  let sel = S.ui.sel.filter(id => runById(id));
  if (!sel.length && !S.ui.selNone) sel = defaultSelection();
  const cand = runs().filter(r => r.has.metrics && runPass(r)).slice().sort((a, b) => b.created - a.created);
  const list = cand.map(r => `<div class="pick ${sel.includes(r.id) ? "on" : ""}" data-pick="${esc(r.id)}" style="--c:${runColor(r)}"><span class="dot"></span><span class="t" title="${esc(r.id)}">${esc(runLabel(r))}</span><span class="chip ${r.status}" style="font-size:9px;padding:0 6px">${r.status}</span></div>`).join("");
  const specs = SPECS.filter(s => !s.more || S.ui.more);
  const seg = (name, opts, cur) => `<div class="seg" data-seg="${name}">${opts.map(([v, t]) => `<button data-v="${v}" class="${cur === v ? "on" : ""}">${t}</button>`).join("")}</div>`;
  let h = `<div class="pageh"><div><h1>Training curves</h1><p>Return, progress, completion and optimiser health per run. Seeds of one config can be averaged with a band.</p></div></div>
    <div class="split"><div class="panel"><h3>Runs <small>${sel.length} selected</small></h3>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px"><button class="btn sm" id="sel-latest">Latest</button><button class="btn sm" id="sel-all">All shown</button><button class="btn sm" id="sel-none">Clear</button></div>
      <div class="picker">${list || '<div class="note">No runs with metrics match the filters.</div>'}</div></div>
    <div><div class="ctl">
      <label>Smoothing <input type="range" id="sm" min="0" max="0.99" step="0.01" value="${S.ui.smooth}"> <b id="smv" style="color:var(--text)">${S.ui.smooth.toFixed(2)}</b></label>
      ${seg("xaxis", [["step", "Steps"], ["update", "Updates"], ["wall", "Wall time"]], S.ui.xaxis)}
      <label><input type="checkbox" id="logy" ${S.ui.log ? "checked" : ""}> log y (loss, KL)</label>
      <label><input type="checkbox" id="grp" ${S.ui.group ? "checked" : ""}> average seeds ± band</label>
      <label><input type="checkbox" id="more" ${S.ui.more ? "checked" : ""}> more charts</label></div>
      <div class="grid charts" id="chartgrid">${chartGrid(specs, sel)}</div></div></div>`;
  return h;
}
function chartGrid(specs, sel) {
  const details = {}; sel.forEach(id => details[id] = S.detail[id]);
  const missing = sel.filter(id => !S.detail[id]);
  if (missing.length) return `<div class="panel empty" style="grid-column:1/-1"><b>Loading curves</b>${missing.length} run${missing.length > 1 ? "s" : ""}...</div>`;
  return specs.map(sp => {
    const series = buildSeries(sp, sel, details);
    return chartPanel({ id: "c-" + sp.id, title: sp.title, series, xKind: S.ui.xaxis, logY: S.ui.log && sp.logable, yFmt: sp.pct ? v => fmtPct(v, 0) : null, ymin: sp.ymin, ymax: sp.ymax });
  }).join("");
}

/* ------------------------------------------------------------------ views: death map */
function heatColor(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[0, [7, 7, 15]], [.15, [43, 16, 94]], [.4, [180, 30, 150]], [.7, [255, 90, 60]], [1, [255, 224, 80]]];
  for (let i = 1; i < stops.length; i++) if (t <= stops[i][0]) {
    const [a, ca] = stops[i - 1], [b, cb] = stops[i], f = (t - a) / (b - a);
    return `rgb(${ca.map((c, k) => Math.round(c + (cb[k] - c) * f)).join(",")})`;
  }
  return "rgb(255,224,80)";
}
function runSelect(id, cur, filterFn) {
  const rs = runs().filter(filterFn).slice().sort((a, b) => b.created - a.created);
  return `<select class="f" id="${id}" style="max-width:340px">${rs.map(r => `<option value="${esc(r.id)}" ${r.id === cur ? "selected" : ""}>${esc(runLabel(r))} - ${esc(r.level_label)} (${r.status})</option>`).join("")}</select>`;
}
function pickRun(cur, fn) {
  const rs = runs().filter(fn);
  if (cur && rs.some(r => r.id === cur)) return cur;
  const ok = rs.filter(runPass);
  const pool = ok.length ? ok : rs;
  const live = pool.filter(r => r.status === "running");
  return (live[0] || pool.slice().sort((a, b) => b.created - a.created)[0] || {}).id || "";
}
function viewDeaths() {
  const fn = r => r.has.curriculum || r.has.episodes;
  S.ui.heatRun = pickRun(S.ui.heatRun, fn);
  const id = S.ui.heatRun, r = id && runById(id);
  let h = `<div class="pageh"><div><h1>Death map</h1><p>Where attempts end along the level, over training time. Bright means many deaths. Star marks the best attempt.</p></div><span class="spacer"></span>${runSelect("heat-run", id, fn)}</div>`;
  if (!r) return h + `<div class="empty"><b>No per-attempt data yet</b>Death maps need an episode log (episodes.jsonl or a curriculum csv). PPO runs on the mock write neither. Try <code>--demo</code> to see an example.</div>`;
  const d = S.detail[id];
  if (!d) return h + `<div class="empty"><b>Loading</b></div>`;
  const D = d.deaths;
  if (!D) return h + `<div class="empty"><b>No attempts recorded for this run</b></div>`;
  h += `<div class="ctl">${`<div class="seg" data-seg="heatNorm">${[["row", "Per time row"], ["global", "Global scale"]].map(([v, t]) => `<button data-v="${v}" class="${S.ui.heatNorm === v ? "on" : ""}">${t}</button>`).join("")}</div>`}
    <span class="note">${fmtNum(D.n)} attempts &middot; ${D.completed} completed &middot; level length ${fmtNum(D.length)} &middot; ${esc(r.level_label)}</span></div>`;
  h += `<div class="panel"><div class="heatwrap" id="heatwrap"><canvas id="heat" width="${D.bx}" height="${D.bt}" style="aspect-ratio:${D.bx}/${Math.max(10, Math.min(D.bt * 3.2, D.bx * .55))}"></canvas>
    <div id="heatover" style="position:absolute;inset:0;pointer-events:none"></div></div>
    <div class="hx"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100% of level</span></div>
    <div class="note" style="margin-top:6px">Top = earliest attempts (${fmtClock(D.t_bins[0][0])}), bottom = latest (${fmtClock(D.t_bins[D.t_bins.length - 1][1])}).</div></div>`;
  const col = Array(D.bx).fill(0); D.grid.forEach(row => row.forEach((v, i) => col[i] += v));
  const mx = Math.max(1, ...col);
  const top = col.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 6).filter(a => a[0] > 0);
  h += `<div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(340px,1fr));margin-top:14px"><div class="panel"><h3>Deaths by position <small>all training</small></h3>
    <svg viewBox="0 0 560 120" style="width:100%">${col.map((v, i) => `<rect x="${i * 560 / D.bx}" y="${118 - v / mx * 112}" width="${560 / D.bx - .6}" height="${v / mx * 112}" fill="${heatColor(.25 + .75 * v / mx)}"/>`).join("")}${(D.starts_x || []).map(x => `<line x1="${x / D.length * 560}" x2="${x / D.length * 560}" y1="0" y2="118" stroke="#00e5ff" stroke-opacity=".35" stroke-dasharray="2 3"/>`).join("")}</svg></div>
    <div class="panel"><h3>Hotspots</h3><table><tbody>${top.map(([v, i]) => `<tr><td>x &asymp; ${fmtNum((i + .5) / D.bx * D.length)}</td><td class="num">${(((i + .5) / D.bx) * 100).toFixed(1)}%</td><td class="num">${v} deaths</td></tr>`).join("")}</tbody></table>
    ${D.best ? `<div class="note" style="margin-top:8px">Best attempt reached x = ${fmtNum(D.best.x)} (${fmtPct(D.best.x / D.length)}) at ${fmtClock(D.best.t)}${D.best.start != null ? `, from start #${D.best.start}` : ""}.</div>` : ""}</div></div>`;
  return h;
}
function drawHeat() {
  const cv = $("#heat"); if (!cv) return;
  const d = S.detail[S.ui.heatRun]; if (!d || !d.deaths) return;
  const D = d.deaths, ctx = cv.getContext("2d");
  const gmax = Math.max(1, ...D.grid.flat());
  D.grid.forEach((row, y) => {
    const rm = S.ui.heatNorm === "row" ? Math.max(1, ...row) : gmax;
    row.forEach((v, x) => { ctx.fillStyle = v ? heatColor(Math.pow(v / rm, .55)) : "#07070f"; ctx.fillRect(x, y, 1, 1); });
  });
  let ov = (D.starts_x || []).map(x => `<div style="position:absolute;left:${x / D.length * 100}%;top:0;bottom:0;border-left:1px dashed rgba(0,229,255,.45)"></div>`).join("");
  if (D.best) {
    const row = Math.min(D.bt - 1, Math.floor(D.best.i / D.n * D.bt));
    ov += `<div style="position:absolute;left:${D.best.x / D.length * 100}%;top:${(row + .5) / D.bt * 100}%;transform:translate(-50%,-50%);color:#7dff4f;font-size:20px;text-shadow:0 0 8px #000,0 0 12px #7dff4f">&#9733;</div>`;
  }
  $("#heatover").innerHTML = ov;
  if (!cv.dataset.b) {
    cv.dataset.b = 1;
    cv.addEventListener("mousemove", e => {
      const r = cv.getBoundingClientRect(), x = Math.floor((e.clientX - r.left) / r.width * D.bx), y = Math.floor((e.clientY - r.top) / r.height * D.bt);
      const v = D.grid[y]?.[x]; if (v == null) return hideTip();
      const tb = D.t_bins[y];
      showTip(`<div class="r">x ${fmtNum(x / D.bx * D.length)} - ${fmtNum((x + 1) / D.bx * D.length)}<b>${v} deaths</b></div><div style="color:var(--mute)">attempts ${tb[2]}-${tb[3]} &middot; ${fmtClock(tb[0])}</div>`, e);
    });
    cv.addEventListener("mouseleave", hideTip);
  }
}

/* ------------------------------------------------------------------ views: curriculum */
function viewCurriculum() {
  const fn = r => r.has.curriculum;
  S.ui.curRun = pickRun(S.ui.curRun, fn);
  const id = S.ui.curRun, r = id && runById(id);
  let h = `<div class="pageh"><div><h1>Curriculum</h1><p>Start points along the level. Each tile is a place the agent can be dropped; the weight is how often it is sampled.</p></div><span class="spacer"></span>${runSelect("cur-run", id, fn)}</div>`;
  if (!r) return h + `<div class="empty"><b>No curriculum runs</b>Runs with <code>env.curriculum</code> in their config show up here. Try <code>--demo</code> for an extreme-demon example.</div>`;
  const d = S.detail[id];
  if (!d) return h + `<div class="empty"><b>Loading</b></div>`;
  const C = d.curriculum;
  if (!C) return h + `<div class="empty"><b>No start list found for this run</b></div>`;
  const counts = { locked: 0, training: 0, mastered: 0 }; C.tiles.forEach(t => counts[t.status]++);
  const flt = S.ui.curStatus;
  h += `<div class="ctl"><div class="seg" data-seg="curStatus">${[["all", `All ${C.tiles.length}`], ["mastered", `Mastered ${counts.mastered}`], ["training", `Training ${counts.training}`], ["locked", `Locked ${counts.locked}`]].map(([v, t]) => `<button data-v="${v}" class="${flt === v ? "on" : ""}">${t}</button>`).join("")}</div>
    <span class="note">${fmtNum(C.episodes)} episodes &middot; mastered = success EMA &ge; ${C.solved} &middot; p(start 0) = ${C.p_first}</span></div>
    <div class="panel" style="margin-bottom:14px"><h3>Level map <small>left = level start</small></h3><div class="strip">${C.tiles.map(t => `<i class="${t.status}" title="${esc(t.name)} @ ${t.x}: ${t.status}"></i>`).join("")}</div></div>
    <div class="grid tiles">${C.tiles.filter(t => flt === "all" || t.status === flt).map(t => {
    const sp = t.spark || []; const pts = sp.map((v, i) => `${(i / Math.max(1, sp.length - 1) * 100).toFixed(1)},${(22 - v * 20).toFixed(1)}`).join(" ");
    const col = t.status === "mastered" ? "var(--lime)" : "var(--cyan)";
    return `<div class="tile ${t.status}" title="best x reached: ${fmtNum(t.best_x)}"><div class="tn"><span>${esc(t.name.replace(/^gdrl-/, ""))}</span><span class="chip ${t.status === "mastered" ? "beaten" : t.status === "training" ? "training" : "untrained"}" style="font-size:9px;padding:0 6px">${t.status}</span></div>
      <div class="tx">x = ${fmtNum(t.x)} ${t.mode ? "&middot; " + esc(t.mode) : ""}</div>
      <div class="sc"><b>${t.attempts ? fmtPct(t.success, 0) : "-"}</b><span class="tx">${t.attempts} tries &middot; ${t.reached} ok</span></div>
      ${sp.length > 1 ? `<svg viewBox="0 0 100 24" preserveAspectRatio="none"><polyline points="${pts}" fill="none" stroke="${col}" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>` : ""}
      <div class="w" title="sampling weight"><i style="width:${Math.min(100, t.weight * 100 * 2.2)}%"></i></div><div class="tx" style="margin-top:3px">weight ${(t.weight * 100).toFixed(1)}%</div></div>`;
  }).join("")}</div>`;
  return h;
}

/* ------------------------------------------------------------------ views: search */
function viewSearch() {
  const fn = r => r.has.generations;
  S.ui.searchRun = pickRun(S.ui.searchRun, fn);
  const id = S.ui.searchRun, r = id && runById(id);
  let h = `<div class="pageh"><div><h1>Search and genetic agents</h1><p>Checkpoint-search generations and the genetic baseline.</p></div><span class="spacer"></span>${r ? runSelect("search-run", id, fn) : ""}</div>`;
  if (r) {
    const d = S.detail[id];
    if (!d) h += `<div class="empty"><b>Loading</b></div>`;
    else if (d.search) {
      const Sx = d.search, ser = Sx.series, res = Sx.result || {};
      const mk = (key, name, color, dash) => ({ name, color, dash, x: ser.generation.filter((_, i) => ser[key][i] != null), y: ser[key].filter(v => v != null) });
      h += `<div class="grid stats">${statTile("Generations", fmtNum(res.generations ?? Sx.last?.generation), res.solved ? "solved" : "in progress")}${statTile("Best progress", fmtPct(Math.max(...ser.best_progress.filter(v => v != null), 0)), "")}${statTile("Rollouts", fmtNum(res.rollouts ?? Sx.last?.rollouts), "attempts")}${statTile("Wall time", fmtDur(res.wall_s ?? r.wall_s), res.verified ? "replay verified" : "")}</div>
      <div class="grid charts" style="margin-top:14px">
      ${chartPanel({ id: "s-fit", title: "Fitness by generation", series: [mk("best_fitness", "best", "#00e5ff"), mk("mean_fitness", "mean", "#ff2bd6", "5 3")], xKind: "gen" })}
      ${chartPanel({ id: "s-prog", title: "Best progress by generation", series: [mk("best_progress", "best progress", "#7dff4f")], xKind: "gen", yFmt: v => fmtPct(v, 0), ymin: 0, ymax: 1 })}
      ${ser.rollouts.some(v => v != null) ? chartPanel({ id: "s-roll", title: "Rollouts per generation", series: [mk("rollouts", "rollouts", "#ffd400")], xKind: "gen" }) : ""}
      ${ser.beam.some(v => v != null) ? chartPanel({ id: "s-beam", title: "Beam survivors", series: [mk("survivors", "survivors", "#9b5cff"), mk("beam", "beam width", "#4d8dff", "4 3")], xKind: "gen" }) : ""}
      </div>`;
    }
  } else h += `<div class="empty"><b>No search runs</b>Runs that write generations.jsonl appear here.</div>`;
  const gen = (S.state.compare.rows || []).filter(x => x.solved != null);
  if (gen.length) {
    h += `<h2>Generations to solve</h2><div class="grid cards2">${gen.map(row => {
      const ls = Object.entries(row.per_level).filter(([, v]) => v.solved);
      const mx = Math.max(1, ...ls.map(([, v]) => v.solved.generations || 0));
      return `<div class="panel"><h3>${esc(row.agent)} <small>${row.solved}/${row.solved_of} solved &middot; mean ${fmtNum(row.gens_mean)} generations</small></h3>
        ${ls.map(([lid, v]) => `<div class="cell" style="min-width:0"><span style="min-width:44px;text-align:left;color:var(--mute)">L${esc(lid)}</span><div class="mini"><i style="width:${(v.solved.generations || 0) / mx * 100}%"></i></div><span>${fmtNum(v.solved.generations)} gens</span></div>`).join("")}</div>`;
    }).join("")}</div>`;
  }
  return h;
}

/* ------------------------------------------------------------------ views: compare */
function pctCell(v, sd) {
  if (v == null) return `<td class="num"><span class="sd">-</span></td>`;
  return `<td><div class="cell"><div class="mini"><i style="width:${v * 100}%"></i></div><span>${(v * 100).toFixed(0)}%${sd != null ? ` <span class="sd">±${(sd * 100).toFixed(0)}</span>` : ""}</span></div></td>`;
}
function viewCompare() {
  const C = S.state.compare;
  let rows = C.rows.filter(r => !S.f.agent || r.agent === S.f.agent);
  if (S.f.q) rows = rows.filter(r => r.agent.toLowerCase().includes(S.f.q.toLowerCase()));
  let h = `<div class="pageh"><div><h1>Agent comparison</h1><p>Completion rate on the mock levels from results/ (train vs held-out, greedy vs sticky actions). Seeds of one agent are averaged (± sd).</p></div></div>`;
  if (!rows.length) return h + `<div class="empty"><b>No result files</b>Drop M8 json files into results/m8/ (see gdrl.tools.m8_compare).</div>`;
  const metric = S.ui.cmpMetric === "progress" ? "_progress" : "";
  h += `<div class="ctl"><div class="seg" data-seg="cmpMetric">${[["completion", "Completion"], ["progress", "Mean progress"]].map(([v, t]) => `<button data-v="${v}" class="${S.ui.cmpMetric === v ? "on" : ""}">${t}</button>`).join("")}</div></div>
  <div class="tw"><table><thead><tr><th>Agent</th><th class="num">Seeds</th><th>Greedy train</th><th>Greedy held-out</th><th>Sticky train</th><th>Sticky held-out</th><th class="num">Solved</th><th class="num">Mean gens</th><th class="num">Transfer*</th></tr></thead><tbody>
  ${rows.map(r => `<tr><td><b>${esc(r.agent)}</b></td><td class="num">${r.n}</td>${pctCell(r["greedy_train" + metric], metric ? null : r.greedy_train_sd)}${pctCell(r["greedy_heldout" + metric], metric ? null : r.greedy_heldout_sd)}${pctCell(r["sticky_train" + metric], metric ? null : r.sticky_train_sd)}${pctCell(r["sticky_heldout" + metric], metric ? null : r.sticky_heldout_sd)}
   <td class="num">${r.solved != null ? `${r.solved}/${r.solved_of}` : "-"}</td><td class="num">${r.gens_mean != null ? fmtNum(r.gens_mean) : "-"}</td><td class="num">${r.greedy_transfer != null ? fmtPct(r.greedy_transfer, 0) : "-"}</td></tr>`).join("")}</tbody></table></div>
  <div class="note" style="margin-top:6px">* Transfer: a genetic agent's press sequence replayed on other levels (mean progress).</div>`;
  const lids = [...new Set(rows.flatMap(r => Object.keys(r.per_level)))].sort((a, b) => +a - +b);
  const mode = S.ui.cmpMode;
  h += `<h2>Per level</h2><div class="ctl"><div class="seg" data-seg="cmpMode">${[["greedy", "Greedy"], ["sticky", "Sticky 0.1"]].map(([v, t]) => `<button data-v="${v}" class="${mode === v ? "on" : ""}">${t}</button>`).join("")}</div></div>
   <div class="tw"><table><thead><tr><th>Agent</th>${lids.map(l => `<th class="num">${l}</th>`).join("")}</tr></thead><tbody>
   ${rows.map(r => `<tr><td>${esc(r.agent)}</td>${lids.map(l => { const v = r.per_level[l]?.[mode] ?? (mode === "greedy" && r.per_level[l]?.solved ? (r.per_level[l].solved.solved ? 1 : 0) : null); return `<td class="num" style="${v == null ? "" : `background:${heatColor(.12 + v * .75).replace("rgb", "rgba").replace(")", ",.5)")}`}">${v == null ? "-" : (v * 100).toFixed(0) + "%"}</td>`; }).join("")}</tr>`).join("")}</tbody></table></div>
   <div class="note" style="margin-top:6px">Columns are mock level ids; 1-5 are training levels, 101-103 held out.</div>`;
  return h;
}

/* ------------------------------------------------------------------ views: ledger */
function viewLedger() {
  const L = S.state.ledger.filter(ledgerPass);
  let h = `<div class="pageh"><div><h1>Beat ledger</h1><p>Every level an agent has beaten, from holistic_progress/level_beats.csv (Pacific time).</p></div></div>`;
  if (!L.length) return h + `<div class="empty"><b>No beats match</b></div>`;
  const names = []; L.forEach(b => { if (!names.includes(b.level)) names.push(b.level); });
  const t0 = Math.min(...L.map(b => b.ts || Infinity)), t1 = Math.max(...L.map(b => b.ts || 0), t0 + 60);
  const pad = Math.max(60, (t1 - t0) * .04), a = t0 - pad, b = t1 + pad;
  const W = 900, rowH = 26, lm = 190, H = names.length * rowH + 34;
  const sx = t => lm + (t - a) / (b - a) * (W - lm - 20);
  const agents = [...new Set(L.map(x => x.agent))]; const ac = x => PALETTE[agents.indexOf(x) % PALETTE.length];
  let svg = "";
  niceTicks(0, 1, 6).forEach(f => { const t = a + (b - a) * f, x = sx(t); svg += `<line x1="${x}" x2="${x}" y1="6" y2="${H - 24}" stroke="var(--line)"/><text x="${x}" y="${H - 8}" text-anchor="middle">${fmtDate(t)}</text>`; });
  names.forEach((n, i) => {
    const y = 14 + i * rowH; svg += `<line x1="${lm}" x2="${W - 20}" y1="${y}" y2="${y}" stroke="var(--line)" stroke-dasharray="2 4"/><text x="${lm - 10}" y="${y + 4}" text-anchor="end">${esc(n.length > 30 ? n.slice(0, 29) + "…" : n)}</text>`;
  });
  L.forEach((bt, k) => { if (!bt.ts) return; const y = 14 + names.indexOf(bt.level) * rowH; svg += `<circle class="bdot" data-k="${k}" cx="${sx(bt.ts)}" cy="${y}" r="7" fill="${ac(bt.agent)}" stroke="var(--bg)" stroke-width="2" style="cursor:pointer"/>`; });
  S._ledgerRows = L;
  h += `<div class="panel tl"><svg viewBox="0 0 ${W} ${H}" id="tlsvg">${svg}</svg><div class="legend">${agents.map(x => `<span><i style="background:${ac(x)}"></i>${esc(x)}</span>`).join("")}</div></div>
  <h2>All beats</h2><div class="tw"><table><thead><tr><th>Beaten</th><th>Level</th><th>Agent</th><th class="num">Attempts</th><th class="num">Gens</th><th class="num">Env steps</th><th class="num">Wall</th><th>Notes</th></tr></thead><tbody>
  ${L.slice().reverse().map(x => `<tr><td style="white-space:nowrap">${esc(x.when || "")}</td><td>${esc(x.level)}</td><td><span class="chip" style="border-color:${ac(x.agent)}">${esc(x.agent)}</span></td><td class="num">${fmtNum(x.attempts)}</td><td class="num">${fmtNum(x.generations)}</td><td class="num">${fmtNum(x.env_steps)}</td><td class="num">${x.wall_hours != null ? fmtDur(x.wall_hours * 3600) : "-"}</td><td class="note" style="max-width:420px">${esc((x.notes || "").slice(0, 160))}</td></tr>`).join("")}</tbody></table></div>`;
  return h;
}

/* ------------------------------------------------------------------ views: live */
function runCard(r) {
  const pct = r.total_steps ? Math.min(1, r.steps / r.total_steps) : null;
  const eta = r.total_steps && r.sps && r.status === "running" ? fmtDur((r.total_steps - r.steps) / r.sps) : null;
  const sp = r.spark || [], pts = sp.map((v, i) => `${(i / Math.max(1, sp.length - 1) * 100).toFixed(1)},${(22 - Math.min(1, v) * 20).toFixed(1)}`).join(" ");
  return `<div class="runcard fade"><div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><span class="chip ${r.status}">${r.status === "running" ? '<span class="pulse" style="margin:0"></span>' : ""}${r.status}</span>${r.demo ? '<span class="chip demo">demo</span>' : ""}<span class="chip">${esc(r.agent)}</span><span class="note">${fmtAgo(r.last_update)}</span></div>
    <div class="t">${esc(r.id)}</div><div class="note">${esc(r.level_label)}</div>
    ${pct != null ? `<div class="bar" style="margin:4px 0"><i style="width:${pct * 100}%"></i></div>` : ""}
    <div class="kv"><div>steps<b>${fmtNum(r.steps)}</b></div><div>steps/s<b>${r.sps ? fmtNum(r.sps) : "-"}</b></div><div>${eta ? "eta" : "wall"}<b>${eta || fmtDur(r.wall_s)}</b></div>
      <div>progress<b>${fmtPct(r.last_progress, 0)}</b></div><div>best<b>${fmtPct(r.best_progress, 0)}</b></div><div>return<b>${r.last_return != null ? r.last_return.toFixed(2) : "-"}</b></div></div>
    ${sp.length > 2 ? `<svg viewBox="0 0 100 24" preserveAspectRatio="none" style="width:100%;height:30px"><polyline points="${pts}" fill="none" stroke="var(--cyan)" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>` : ""}
    ${r.log_tail && r.log_tail.length ? `<div class="log">${esc(r.log_tail.slice(-3).join("\n"))}</div>` : ""}${r.log_bad && r.log_bad.length && r.status !== "running" && r.status !== "finished" ? `<div class="log" style="color:var(--red)">${esc(r.log_bad.slice(-1)[0])}</div>` : ""}</div>`;
}
function viewLive() {
  const st = S.state, rs = st.runs.filter(runPass);
  const active = rs.filter(r => r.status === "running");
  const recent = rs.filter(r => r.status !== "running").sort((a, b) => b.last_update - a.last_update).slice(0, 6);
  const feed = st.feed.filter(e => (!S.f.run || e.run === S.f.run) && (!S.f.agent || e.agent === S.f.agent) && dateOk(e.t));
  const fresh = S.firstFeed ? new Set() : null;
  const evHtml = feed.slice(0, 40).map(e => {
    const key = e.run + e.t + e.kind; const isNew = !S.seen.has(key) && !S.firstFeed; S.seen.add(key);
    const r = runById(e.run);
    if (e.kind === "update") return `<div class="ev ${isNew ? "new" : ""}"><span class="t">${fmtClock(e.t)}</span><span>update &middot; ${esc(r?.base || e.run)} &middot; ${fmtNum(e.steps)} steps &middot; ${e.sps ? fmtNum(e.sps) + "/s" : ""}</span><span class="p">${fmtPct(e.progress, 0)}</span></div>`;
    return `<div class="ev ${e.completed ? "win" : ""} ${isNew ? "new" : ""}"><span class="t">${fmtClock(e.t)}</span><span>${e.completed ? "&#9733; completed" : "died"}${e.x != null ? " at x " + fmtNum(e.x) : ""}${e.start != null ? " &middot; start " + e.start : ""} <span class="note">${esc(r?.base || e.run)}</span></span><span class="p">${fmtPct(e.progress, 1)}</span></div>`;
  }).join("");
  S.firstFeed = false;
  return `<div class="pageh"><div><h1>Live feed</h1><p>Run status from metrics and logs; latest episodes. Refreshes every few seconds.</p></div></div>
  <div class="split" style="grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr)"><div>
    <h2 style="margin-top:0">Running (${active.length})</h2>${active.length ? `<div class="grid cards2">${active.map(runCard).join("")}</div>` : '<div class="empty"><b>Nothing is training right now</b>Runs count as live while their files change.</div>'}
    <h2>Recent runs</h2><div class="grid cards2">${recent.map(runCard).join("")}</div></div>
    <div><h2 style="margin-top:0">Latest episodes</h2><div class="feed">${evHtml || '<div class="empty">No episodes yet.</div>'}</div></div></div>
  <h2>All runs (${rs.length})</h2><div class="tw"><table><thead><tr><th>Status</th><th>Run</th><th>Level</th><th>Agent</th><th class="num">Steps</th><th class="num">Best</th><th class="num">Wall</th><th>Updated</th></tr></thead><tbody>
  ${rs.slice().sort((a, b) => b.created - a.created).map(r => `<tr><td><span class="chip ${r.status}">${r.status}</span></td><td style="word-break:break-all">${esc(r.id)}${r.demo ? ' <span class="chip demo">demo</span>' : ""}</td><td>${esc(r.level_label)}</td><td>${esc(r.agent)}</td><td class="num">${fmtNum(r.steps)}</td><td class="num">${fmtPct(r.best_progress, 0)}</td><td class="num">${fmtDur(r.wall_s)}</td><td>${fmtAgo(r.last_update)}</td></tr>`).join("")}</tbody></table></div>`;
}

/* ------------------------------------------------------------------ render loop */
const VIEWFN = { overview: viewOverview, curves: viewCurves, deaths: viewDeaths, curriculum: viewCurriculum, search: viewSearch, compare: viewCompare, ledger: viewLedger, live: viewLive };
function needDetails() {
  const ids = [];
  if (S.view === "curves") { let sel = S.ui.sel.filter(id => runById(id)); if (!sel.length && !S.ui.selNone) sel = defaultSelection(); ids.push(...sel); }
  if (S.view === "deaths") ids.push(S.ui.heatRun || pickRun("", r => r.has.curriculum || r.has.episodes));
  if (S.view === "curriculum") ids.push(S.ui.curRun || pickRun("", r => r.has.curriculum));
  if (S.view === "search") ids.push(S.ui.searchRun || pickRun("", r => r.has.generations));
  return ids.filter(Boolean);
}
let lastHTML = "", rendering = false;
async function render(keepScroll) {
  if (!S.state || rendering) return;
  rendering = true;
  try {
    renderNav(); renderFilters(); renderLivebox();
    const view = $("#view");
    const ids = needDetails();
    const stale = ids.filter(id => !S.detail[id] || (!SNAP && S.detailVer[id] !== S.version && runById(id)?.status === "running"));
    if (stale.length) await Promise.all(stale.map(loadDetail));
    const html = VIEWFN[S.view]();
    if (html !== lastHTML) {
      const y = scrollY;
      view.innerHTML = html; lastHTML = html;
      if (keepScroll !== false) scrollTo(0, y);
      postRender();
    }
  } catch (e) { console.error(e); $("#view").innerHTML = `<div class="empty"><b>Render error</b>${esc(e.message)}</div>`; }
  rendering = false;
}
function postRender() { if (S.view === "deaths") drawHeat(); }

/* ------------------------------------------------------------------ events */
document.addEventListener("click", e => {
  const t = e.target;
  const card = t.closest(".card[data-key]"); if (card) return openDrawer(card.dataset.key);
  if (t.id === "dx") return closeDrawer();
  const gb = t.closest("[data-go]"); if (gb) { const v = gb.dataset.go; const run = gb.dataset.run; closeDrawer(); if (run) { if (v === "curves") S.ui.sel = [run]; if (v === "deaths") S.ui.heatRun = run; } saveUi(); location.hash = "#" + v; if (S.view === v) render(); return; }
  const gh = t.closest(".ghead"); if (gh) { const g = gh.dataset.g; const items = S.state.levels.filter(l => l.group === g && levelPass(l)); const def = items.every(l => l.status === "untrained"); S.ui.collapsed[g] = !(S.ui.collapsed[g] ?? def); saveUi(); lastHTML = ""; return render(); }
  const pk = t.closest(".pick"); if (pk) { const id = pk.dataset.pick; let sel = S.ui.sel.filter(runById); if (!sel.length && !S.ui.selNone) sel = defaultSelection(); S.ui.selNone = false; S.ui.sel = sel.includes(id) ? sel.filter(x => x !== id) : [...sel, id]; saveUi(); lastHTML = ""; return render(); }
  const sg = t.closest(".seg button"); if (sg) { const name = sg.parentElement.dataset.seg; S.ui[name] = sg.dataset.v; saveUi(); lastHTML = ""; return render(); }
  if (t.id === "sel-latest") { S.ui.sel = []; S.ui.selNone = false; saveUi(); lastHTML = ""; return render(); }
  if (t.id === "sel-all") { S.ui.selNone = false; S.ui.sel = runs().filter(r => r.has.metrics && runPass(r)).map(r => r.id); saveUi(); lastHTML = ""; return render(); }
  if (t.id === "sel-none") { S.ui.sel = []; S.ui.selNone = true; saveUi(); lastHTML = ""; return render(); }
  const bd = t.closest(".bdot"); if (bd) { const b = S._ledgerRows[+bd.dataset.k]; showTip(`<b>${esc(b.level)}</b><div style="color:var(--mute)">${esc(b.when)}</div><div>${esc(b.agent)} &middot; ${fmtNum(b.attempts)} attempts</div>`, e); }
});
document.addEventListener("keydown", e => { if (e.key === "Escape") closeDrawer(); if ((e.key === "Enter" || e.key === " ") && e.target.classList?.contains("card")) { e.preventDefault(); openDrawer(e.target.dataset.key); } });
document.addEventListener("change", e => {
  const t = e.target;
  if (t.id === "hide-un") { S.ui.hideUntrained = t.checked; saveUi(); lastHTML = ""; render(); }
  if (t.id === "logy") { S.ui.log = t.checked; saveUi(); lastHTML = ""; render(); }
  if (t.id === "grp") { S.ui.group = t.checked; saveUi(); lastHTML = ""; render(); }
  if (t.id === "more") { S.ui.more = t.checked; saveUi(); lastHTML = ""; render(); }
  if (t.id === "heat-run") { S.ui.heatRun = t.value; saveUi(); lastHTML = ""; render(); }
  if (t.id === "cur-run") { S.ui.curRun = t.value; saveUi(); lastHTML = ""; render(); }
  if (t.id === "search-run") { S.ui.searchRun = t.value; saveUi(); lastHTML = ""; render(); }
});
document.addEventListener("input", e => {
  if (e.target.id === "sm") { S.ui.smooth = +e.target.value; $("#smv").textContent = S.ui.smooth.toFixed(2); saveUi(); clearTimeout(S._smt); S._smt = setTimeout(() => { lastHTML = ""; render(); }, 90); }
});
document.addEventListener("mousemove", e => { if (S.view === "curves" || S.view === "search") chartHover(e); });
document.addEventListener("mouseover", e => { if (!e.target.closest || (!e.target.closest(".chart") && !e.target.closest("#heat") && !e.target.closest(".bdot"))) hideTip(); });
window.addEventListener("hashchange", () => { route(); lastHTML = ""; render(false); scrollTo(0, 0); });

/* ------------------------------------------------------------------ boot */
async function tick() {
  if (document.hidden) return;
  const changed = await loadState();
  if (changed) { lastHTML = lastHTML; render(); } else renderLivebox();
}
(async function boot() {
  applyTheme(); route(); bindFilters();
  $("#nav").addEventListener("click", e => { const a = e.target.closest("a[data-v]"); if (a) { closeDrawer(); } });
  await loadState();
  if (!S.state) { $("#view").innerHTML = `<div class="empty"><b>Cannot reach the dashboard server</b>${esc(S.err || "")}</div>`; }
  else await render(false);
  if (!SNAP) setInterval(tick, 4000);
  if (!SNAP) setInterval(() => { if (S.state && S.view === "live") { lastHTML = ""; render(); } }, 15000);
})();
