// Hand-drawn (Excalidraw-style) architecture diagram: rough.js shapes + the Virgil font.
import rough from "roughjs";
import fs from "fs";

const W = 1640, H = 830;
const gen = rough.generator();
let seed = 7;
const out = [];
const INK = "#1e1e1e";
const C = { yellow: "#ffec99", blue: "#a5d8ff", green: "#b2f2bb", red: "#ffc9c9", violet: "#d0bfff", orange: "#ffd8a8", grey: "#e9ecef" };

function emit(drawable) {
  for (const p of gen.toPaths(drawable)) {
    out.push(`<path d="${p.d}" stroke="${p.stroke}" stroke-width="${p.strokeWidth}" fill="${p.fill || "none"}" stroke-linecap="round" stroke-linejoin="round"/>`);
  }
}
const opt = (o = {}) => ({ seed: seed++, roughness: 1.3, bowing: 1.2, stroke: INK, strokeWidth: 2, ...o });
function box(x, y, w, h, fill, o = {}) {
  emit(gen.rectangle(x, y, w, h, opt({ fill, fillStyle: "hachure", hachureGap: 7, fillWeight: 1.2, hachureAngle: -41, ...o })));
}
function ellipse(cx, cy, w, h, fill) {
  emit(gen.ellipse(cx, cy, w, h, opt({ fill, fillStyle: "solid", strokeWidth: 1.6, roughness: 1 })));
}
function text(x, y, s, { size = 22, anchor = "start", color = INK, weight = "normal" } = {}) {
  const esc = s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  out.push(`<text x="${x}" y="${y}" font-family="Virgil" font-size="${size}" fill="${color}" text-anchor="${anchor}" font-weight="${weight}">${esc}</text>`);
}
function arrow(pts, { color = INK, width = 2, head = true } = {}) {
  emit(gen.curve(pts, opt({ stroke: color, strokeWidth: width, roughness: 1 })));
  if (!head) return;
  const [x2, y2] = pts[pts.length - 1], [x1, y1] = pts[pts.length - 2];
  const a = Math.atan2(y2 - y1, x2 - x1), L = 16;
  for (const d of [0.45, -0.45]) {
    emit(gen.line(x2, y2, x2 - L * Math.cos(a + d), y2 - L * Math.sin(a + d), opt({ stroke: color, strokeWidth: width, roughness: 0.8 })));
  }
}

// title
text(40, 58, "how the bot plays (and learns) Geometry Dash", { size: 34 });
emit(gen.curve([[40, 72], [300, 76], [560, 70], [700, 74]], opt({ stroke: "#e8590c", strokeWidth: 3, roughness: 1.6 })));

// 1. Geometry Dash + bridge mod
box(40, 120, 380, 300, C.yellow);
text(64, 164, "Geometry Dash", { size: 30 });
text(64, 196, "+ our C++ bridge mod", { size: 22, color: "#5c5f66" });
text(64, 246, "• freezes the game clock", { size: 21 });
text(64, 280, "• runs exactly 4 physics ticks", { size: 21 });
text(84, 306, "per STEP, then replies", { size: 21 });
text(64, 340, "• save / load the whole game", { size: 21 });
text(84, 366, "in under a millisecond", { size: 21 });
text(64, 400, "~16x faster than real time", { size: 19, color: "#e8590c" });

// 2. Python environment
box(600, 120, 360, 300, C.blue);
text(624, 164, "Python environment", { size: 30 });
text(624, 196, "(Gymnasium-style)", { size: 22, color: "#5c5f66" });
text(624, 246, "• encoder: game state", { size: 21 });
text(644, 272, "→ 234 numbers", { size: 21 });
text(624, 306, "• reward = progress", { size: 21 });
text(624, 340, "• curriculum: practice from", { size: 21 });
text(644, 366, "start positions", { size: 21 });

// TCP arrows between 1 and 2
arrow([[420, 230], [510, 218], [600, 228]]);
text(510, 205, "state", { size: 20, anchor: "middle" });
arrow([[600, 320], [510, 334], [420, 322]]);
text(510, 362, "STEP: hold?", { size: 20, anchor: "middle" });
text(510, 288, "TCP", { size: 17, anchor: "middle", color: "#868e96" });

// 3. Policy network with neurons
box(1160, 120, 380, 300, C.green);
text(1184, 164, "policy network", { size: 30 });
const cols = [[1205, 4, "234"], [1285, 5, "128"], [1365, 5, "128"], [1440, 2, ""]];
const pos = cols.map(([x, n]) => Array.from({ length: n }, (_, i) => [x, 205 + (i + 0.5) * (170 / n) + 8]));
for (let c = 0; c < pos.length - 1; c++)
  for (const a of pos[c]) for (const b of pos[c + 1])
    emit(gen.line(a[0] + 9, a[1], b[0] - 9, b[1], opt({ stroke: "#868e96", strokeWidth: 0.8, roughness: 0.6 })));
pos.forEach((col, c) => col.forEach(([x, y], i) => ellipse(x, y, 18, 18, c === 3 ? (i ? C.orange : C.blue) : "#ffffff")));
cols.forEach(([x, , label]) => label && text(x, 404, label, { size: 18, anchor: "middle" }));
text(1460, 263, "release", { size: 17, color: "#495057" });
text(1460, 348, "jump", { size: 17, color: "#495057" });

// env <-> policy
arrow([[960, 230], [1060, 218], [1160, 228]]);
text(1060, 205, "234 numbers", { size: 20, anchor: "middle" });
arrow([[1160, 320], [1060, 334], [960, 322]]);
text(1060, 362, "jump / release", { size: 20, anchor: "middle" });
text(1060, 386, "60× a second", { size: 17, anchor: "middle", color: "#868e96" });

// 4. PPO training
box(1160, 520, 380, 150, C.violet);
text(1184, 562, "PPO training", { size: 28 });
text(1184, 598, "try, see what worked,", { size: 21 });
text(1184, 626, "make it more likely next time", { size: 21 });
arrow([[1280, 420], [1270, 470], [1280, 520]]);
text(1255, 478, "experience", { size: 18, anchor: "end" });
arrow([[1420, 520], [1432, 470], [1420, 420]]);
text(1446, 478, "better weights", { size: 18 });

// 5. Checkpoint search + behaviour cloning
box(40, 520, 380, 150, C.red);
text(64, 562, "checkpoint search", { size: 28 });
text(64, 598, "rewinds the game thousands", { size: 21 });
text(64, 626, "of times to find a route", { size: 21 });
arrow([[230, 420], [222, 470], [230, 520]]);
text(244, 478, "save / load", { size: 18 });

box(600, 520, 360, 150, C.orange);
text(624, 562, "behaviour cloning", { size: 28 });
text(624, 598, "copy the found route", { size: 21 });
text(624, 626, "into the network", { size: 21 });
arrow([[420, 596], [510, 586], [600, 596]]);
text(510, 572, "route", { size: 20, anchor: "middle" });
arrow([[960, 596], [1060, 606], [1160, 596]]);
text(1060, 640, "keeps PPO from", { size: 17, anchor: "middle", color: "#495057" });
text(1060, 662, "forgetting it", { size: 17, anchor: "middle", color: "#495057" });

// side note
text(40, 750, "the hard parts of extreme demons (Bloodbath, Rauchkammer) needed the search path at the bottom;", { size: 20, color: "#495057" });
text(40, 782, "Stereo Madness was learned by PPO alone.", { size: 20, color: "#495057" });

const font = fs.readFileSync("Virgil.woff2").toString("base64");
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
<style>@font-face{font-family:Virgil;src:url(data:font/woff2;base64,${font}) format("woff2");}</style>
<rect width="100%" height="100%" fill="#ffffff"/>
${out.join("\n")}
</svg>`;
fs.writeFileSync("architecture.svg", svg);
fs.writeFileSync("architecture.html", `<!doctype html><html><body style="margin:0;background:#fff">${svg}</body></html>`);
console.log("ok", out.length);
