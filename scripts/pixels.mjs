// Pixel art for xingliu.me: the four 88x31 buttons and the star favicon, as crisp SVG. No browser
// and no fonts are needed: every letter is drawn from the bitmap fonts below.
//
// The output is committed to public/, so run this by hand, only when the art changes:
//   node scripts/pixels.mjs
// The Open Graph cards (src/integrations/og-render.mjs) draw the buttons with badgeSvg() at build
// time. What each button claims, and its alt text, live in src/components/Badges.astro.
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Resvg } from "@resvg/resvg-js";

// 5x7 capitals and digits ("#" = ink). Classic dot-matrix shapes.
const BIG = {
  A: [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
  B: ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
  C: [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
  D: ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
  E: ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
  F: ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
  G: [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"],
  H: ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
  I: ["###", ".#.", ".#.", ".#.", ".#.", ".#.", "###"],
  J: ["..###", "...#.", "...#.", "...#.", "#..#.", "#..#.", ".##.."],
  K: ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
  L: ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
  M: ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
  N: ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"],
  O: [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
  P: ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
  Q: [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
  R: ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
  S: [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
  T: ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
  U: ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
  V: ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
  W: ["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "#.#.#", ".#.#."],
  X: ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
  Y: ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
  Z: ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
  0: [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
  1: ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
  2: [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
  3: ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
  4: ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
  5: ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
  6: ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
  7: ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
  8: [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
  9: [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
  ".": [".", ".", ".", ".", ".", ".", "#"],
  "!": ["#", "#", "#", "#", "#", ".", "#"],
  "-": ["...", "...", "...", "###", "...", "...", "..."],
  " ": ["..", "..", "..", "..", "..", "..", ".."],
};

// 3x5 small capitals (M, N and W are wider so they stay legible).
const SMALL = {
  A: [".#.", "#.#", "###", "#.#", "#.#"],
  B: ["##.", "#.#", "##.", "#.#", "##."],
  C: [".##", "#..", "#..", "#..", ".##"],
  D: ["##.", "#.#", "#.#", "#.#", "##."],
  E: ["###", "#..", "##.", "#..", "###"],
  F: ["###", "#..", "##.", "#..", "#.."],
  G: [".##", "#..", "#.#", "#.#", ".##"],
  H: ["#.#", "#.#", "###", "#.#", "#.#"],
  I: ["###", ".#.", ".#.", ".#.", "###"],
  K: ["#.#", "#.#", "##.", "#.#", "#.#"],
  L: ["#..", "#..", "#..", "#..", "###"],
  M: ["#...#", "##.##", "#.#.#", "#...#", "#...#"],
  N: ["#..#", "##.#", "#.##", "#..#", "#..#"],
  O: ["###", "#.#", "#.#", "#.#", "###"],
  P: ["##.", "#.#", "##.", "#..", "#.."],
  R: ["##.", "#.#", "##.", "#.#", "#.#"],
  S: [".##", "#..", ".#.", "..#", "##."],
  T: ["###", ".#.", ".#.", ".#.", ".#."],
  U: ["#.#", "#.#", "#.#", "#.#", "###"],
  V: ["#.#", "#.#", "#.#", "#.#", ".#."],
  W: ["#...#", "#...#", "#.#.#", "#.#.#", ".#.#."],
  Y: ["#.#", "#.#", ".#.", ".#.", ".#."],
  " ": ["..", "..", "..", "..", ".."],
};

/**
 * @param {Record<string, string[]>} font
 * @param {string} text
 */
function textPixels(font, text) {
  /** @type {[number, number][]} */
  const out = [];
  let x = 0;
  for (const ch of text) {
    const g = font[ch];
    if (!g) throw new Error(`no glyph for "${ch}"`);
    g.forEach((row, y) => [...row].forEach((c, dx) => c === "#" && out.push([x + dx, y])));
    x += g[0].length + 1;
  }
  return { pixels: out, width: x - 1 };
}

/**
 * Merges each row's pixels into runs, so the SVG stays small.
 * @param {[number, number][]} pixels
 */
function runs(pixels) {
  /** @type {Map<number, number[]>} */
  const byRow = new Map();
  for (const [x, y] of pixels) (byRow.get(y) ?? byRow.set(y, []).get(y))?.push(x);
  const rects = [];
  for (const [y, xs] of byRow) {
    xs.sort((a, b) => a - b);
    let start = xs[0], prev = xs[0];
    for (const x of xs.slice(1).concat([Infinity])) {
      if (x === prev + 1) { prev = x; continue; }
      rects.push(`M${start} ${y}h${prev - start + 1}v1h-${prev - start + 1}z`);
      start = prev = x;
    }
  }
  return rects.join("");
}

/** @typedef {{ file: string, top: string, bottom: string, bg: string, light: string, dark: string, topInk: string, ink: string }} Badge */

/**
 * One 88x31 button: a raised two-pixel bevel, a small-capitals line over a large one.
 * @param {Badge} badge
 */
function badge({ top, bottom, bg, light, dark, topInk, ink }) {
  const t = textPixels(SMALL, top);
  const b = textPixels(BIG, bottom);
  const tx = Math.floor((88 - t.width) / 2), bx = Math.floor((88 - b.width) / 2);
  /** @type {(ps: [number, number][], dx: number, dy: number) => [number, number][]} */
  const shift = (ps, dx, dy) => ps.map(([x, y]) => [x + dx, y + dy]);
  return `<svg xmlns="http://www.w3.org/2000/svg" width="88" height="31" viewBox="0 0 88 31" shape-rendering="crispEdges">`
    + `<rect width="88" height="31" fill="${bg}"/>`
    + `<path fill="#ffffff" d="M0 0h88v1h-87v30h-1z"/>`
    + `<path fill="#000000" d="M87 0h1v31h-88v-1h87z"/>`
    + `<path fill="${light}" d="M1 1h86v1h-85v28h-1z"/>`
    + `<path fill="${dark}" d="M86 1h1v29h-86v-1h85z"/>`
    + `<path fill="${topInk}" d="${runs(shift(t.pixels, tx, 8))}"/>`
    + `<path fill="${ink}" d="${runs(shift(b.pixels, bx, 16))}"/>`
    + `</svg>\n`;
}

/** The four buttons, by the key badgeSvg() takes. @type {Record<"rss" | "astro" | "pages" | "any", Badge>} */
const BADGES = {
  rss: { file: "rss.svg", top: "SUBSCRIBE", bottom: "RSS FEED", bg: "#ff6600", light: "#ff9966", dark: "#993300", topInk: "#000000", ink: "#000000" },
  astro: { file: "astro.svg", top: "BUILT WITH", bottom: "ASTRO", bg: "#000080", light: "#3333cc", dark: "#000033", topInk: "#c0c0c0", ink: "#ffffff" },
  pages: { file: "github-pages.svg", top: "HOSTED ON", bottom: "GITHUB PAGES", bg: "#000000", light: "#666666", dark: "#000000", topInk: "#c0c0c0", ink: "#ffffff" },
  any: { file: "any-browser.svg", top: "VIEWABLE WITH", bottom: "ANY BROWSER", bg: "#008000", light: "#33cc33", dark: "#003300", topInk: "#ffffff", ink: "#ffffff" },
};

/**
 * One button as SVG, 88x31.
 * @param {keyof typeof BADGES} key
 */
export const badgeSvg = (key) => badge(BADGES[key]);

// The favicon: a 16x16 star (星 means star), outlined in black with a lit top-left edge.
const STAR = [
  ".......##.......",
  ".......##.......",
  "......####......",
  "......####......",
  ".....######.....",
  "################",
  ".##############.",
  "..############..",
  "...##########...",
  "....########....",
  "....########....",
  "...####..####...",
  "...###....###...",
  "..###......###..",
  "..##........##..",
  ".#............#.",
];

function starSvg() {
  /** @type {(x: number, y: number) => boolean} */
  const inside = (x, y) => y >= 0 && y < 16 && x >= 0 && x < 16 && STAR[y][x] === "#";
  /** @type {Record<string, [number, number][]>} */
  const groups = { "#000000": [], "#ffcc00": [], "#ffff99": [], "#cc6600": [] };
  for (let y = 0; y < 16; y++)
    for (let x = 0; x < 16; x++) {
      if (!inside(x, y)) continue;
      const edge = !inside(x - 1, y) || !inside(x + 1, y) || !inside(x, y - 1) || !inside(x, y + 1);
      if (edge) groups["#000000"].push([x, y]);
      else if (!inside(x - 1, y - 1) || !inside(x - 1, y) || !inside(x, y - 1)) groups["#ffff99"].push([x, y]);
      else if (!inside(x + 1, y + 1) || !inside(x + 1, y) || !inside(x, y + 1)) groups["#cc6600"].push([x, y]);
      else groups["#ffcc00"].push([x, y]);
    }
  return `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" shape-rendering="crispEdges">`
    + Object.entries(groups).filter(([, p]) => p.length).map(([c, p]) => `<path fill="${c}" d="${runs(p)}"/>`).join("")
    + `</svg>\n`;
}

/**
 * A Windows .ico holding PNG images (valid since Windows Vista, and in every browser).
 * @param {{ size: number, data: Buffer }[]} pngs
 */
function icoFromPngs(pngs) {
  const header = Buffer.alloc(6 + 16 * pngs.length);
  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(pngs.length, 4);
  let offset = header.length;
  pngs.forEach(({ size, data }, i) => {
    const e = 6 + 16 * i;
    header.writeUInt8(size >= 256 ? 0 : size, e);
    header.writeUInt8(size >= 256 ? 0 : size, e + 1);
    header.writeUInt8(0, e + 2);
    header.writeUInt8(0, e + 3);
    header.writeUInt16LE(1, e + 4);
    header.writeUInt16LE(32, e + 6);
    header.writeUInt32LE(data.length, e + 8);
    header.writeUInt32LE(offset, e + 12);
    offset += data.length;
  });
  return Buffer.concat([header, ...pngs.map((p) => p.data)]);
}

// Run directly (not imported by the card renderer): write the buttons and icons into public/.
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const out = (/** @type {string} */ name) => fileURLToPath(new URL(`../public/${name}`, import.meta.url));
  /** @type {(svg: string, width: number) => Buffer} */
  const png = (svg, width) => new Resvg(svg, { fitTo: { mode: "width", value: width }, shapeRendering: 0 }).render().asPng();

  for (const b of Object.values(BADGES)) writeFileSync(out(`badges/${b.file}`), badge(b));
  const star = starSvg();
  writeFileSync(out("favicon.svg"), star);
  writeFileSync(out("favicon.ico"), icoFromPngs([16, 32, 48].map((size) => ({ size, data: png(star, size) }))));
  // The Apple touch icon: the star at 9x on the grey page, with room for iOS's rounded corners.
  const touch = `<svg xmlns="http://www.w3.org/2000/svg" width="180" height="180" viewBox="0 0 180 180" shape-rendering="crispEdges"><rect width="180" height="180" fill="#c0c0c0"/><g transform="translate(18 18) scale(9)">${star.replace(/^<svg[^>]*>/, "").replace(/<\/svg>\s*$/, "")}</g></svg>`;
  writeFileSync(out("apple-touch-icon.png"), png(touch, 180));
  console.log("wrote public/badges/*.svg, favicon.svg, favicon.ico and apple-touch-icon.png");
}
