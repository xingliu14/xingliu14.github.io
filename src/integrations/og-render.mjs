// The Open Graph cards, drawn without a browser: Satori lays each card out as SVG and resvg turns it
// into a 1200x630 PNG. src/integrations/og-cards.mjs calls renderCard() once per entry of
// /og/cards.json (src/pages/og/cards.json.ts).
//
// Each card is a picture of its page as Navigator drew it: the grey page, the bracketed nav, the
// shaded rule, the title in Times bold, the byline in Verdana, and three of the 88x31 buttons. The
// fonts come from npm devDependencies (metric twins of the Core Fonts, plus Noto Serif SC cut down
// to each card's characters); they are read at build time only and never served.
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { Resvg } from "@resvg/resvg-js";
import satori from "satori";
import subsetFont from "subset-font";
import { badgeSvg } from "../../scripts/pixels.mjs";

/**
 * One entry of /og/cards.json. A "home" card (Home and About) shows the avatar beside `title`,
 * `sub` and `text`; "section" and "post" cards show `crumbs`, `title`, a `byline` (posts only) and
 * the `description`.
 * @typedef {{
 *   file: string,
 *   kind: "home" | "section" | "post",
 *   current: "Home" | "Writing" | "Projects" | "About",
 *   lang?: "en" | "zh-CN",
 *   title: string,
 *   sub?: string,
 *   text?: string,
 *   crumbs?: string[],
 *   byline?: (string | { link: string })[],
 *   description?: string,
 * }} Card
 */

const require = createRequire(import.meta.url);
/** @param {string} file a path inside node_modules */
const font = (file) => readFileSync(require.resolve(file));

/** @type {Record<string, Buffer> | undefined} */
let fonts;
// Loaded on the first card, not on import: astro.config.mjs imports this for every command.
function loadFonts() {
  fonts ??= {
    tinosBold: font("@expo-google-fonts/tinos/700Bold/Tinos_700Bold.ttf"),
    gelasio: font("@expo-google-fonts/gelasio/400Regular/Gelasio_400Regular.ttf"),
    gelasioItalic: font("@expo-google-fonts/gelasio/400Regular_Italic/Gelasio_400Regular_Italic.ttf"),
    dejavu: font("dejavu-fonts-ttf/ttf/DejaVuSans.ttf"),
    dejavuBold: font("dejavu-fonts-ttf/ttf/DejaVuSans-Bold.ttf"),
    notoSerifScBold: font("@expo-google-fonts/noto-serif-sc/700Bold/NotoSerifSC_700Bold.ttf"),
    notoSerifSc: font("@expo-google-fonts/noto-serif-sc/400Regular/NotoSerifSC_400Regular.ttf"),
  };
  return fonts;
}

// Only the Han characters and CJK punctuation a card uses are kept, so a card's Chinese font costs
// kilobytes, not the full 15 MB.
const CJK = /[⺀-鿿　-〿＀-￯“”—…·]/gu;
/** @param {string} text */
async function cjkFonts(text) {
  const chars = [...new Set(text.match(CJK) ?? [])].join("");
  if (!chars) return [];
  const { notoSerifScBold, notoSerifSc } = loadFonts();
  const [bold, regular] = await Promise.all([
    subsetFont(notoSerifScBold, chars, { targetFormat: "sfnt" }),
    subsetFont(notoSerifSc, chars, { targetFormat: "sfnt" }),
  ]);
  return [
    { name: "Noto Serif SC", data: bold, weight: 700, style: "normal" },
    { name: "Noto Serif SC", data: regular, weight: 400, style: "normal" },
  ];
}

/**
 * A Satori element: like React's, without React.
 * @param {string} type
 * @param {Record<string, unknown>} style
 * @param {...unknown} children
 */
const h = (type, style, ...children) => ({ type, props: { style, children: children.flat().filter((c) => c != null && c !== false) } });
/** @param {string} mime @param {string | Buffer} data */
const dataUri = (mime, data) => `data:${mime};base64,${Buffer.from(data).toString("base64")}`;
/** @param {"rss" | "astro" | "any"} key */
const badge = (key) => ({ type: "img", props: { src: dataUri("image/svg+xml", badgeSvg(key)), width: 176, height: 62, style: { marginLeft: 18 } } });

const C = { page: "#c0c0c0", ink: "#000000", link: "#0000ee", lo: "#808080", hi: "#ffffff" };
const SERIF = "Tinos, Noto Serif SC";
const TEXT = "Gelasio, Noto Serif SC";
const SANS = "DejaVu Sans, Noto Serif SC";
/** @param {string} text @param {Record<string, unknown>} [extra] */
const link = (text, extra = {}) => h("span", { color: C.link, textDecoration: "underline", ...extra }, text);
/**
 * A run of text: strings stay plain, { link } becomes a blue underlined link. Spaces inside a part
 * become no-break spaces, so a line can only break between parts.
 * @param {(string | { link: string })[]} parts
 */
const run = (parts) => parts.map((p) => (typeof p === "string" ? p.replace(/ /g, "\u00a0") : link(p.link)));

/** @param {Card["current"]} current */
function navRow(current) {
  const items = ["Home", "Writing", "Projects", "About", "Search"];
  const parts = [h("span", { marginRight: 12 }, "[")];
  items.forEach((it, i) => {
    parts.push(it === current ? h("span", { fontWeight: 700, color: C.ink }, it) : link(it));
    parts.push(h("span", { margin: "0 12px" }, i === items.length - 1 ? "]" : "|"));
  });
  return h("div", { display: "flex", justifyContent: "center", fontFamily: TEXT, fontSize: 30, color: C.ink }, parts);
}
const rule = () => h("div", { display: "flex", height: 0, borderTop: `2px solid ${C.lo}`, borderBottom: `2px solid ${C.hi}`, margin: "20px 0 0" });
const footRow = () =>
  h("div", { position: "absolute", left: 70, right: 70, bottom: 38, display: "flex", alignItems: "flex-end", justifyContent: "space-between" },
    h("span", { fontFamily: "Gelasio", fontStyle: "italic", fontSize: 34, color: C.link, textDecoration: "underline" }, "xingliu.me"),
    h("div", { display: "flex" }, badge("rss"), badge("astro"), badge("any")));
/** @param {unknown[]} children */
const frame = (children) =>
  h("div", { width: 1200, height: 630, display: "flex", background: C.page, position: "relative", color: C.ink },
    h("div", { display: "flex", flexDirection: "column", width: 1060, marginLeft: 70, marginTop: 34 }, children),
    footRow());
/** @param {string[]} crumbs */
const crumbRow = (crumbs) =>
  h("div", { display: "flex", marginTop: 26, fontFamily: SANS, fontSize: 24 }, crumbs.flatMap((c, i) => (i ? [h("span", { margin: "0 10px" }, "›"), link(c)] : [link(c)])));

/**
 * The title's size by its length: 88, 74 or 62 px. Han characters are about twice as wide as
 * Latin letters, so they count double.
 * @param {string} title
 */
function titleSize(title) {
  const width = [...title].reduce((n, ch) => n + (/[⺀-鿿＀-￯]/u.test(ch) ? 2 : 1), 0);
  return width <= 22 ? 88 : width <= 36 ? 74 : 62;
}

/**
 * A short Latin subtitle after a full-width colon stays on one line, as in the post's own h1
 * (src/pages/[year]/[month]/[day]/[slug].astro): "侦探小说：" may end a line, "He’s Replaceable"
 * never breaks.
 * @param {string} title
 */
function titleParts(title) {
  const subtitle = /^(.+：)([^\p{Script=Han}：]{1,24})$/u.exec(title);
  return subtitle ? [subtitle[1], h("span", { whiteSpace: "nowrap" }, subtitle[2])] : [title];
}

/**
 * "post" and "section" cards: crumbs, title, the byline (posts only) and the description. A
 * Chinese description is balanced across its lines, so its last line never holds a lone character.
 * @param {Card} card
 */
function textCard({ current, lang, crumbs = [], title, byline, description }) {
  return frame([
    navRow(current),
    rule(),
    crumbRow(crumbs),
    h("div", { display: "flex", flexWrap: "wrap", marginTop: 10, fontFamily: SERIF, fontWeight: 700, fontSize: titleSize(title), lineHeight: 1.08, letterSpacing: -0.5 }, titleParts(title)),
    byline ? h("div", { display: "flex", flexWrap: "wrap", marginTop: 16, fontFamily: SANS, fontSize: 26 }, run(byline)) : null,
    description ? h("div", { display: "flex", marginTop: 18, fontFamily: TEXT, fontSize: 33, lineHeight: 1.4, maxWidth: 1040, ...(lang === "zh-CN" && { textWrap: "balance" }) }, description) : null,
  ]);
}

/**
 * "home" cards (Home and About): the avatar in a sunken frame beside the title and one sentence.
 * @param {Card} card
 * @param {Buffer} avatar
 */
function homeCard({ current, title, sub, text }, avatar) {
  return frame([
    navRow(current),
    rule(),
    h("div", { display: "flex", marginTop: 40, alignItems: "flex-start" },
      h("div", { display: "flex", borderTop: `3px solid ${C.lo}`, borderLeft: `3px solid ${C.lo}`, borderRight: `3px solid ${C.hi}`, borderBottom: `3px solid ${C.hi}` },
        { type: "img", props: { src: dataUri("image/jpeg", avatar), width: 216, height: 216 } }),
      h("div", { display: "flex", flexDirection: "column", marginLeft: 48, marginTop: 6 },
        h("div", { display: "flex", fontFamily: SERIF, fontWeight: 700, fontSize: 82, lineHeight: 1.05 }, title),
        h("div", { display: "flex", marginTop: 8, fontFamily: "Noto Serif SC", fontSize: 40, letterSpacing: 4 }, sub),
        h("div", { display: "flex", marginTop: 22, fontFamily: TEXT, fontSize: 32, lineHeight: 1.4, maxWidth: 760 }, text))),
  ]);
}

/**
 * Draws one card as PNG bytes.
 * @param {Card} card
 * @param {{ avatar: Buffer }} assets  the avatar JPEG that home cards show
 */
export async function renderCard(card, { avatar }) {
  const element = card.kind === "home" ? homeCard(card, avatar) : textCard(card);
  const f = loadFonts();
  const svg = await satori(element, {
    width: 1200,
    height: 630,
    fonts: [
      { name: "Tinos", data: f.tinosBold, weight: 700, style: "normal" },
      { name: "Gelasio", data: f.gelasio, weight: 400, style: "normal" },
      { name: "Gelasio", data: f.gelasioItalic, weight: 400, style: "italic" },
      { name: "DejaVu Sans", data: f.dejavu, weight: 400, style: "normal" },
      { name: "DejaVu Sans", data: f.dejavuBold, weight: 700, style: "normal" },
      ...(await cjkFonts(JSON.stringify(card))),
    ],
  });
  return new Resvg(svg, { fitTo: { mode: "width", value: 1200 } }).render().asPng();
}
