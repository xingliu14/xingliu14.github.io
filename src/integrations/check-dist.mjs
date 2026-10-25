// The build's last step: checks what is about to be deployed, and fails the build with a list of
// every problem it finds. It keeps the site's promises: old URLs and heading anchors keep working, no
// path takes a name a repository's own Pages site may use (they share xingliu.me), every page
// carries its metadata and its card, nothing unfinished ships, the HTML is valid, and the
// stylesheet and script stay within their byte budgets.
import { readdir, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";
import { HtmlValidate } from "html-validate";
import htmlValidateConfig from "../../.htmlvalidate.json" with { type: "json" };
import headingIds from "../../scripts/heading-ids.json" with { type: "json" };
import { publishedPostFiles } from "../lib/post-files.mjs";

const SITE = "https://xingliu.me";

// The only names allowed at the top of dist/. Any other name under xingliu.me/ may be a repository's
// Pages site (/GameKingdom/, /YOY-Labs-Website/, …), so a new one fails the build until it is added
// here on purpose. Year folders (2020…2099) hold the posts.
const TOP_LEVEL = new Set([
  "_astro", "writing", "projects", "about", "og", "badges", "resume",
  "404.html", "index.html", "feed.xml", "search.json", "sitemap-index.xml", "sitemap-0.xml", "robots.txt",
  "favicon.ico", "favicon.svg", "apple-touch-icon.png", "avatar-240.jpg", "avatar-320.jpg", "style.css", "site.js",
]);
const YEAR = /^20\d\d$/;

// Besides every post: the pages people and robots expect, and the résumé's permanent address.
const REQUIRED = [
  "index.html", "writing/index.html", "projects/index.html", "about/index.html", "404.html",
  "feed.xml", "search.json", "sitemap-index.xml", "robots.txt", "resume/Xing_Liu_Resume_20250924.pdf",
  "style.css", "site.js",
];

// Fields of each search.json entry: URL, title, date, language, topic, summary and text.
const SEARCH_FIELDS = ["u", "t", "d", "l", "k", "s", "x"];

// Page chrome that must not reach feed readers: the contents box, back links, section numbers,
// table scrollers, tab stops and scripts (src/lib/feed-html.mjs removes them).
const FEED_CHROME = /<script\b|<[a-z][^>]*\stabindex=|<[a-z][^>]*\sclass="(?:[^"]*\s)?(?:toc|top|secno|table-scroll)(?:\s[^"]*)?"/i;

// The two files every page loads, served as written with their comments, and their budgets in
// bytes: raw, and gzipped at level 9 (CLAUDE.md).
const BUDGETS = { "style.css": [36_000, 11_000], "site.js": [18_000, 7_000] };

// An unfilled placeholder such as __TEXT_KB__ (src/integrations/page-weight.mjs), or filler text.
const UNFINISHED = /__[A-Z][A-Z0-9_]*__|\b[Ll]orem [Ii]psum\b/;

/** @type {Record<string, string>} */
const ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" };
/** Decodes HTML and XML character references, in one pass. @param {string} text */
const decode = (text) =>
  text.replace(/&(?:#x([\da-f]+)|#(\d+)|(amp|lt|gt|quot|apos));/gi, (_, hex, dec, name) =>
    hex ? String.fromCodePoint(parseInt(hex, 16)) : dec ? String.fromCodePoint(Number(dec)) : ENTITIES[name.toLowerCase()]);

/**
 * The value of attribute `name` on the first `<tag>` that has `has` among its attributes.
 * @param {string} html
 * @param {string} tag
 * @param {string} has  e.g. 'name="description"'; "" for the first `<tag>` of all
 * @param {string} name
 */
function attribute(html, tag, has, name) {
  for (const [element] of html.matchAll(new RegExp(`<${tag}\\s[^>]*>`, "g"))) {
    if (!element.includes(has)) continue;
    const value = new RegExp(`\\s${name}="([^"]*)"`).exec(element)?.[1];
    return value === undefined ? null : decode(value);
  }
  return null;
}

/**
 * What every check gets: dist/, every path in it and its HTML pages (relative to it), and the
 * published posts.
 * @typedef {{
 *   dir: URL,
 *   files: Set<string>,
 *   pages: string[],
 *   posts: { url: string }[],
 *   read: (path: string) => Promise<string | null>,
 * }} Dist
 */

/** 1. Only allowlisted names at the top level. @param {Dist} dist */
async function topLevel({ dir }) {
  return (await readdir(dir))
    .filter((name) => !TOP_LEVEL.has(name) && !YEAR.test(name))
    .map((name) => `/${name} is not on the TOP_LEVEL list: a repository's Pages site may need that path`);
}

/** 2. Every post, and the other pages that must exist. @param {Dist} dist */
async function required({ files, posts }) {
  return [...posts.map((post) => `${post.url.slice(1)}index.html`), ...REQUIRED]
    .filter((path) => !files.has(path))
    .map((path) => `${path} is missing`);
}

/** 3. The heading ids other sites link to. @param {Dist} dist */
async function anchors({ read }) {
  const problems = [];
  for (const [url, ids] of Object.entries(headingIds)) {
    if (!ids.length) continue;
    const html = await read(`${url.slice(1)}index.html`);
    if (html === null) {
      problems.push(`${url} is gone, and with it ${ids.length} heading ids other pages may link to`);
      continue;
    }
    const present = new Set([...html.matchAll(/\sid="([^"]*)"/g)].map((m) => decode(m[1])));
    const lost = ids.filter((id) => !present.has(id));
    if (lost.length) problems.push(`${url} lost heading ids: ${lost.join(", ")}`);
  }
  return problems;
}

/**
 * 4. Every page: one h1, a title, a description, a canonical link (or noindex on Not Found), its
 * card, its language, and nothing unfinished.
 * @param {Dist} dist
 */
async function eachPage({ files, pages, read }) {
  const problems = [];
  for (const path of pages) {
    const html = (await read(path)) ?? "";
    const h1s = html.match(/<h1[\s>]/g)?.length ?? 0;
    if (h1s !== 1) problems.push(`${path} has ${h1s} <h1> elements`);
    if (!/<title>[^<]*\S[^<]*<\/title>/.test(html)) problems.push(`${path} has no <title>`);
    if (!attribute(html, "meta", 'name="description"', "content")?.trim()) problems.push(`${path} has no meta description`);
    const canonical = attribute(html, "link", 'rel="canonical"', "href");
    if (path === "404.html") {
      if (canonical) problems.push(`${path} has a canonical link`);
      if (!attribute(html, "meta", 'name="robots"', "content")?.includes("noindex")) problems.push(`${path} is not marked noindex`);
    } else {
      if (!canonical) problems.push(`${path} has no canonical link`);
      const image = attribute(html, "meta", 'property="og:image"', "content");
      if (!image) problems.push(`${path} has no og:image`);
      else if (!image.startsWith(`${SITE}/og/`) || !files.has(image.slice(SITE.length + 1))) problems.push(`${path}: its og:image ${image} is not a card in dist/og/`);
    }
    if (!attribute(html, "html", "", "lang")) problems.push(`${path} has no <html lang>`);
    // Code may print anything, __init__ or __FILE__ included.
    const unfinished = UNFINISHED.exec(html.replace(/<(pre|code)\b[\s\S]*?<\/\1>/g, ""));
    if (unfinished) problems.push(`${path} still says "${unfinished[0]}"`);
  }
  return problems;
}

/** 5. The feed: every post once, in full, without the page's chrome. @param {Dist} dist */
async function feed({ posts, read }) {
  const xml = await read("feed.xml");
  if (xml === null) return [];
  const problems = [];
  const items = [...xml.matchAll(/<item>([\s\S]*?)<\/item>/g)].map((m) => m[1]);
  if (items.length !== posts.length) problems.push(`feed.xml has ${items.length} items for ${posts.length} posts`);
  for (const post of posts) {
    if (!xml.includes(`<link>${SITE}${post.url}</link>`)) problems.push(`feed.xml has no item for ${post.url}`);
  }
  for (const item of items) {
    const content = /<content:encoded>([\s\S]*)<\/content:encoded>/.exec(item)?.[1] ?? "";
    const html = content.startsWith("<![CDATA[") ? content.slice(9, -3) : decode(content);
    const chrome = FEED_CHROME.exec(html);
    if (chrome) problems.push(`feed.xml: the item for ${/<link>([^<]*)/.exec(item)?.[1]} carries page chrome: ${chrome[0]}`);
  }
  return problems;
}

/** 6. The search index: one entry per post, each with every field. @param {Dist} dist */
async function search({ posts, read }) {
  const json = await read("search.json");
  if (json === null) return [];
  /** @type {Record<string, unknown>[]} */
  let entries;
  try {
    entries = JSON.parse(json);
    if (!Array.isArray(entries)) throw new Error("not a list");
  } catch (error) {
    return [`search.json is not a list of entries: ${error}`];
  }
  const problems = [];
  if (entries.length !== posts.length) problems.push(`search.json has ${entries.length} entries for ${posts.length} posts`);
  const urls = new Set(entries.map((entry) => entry.u));
  for (const post of posts) if (!urls.has(post.url)) problems.push(`search.json has no entry for ${post.url}`);
  for (const entry of entries) {
    const missing = SEARCH_FIELDS.filter((field) => typeof entry[field] !== "string");
    if (missing.length) problems.push(`search.json: the entry for ${entry.u} lacks ${missing.join(", ")}`);
  }
  return problems;
}

/**
 * 7. The sitemap leaves the Not Found page out, and dates pages by the day: a time would be made
 * up (src/integrations/sitemap-dates.mjs).
 * @param {Dist} dist
 */
async function sitemap({ files, read }) {
  const problems = [];
  for (const path of [...files].filter((file) => /^sitemap-.*\.xml$/.test(file))) {
    const xml = (await read(path)) ?? "";
    for (const [, loc] of xml.matchAll(/<loc>([^<]*)<\/loc>/g)) {
      if (/\/404(\.html)?\/?$/.test(new URL(loc).pathname)) problems.push(`${path} lists ${loc}`);
    }
    for (const [, lastmod] of xml.matchAll(/<lastmod>([^<]*)<\/lastmod>/g)) {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(lastmod)) problems.push(`${path} has a <lastmod> that is not a plain date: ${lastmod}`);
    }
  }
  return problems;
}

/**
 * 8. Valid HTML. JSON has no comments, so here is why .htmlvalidate.json relaxes five rules:
 * - valid-id: post headings like id="1" are valid HTML5, and other sites link to them;
 * - prefer-native-element (region): <div role="region" tabindex="0"> is the standard scrollable
 *   table;
 * - no-redundant-role: the projects table's explicit roles keep its semantics on phones;
 * - long-title: several post titles are longer than 70 characters;
 * - void-style: posts keep the <br/> their author typed, as valid as <br>.
 * @param {Dist} dist
 */
async function validHtml({ dir, pages }) {
  const report = await new HtmlValidate(htmlValidateConfig).validateMultipleFiles(pages.map((path) => fileURLToPath(new URL(path, dir))));
  const root = fileURLToPath(dir);
  return report.results.flatMap(({ filePath, messages }) =>
    messages
      .filter(({ severity }) => severity === 2)
      .map(({ line, column, message, ruleId }) => `${filePath.slice(root.length)}:${line}:${column} ${message} (${ruleId})`));
}

/** 9. The stylesheet and the script stay within their budgets. @param {Dist} dist */
async function budgets({ read }) {
  const problems = [];
  for (const [name, [raw, gzipped]] of Object.entries(BUDGETS)) {
    const text = await read(name);
    if (text === null) continue; // required() reports a missing file
    const bytes = Buffer.byteLength(text);
    const packed = gzipSync(text, { level: 9 }).length;
    if (bytes > raw) problems.push(`${name} is ${bytes} bytes, over its budget of ${raw}`);
    if (packed > gzipped) problems.push(`${name} is ${packed} bytes gzipped, over its budget of ${gzipped}`);
  }
  return problems;
}

const CHECKS = [topLevel, required, anchors, eachPage, feed, search, sitemap, validHtml, budgets];

/** @returns {import("astro").AstroIntegration} */
export default function checkDist() {
  return {
    name: "check-dist",
    hooks: {
      "astro:build:done": async ({ dir, logger }) => {
        const files = new Set(await readdir(dir, { recursive: true }));
        /** @type {Dist} */
        const dist = {
          dir,
          files,
          pages: [...files].filter((path) => path.endsWith(".html")),
          posts: publishedPostFiles(),
          read: (path) => readFile(new URL(path, dir), "utf8").catch(() => null),
        };
        const problems = [];
        for (const check of CHECKS) problems.push(...(await check(dist)));
        if (problems.length) {
          const list = problems.map((problem) => `  - ${problem}`).join("\n");
          throw new Error(`check-dist found ${problems.length} problem${problems.length === 1 ? "" : "s"} in dist/:\n${list}`);
        }
        logger.info(`${dist.pages.length} pages and ${dist.posts.length} posts checked: all good`);
      },
    },
  };
}
