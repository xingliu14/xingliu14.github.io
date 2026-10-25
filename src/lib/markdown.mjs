// The Markdown pipeline for posts, run by Sätteri (astro.config.mjs):
//   mdast: normalizeHeadings   the post's outline starts at h2, under the page's own h1;
//   hast:  rougeCode           code keeps the markup kramdown and Rouge emitted, highlighted by Prism;
//          postStructure       heading ids and sizes, table regions, the contents box, section
//                              numbers, back links and the reading time.
// Templates never split post HTML: everything a post needs inside its body is written here.
//
// Astro's content cache does not notice changes to this file or to src/data/posts.json (which
// postStructure reads), so builds run with --force; use `astro dev --force` after editing either.
import Prism from "prismjs";
import loadLanguages from "prismjs/components/index.js";
import Slugger from "github-slugger";
import { isFiction } from "./post-meta.mjs";

// ---------------------------------------------------------------------------------- rougeCode

// Load every grammar up front: loading on demand makes a grammar's optional parts (regex inside
// JavaScript, say) depend on which post happened to render first.
loadLanguages.silent = true;
loadLanguages();

// Fence names Rouge understood that Prism calls something else.
const PRISM_ALIASES = {
  console: "shell-session",
  terminal: "shell-session",
  shell_session: "shell-session",
  zsh: "bash",
  "c++": "cpp",
  cuda: "cpp",
  golang: "go",
  make: "makefile",
};

const escapeHtml = (text) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const escapeAttr = (text) => escapeHtml(text).replace(/"/g, "&quot;");
const clamp = (n, min, max) => Math.max(min, Math.min(max, n));

const warned = new Set();

function highlight(code, lang) {
  if (lang === "plaintext") return escapeHtml(code);
  const prismLang = PRISM_ALIASES[lang] ?? lang;
  const grammar = Prism.languages[prismLang];
  if (!grammar) {
    if (!warned.has(lang)) console.warn(`[rouge-code] no Prism grammar for "${lang}"; rendering it as plain text`);
    warned.add(lang);
    return escapeHtml(code);
  }
  return Prism.highlight(code, grammar, prismLang);
}

function languageOf(code) {
  const classes = code.properties?.className;
  const cls = Array.isArray(classes) ? classes.find((c) => typeof c === "string" && c.startsWith("language-")) : undefined;
  return cls ? cls.slice("language-".length) : "plaintext";
}

/**
 * Fenced code becomes
 * `<div class="language-x highlighter-rouge"><div class="highlight"><pre class="highlight" tabindex="0"><code>…`
 * and inline code gets `class="language-plaintext highlighter-rouge"`, as kramdown emitted. The
 * tabindex lets keyboard users scroll a well whose lines are wider than the column.
 */
export const rougeCode = {
  name: "rouge-code",
  element: [
    {
      filter: ["pre"],
      visit(node, ctx) {
        const code = node.children?.find((c) => c.type === "element" && c.tagName === "code");
        if (!code) return;
        const lang = languageOf(code);
        const html = highlight(ctx.textContent(code), lang);
        return {
          type: "raw",
          value: `<div class="language-${escapeAttr(lang)} highlighter-rouge"><div class="highlight"><pre class="highlight" tabindex="0"><code>${html}</code></pre></div></div>`,
        };
      },
    },
    {
      filter: ["code"],
      visit(node, ctx) {
        const parent = ctx.parent(node);
        if (parent?.type === "element" && parent.tagName === "pre") return;
        ctx.setProperty(node, "className", ["language-plaintext", "highlighter-rouge"]);
      },
    },
  ],
};

// ---------------------------------------------------------------------------- normalizeHeadings

/**
 * Renumbers in-body headings so the post's outline starts at h2, under the page's h1, keeping
 * their relative structure. Posts that start at `#` are demoted one level; posts that start lower
 * (`##`, `####`) move up, where a plain demotion would skip levels. The shift is left in
 * `ctx.data.headingShift` for postStructure, which keeps each heading's original size.
 */
export const normalizeHeadings = {
  name: "normalize-headings",
  before(root, ctx) {
    let min = 7;
    const walk = (node) => {
      if (node.type === "heading") min = Math.min(min, node.depth);
      (node.children ?? []).forEach(walk);
    };
    walk(root);
    ctx.data.headingShift = min === 7 ? 0 : 2 - min;
  },
  heading(node, ctx) {
    const depth = clamp(node.depth + ctx.data.headingShift, 2, 6);
    if (depth !== node.depth) ctx.setProperty(node, "depth", depth);
  },
};

// -------------------------------------------------------------------------------- postStructure

const isHeading = (node) => node.type === "element" && /^h[2-6]$/.test(node.tagName);

const HAN = /\p{Script=Han}/gu;
// A Latin word inside Chinese text: "Slack", "PoC", "North", "it's", "v1.0", "e-mail".
const LATIN_WORD = /[A-Za-z0-9]+(?:['’.\-][A-Za-z0-9]+)*/g;

/**
 * Reading time. English counts whitespace-separated words at 220 a minute; Chinese counts Han
 * characters plus Latin words at 400 a minute.
 * @returns {{ minutes: number, count: number, unit: "en" | "zh" }}
 */
function readingTime(text, lang) {
  if (lang.startsWith("zh")) {
    const han = (text.match(HAN) ?? []).length;
    const latin = (text.replace(HAN, " ").match(LATIN_WORD) ?? []).length;
    const count = han + latin;
    return { minutes: Math.max(1, Math.ceil(count / 400)), count, unit: "zh" };
  }
  const count = text.split(/\s+/).filter(Boolean).length;
  return { minutes: Math.max(1, Math.ceil(count / 220)), count, unit: "en" };
}

/**
 * Closes the contents box before first paint on phones (the em width tracks the reader's default
 * font size, like the stylesheet's breakpoints), so the prose starts on the first screen. Without
 * JavaScript the box stays open.
 */
const CLOSE_ON_PHONES =
  '<script>(function(d){if(window.matchMedia&&matchMedia("(max-width: 40em)").matches)d.open=false})(document.currentScript.previousElementSibling)</script>';

/**
 * Hast plugin factory: Sätteri calls it once per document, so the slugger and counters start
 * fresh for each post. It reads the post's `lang` from its frontmatter and, from the file name,
 * whether the post is fiction (src/data/posts.json or the frontmatter `topic`).
 */
export function postStructure(fctx) {
  const frontmatter = fctx.data?.astro?.frontmatter ?? {};
  const lang = typeof frontmatter.lang === "string" ? frontmatter.lang : "en";
  const zh = lang.startsWith("zh");
  const id = fctx.fileURL ? decodeURIComponent(fctx.fileURL.pathname.split("/").pop()).replace(/\.md$/, "") : null;
  const fiction = id !== null && isFiction(id, frontmatter);
  const slugger = new Slugger();

  return {
    name: "post-structure",
    element: [
      {
        // GFM column alignment arrives as an inline style. It becomes a class, so the page carries
        // no inline styles, and header cells get scope="col".
        filter: ["th", "td"],
        visit(node, ctx) {
          const align = /text-align:\s*(left|center|right)/.exec(String(node.properties?.style ?? ""));
          if (align) {
            ctx.setProperty(node, "style", null);
            ctx.setProperty(node, "className", ["al-" + align[1]]);
          }
          if (node.tagName === "th") ctx.setProperty(node, "scope", "col");
        },
      },
      {
        filter: ["h1", "h2", "h3", "h4", "h5", "h6"],
        visit(node, ctx) {
          // Astro's own heading-ids pass runs later with the same algorithm and keeps any id it
          // finds, so ids are set here first: the contents box links to them and other sites link
          // to them too. A heading that already has one still counts, keeping duplicates in step.
          if (typeof node.properties?.id === "string") slugger.slug(node.properties.id);
          else ctx.setProperty(node, "id", slugger.slug(ctx.textContent(node)));
          // A renumbered heading keeps the size its Markdown level implies.
          const level = Number(node.tagName[1]);
          const size = clamp(level - (ctx.data.headingShift ?? 1) + 1, 2, 6);
          if (size !== level) ctx.setProperty(node, "className", ["s" + size]);
        },
      },
    ],
    after(root, ctx) {
      const blocks = root.children.filter((c) => c.type === "element" || c.type === "raw");
      const headings = blocks.filter(isHeading);

      // 1. Reading time. textContent covers text nodes only, so code blocks, which rougeCode has
      // already turned into raw HTML, are not counted; inline code is.
      const time = readingTime(ctx.textContent(root), lang);

      // The box needs a long post with a repeated heading level; the shallowest such level is the
      // main level, the one the box lists.
      const perLevel = {};
      for (const h of headings) perLevel[h.tagName] = (perLevel[h.tagName] ?? 0) + 1;
      const main = ["h2", "h3", "h4", "h5", "h6"].find((tag) => perLevel[tag] > 1);
      const toc = headings.length >= 6 && time.minutes >= 3 && main !== undefined;
      if (ctx.data.astro) ctx.data.astro.frontmatter.readingTime = { ...time, toc };

      // 2. Tables become labelled regions that scroll sideways on their own (and by keyboard),
      // named after the nearest heading above them.
      let lastHeading = "";
      let tables = 0;
      for (const node of root.children) {
        if (isHeading(node)) lastHeading = ctx.textContent(node);
        if (node.type === "element" && node.tagName === "table") {
          tables++;
          const label = lastHeading ? `Table ${tables}: ${lastHeading}` : `Table ${tables}`;
          ctx.wrapNode(node, { raw: `<div class="table-scroll" role="region" tabindex="0" aria-label="${escapeAttr(label)}"></div>` });
        }
      }

      if (!toc) return;

      // 3. The contents box, after the opening paragraph when the post opens with one. Fiction
      // lists chapters that carry their own numbers (第一章…), so it gets a <ul> and no numbers.
      const sections = headings.filter((h) => h.tagName === main);
      const label = zh ? "目录" : "Contents";
      const count = zh ? `（${sections.length} ${fiction ? "章" : "节"}）` : ` (${sections.length} sections)`;
      // A path like engine/block_manager.py may break after its slashes, never inside a name.
      const items = sections
        .map((h) => `<li><a href="#${escapeAttr(h.properties.id)}">${escapeAttr(ctx.textContent(h)).replace(/\/(?=\S)/g, "/<wbr>")}</a></li>`)
        .join("");
      const list = fiction ? `<ul>${items}</ul>` : `<ol>${items}</ol>`;
      const box = `<nav class="toc" id="toc" aria-label="${label}"><details open><summary>${label}<span class="toc-n">${count}</span></summary>${list}</details>${CLOSE_ON_PHONES}</nav>`;
      const first = blocks[0];
      if (first.type === "element" && first.tagName === "p") ctx.insertAfter(first, { type: "raw", value: box });
      else ctx.insertBefore(first, { type: "raw", value: box });

      // 4. Section numbers, written into the HTML so they always match the box's <ol> (no CSS
      // counters), and a link back to the box before every section but the first and at the end.
      const back = `<p class="top"><a href="#toc"><span aria-hidden="true">↑</span> ${label}</a></p>`;
      sections.forEach((h, i) => {
        if (!fiction) {
          ctx.prependChild(h, [
            { type: "element", tagName: "span", properties: { className: ["secno"] }, children: [{ type: "text", value: `${i + 1}.` }] },
            { type: "text", value: " " },
          ]);
        }
        if (i > 0) ctx.insertBefore(h, { type: "raw", value: back });
      });
      ctx.appendChild(root, { type: "raw", value: back });
    },
  };
}
