// Sätteri HAST plugins that reproduce the code markup Jekyll (kramdown + Rouge)
// produced, so posts keep their URLs' look and the existing styles still apply.
import Prism from "prismjs";
import loadLanguages from "prismjs/components/index.js";

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
 * `<div class="language-x highlighter-rouge"><div class="highlight"><pre class="highlight"><code>…`
 * and inline code gets `class="language-plaintext highlighter-rouge"`, as kramdown emitted.
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
          value: `<div class="language-${escapeAttr(lang)} highlighter-rouge"><div class="highlight"><pre class="highlight"><code>${html}</code></pre></div></div>`,
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
