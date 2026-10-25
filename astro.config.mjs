import { defineConfig, envField } from "astro/config";
import { satteri } from "@astrojs/markdown-satteri";
import { normalizeHeadings, postStructure, rougeCode } from "./src/lib/markdown.mjs";

// Served at the custom domain set in Settings → Pages (Actions deployments ignore a CNAME file).
// Project sites without their own domain (e.g. /GameKingdom/) are served under it too.
export default defineConfig({
  site: "https://xingliu.me",
  // Keep whitespace as written: the single spaces between the navigation's items (src/lib/nav.ts)
  // and between other inline elements render as spaces, and are where a narrow line may wrap.
  compressHTML: false,
  env: {
    schema: {
      // Optional; CI sets it so the build's GitHub API calls aren't rate limited.
      GITHUB_TOKEN: envField.string({ context: "server", access: "secret", optional: true }),
    },
  },
  vite: {
    build: {
      // public/style.css is served as it is, but any CSS Vite processes must still work in Safari
      // 14: with Vite's default target the minifier rewrites `(max-width: 40em)` as `(width <= 40em)`,
      // which Safari before 16.4 ignores.
      cssTarget: ["chrome87", "edge88", "firefox78", "safari14"],
    },
  },
  markdown: {
    // rougeCode does the highlighting so code keeps Jekyll's markup and colours; the other two
    // plugins shape the post body (src/lib/markdown.mjs).
    syntaxHighlight: false,
    processor: satteri({ mdastPlugins: [normalizeHeadings], hastPlugins: [rougeCode, postStructure] }),
  },
});
