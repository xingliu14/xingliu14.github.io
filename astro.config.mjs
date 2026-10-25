import { defineConfig, envField } from "astro/config";
import { satteri } from "@astrojs/markdown-satteri";
import { rougeCode } from "./src/lib/markdown.mjs";

// Switching to a custom domain: set `site` to it, then add the domain under
// Settings → Pages → Custom domain (Actions deployments need no CNAME file). A domain on this
// user site also moves every project site without its own domain (e.g. /GameKingdom/) onto it.
export default defineConfig({
  site: "https://xingliu14.github.io",
  // Keep the whitespace between inline elements (e.g. an icon and its label): it renders as a
  // space, and the layout was designed with it there, as Jekyll emitted it.
  compressHTML: false,
  env: {
    schema: {
      // Optional; CI sets it so the build's GitHub API calls aren't rate limited.
      GITHUB_TOKEN: envField.string({ context: "server", access: "secret", optional: true }),
    },
  },
  vite: {
    build: {
      // Vite's default target lets the minifier rewrite `(max-width: 768px)` as `(width <= 768px)`,
      // which Safari before 16.4 ignores, dropping the mobile layout. Target older browsers.
      cssTarget: ["chrome87", "edge88", "firefox78", "safari14"],
    },
    css: {
      // Primer 11 still carries old IE hacks (`@media (min-width: 0\0)`) that browsers ignore;
      // let the minifier drop them instead of failing.
      lightningcss: { errorRecovery: true },
    },
  },
  markdown: {
    // rougeCode does the highlighting so code keeps Jekyll's markup and colours.
    syntaxHighlight: false,
    processor: satteri({ hastPlugins: [rougeCode] }),
  },
});
