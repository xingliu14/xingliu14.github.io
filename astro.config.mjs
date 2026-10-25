import { defineConfig, envField } from "astro/config";
import { satteri } from "@astrojs/markdown-satteri";
import sitemap from "@astrojs/sitemap";
import checkDist from "./src/integrations/check-dist.mjs";
import ogCards from "./src/integrations/og-cards.mjs";
import pageWeight from "./src/integrations/page-weight.mjs";
import sitemapDates from "./src/integrations/sitemap-dates.mjs";
import { sitemapLastmod } from "./src/lib/last-modified.mjs";
import { normalizeHeadings, postStructure, rougeCode } from "./src/lib/markdown.mjs";

// Served at the custom domain set in Settings → Pages (Actions deployments ignore a CNAME file).
// Project sites without their own domain (e.g. /GameKingdom/) are served under it too.
export default defineConfig({
  site: "https://xingliu.me",
  // Keep whitespace as written: the single spaces between the navigation's items (src/lib/nav.ts)
  // and between other inline elements render as spaces, and are where a narrow line may wrap.
  compressHTML: false,
  // Each one runs when the build is done, in this order, so check-dist sees the finished site.
  integrations: [
    sitemap({
      // Every page but Not Found. Posts and /writing/ say when they last changed; other pages give
      // no date rather than an imprecise one.
      filter: (page) => !/\/404(\.html)?\/?$/.test(new URL(page).pathname),
      serialize: (item) => {
        const lastmod = sitemapLastmod(new URL(item.url).pathname);
        if (lastmod) item.lastmod = lastmod;
        return item;
      },
    }),
    sitemapDates(), // the sitemap's <lastmod> as plain dates, not midnight UTC
    ogCards(), // dist/og/cards.json → dist/og/*.png, then deletes cards.json
    pageWeight(), // fills in the weight sentence in dist/about/index.html
    checkDist(), // last: fails the build if anything in dist/ is wrong
  ],
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
