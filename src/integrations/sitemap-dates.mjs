// Puts plain dates back into the sitemap. Posts and /writing/ give their <lastmod> as a day
// ("2026-03-04", src/lib/last-modified.mjs), but @astrojs/sitemap writes it through
// new Date(…).toISOString(), as midnight UTC ("2026-03-04T00:00:00.000Z"): a precise instant that
// comes before the edit it dates, and before a post published in California existed. The sitemap
// protocol takes a plain date, so every sitemap file, the index included, gets one.
import { readdir, readFile, writeFile } from "node:fs/promises";

const MIDNIGHT = /(<lastmod>\d{4}-\d{2}-\d{2})T00:00:00\.000Z(<\/lastmod>)/g;

/** @returns {import("astro").AstroIntegration} */
export default function sitemapDates() {
  return {
    name: "sitemap-dates",
    hooks: {
      "astro:build:done": async ({ dir }) => {
        for (const name of (await readdir(dir)).filter((file) => /^sitemap-.*\.xml$/.test(file))) {
          const file = new URL(name, dir);
          await writeFile(file, (await readFile(file, "utf8")).replace(MIDNIGHT, "$1$2"));
        }
      },
    },
  };
}
