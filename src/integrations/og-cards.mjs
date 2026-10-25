// Draws the Open Graph cards after the build: reads the list that src/pages/og/cards.json.ts wrote
// to dist/og/cards.json, renders each entry to dist/og/<file> (src/integrations/og-render.mjs), and
// deletes the list. Nothing here needs a browser, so it runs in CI on a bare Ubuntu runner.
import { readFile, rm, writeFile } from "node:fs/promises";

/** @returns {import("astro").AstroIntegration} */
export default function ogCards() {
  return {
    name: "og-cards",
    hooks: {
      "astro:build:done": async ({ dir, logger }) => {
        // Loaded here rather than at the top, so other commands (dev, check) skip Satori and resvg.
        const { renderCard } = await import("./og-render.mjs");
        const list = new URL("og/cards.json", dir);
        /** @type {import("./og-render.mjs").Card[]} */
        const cards = JSON.parse(await readFile(list, "utf8"));
        const avatar = await readFile(new URL("avatar-240.jpg", dir));
        const start = performance.now();
        for (const card of cards) await writeFile(new URL(`og/${card.file}`, dir), await renderCard(card, { avatar }));
        await rm(list);
        logger.info(`${cards.length} cards drawn in ${((performance.now() - start) / 1000).toFixed(1)}s`);
      },
    },
  };
}
