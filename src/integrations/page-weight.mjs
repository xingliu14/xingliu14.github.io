// Fills in the colophon's weight sentence on /about/ after the build: "This page, its stylesheet and
// its script weigh __TEXT_KB__ KB as sent, plus __PIC_KB__ KB for the picture: about __SECONDS__
// seconds on a 28.8k modem." The numbers are measured on the files about to be deployed, so the
// claim stays true whatever changes. check-dist fails the build if a placeholder survives.
import { readFile, stat, writeFile } from "node:fs/promises";
import { gzipSync } from "node:zlib";

// The portrait the About page shows.
const PICTURE = "avatar-320.jpg";
const MODEM_BITS_PER_SECOND = 28_800;
const PLACEHOLDER = /__(TEXT_KB|PIC_KB|SECONDS)__/g;

/** Bytes as sent: GitHub Pages compresses text with gzip. @param {string | Buffer} data */
const sent = (data) => gzipSync(data, { level: 9 }).length;
/** @param {number} bytes */
const kb = (bytes) => Math.round(bytes / 1024);

/** @returns {import("astro").AstroIntegration} */
export default function pageWeight() {
  return {
    name: "page-weight",
    hooks: {
      "astro:build:done": async ({ dir, logger }) => {
        const page = new URL("about/index.html", dir);
        const template = await readFile(page, "utf8").catch(() => "");
        if (template.search(PLACEHOLDER) === -1) {
          logger.warn("about/index.html has no weight sentence to fill in");
          return;
        }
        if (!template.includes(`src="/${PICTURE}"`)) throw new Error(`page-weight: about/index.html no longer shows /${PICTURE}; update PICTURE`);

        const assets = sent(await readFile(new URL("style.css", dir))) + sent(await readFile(new URL("site.js", dir)));
        const picBytes = (await stat(new URL(PICTURE, dir))).size;
        /** @param {string} html @returns {Record<string, number>} */
        const measure = (html) => {
          const textBytes = sent(html) + assets;
          return { TEXT_KB: kb(textBytes), PIC_KB: kb(picBytes), SECONDS: Math.round(((textBytes + picBytes) * 8) / MODEM_BITS_PER_SECOND) };
        };
        /** @param {Record<string, number>} numbers */
        const fill = (numbers) => template.replace(PLACEHOLDER, (_, name) => String(numbers[name]));

        // The sentence is part of the page it weighs, so fill it in again until the numbers it states
        // are the numbers it measures: twice, nearly always.
        let numbers = measure(template);
        for (let pass = 1; ; pass++) {
          const html = fill(numbers);
          const measured = measure(html);
          if (JSON.stringify(measured) === JSON.stringify(numbers)) {
            await writeFile(page, html);
            break;
          }
          if (pass === 5) throw new Error("page-weight: the weight sentence never settles on one set of numbers");
          numbers = measured;
        }
        logger.info(`about/index.html weighs ${numbers.TEXT_KB} KB as sent, plus ${numbers.PIC_KB} KB for the picture: about ${numbers.SECONDS} s at 28.8k`);
      },
    },
  };
}
