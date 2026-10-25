// The published posts, read straight from their files, for code that runs outside Astro's content
// layer: the sitemap's lastmod (src/lib/last-modified.mjs) and check-dist. Like src/lib/posts.ts,
// which the pages use, it takes each post's URL and date from its file name.
import { readdirSync, readFileSync } from "node:fs";

// Relative to the repository root, which is where Astro runs.
const DIR = "src/content/posts";
const FILE_NAME = /^(\d{4})-(\d{2})-(\d{2})-(.+)\.md$/;

/**
 * Every post the build publishes: each src/content/posts/*.md whose frontmatter does not say
 * `published: false`, with its path, its URL and its publication date ("YYYY-MM-DD").
 * @returns {{ file: string, url: string, date: string }[]}
 */
export function publishedPostFiles() {
  return readdirSync(DIR).flatMap((name) => {
    // src/lib/posts.ts fails the build on a file name without a date.
    const match = FILE_NAME.exec(name);
    if (!match) return [];
    const file = `${DIR}/${name}`;
    const frontmatter = /^---\r?\n([\s\S]*?)\r?\n---/.exec(readFileSync(file, "utf8"))?.[1] ?? "";
    if (/^published:\s*false\s*(#.*)?$/im.test(frontmatter)) return [];
    const [, year, month, day, slug] = match;
    return [{ file, url: `/${year}/${month}/${day}/${slug}/`, date: `${year}-${month}-${day}` }];
  });
}
