// "Last modified" dates from git history, so they stay true on a static host: a scheduled rebuild,
// a redesign or a file rename does not move them, and neither does any commit listed in
// .git-blame-ignore-revs (the file GitHub's blame view also reads).
//
// Paths are relative to the repository root, which is where Astro runs. Without usable history
// (not a repository, or a shallow clone) postLastModified and pathsLastModified return null after
// one warning, and callers fall back to publication or build dates. CI checks out with
// fetch-depth: 0.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { publishedPostFiles } from "./post-files.mjs";

/**
 * A commit's author date: `date` is the calendar date where the author was, `iso` the full time.
 * @typedef {{ date: string, iso: string }} Modified
 */

/** @param {string[]} args */
function git(args) {
  return execFileSync("git", ["-c", "core.quotePath=false", ...args], {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    stdio: ["ignore", "pipe", "ignore"],
  });
}

/** @type {boolean | undefined} */
let usable;

/** Whether the full git history is available. */
export function gitUsable() {
  if (usable === undefined) {
    try {
      usable = git(["rev-parse", "--is-shallow-repository"]).trim() === "false";
    } catch {
      usable = false;
    }
    if (!usable) {
      console.warn("[last-modified] no full git history here (not a repository, or a shallow clone); using publication and build dates instead");
    }
  }
  return usable;
}

/** @type {Set<string> | undefined} */
let ignored;

/** Full hashes of commits that changed files without changing what they say. */
function ignoredCommits() {
  if (!ignored) {
    let text = "";
    try {
      text = readFileSync(".git-blame-ignore-revs", "utf8");
    } catch {
      // No ignore file: every commit counts.
    }
    ignored = new Set(text.split("\n").map((line) => line.replace(/#.*/, "").trim()).filter(Boolean));
  }
  return ignored;
}

/**
 * What a post says: its title and its body, ignoring line endings, trailing spaces and the rest of
 * the frontmatter.
 * @param {string | null} source
 */
function postText(source) {
  if (source === null) return null;
  const text = source.replace(/\r\n/g, "\n");
  const frontmatter = /^---\n([\s\S]*?)\n---(?:\n|$)/.exec(text);
  const title = frontmatter ? (/^title:[ \t]*(.*)$/m.exec(frontmatter[1])?.[1] ?? "") : "";
  const body = frontmatter ? text.slice(frontmatter[0].length) : text;
  // A title retyped from 'quoted' to "quoted" or bare still says the same thing.
  const unquoted = title.trim().replace(/^(["'])(.*)\1$/, "$2");
  return `${unquoted}\n${body}`.replace(/[ \t]+$/gm, "").trim();
}

/** @param {string} rev  e.g. "<hash>:<path>"; null when the file does not exist there. */
function show(rev) {
  try {
    return git(["show", rev]);
  } catch {
    return null;
  }
}

/** @type {Map<string, Modified | null>} */
const postCache = new Map();

/**
 * When a post's words last changed, if that is later than its publication date; otherwise null,
 * and the post's last modification is its publication.
 *
 * Walks the file's history newest first, across renames. Commits in .git-blame-ignore-revs are
 * skipped, and so are commits after which the post's text (title and body) reads the same as
 * before: renames and frontmatter-only edits. The commit that added the file is the publication,
 * not an edit, so the walk stops there.
 * @param {string} file  e.g. "src/content/posts/2026-01-23-nano-vllm-notes.md"
 * @param {string} published  "YYYY-MM-DD"
 * @returns {Modified | null}
 */
export function postLastModified(file, published) {
  if (!gitUsable()) return null;
  const key = `${file}\n${published}`;
  if (!postCache.has(key)) postCache.set(key, lastTextChange(file, published));
  return postCache.get(key) ?? null;
}

/**
 * @param {string} file
 * @param {string} published
 * @returns {Modified | null}
 */
function lastTextChange(file, published) {
  const log = git(["log", "--follow", "-M", "--format=%x00%H %aI", "--name-status", "--", file]);
  for (const entry of log.split("\0").slice(1)) {
    const [head, ...rest] = entry.split("\n").filter(Boolean);
    const [hash, iso] = head.split(" ");
    const status = rest[0]?.split("\t");
    // Merges list no files; ignored commits never count.
    if (!status || ignoredCommits().has(hash)) continue;
    if (status[0].startsWith("A")) return null;
    const [oldPath, newPath] = status.length > 2 ? [status[1], status[2]] : [status[1], status[1]];
    if (postText(show(`${hash}:${newPath}`)) === postText(show(`${hash}^:${oldPath}`))) continue;
    const date = iso.slice(0, 10);
    return date > published ? { date, iso } : null;
  }
  return null;
}

/** @type {Map<string, Modified | null>} */
const pathsCache = new Map();

/**
 * The newest commit that touches any of these files or folders, leaving out
 * .git-blame-ignore-revs. Pages take the later of this and their own content dates.
 * @param {string[]} paths
 * @returns {Modified | null}
 */
export function pathsLastModified(paths) {
  if (!gitUsable()) return null;
  const key = paths.join("\n");
  if (!pathsCache.has(key)) {
    const log = git(["log", "--format=%H %aI", "--", ...paths]);
    const line = log.split("\n").find((l) => l && !ignoredCommits().has(l.split(" ")[0]));
    const iso = line?.split(" ")[1];
    pathsCache.set(key, iso ? { date: iso.slice(0, 10), iso } : null);
  }
  return pathsCache.get(key) ?? null;
}

/** @type {Map<string, string> | undefined} */
let lastmods;

/**
 * The sitemap's <lastmod> for a page: a post's "Last modified" date, and for /writing/ the newest
 * of those. Other pages get none rather than an imprecise one. This runs in astro.config.mjs,
 * outside Astro's content layer, so it finds the posts by their file names.
 * @param {string} pathname  e.g. "/2026/01/23/nano-vllm-notes/"
 * @returns {string | null}  "YYYY-MM-DD"
 */
export function sitemapLastmod(pathname) {
  if (!lastmods) {
    lastmods = new Map(publishedPostFiles().map(({ file, url, date }) => [url, postLastModified(file, date)?.date ?? date]));
    const newest = [...lastmods.values()].sort().at(-1);
    if (newest) lastmods.set("/writing/", newest);
  }
  return lastmods.get(decodeURIComponent(pathname)) ?? null;
}
