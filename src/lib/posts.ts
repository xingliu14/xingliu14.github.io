import { getCollection, type CollectionEntry } from "astro:content";
import { plainText } from "./feed-html.mjs";
import { postLastModified } from "./last-modified.mjs";
import { descriptionFor, topicFor, type Topic } from "./post-meta.mjs";

export interface Post {
  entry: CollectionEntry<"posts">;
  /** The file name without `.md`, e.g. "2026-01-23-nano-vllm-notes". */
  id: string;
  year: string;
  month: string;
  day: string;
  slug: string;
  /** Jekyll's permalink pattern, /:year/:month/:day/:title/, which old links point to. */
  url: string;
  /** Publication date, "YYYY-MM-DD", from the file name. */
  date: string;
  /** The title as every page, card and feed shows it, with curly apostrophes and quotes. */
  title: string;
  lang: "en" | "zh-CN";
  zh: boolean;
  topic: Topic;
  description: string;
  /** Reading time, and what it counts: words ("en") or Han characters plus Latin words ("zh"). */
  minutes: number;
  count: number;
  unit: "en" | "zh";
  /** Whether the pipeline gave the post a contents box. */
  hasToc: boolean;
  /** When its words last changed, if after publication (src/lib/last-modified.mjs). */
  modified: { date: string; iso: string } | null;
  /** `modified`, or else the publication date (with no time). */
  lastModified: { date: string; iso: string | null };
  /** The article HTML, contents box and section numbers included. */
  html: string;
}

/** What src/lib/markdown.mjs writes into the rendered frontmatter. */
interface ReadingTime {
  minutes: number;
  count: number;
  unit: "en" | "zh";
  toc: boolean;
}

const FILE_NAME = /^(\d{4})-(\d{2})-(\d{2})-(.+)$/;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DESCRIPTION_LENGTH = 160;

/** Typewriter apostrophes between letters become ’, and paired straight double quotes “…”. */
function displayTitle(title: string): string {
  return title.replace(/(\p{L}|\p{N})'(\p{L})/gu, "$1’$2").replace(/"([^"]*)"/g, "“$1”");
}

/**
 * A description for a post that has none: the text of its opening paragraph, cut after a whole
 * word (Chinese included) to at most 160 characters.
 */
function openingText(html: string, lang: string): string {
  const text = plainText(/<p>([\s\S]*?)<\/p>/.exec(html)?.[1] ?? "");
  if (Array.from(text).length <= DESCRIPTION_LENGTH) return text;
  let cut = "";
  for (const { segment } of new Intl.Segmenter(lang, { granularity: "word" }).segment(text)) {
    if (Array.from(cut + segment).length > DESCRIPTION_LENGTH) break;
    cut += segment;
  }
  // "as well,…" reads badly: drop a dangling space or comma before the ellipsis.
  return `${cut.replace(/[\s,;:，、；：]+$/u, "")}…`;
}

function toPost(entry: CollectionEntry<"posts">): Post {
  const { id, data, rendered } = entry;
  const match = FILE_NAME.exec(id);
  if (!match) throw new Error(`Post file names must look like YYYY-MM-DD-slug.md, got "${id}.md"`);
  const [, year, month, day, slug] = match;
  const at = new Date(Date.UTC(Number(year), Number(month) - 1, Number(day)));
  if (at.getUTCMonth() !== Number(month) - 1 || at.getUTCDate() !== Number(day)) {
    throw new Error(`"${id}.md" has no such date: ${year}-${month}-${day}`);
  }
  const date = `${year}-${month}-${day}`;

  // The pipeline leaves the reading time in the rendered frontmatter (remarkPluginFrontmatter).
  const time = (rendered?.metadata?.frontmatter as { readingTime?: ReadingTime } | undefined)?.readingTime;
  if (!time) {
    throw new Error(`Post "${id}" has no reading time: its HTML predates src/lib/markdown.mjs. Build with --force.`);
  }
  const html = rendered?.html ?? "";
  const lang = data.lang ?? "en";
  const description = descriptionFor(id, data) ?? openingText(html, lang);
  if (!description) throw new Error(`Post "${id}" needs a description: add "description: …" to its frontmatter.`);
  const modified = entry.filePath ? postLastModified(entry.filePath, date) : null;

  return {
    entry,
    id,
    year,
    month,
    day,
    slug,
    url: `/${year}/${month}/${day}/${slug}/`,
    date,
    title: displayTitle(data.title),
    lang,
    zh: lang === "zh-CN",
    topic: topicFor(id, data),
    description,
    minutes: time.minutes,
    count: time.count,
    unit: time.unit,
    hasToc: time.toc,
    modified,
    lastModified: modified ?? { date, iso: null },
    html,
  };
}

/** Published posts, newest first. */
export async function getPosts(): Promise<Post[]> {
  const entries = await getCollection("posts", ({ data }) => data.published);
  return entries.map(toPost).sort((a, b) => b.id.localeCompare(a.id));
}

/** "Mar 10, 2026", the format the Jekyll templates used. */
export function formatDate({ year, month, day }: Post): string {
  return `${MONTHS[Number(month) - 1]} ${day}, ${year}`;
}
