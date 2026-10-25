import { getCollection, type CollectionEntry } from "astro:content";

export interface Post {
  entry: CollectionEntry<"posts">;
  year: string;
  month: string;
  day: string;
  slug: string;
  /** Jekyll's permalink pattern, /:year/:month/:day/:title/, which old links point to. */
  url: string;
}

const FILE_NAME = /^(\d{4})-(\d{2})-(\d{2})-(.+)$/;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function toPost(entry: CollectionEntry<"posts">): Post {
  const match = FILE_NAME.exec(entry.id);
  if (!match) throw new Error(`Post file names must look like YYYY-MM-DD-slug.md, got "${entry.id}.md"`);
  const [, year, month, day, slug] = match;
  const date = new Date(Date.UTC(Number(year), Number(month) - 1, Number(day)));
  if (date.getUTCMonth() !== Number(month) - 1 || date.getUTCDate() !== Number(day)) {
    throw new Error(`"${entry.id}.md" has no such date: ${year}-${month}-${day}`);
  }
  return { entry, year, month, day, slug, url: `/${year}/${month}/${day}/${slug}/` };
}

/** Published posts, newest first. */
export async function getPosts(): Promise<Post[]> {
  const entries = await getCollection("posts", ({ data }) => data.published);
  return entries.map(toPost).sort((a, b) => b.entry.id.localeCompare(a.entry.id));
}

/** "Mar 10, 2026", the format the Jekyll templates used. */
export function formatDate({ year, month, day }: Post): string {
  return `${MONTHS[Number(month) - 1]} ${day}, ${year}`;
}

/** Plain-text summary for <meta name="description">, at most `length` characters. */
export function summarize(html: string, length = 300): string {
  const text = html
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<(script|style)\b[^>]*>[\s\S]*?<\/\1>/gi, "")
    .replace(/<[^>]*>/g, "")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();
  // Count code points, so an emoji is never cut in half.
  const chars = Array.from(text);
  return chars.length > length ? `${chars.slice(0, length - 3).join("")}...` : text;
}
