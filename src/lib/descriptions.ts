// The Writing and Projects descriptions, counted from data. Each is the page's meta and Open Graph
// description and also the text on its social card (src/pages/og/cards.json.ts), so the page and the
// card should both take it from here.
import { monthYear } from "./dates";
import type { Project } from "./github";
import type { Post } from "./posts";

/** "All 13 posts by Xing Liu since December 2020, by date and by topic: …" */
export function writingDescription(posts: Post[]): string {
  // Posts are newest first, so the last one is where the archive starts.
  const since = monthYear(posts[posts.length - 1].date);
  return `All ${posts.length} posts by Xing Liu since ${since}, by date and by topic: notes on LLM inference, books, crypto and life, and a detective story in Chinese.`;
}

/**
 * "The 7 public repositories Xing Liu has started on GitHub (forks left out): …". GitHub's own count
 * includes the forks, which the Projects page leaves out, so the sentence says so.
 */
export function projectsDescription(projects: Project[]): string {
  return `The ${projects.length} public repositories Xing Liu has started on GitHub (forks left out): what each one is, its language, its stars and when it last changed.`;
}
