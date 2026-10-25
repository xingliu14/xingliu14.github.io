// The home page's What's New log: the three newest posts, plus the two newest other events (new
// repositories, and site news from src/data/site-log.json), merged newest first. It is a digest
// of recent activity, never the archive.
import { addDays, laDate } from "./dates";
import type { Project } from "./github";
import type { Post } from "./posts";

/** An entry of src/data/site-log.json; `html` is trusted, hand-written markup. */
export interface SiteLogEntry {
  date: string;
  html: string;
}

type LogLine = { kind: "site"; date: string; html: string } | { kind: "project"; date: string; project: Project } | { kind: "post"; date: string; post: Post };

export type WhatsNewItem = LogLine & {
  /** Whether NEW! shows: the item is less than NEW_FOR_DAYS old at build time. */
  isNew: boolean;
  /** The day NEW! expires; site.js removes the tag from then on, by the reader's own clock. */
  until: string;
};

const NEW_FOR_DAYS = 30;
// On the same day, site news comes first, then a repository, then a post.
const RANK = { site: 0, project: 1, post: 2 };

const newestFirst = (a: LogLine, b: LogLine) => b.date.localeCompare(a.date) || RANK[a.kind] - RANK[b.kind];

export function getWhatsNew({ posts, projects, siteLog, today }: { posts: Post[]; projects: Project[]; siteLog: SiteLogEntry[]; today: string }): WhatsNewItem[] {
  const others: LogLine[] = [
    ...siteLog.map((entry): LogLine => ({ kind: "site", date: entry.date, html: entry.html })),
    // A repository's date is the day it was created, in the site's time zone.
    ...projects.map((project): LogLine => ({ kind: "project", date: laDate(project.created_at), project })),
  ];
  const newestPosts = posts.slice(0, 3).map((post): LogLine => ({ kind: "post", date: post.date, post }));
  return [...others.sort(newestFirst).slice(0, 2), ...newestPosts].sort(newestFirst).map((event) => {
    const until = addDays(event.date, NEW_FOR_DAYS);
    return { ...event, isNew: today < until, until };
  });
}
