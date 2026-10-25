// Profile and repositories from the GitHub API at build time. Falls back to
// src/data/github-snapshot.json if the API can't be reached, and says so (`live`).
import { GITHUB_TOKEN } from "astro:env/server";
import { GITHUB_USER, PROJECTS } from "../config";
import snapshot from "../data/github-snapshot.json";
import repoNotes from "../data/repo-notes.json";

export interface Owner {
  login: string;
  name: string | null;
  bio: string | null;
  avatar_url: string;
  email: string | null;
  location: string | null;
  hireable: boolean | null;
}

export interface Repo {
  name: string;
  html_url: string;
  description: string | null;
  /** The website set on the repository; GitHub sends "" or null when there is none. */
  homepage: string | null;
  language: string | null;
  stargazers_count: number;
  forks_count: number;
  fork: boolean;
  archived: boolean;
  /** Whether the repository publishes a GitHub Pages site, served under this site's domain. */
  has_pages: boolean;
  created_at: string;
  pushed_at: string | null;
}

interface GitHubData {
  owner: Owner;
  repos: Repo[];
  /** When the data was fetched: during this build, or when the snapshot was saved. */
  fetchedAt: string;
  /** False when the API could not be reached and the data comes from the snapshot. */
  live: boolean;
}

/** A repository as the Projects page and What's New show it. */
export interface Project extends Repo {
  /** GitHub's description, else the note in src/data/repo-notes.json; null when there is neither. */
  about: string | null;
  /** The site to link to: the repository's website, else the note's, else its Pages site. */
  homepage: string | null;
}

interface RepoNote {
  note?: string;
  homepage?: string;
}

const NOTES: Record<string, RepoNote> = repoNotes;

async function get<T>(path: string): Promise<T> {
  const headers: Record<string, string> = {
    Accept: "application/vnd.github+json",
    "User-Agent": `${GITHUB_USER}.github.io`,
  };
  // Set in CI to avoid the 60 requests/hour anonymous limit.
  if (GITHUB_TOKEN) headers.Authorization = `Bearer ${GITHUB_TOKEN}`;
  const res = await fetch(`https://api.github.com${path}`, { headers, signal: AbortSignal.timeout(15_000) });
  if (!res.ok) throw new Error(`GET ${path} returned ${res.status}`);
  return res.json() as Promise<T>;
}

async function load(): Promise<GitHubData> {
  try {
    const [owner, repos] = await Promise.all([
      get<Owner>(`/users/${GITHUB_USER}`),
      get<Repo[]>(`/users/${GITHUB_USER}/repos?per_page=100&type=owner`),
    ]);
    if (typeof owner?.login !== "string" || !Array.isArray(repos)) throw new Error("unexpected API response shape");
    return { owner, repos, fetchedAt: new Date().toISOString(), live: true };
  } catch (err) {
    console.warn(`[github] ${(err as Error).message}; using src/data/github-snapshot.json`);
    return { owner: snapshot.owner, repos: snapshot.repos as Repo[], fetchedAt: snapshot.fetched_at, live: false };
  }
}

let data: Promise<GitHubData> | undefined;

export function getGitHubData(): Promise<GitHubData> {
  data ??= load();
  return data;
}

/** Every public repository not excluded by PROJECTS (forks, by default), most recently pushed first. */
export async function getProjects(): Promise<Project[]> {
  const { repos } = await getGitHubData();
  const { exclude } = PROJECTS;
  return repos
    .filter((r) => !(exclude.forks && r.fork) && !exclude.projects.includes(r.name))
    .sort((a, b) => (b.pushed_at ?? "").localeCompare(a.pushed_at ?? ""))
    .map((r) => {
      const note = NOTES[r.name];
      // A project's own Pages site lives under this site's domain, e.g. /GameKingdom/.
      const pages = r.has_pages ? new URL(`/${r.name}/`, import.meta.env.SITE).href : null;
      return { ...r, about: r.description || note?.note || null, homepage: r.homepage || note?.homepage || pages };
    });
}
