// Profile and repositories from the GitHub API at build time (what jekyll-github-metadata
// provided). Falls back to src/data/github-snapshot.json if the API can't be reached.
import { GITHUB_TOKEN } from "astro:env/server";
import { GITHUB_USER, PROJECTS } from "../config";
import snapshot from "../data/github-snapshot.json";

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
  language: string | null;
  stargazers_count: number;
  forks_count: number;
  fork: boolean;
  archived: boolean;
  pushed_at: string | null;
}

interface GitHubData {
  owner: Owner;
  repos: Repo[];
}

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
    return { owner, repos };
  } catch (err) {
    console.warn(`[github] ${(err as Error).message}; using src/data/github-snapshot.json`);
    return snapshot as GitHubData;
  }
}

let data: Promise<GitHubData> | undefined;

export function getGitHubData(): Promise<GitHubData> {
  data ??= load();
  return data;
}

export async function getOwner(): Promise<Owner> {
  return (await getGitHubData()).owner;
}

/** Public repositories for the Projects section, filtered, sorted and limited per PROJECTS. */
export async function getProjects(): Promise<Repo[]> {
  const { repos } = await getGitHubData();
  const { sortBy, limit, exclude } = PROJECTS;
  return repos
    .filter((r) => !(exclude.archived && r.archived) && !(exclude.forks && r.fork))
    .filter((r) => !exclude.projects.includes(r.name))
    .sort((a, b) =>
      sortBy === "stars"
        ? b.stargazers_count - a.stargazers_count
        : (b.pushed_at ?? "").localeCompare(a.pushed_at ?? ""),
    )
    .slice(0, limit);
}
