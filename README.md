# xingliu.me

My personal website and blog: https://xingliu.me (repo `xingliu14.github.io`)

Built with [Astro](https://astro.build). It started from GitHub's [personal-website](https://github.com/github/personal-website) Jekyll template, moved to Astro in October 2026 with the same post URLs, and then got its 1997 look ("Navigator Grey"). The colophon on the About page explains the design.

## Writing a post

Add `src/content/posts/YYYY-MM-DD-slug.md`:

```md
---
title: "My post"
topic: llm-inference
description: "One sentence on what the post is about."
---

Post body in Markdown.
```

- It is published at `/YYYY/MM/DD/slug/`. Once it is out, keep the file name: other sites link to it.
- `topic` files the post under one topic on the Writing page. Use one of the ids in `src/data/posts.json`: `llm-inference`, `books`, `crypto`, `life`, `fiction` or `meta`. The build stops with a message if a post has no topic.
- `description` is the one-sentence summary that search engines, social cards, the RSS feed and What's New show. Without one, the site uses the start of the first paragraph.
- Optional: `published: false` keeps a draft out of the build, and `lang: zh-CN` marks a post written in Chinese.
- The newest posts show up in What's New on the home page by themselves.

A post's "Updated" date comes from git: the last commit that changed its title or text. Moving a heading to another level doesn't count. If you ever change post files without changing what they say (reformatting, say), add that commit's full hash to `.git-blame-ignore-revs`, so the posts keep their dates.

## Site news

To announce something about the site itself in What's New, add an entry at the top of `src/data/site-log.json`:

```json
{ "date": "2026-10-25", "html": "What changed, with <a href=\"/about/\">links</a> if useful." }
```

What's New shows the three newest posts and the two newest other events (new GitHub repositories and these entries), newest first, and marks anything under 30 days old NEW!.

## Running it locally

Needs Node 22.23.4 or newer (`.node-version` pins 24).

```sh
npm install
npm run dev       # http://localhost:4321
npm run build     # type-check, build into dist/, then check the result
npm run preview   # serve dist/
```

`npm run build` fails if the finished site breaks one of its promises: a missing page, a lost heading anchor, invalid HTML, a page without its title, description or social card, and so on. The error lists every problem.

## Deploying

Every push to `master` builds the site with GitHub Actions and publishes it to GitHub Pages (Settings → Pages → Source: GitHub Actions), with `.github/workflows/deploy.yml`. A second workflow, Daily rebuild (`.github/workflows/daily.yml`), runs that deploy every day at 09:17 UTC, so Projects and What's New stay current with no commit.

GitHub disables the daily rebuild after 60 days without a commit; pushes still deploy. To turn it back on, open Actions → Daily rebuild → Enable workflow, or run `gh workflow enable daily.yml`.

## Reserved names

GitHub serves each of my repositories' Pages sites under this domain, at `xingliu.me/<repository>/` (for example `/GameKingdom/`). So never name a repository `writing`, `projects`, `about`, `og`, `badges`, `resume`, or a four-digit year: those paths belong to this site. The build refuses to put anything else at the top level until it is added to the list in `src/integrations/check-dist.mjs`.

## Settings

- `src/config.ts`: the résumé link, the Elsewhere links, which GitHub repositories Projects leaves out, and the visitor map.
- `src/components/AboutText.astro`: the About text, shown on the home page and the About page.
- The job (Annapurna Labs of AWS, before that Google and Yahoo) is also written into these, which a new job means updating too:
  - the meta descriptions of the home page (`src/pages/index.astro`) and About (`src/pages/about/index.astro`);
  - About's short bio and résumé list (`src/pages/about/index.astro`);
  - `worksFor` in the structured data search engines read (`src/lib/seo.ts`);
  - the text on the home and About social cards (`src/pages/og/cards.json.ts`).
- `src/data/repo-notes.json`: a one-line note for a repository with no description on GitHub, or a homepage to show instead of the one set on GitHub (`null` shows none, for a site that is gone).
- `public/resume/`: the résumé PDF and its LaTeX source. Add a new résumé as `public/resume/Xing_Liu_Resume_YYYYMMDD.pdf` (About reads the month from the name) and point `RESUME_URL` in `src/config.ts` at it. Keep the old PDF: its address must keep working, and the build checks that it does.
- `resume/` (at the repo root) only keeps the 2025 PDF where the old site's GitHub link expects it; the build doesn't use it.

The Projects page and the new-repository lines in What's New are read from GitHub when the site builds.
