# xingliu.me

My personal website and blog: https://xingliu.me (repo `xingliu14.github.io`)

Built with [Astro](https://astro.build). It started from GitHub's [personal-website](https://github.com/github/personal-website) Jekyll template; in October 2026 it moved to Astro with the same design and the same post URLs.

## Writing a post

Add `src/content/posts/YYYY-MM-DD-slug.md`:

```md
---
title: "My post"
---

Post body in Markdown.
```

It is published at `/YYYY/MM/DD/slug/`. Set `published: false` to keep a draft out of the build, and `lang: zh-CN` for a post written in Chinese.

## Running it locally

Needs Node 22.23.4 or newer (`.node-version` pins 24).

```sh
npm install
npm run dev       # http://localhost:4321
npm run build     # type-check, then build into dist/
npm run preview   # serve dist/
```

## Deploying

Every push to `master` builds the site with GitHub Actions and publishes it to GitHub Pages (Settings → Pages → Source: GitHub Actions).

## Settings

- `src/config.ts`: resume link, social accounts, which GitHub repositories appear under Projects.
- `src/components/About.astro`: the About text.
- `public/resume/`: the resume PDF and its LaTeX source. Point `RESUME_URL` in `src/config.ts` at a new PDF when you add one.
- `resume/` (at the repo root) only keeps the 2025 PDF where the old site's GitHub link expects it; the build doesn't use it.

The profile in the sidebar (name, bio, avatar) and the Projects cards are read from GitHub when the site builds.
