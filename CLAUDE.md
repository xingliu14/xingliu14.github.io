# CLAUDE.md

Personal site and blog for Xing Liu, built with Astro and deployed to GitHub Pages by `.github/workflows/deploy.yml` on every push to `master`. It was migrated from a Jekyll site (GitHub's personal-website template) in October 2026, keeping the same look and URLs.

`npm run build` type-checks (`astro check`) and then builds into `dist/` with `--force`: Astro's content cache does not notice changes to `src/lib/markdown.mjs`, so a cached build would keep post HTML from the old plugin. Run `astro dev --force` after editing that file too.

## Posts

- A post is `src/content/posts/YYYY-MM-DD-slug.md` with `title` (and optionally `published: false`, `lang: zh-CN`) in its frontmatter.
- The file name is the URL: `/YYYY/MM/DD/slug/`, case preserved. Published posts are linked from elsewhere, so their file names are permanent.
- Markdown is rendered by Astro's default Sätteri processor (CommonMark + GFM + smart punctuation), not kramdown. Two habits keep posts rendering as intended:
  - Put an inline HTML tag like `<br>` on the same line as the text it belongs to; on a line by itself it starts a raw HTML block and the text after it loses its `<p>`.
  - In Chinese prose, type curly quotes “…” directly; smart punctuation can't tell opening from closing quotes next to CJK characters.
- Emoji go in as Unicode characters (Jekyll's `:shortcode:` emoji are not supported).

## Look and markup

- The design is Primer 11 (`primer/build/build.css`, imported in `src/layouts/Base.astro`) plus `src/styles/global.scss`. Each page's `<style is:global>` block is the layout CSS carried over from the Jekyll layouts.
- `src/lib/markdown.mjs` makes code blocks and inline code emit the same markup kramdown/Rouge did and highlights with Prism; token colours live in `src/styles/_highlight-syntax.scss`.
- `compressHTML: false` is deliberate: the whitespace between inline elements (an icon and its label) renders as a space the layout relies on.
- `vite.build.cssTarget` targets older browsers on purpose: with Vite's default the minifier rewrites media queries into range syntax (`width <= 768px`), which Safari before 16.4 ignores.

## Files

- `public/resume/` holds the resume the site links to (`RESUME_URL` in `src/config.ts`). `resume/` at the repo root only keeps the 2025 PDF at the GitHub path the Jekyll site linked to; the build ignores it.

## Data

- `src/config.ts` holds the resume path, social accounts, and Projects settings.
- The sidebar profile and the Projects cards come from the GitHub API at build time (`src/lib/github.ts`); when the API is unreachable the build falls back to `src/data/github-snapshot.json`. CI passes `GITHUB_TOKEN` so it isn't rate limited.
