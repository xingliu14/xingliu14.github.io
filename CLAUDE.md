# CLAUDE.md

Personal site and blog for Xing Liu at https://xingliu.me, built with Astro and deployed to GitHub Pages by `.github/workflows/deploy.yml` on every push to `master`. It was migrated from a Jekyll site (GitHub's personal-website template) in October 2026, keeping the same look and URLs.

`npm run build` type-checks (`astro check`) and then builds into `dist/` with `--force`: Astro's content cache does not notice changes to `src/lib/markdown.mjs`, so a cached build would keep post HTML from the old plugin. Run `astro dev --force` after editing that file too.

## Posts

- A post is `src/content/posts/YYYY-MM-DD-slug.md` with `title` (and optionally `published: false`, `lang: zh-CN`) in its frontmatter.
- The file name is the URL: `/YYYY/MM/DD/slug/`, case preserved. Published posts are linked from elsewhere, so their file names are permanent.
- Markdown is rendered by Astro's default Sätteri processor (CommonMark + GFM + smart punctuation), not kramdown. Two habits keep posts rendering as intended:
  - Put an inline HTML tag like `<br>` on the same line as the text it belongs to; on a line by itself it starts a raw HTML block and the text after it loses its `<p>`.
  - In Chinese prose, type curly quotes “…” directly; smart punctuation can't tell opening from closing quotes next to CJK characters.
- Emoji go in as Unicode characters (Jekyll's `:shortcode:` emoji are not supported).

## Look and markup

- The design is "Navigator Grey": a 1997 Netscape-style home page, typeset with care. The rule behind every choice: only what Navigator could draw, set the way a typographer would.
- `public/style.css` and `public/site.js` are served verbatim: no Sass, no bundling. `src/layouts/Base.astro` links them with a `?v=` hash of their contents and writes every `<script>` with `is:inline`, so Astro emits no `_astro/` folder. Their budgets include the comments, and both files are close to them: style.css at most 36,000 bytes (11,000 gzipped), site.js at most 18,000 (7,000 gzipped).
- Breakpoints are in em (`40em` is 640px at the 16px default), so a reader's larger default font size never pushes the layout off screen. site.js's `(min-width: 77.5em)` and the contents box's `(max-width: 40em)` script match them.
- `src/lib/markdown.mjs` makes code blocks and inline code emit the same markup kramdown/Rouge did and highlights with Prism; token colours live in `public/style.css`.
- `compressHTML: false` is deliberate: the single spaces between navigation items (`src/lib/nav.ts`) and other inline elements render as spaces, and are where a narrow line may wrap.
- `vite.build.cssTarget` targets older browsers on purpose: with Vite's default the minifier rewrites media queries into range syntax (`width <= 40em`), which Safari before 16.4 ignores.

## Files

- `public/resume/` holds the resume the site links to (`RESUME_URL` in `src/config.ts`). `resume/` at the repo root only keeps the 2025 PDF at the GitHub path the Jekyll site linked to; the build ignores it.

## Data

- `src/config.ts` holds the resume path, social accounts, and Projects settings.
- The Projects page and What's New's new repositories come from the GitHub API at build time (`src/lib/github.ts`); when the API is unreachable the build falls back to `src/data/github-snapshot.json`. CI passes `GITHUB_TOKEN` so it isn't rate limited.
