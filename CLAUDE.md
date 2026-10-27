# CLAUDE.md

Personal site and blog for Xing Liu at https://xingliu.me, built with Astro and deployed to GitHub Pages by `.github/workflows/deploy.yml` on every push to `master`. `.github/workflows/daily.yml` starts that deploy daily at 09:17 UTC (refreshing Projects and What's New, and retiring NEW! tags); it holds the schedule on its own because GitHub disables a scheduled workflow after 60 days without a commit, and that must never stop pushes from deploying. It moved from Jekyll in October 2026 and was then redesigned; every post kept its URL through both.

Read [DESIGN.md](DESIGN.md) before changing the site's look, structure or behavior: it records the design's rule, the system as built, and why each decision was made. Keep it current: a change that makes it wrong updates it in the same commit.

`npm run build` type-checks (`astro check`) and then builds into `dist/` with `--force`: Astro's content cache does not notice changes to `src/lib/markdown.mjs` or to `src/data/posts.json` (the pipeline reads it), so a cached build would keep stale post HTML. Run `astro dev --force` after editing either. CI builds with `cache: false` for the same reason.

## Posts

- A post is `src/content/posts/YYYY-MM-DD-slug.md` with `title` (and optionally `published: false`, `lang: zh-CN`, `topic`, `description`) in its frontmatter. A new post gives `topic` (an id from `src/data/posts.json`) and `description`; the older posts take both from `posts.json`, and the build fails on a post with no topic.
- The file name is the URL: `/YYYY/MM/DD/slug/`, case preserved. Published posts are linked from elsewhere, so their file names are permanent, and so are their heading ids (`scripts/heading-ids.json`).
- A post's "Last modified" date is the last commit that changed its title or body (`src/lib/last-modified.mjs`). Renames, edits to frontmatter other than the title, and headings moved to another level are skipped automatically. A commit that changes a post's text without changing what it says (rewrapped lines, retyped quotes, a mass reformat) needs its full hash listed in `.git-blame-ignore-revs`, or it becomes the date of every post it touches.
- Markdown is rendered by Astro's default Sätteri processor (CommonMark + GFM + smart punctuation), not kramdown. Two habits keep posts rendering as intended:
  - Put an inline HTML tag like `<br>` on the same line as the text it belongs to; on a line by itself it starts a raw HTML block and the text after it loses its `<p>`.
  - In Chinese prose, type curly quotes “…” directly; smart punctuation can't tell opening from closing quotes next to CJK characters.
- Emoji go in as Unicode characters (Jekyll's `:shortcode:` emoji are not supported).

## Look and markup

- The design is "Navigator Grey": a 1997 Netscape-style home page, typeset with care. The rule behind every choice: only what Navigator could draw, set the way a typographer would.
- `public/style.css` and `public/site.js` are served verbatim: no Sass, no bundling. `src/layouts/Base.astro` links them with a `?v=` hash of their contents and writes every `<script>` with `is:inline`, so Astro emits no `_astro/` folder. Their budgets include the comments, both files are close to them, and check-dist fails the build past them: style.css at most 36,000 bytes (11,000 gzipped at level 9), site.js at most 18,000 (7,000 gzipped).
- Breakpoints are in em (`40em` is 640px at the 16px default), so a reader's larger default font size never pushes the layout off screen. site.js's `(min-width: 77.5em)` and the contents box's `(max-width: 40em)` script match them.
- Everything a reader sees or hears (text, alt text, aria labels, meta descriptions) uses American spelling, as the owner writes: "Page colors:", "Netscape gray". Class names, custom properties and other identifiers (`colours`, the `grey` theme value) keep their spelling, because renaming them only churns the code.
- `compressHTML: false` is deliberate: the single spaces between navigation items (`src/lib/nav.ts`) and other inline elements render as spaces, and are where a narrow line may wrap.
- `vite.build.cssTarget` targets older browsers on purpose: with Vite's default the minifier rewrites media queries into range syntax (`width <= 40em`), which Safari before 16.4 ignores.

## Markdown pipeline (`src/lib/markdown.mjs`)

- `normalizeHeadings` shifts every post's headings so the top level is `<h2>`: the page's only `<h1>` is the post title. Ids stay as they were.
- `rougeCode` gives code blocks and inline code the markup kramdown/Rouge did and highlights with Prism; token colours live in `public/style.css`.
- `postStructure` adds the heading ids (Astro's slugger), table scroll boxes, the contents box, section numbers (not in fiction) and back links, and leaves the reading time in the rendered frontmatter. It also marks referral links `rel="sponsored"` (the `SPONSORED` list; add new ones there) and lets long dotted names in prose break before their dots (`<wbr>`).

## Data

- `src/config.ts`: the GitHub user, the résumé path, the Elsewhere links, which repositories Projects leaves out, and the visitor map's script.
- `src/data/posts.json`: the six topics, and the topic and description of each older post.
- `src/data/site-log.json`: hand-written site news for What's New, newest first.
- `src/data/repo-notes.json`: a note for each repository with no description on GitHub, and homepages that replace GitHub's (`null` hides a site that is gone).
- The Projects page and What's New's new repositories come from the GitHub API at build time (`src/lib/github.ts`); when the API is unreachable the build falls back to `src/data/github-snapshot.json`. CI passes `GITHUB_TOKEN` so it isn't rate limited.

## Build output

- After the pages are built, the integrations in `astro.config.mjs` run in order: the sitemap (`lastmod` on posts and `/writing/`), `sitemap-dates` (turns the sitemap's midnight-UTC `lastmod` back into plain dates), `og-cards` (draws `/og/<key>.png` for every page from the list at `/og/cards.json` with Satori and resvg; no browser), `page-weight` (fills in the weight sentence on `/about/`), and `check-dist`, last.
- `src/integrations/check-dist.mjs` fails the build with a list of problems. It checks the top-level names in `dist/`, that every post and required page exists, the heading ids, each page's `<h1>`, title, description, canonical link, card and `lang`, leftover `__PLACEHOLDERS__`, the feed and search index, the sitemap (no Not Found, plain dates), html-validate with `.htmlvalidate.json`, and the style.css and site.js budgets.
- Other repositories' Pages sites share the domain (`/GameKingdom/`, `/YOY-Labs-Website/`), so `dist/` may only hold the top-level names in check-dist's `TOP_LEVEL` list. A new page or file at the top level goes on that list on purpose, and a new repository never takes one of those names.
- `scripts/pixels.mjs` draws the 88×31 buttons and the favicons. Run `node scripts/pixels.mjs` only when the art changes, and commit what it writes to `public/`.

## Files

- `public/resume/` holds the résumé PDF the site links to (`RESUME_URL` in `src/config.ts`). Its LaTeX source is not published; the last version is in git history (`git show 14153b4:public/resume/main.tex`).
