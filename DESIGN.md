# DESIGN.md

The design of xingliu.me as built: the rule it follows, the system that carries it out, and the reasons for the decisions most likely to be undone by accident. Read it before changing the site's look, structure or behavior. It describes the current state only. Where it and the code disagree, the code is right: fix this file in the same commit.

Two files own other subjects, and this one links to them instead of repeating them. [CLAUDE.md](CLAUDE.md) holds the working conventions and gotchas: build flags, the budget rule, spelling, Markdown habits, the data files, the integration order and what check-dist covers. [README.md](README.md) holds the owner's how-tos: writing a post, site news, running and deploying, reserved names, settings and a job change.

Update this file whenever a change touches a token, breakpoint, component, page, URL, script behavior, check, budget, or anything the site says about itself. Figures marked *today* describe the build of 2026-10-26 (12 posts) and move with the content. Everything else is a rule in the code.

Contents: 1 The rule · 2 Principles · 3 Information architecture · 4 Pages · 5 Visual system · 6 Behavior · 7 Content model · 8 Search engines, feeds and cards · 9 Build gates, budgets and CI · 10 Decisions · 11 Checklists

## 1. The rule

> **Only what Navigator could draw, set the way a typographer would.**

"Navigator Grey" is Xing Liu's Home Page as Netscape Navigator 3 drew a page in 1997 (gray page, Times headings, blue links that turn purple once read, shaded rules, bevelled tables), then typeset with care. The rule is written word for word in three places, which must agree: the header of `public/style.css`, the colophon on `/about/` ("One rule decides everything: …") and CLAUDE.md. The design's name keeps its spelling; visible copy says "gray".

- **The first half is the vocabulary:** HTML 3.2 as Navigator 3 drew it with its defaults. That means the `#c0c0c0` page and the default link colors, the two-tone bevel, the shaded `<hr>`, `<table border=1 cellspacing=2>`, sunken fields and raised buttons, plain lists, `<dl>` and `<address>`, a bracketed text navigation, 88×31 buttons, a red NEW!, a dated What's New log, Apache's error page, and the 1996 Core Fonts for the Web.
- **It rules out** what Navigator could not draw: cards, blurred shadows, visible gradients, rounded corners and web fonts. By choice, it also rules out the period clutter Navigator could draw: icons, GIFs, tiled backgrounds, animation such as `<blink>`, fake counters and "under construction".
- **The second half is the craft:** a measure of about 70 characters, leading of 1.6 (1.85 for Chinese), 11.54:1 body contrast, curly quotes, middle dots that never start a line, names and short subtitles that never break, full-width Chinese punctuation with two-character indents, and every type size and breakpoint in rem or em (print type is in pt, and fixed details such as bevels, tap-target heights, the phone gutters, the images and the map well are in px).
- **View Source is part of it.** Every page opens with a greeting comment (`GREETING` in `src/layouts/Base.astro`): "xingliu.me, set like a 1997 home page: no framework on the page, no web fonts, and no tracking except the visitor map on the home page. View Source was how everyone learned HTML. It still works here." `style.css` and `site.js` are served unminified, with their comments.

### 1.1 The brief

The owner's request, verbatim: 把个人站的SEO和审美设计全面升级，我相信你能把它弄得巨屌。这样，我想设计成这样子，就突然打开感觉有点像90年代，00年代的网站，很简陋很朴素，但里面的功能其实都有，而且其实让人很容易阅读，有一种精心设计的复古的美学。然后现在这种把blog全部列出来啥的其实很蠢，你来全部重新设计. Each phrase is a requirement the site meets:

| Phrase | What the site does |
|---|---|
| Upgrade the SEO and the look, and make it insanely good (巨屌) | Unique titles, descriptions and canonical links, a card per page drawn as a Navigator page, JSON-LD, a sitemap with true dates and a full-text feed. Memorable details, all true: purple that remembers what you read, four honest buttons, the navy visitor map, a colophon that weighs the page in modem seconds, an Apache 404 signed by the real server |
| It should feel like the '90s or 2000s the moment it opens | By default, every first screen shows the gray page (Night's `#333333` when the system prefers dark), the bracketed nav, a shaded rule, a Times bold `<h1>` and underlined links |
| Very plain (很简陋很朴素) | One centered column. The avatar is the only picture in the content. Ornament is HTML's own rules, bevels and lists |
| But everything is there | Search in English and 中文, a full-text feed, the archive by date and by topic, contents boxes, share links, copy buttons, live projects, three color schemes, print styles and a 404 that helps |
| Very easy to read | Principle 4 |
| A carefully designed retro aesthetic, not a parody | One written rule, a closed token system, devices that each do a job, and this record of decisions |
| Listing every post is dumb; redesign it all | Home is a front door. The archive is its own page, `/writing/` |

### 1.2 Precedents: when the two halves pull apart

Use these as precedent for new work.

| Need | Period device chosen | Why |
|---|---|---|
| Mark the section being read | A 1997 list-box selection: `--mark-ink` on `--mark`, at the same weight | Bold would reflow the rail |
| Flag a Chinese post | The 中文 chip: flat, with a 1px `--lo` border | Raised means "press me" |
| Search results | AltaVista's: a count, numbered hits with context, the match in bold, the address in small print | Period-true and helpful |
| Contents | A Navigator group box (an etched frame with its title on the top rule), built on `<details>` | It folds without JavaScript |
| Contents on wide screens | A framed site's Contents frame, without the frameset | It uses the empty gray margin |
| Topic index and back links | A run-in index, and "↑ Contents" closing each section | How 1990s FAQs indexed themselves |
| Where a jump landed | A static band in the well color (white, black in Night), like a selected list item | The Yellow Fade is animation |
| Sharing | "Tell a friend about this page", with plain links | No third-party script or cookie, and it works without JavaScript |
| A search field | No clear button | Navigator had none |
| Print | A typed manuscript: black on white, no navigation, URLs after links | A link can't be followed on paper, so its address prints after it |

## 2. Principles

| # | Principle | How the code keeps it | Why |
|---|---|---|---|
| 1 | One design language | Every color is a token taken from Navigator's defaults, HTML 3.2's color names or the 216 browser-safe colors. Three neutrals are added for reading (`#e6e6e6`, `#595959`, `#e0e0e0`) and two for print (`#eeeeee`, `#444444`). Frames come from the bevel vocabulary (§5.8), and type from five stacks (§5.4) | One closed system reads as a homage; mixing eras makes a collage |
| 2 | Every period device does a job | Visited purple marks what you have read, and the Writing lede says so. The bevelled table holds records, the sunken field holds the permalink, and the group box is the contents. What's New is the front page's digest, and NEW! flags what is under 30 days old. Each 88×31 button links to what it claims. The Apache 404 gets you back, and the 28.8k-modem sentence reports About's real weight | A device without a job is a costume, and costume is the parody the owner ruled out |
| 3 | Every claim is true, and stays true between builds | Counts, months, sizes, reading times and the page weight are computed, never typed. "Last modified" comes from git. Projects names its own fetch date, and NEW! expires by itself. No date the site states depends on the daily schedule (§7.7) | The brief bans fake counters and false badges. A homage that lies about itself is a parody |
| 4 | Reading beats nostalgia | Georgia at 19px/1.6 on a 33em column, Paper white one menu away, Chinese set in Song with its own rules (§5.5), em breakpoints, 16px fields on phones, and code and tables that scroll in their own boxes | The owner asked for a site that is very easy to read. Comfort wins wherever the two conflict |
| 5 | It works without JavaScript | Every page reads, navigates and searches with scripts off. Scripts only add (§6.1) | A brief requirement, and the "Viewable with any browser" button and the colophon both claim it |
| 6 | Safari 14 keeps the full layout | Never used: flex `gap`, `:has()`, container queries, `aspect-ratio`, `overflow-wrap: anywhere`, range media queries. Enhancement only: `text-wrap: balance`, `:focus-visible`, `@page` margin boxes, `text-autospace`, `text-justify`, `content: … / ""` alt text and `forced-colors`. `fit-content` and `box-decoration-break` follow their `-webkit-` forms, and `image-rendering: pixelated` follows `-webkit-optimize-contrast`. site.js guards newer APIs (§6.1) | The brief's floor is current evergreen browsers plus Safari 14 |
| 7 | Small | No web fonts (`style.css` has no `url(`). Every page has one stylesheet and one deferred script, plus a 460-byte head script; posts with a contents box add a 136-byte one (477 and 153 bytes with their `<script>` tags). check-dist enforces the budgets (§9.4). The only third-party code loads last, on Home only | Fast pages. The modem sentence on About only works if the page really is light |
| 8 | Posts stay untouched | No design change edits `src/content/posts/`. Older posts' topics and descriptions live in `src/data/posts.json`, the pipeline shapes their HTML at render time, heading ids are frozen in `scripts/heading-ids.json`, and dates ignore commits that do not change the words | The posts are the owner's words, and other sites link to them and to their anchors |

The owner's hard constraints are held the same way: old URLs and anchors by check-dist (§9.2); the paths other repositories need by `TOP_LEVEL` (§3.4); no browser at build time by Satori and html-validate (§8.4, §9); 320–430px screens, with no page-level horizontal scroll and 24px targets, by §5.7 and §5.9; and no tracking beyond the visitor map, with no cookies of the site's own, by keeping the map on `/`, the color choice in `localStorage`, and `/search.json` fetched with `credentials: "omit"`.

## 3. Information architecture

### 3.1 Places

| Place | URL | Answers |
|---|---|---|
| Home | `/` | Who is this, what's new, and where can I go? A 1997 home page answered exactly these |
| Writing | `/writing/` | What has he written? Every post, by date and by topic, with search |
| Post | `/YYYY/MM/DD/slug/` | The text itself, and where to go next |
| Projects | `/projects/` | What has he built? Every non-fork public repository |
| About Me | `/about/` | Who is he, in full, and how is this site made (the colophon)? |
| Not Found | `/404.html`, served at any missing path | What happened, and how do I get back? |

Post URLs are flat and dated, but a post belongs to Writing: its crumbs read `Xing Liu’s Home Page › Writing › {YYYY}`, with the year linking to `/writing/#y{YYYY}`; the nav marks Writing; and its pager ends with "[ Up to the Writing archive ]".

### 3.2 Navigation

- **The nav.** `navHtml(current, section, label)` in `src/lib/nav.ts` builds `[ Home | Writing | Projects | About | Search ]` as one string with exact whitespace. Every page shows it twice: in the masthead (`aria-label="Main"`) and in the footer (`aria-label="Footer"`), so a reader at the end of a long page need not scroll back.
- **Current state.** The current page is `<b aria-current="page">`: bold, and not a link, because a link to the page you are on goes nowhere. On posts the section is `<a class="here" href="/writing/" aria-current="true">`, a bold link. Search (`/writing/#search`) is never current, and Not Found (`current="404"`) marks nothing.
- **Markup.** The brackets and pipes are real text, so Lynx shows them, and screen readers skip them (`aria-hidden`). Each item is a `.ni` nowrap group that carries its own separator, and the single spaces between groups, kept by `compressHTML: false`, are the only places a line may wrap.
- **Breadcrumbs** (`p.crumbs`) appear on Writing, Projects, About and posts, and not on Home or Not Found. The separator is `›`. On Writing, Projects and About the current page ends the trail as plain text. On posts the trail ends with the year, as a link, and leaves the post out. Post crumbs carry `lang="en"`.
- **Search is one place.** It lives on `/writing/` (`form#search`), and there is no `/search/` page. The nav's Search item and Home's "Search (English or 中文)" both link to `/writing/#search`. The footer's mini form appears on every page except `/writing/` and `/404.html`, which carry the full form in `<main>`. With JavaScript, the footer and 404 forms go to `/writing/?q=…#search`; without it, every form asks DuckDuckGo.
- **Cross-links.** Home links to the newest posts, new repositories and site news (What's New), to "all {n} posts · all {m} projects · RSS feed", and to `/about/#colophon` ("How this site is made"). A post's byline links the author to `/about/` (`rel="author"`) and the topic to `/writing/#topic-{id}`. Its pager links the next-older post (`rel="prev"`) and the next-newer one (`rel="next"`). Every footer carries the nav, four buttons and an address that links `/`. Every footer except those of `/writing/` and `/404.html` also carries the mini search form.

### 3.3 URL map

| URL | Source | Indexed |
|---|---|---|
| `/` | `src/pages/index.astro` | yes |
| `/writing/` | `src/pages/writing/index.astro` | yes |
| `/YYYY/MM/DD/slug/` | `src/pages/[year]/[month]/[day]/[slug].astro` | yes |
| `/projects/` | `src/pages/projects/index.astro` | yes |
| `/about/` | `src/pages/about/index.astro` | yes |
| `/404.html` | `src/pages/404.astro` | no: `noindex`, no canonical link, no Open Graph or JSON-LD, and not in the sitemap |

Pages are emitted as `<path>/index.html`, so canonical URLs end in `/`. Paths are case-sensitive on GitHub Pages. A post's file name is its URL (CLAUDE.md): Jekyll's `/:year/:month/:day/:title/`, so every old link still works.

| File | Made by | What it is |
|---|---|---|
| `/feed.xml` | `src/pages/feed.xml.ts` | RSS 2.0, every post in full (§8.3) |
| `/search.json` | `src/pages/search.json.ts` | The search index (§6.2) |
| `/sitemap-index.xml`, `/sitemap-0.xml` | `@astrojs/sitemap`, then `src/integrations/sitemap-dates.mjs` | Every page except Not Found (§8.3) |
| `/robots.txt` | `public/robots.txt` | Allows everything and names the sitemap |
| `/og/<key>.png` | `src/integrations/og-cards.mjs` | A 1200×630 card for each indexed page. Keys: `home`, `about`, `writing`, `projects` and each post's id (§8.4) |
| `/og/cards.json` | `src/pages/og/cards.json.ts` | The list of cards to draw. It is deleted after drawing and never deployed |
| `/style.css?v=…`, `/site.js?v=…` | `public/` | Served verbatim. `?v=` is the first 8 hex digits of the file's SHA-256, so a changed file gets a new URL |
| `/favicon.ico`, `/favicon.svg`, `/apple-touch-icon.png`, `/badges/*.svg` | `public/`, drawn by `scripts/pixels.mjs` | The 16-pixel star, and the four 88×31 buttons |
| `/avatar-240.jpg`, `/avatar-320.jpg` | `public/` | The avatar, self-hosted. The 240 file is shown at 120×120 on Home and drawn on the Home and About cards; the 320 file is shown at 160×160 on About and is the JSON-LD `Person.image` |
| `/resume/Xing_Liu_Resume_20250924.pdf` | `public/resume/` (`RESUME_URL` in `src/config.ts`) | A permanent address |

| Address | What it is |
|---|---|
| `#main` | `<main id="main" tabindex="-1">`, the target of "Skip to content" on every page |
| `/writing/#search`, `/writing/?q={query}#search` | The search form, and a search anyone can link to (`+` reads as a space) |
| `/writing/#by-date`, `#by-topic`, `#y{YYYY}`, `#topic-{id}` | The archive's two indexes. Only years and topics that have posts get an anchor |
| `/#whats-new`, `/#visitors` | Home's sections. The colophon's Privacy item links `/#visitors` |
| `/about/#bio`, `#resume`, `#elsewhere`, `#colophon`, `#buttons` | About's sections. Home and the site-log entry link `#colophon` |
| `#toc`, and each post's heading ids | The contents box and the sections it lists. The ids are frozen in `scripts/heading-ids.json` |

Outside addresses: DuckDuckGo for search without JavaScript (`action="https://duckduckgo.com/"`, with `q` and a hidden `sites=xingliu.me`; the script's fallback links are `https://duckduckgo.com/?q={query}%20site%3Axingliu.me`); each service's own share form (LinkedIn, X, Hacker News, Reddit, and `mailto:`); the owner's accounts (`ELSEWHERE` in `src/config.ts`, which About lists with `rel="me"` and adds `https://github.com/xingliu-aws`); and the map script (`VISITOR_MAP_SRC`).

### 3.4 The shared domain and permanent addresses

- The top level of `dist/` is an allowlist, because other repositories' Pages sites share the domain (CLAUDE.md; README, Reserved names): check-dist's `TOP_LEVEL`, plus year folders matching `YEAR = /^20\d\d$/`. `_astro` is on it although nothing emits it. A `CNAME` file would fail the build: the custom domain is set in Settings → Pages, and Actions deployments ignore the file.
- **Permanent, because others link to them:** every post URL (case-sensitive), every heading id in `scripts/heading-ids.json`, `/resume/Xing_Liu_Resume_20250924.pdf`, `/`, `/404.html`, `/feed.xml` (feed readers subscribe to it) and `/sitemap-index.xml` (named in `robots.txt`). Topic ids appear in addresses too, as `/writing/#topic-{id}`.

## 4. Pages

### 4.1 The frame (`src/layouts/Base.astro`)

- **Order.** The doctype, the greeting comment, then `<html lang>` (`en`, or `zh-CN` on a Chinese post). Then the head, and a body whose `div.page` holds `a.skip` ("Skip to content"), `header.masthead[lang=en]` (the nav, then `<hr>`), `main#main[tabindex=-1]` and `footer.foot[lang=en]`.
- **The head, in order:** charset; viewport `width=device-width, initial-scale=1` (no `maximum-scale`, so readers can zoom); `<title>`; description; either a canonical link plus `<meta name="robots" content="max-image-preview:large">`, or `noindex` on Not Found; author; generator; `<meta name="color-scheme" content="light dark">`; two `theme-color` metas (`#c0c0c0` light, `#333333` dark); the inline head script; the stylesheet; the icons; the RSS autodiscovery link; Open Graph and Twitter tags; JSON-LD; and `site.js` with `defer`. The head script comes before the stylesheet, so a stored color scheme applies before first paint.
- **The footer** (`Footer.astro`): `<hr>`, the nav labeled "Footer", the mini search form (when `footerSearch`), the four buttons, an `<address>` reading "Xing Liu (刘星) · xingliu.me" (the domain links `/`), "Last modified: {Weekday, Month D, YYYY}" in a `<time datetime>`, and "Page colors:" with `select#colours`, shown only with JavaScript.

| Page | `<title>` | `og:title` | Description | `current` | `ogType` / `ogKey` | Footer search |
|---|---|---|---|---|---|---|
| Home | Xing Liu (刘星) — Home Page | Xing Liu’s Home Page | Written in the page | `home` | `website` / `home` | yes |
| Writing | Writing — Xing Liu | Writing | `writingDescription(posts)` | `writing` | `website` / `writing` | no |
| Post | {title} — Xing Liu | {title} | The post's description (§7.1) | `post`, `section="writing"` | `article` / the post id | yes |
| Projects | Projects — Xing Liu | Projects | `projectsDescription(projects)` | `projects` | `website` / `projects` | yes |
| About | About Xing Liu (刘星) | the same | Written in the page | `about` | `profile` / `about` | yes |
| Not Found | Not Found — Xing Liu | none | Written in the page | `404` | none (`noindex`) | no |

**"Last modified"** shows in every footer. It is never the build time.

| Page | Date |
|---|---|
| Home | The later of `pathsLastModified(["src/pages/index.astro", "src/components/AboutText.astro", "src/data/site-log.json"])` and the newest What's New date |
| Writing | `writingLastModified()`: the later of the last commit to `src/pages/writing/index.astro` or `src/data/posts.json` and every post's last change. Its sitemap `lastmod` is the same date |
| Post | Its last text change, if that is later than publication; otherwise its publication date |
| Projects | The later of `pathsLastModified(["src/pages/projects/index.astro", "src/data/repo-notes.json"])` and the newest `pushed_at`, as a Los Angeles day |
| About | `pathsLastModified(["src/pages/about/index.astro", "src/components/AboutText.astro", "public/resume"])` |
| Not Found | The later of `pathsLastModified(["src/pages/404.astro"])` and the newest post's date |

The layout, the post template, the shared components other than `AboutText.astro`, `style.css` and `site.js` are in no list, so a change confined to them moves no page's date. Editing Home's, Writing's, Projects', About's or Not Found's own `.astro` file moves that page's date, and editing `AboutText.astro` moves Home's and About's. Without usable git history, posts fall back to their publication dates, Writing to its newest post, and the other pages to the build day.

### 4.2 Home `/`: the front door

In order: `div.hello` (the avatar `/avatar-240.jpg` at 120×120 in a sunken frame, alt "Vault Boy badge reading “My mom thinks I’m S.P.E.C.I.A.L.”"; `<h1>Xing Liu’s Home Page</h1>`; and `p.sub[lang=zh-CN]` 刘星的个人主页); `AboutText.astro`, the owner's own two paragraphs, word for word, shared with About; What's New (`#whats-new`): five dated lines, then "Older news: all {n} posts · all {m} projects · RSS feed"; `div.columns` with "On this site" (Writing, Projects, About me, the résumé PDF with its size, Search) and "Elsewhere" (GitHub, LinkedIn, Instagram, "RSS feed (full text)", "How this site is made"); and last, Visitors (`#visitors`), the map.

Why: a 1997 home page answered who, what's new and where to go, in that order, and the owner called listing every post dumb, so Home never lists the archive. The map is the only third-party code, so it comes last, on this page only, loaded after everything else into a well that already has its height (§6.4).

### 4.3 Writing `/writing/`: the archive

In order: crumbs; `<h1>Writing</h1>`; `p.lede` "{n} posts since {Month YYYY}: {WRITING_SUMMARY}. Newest first. Links you have already visited turn purple, so the list remembers what you have read.", with "purple" set in the visited color (`.visited-demo`); the main search form and `#results`; `p.views` "Index: by date · by topic · RSS feed (full text)"; "By date" (`#by-date`), where each year with posts gets `h3.year#y{YYYY}` "{YYYY} ({n})" and a `ul.archive`; `<hr>`; and "By topic" (`#by-topic`), where each topic with posts, in `posts.json` order, gets `div.topic` with `h3#topic-{id}` "{name} ({n})" and a `ul.runin`.

Why: the archive is its own page, never the home page. It has two indexes because readers browse by time and by subject. The posts fit on one page, so there is no pagination. Empty years and topics are left out (*today* `meta` has no posts, so five topics show). The footer's mini form is left out because the full form is on the page.

### 4.4 Post `/YYYY/MM/DD/slug/`: reading

In order: `article.post` (with `lang="zh-CN"` on a Chinese post), holding `header.post-head` (crumbs, the `<h1>`, the byline, and on posts of 10 minutes or more `p.colours-line` with a second color menu, `select#colours-top`), then `div.article` (plus `fiction` for the fiction topic) with the pipeline's output (§7.3), `p.end` "* * *" and `p.print-url`, shown only in print. Then `<hr>`, the share box (`section.share`), `<hr>`, and the pager (`nav.pager`, `aria-label="Older and newer posts"`) followed by `p.up` "[ Up to the Writing archive ]".

- **English byline:** "Posted {Weekday, Month D, YYYY} by Xing Liu · Updated {Month D, YYYY} · {n} minute read · {Topic}". "Updated" appears only when the words changed on a later day than publication (§7.4).
- **Chinese byline:** "{YYYY年M月D日}（{星期X}）发表 · 作者 刘星 · 更新于 {…} · 约 {n} 分钟读完（{count} 字） · {topic.zh}". The author links to `/about/` and the topic to `/writing/#topic-{id}`.

Why: the title is the page's only `<h1>`. The color menu sits under the byline only on long reads: those readers find Paper white near the text, and everywhere else it stays out of the way between title and text. Share links are plain links to each service's own form, so there is no third-party script or cookie, and they work without JavaScript. Each end of the pager says so ("This is the first post.", "This is the newest post."), which is clearer than an empty cell. The Chinese reader gets Chinese wherever it belongs to the article (the byline, 目录, ↑ 目录), while the site's chrome (nav, crumbs, share box, pager, footer and skip link) stays English with `lang="en"`.

### 4.5 Projects `/projects/`

In order: crumbs; `<h1>Projects</h1>`; "Every public repository I have started on GitHub, most recently updated first. Forks are left out. The list comes from the GitHub API when the site is built; the date is at the foot of the table."; then `div.table-scroll.projects-wrap[role=region][tabindex=0][aria-label=Projects]` around `table.projects[role=table]`, whose rows and cells carry explicit roles. The columns are Repository | About | Language | Stars | Updated. A repository with neither a description nor a note reads "No description on GitHub." in italics. The foot line (`p.small`) reads "Fetched from api.github.com on {date}." when the data is live, or "GitHub could not be reached when the site was built, so this list comes from a saved copy made on {date}." from the snapshot, then "Forks and everything else are at github.com/xingliu14."

Why: the brief asked for every non-fork public repository, not a top three. The bevelled table is the period device for a list of records. Live data names its own date and admits when it is a saved copy. The description says "(forks left out)" because GitHub's own count includes them.

### 4.6 About `/about/`

In order: crumbs; `<h1>About Me</h1>`; `img.portrait` (`/avatar-320.jpg`, 160×160, floated right); `AboutText`; "Short bio" (`#bio`): "For introductions:", then a third-person, pronoun-free `blockquote.bio`; "Résumé" (`#resume`): three employers, then "A longer résumé, written in {Month YYYY}, as a PDF: {file} ({kb} KB)." from `RESUME` in `src/lib/resume.ts`; "Elsewhere" (`#elsewhere`): `dl.elsewhere`, with `rel="me"` on each account; `<hr>`; "Colophon" (`#colophon`): the rule, then nine items with bold labels: Color, Type, Machinery, Search, Dates, Feeds and addresses, Previews, Privacy and The star; "The buttons" (`h3#buttons`), whose `ul.claims` restates each button's claim in text; and `p.weight`, the weight sentence, filled in after the build (§9.4).

Why: the colophon is where the design explains itself, so readers see the look as a choice. A comment in the page says every sentence in it must stay true of the shipped site (§7.7). The résumé's size is read from the PDF and its month from the file name (`…_YYYYMMDD.pdf`), so a new résumé needs only a dated file name and `RESUME_URL` pointed at it (README). The short bio gives people introducing the owner something to paste.

### 4.7 Not Found `/404.html`

In order: `<h1>Not Found</h1>`; `p#missing` "The requested URL was not found on this server."; `p#didyoumean`, hidden until a script fills it; "Perhaps it moved, or the link has a typo. One of these may help:"; the 404 search form; "The five newest posts" (`#newest`), as brief archive rows dated `YYYY-MM`; "Or start again from the home page or the Writing archive."; `<hr>`; and `<address class="server">GitHub Pages server at xingliu.me Port 443</address>`, italic like Apache's signature.

It passes `noindex`, `current="404"` and `footerSearch={false}` to Base. GitHub Pages serves this one file at any missing path, at any depth, so no link on it is path-relative: the site's own links are root-relative, the skip link is `#main`, and the rest are absolute. Why: Apache's error page is the period device, and it has a job, getting the reader back (§6.5). Its signature is true: GitHub Pages does serve the file, over HTTPS.

## 5. Visual system

### 5.1 Where the look is defined

| File | What it decides |
|---|---|
| `public/style.css` | All of it. Its header names its sections: 0 Faces, 1 Tokens, 2 Page, 3 Text, 4 Code, 5 Tables, 6 Forms, 7 Masthead, 8 Home, 9 Writing, 10 Post, 11 Contents box, 12 Chinese posts, 13 Projects, 14 About, 15 Search, 16 Not Found, 17 Footer, 18 Small screens, 19 Print |
| `src/layouts/Base.astro` | The `color-scheme` and `theme-color` metas, the head script, and the frame |
| `src/lib/nav.ts` | The nav, as one string |
| `src/lib/markdown.mjs` | Every class inside a post body |
| `src/components/`, `src/pages/` | Every other class |
| `public/site.js` | What appears at run time: `.code-bar`, `.scroll-hint`, the rail's mark, search results, the map, NEW! removal and scheme changes |
| `scripts/pixels.mjs` | The 88×31 buttons, the favicons and the touch icon, as committed pixel art |

### 5.2 Color

Components use tokens only, all defined on `:root`. Paper white redefines four of them. Night is defined twice, with the same 20 tokens and `color-scheme: dark` (§5.3), and the two blocks must stay identical. Print resets 16 tokens.

| Token | Netscape gray | Paper white | Night | Print | Used for |
|---|---|---|---|---|---|
| `--page` | `#c0c0c0` | `#ffffff` | `#333333` | `#ffffff` | The page; button faces; the skip link; behind the contents title |
| `--ink` | `#000000` | = | `#e6e6e6` | `#000000` | Text; the focus ring; the URL in search results |
| `--link` | `#0000ee` | = | `#99ccff` | `#000000` | Links |
| `--visited` | `#551a8b` | = | `#cc99ff` | `#000000` | `a:visited`, and the word "purple" in the Writing lede |
| `--active` | `#800000` | = | `#ff9966` | `#000000` | `a:active` |
| `--hi` | `#ffffff` | `#ffffff` | `#808080` | `#ffffff` | Bevel highlights; the lower line of `hr` |
| `--lo` | `#808080` | `#999999` | `#000000` | `#999999` | Bevel shadows; the upper line of `hr`; etched frames; the chip's border |
| `--edge` | `#000000` | = | `#000000` | not reset | The hard 1px edge of buttons, fields and wells |
| `--well` | `#ffffff` | = | `#000000` | `#ffffff` | Code wells, fields, table cells, Projects wells, and the `:target` band where `--band` is unset |
| `--band` | unset | `#e0e0e0` | unset | unset | The `:target` band in Paper white, where `--well` matches the page |
| `--well-ink` | `#000000` | = | `#e6e6e6` | `#000000` | Text in wells; the select's arrow |
| `--head` | `#c0c0c0` | `#e0e0e0` | `#333333` | `#eeeeee` | Table header cells |
| `--hint` | `#595959` | = | `#999999` | not reset | Placeholders (`opacity: 1`) |
| `--mark` | `#000080` | = | `#e6e6e6` | not reset | `::selection` and the current contents entry: background |
| `--mark-ink` | `#ffffff` | = | `#000000` | not reset | The same: text |
| `--new`, `--new-ink` | `#cc0000`, `#ffffff` | = | = | not reset | NEW! |
| `--kw` | `#0000ff` | = | `#6699ff` | `#000000` | Code: keyword, important, atrule, boolean |
| `--cm` | `#008000` | = | `#66cc66` | `#444444` | Code: comment, prolog, doctype, cdata (italic); inserted |
| `--st` | `#800000` | = | `#ff9966` | `#000000` | Code: string, char, attr-value, url, regex; deleted |
| `--nu` | `#800080` | = | `#cc99ff` | `#000000` | Code: number, symbol, constant, entity |
| `--ty` | `#008080` | = | `#66cccc` | `#000000` | Code: class-name, builtin, namespace, attr-name, property, variable, key |
| `--fn` | `#000080` | = | `#99ccff` | `#000000` | Code: function, tag, selector |

`=` means the same as Netscape gray. Night's rules are screen-only, so print starts from `:root`. NEW! uses `--new` and `--new-ink`, which no scheme or print block redefines, so it is white on `#cc0000` everywhere.

- **Fixed colors outside the tokens.** The map well is `#000080` with `#ffffff` text and links in every scheme, matching the widget's own `co=000080`, `cl=c0c0c0` and `ct=ffffff`. The buttons and the favicon are images with their own palettes (§5.9). Print hard-codes `#ffffff` and `#000000`, and gives `pre` a `#999999` border.
- **Contrast** (WCAG 2.x, from the token values). Body text on the page is 11.54:1 in gray, 21:1 in Paper white and 10.12:1 in Night. Links are 5.17, 9.40 and 7.48:1, and visited links 6.05, 11.01 and 5.76:1. The lowest text pair is teal `--ty` on a white well, at 4.77:1. NEW! is 5.89:1, map text 16.01:1, and the selection and contents mark are 16.01:1 (16.83:1 in Night).
- **Boundaries.** In the light schemes a field or button is found by its black `--edge` (11.54:1 against gray). In Night `--edge` is only 1.66:1 against `#333333`, so the `#808080` `--hi` highlight (3.20:1) carries the boundary.
- **Rule for a new color:** take it from the three sources, make it a token, and give it values in `:root`, in both Night blocks, in Paper white if it depends on the page color, and in print if it touches text. Check text at 4.5:1 and UI boundaries at 3:1 in all three schemes. Each token added to Night costs its bytes twice.

### 5.3 Color schemes

| Menu choice | `<html data-theme>` | Result |
|---|---|---|
| Automatic | absent | Netscape gray, or Night when the system prefers dark, on screen only |
| Netscape gray | `grey` | Gray, whatever the system says |
| Paper white | `white` | White. It is never chosen automatically |
| Night | `night` | Night, whatever the system says |

- **Selectors.** Gray is the plain `:root` block. Paper white is `:root[data-theme="white"]`. Automatic Night is `@media screen and (prefers-color-scheme: dark) { :root:not([data-theme="grey"]):not([data-theme="white"]) {…} }`, and chosen Night is `@media screen { :root[data-theme="night"] {…} }`.
- **Before first paint.** The head script (`HEAD_SCRIPT` in Base) adds `js` to `<html>`. It then reads `localStorage.getItem("colours")` inside `try`, accepts only `grey`, `white` or `night`, sets `data-theme`, sets every `theme-color` meta to `#c0c0c0`, `#ffffff` or `#333333`, and sets the `color-scheme` meta to `dark` for Night and `light` otherwise. A stored choice therefore never flashes the wrong colors; with storage blocked, the page stays on Automatic.
- **The menus** (`ColoursSelect.astro`): Automatic, Netscape gray, Paper white and Night. One sits in every footer (`#colours`), and posts of 10 minutes or more repeat it under the byline (`#colours-top`). Both are hidden until `.js` is set, since without scripts they would do nothing. A change does what the head script does, writes or removes the stored choice inside `try` (for private windows), restores each meta's default for Automatic, and sets every menu on the page to the new value.
- **Why.** Gray is the identity: Navigator's page color, at 11.54:1. Paper white is the reader's alternative for long reading, one menu away on every page and right under the title of long reads. Night is "the same browser with the lights off, in browser-safe colours" (the `style.css` comment). It is screen-only, so print is always black on white. It lightens `--mark`, because navy on `#333333` is 1.27:1. The bevel shades are tokens because Navigator derived both from the page color.

### 5.4 Type

| Token | Stack | Used for |
|---|---|---|
| `--text` | `Georgia, "Gelasio", "Noto Serif", "DejaVu Serif", "Songti SC", STSong, SimSun, serif` | Body text, the nav, the footer address |
| `--display` | `"Times New Roman", Times, Tinos, "Liberation Serif", "Noto Serif", "Songti SC", STSong, SimSun, serif` | Headings, the contents title, fiction scene numbers |
| `--small` | `Verdana, "DejaVu Sans", "Bitstream Vera Sans", Tahoma, Geneva, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif` | Small print and controls: bylines, crumbs, tables, fields, buttons, captions, the footer |
| `--mono` | `Monaco, "Lucida Console", "Andale Mono", "DejaVu Sans Mono", Consolas, "Courier New", "PingFang SC", monospace` | Code, list dates, printed URLs |
| `--cjk` | `"ZH Punct", "Times New Roman", Times, Tinos, "Songti SC", STSong, SimSun, "Noto Serif CJK SC", "Source Han Serif SC", serif` | Chinese: Times for Latin and figures, Song for Han characters |

- **Order.** The Core Fonts for the Web come first, then stand-ins: metric-compatible ones where they exist (Gelasio for Georgia, Tinos and Liberation Serif for Times), which keep the same line lengths on systems that have them, then similar faces that are not metric-compatible (Noto Serif, which is Android's serif, DejaVu Serif, and DejaVu Sans and Bitstream Vera Sans for Verdana), then the CJK faces.
- **Faces named outside the tokens.** The 中文 chip names `"PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC"` before `var(--small)`. The map's placeholder link uses `Arial, Helvetica, sans-serif`, and the `@page` margin boxes use `8pt Verdana, sans-serif`.
- **"ZH Punct"** is the only `@font-face`, made of two rules with `local()` sources only. U+201C–201D (“ ”) come from Hiragino Sans GB, Heiti SC or SimSun, which draw them a full em wide; Times and Song draw them about 0.4em. U+2014 and U+2026 (— …) come from Song, which centers them like Han text. ‘ ’ stay with Times, because ’ is also the English apostrophe, and the interpunct · stays narrow. Without these faces the rule is skipped, and the quotes stay narrow but correct.
- **Scale.** `html { font-size: 100% }` keeps the reader's default size. The body is `1.1875rem/1.6` (19px at the default), `1.125rem` with 1.58 leading at `(max-width: 40em)` (18px), and `1.0625rem` at `(max-width: 22.5em)` (17px). Headings are Times bold at `line-height: 1.18`: `h1` is `2.3em` (`2em` on phones, `1.85em` at 22.5em), `h2` `1.62em` (`.article h2` `1.5em`), `h3` `1.32em`, `h4` `1.14em`, `h5` `1.04em`, and `h6` `1em` italic. Heading margins are `1.55em 0 0.45em`, so a heading sits closer to what follows it.
- **Small print.** Verdana at 14px (`0.875rem`) sets buttons, bylines, `.more`, `.views`, the share line and `.up`; the Monaco list dates are 14px too (13px in archive rows on phones, Not Found's included). Verdana at 13px (`0.8125rem`) sets crumbs, captions, hints, pager labels, `.meta`, `.claims`, `.weight` and the footer. 12px (`0.75rem`) is the floor: `.new`, `.code-bar`, `button.small`, `.article .top` and `.mm-wait`. The one exception is `table code` at `0.92em` of a table, which is 11.96px on phones.
- **Measure.** `.page { max-width: 33em; margin: 0 auto; padding: 0 1.5rem }` gives 627px at 19px, about 70 characters of Georgia. On phones the gutters are `16px`.
- **Why.** Georgia was drawn by Matthew Carter for screens and given away by Microsoft in 1996; Times is every browser's default; Verdana comes from the same Core Fonts set; Monaco and Lucida Console are the period's system monospaces. Monaco also lines up the list dates, whose days are space-padded ("Jan  1", from `monthDay()` with `white-space: pre`). Sizes in rem and em honor the reader's default font size.

### 5.5 Chinese typesetting

- A Chinese post carries `lang="zh-CN"` on `<html>` and on `article.post`. The site chrome inside it carries `lang="en"`. Any Chinese inside English text is marked `lang="zh-CN"`. The 中文 chip marks only its text, because its `title` tooltip ("Written in Chinese") is English.
- `--cjk` sets the Chinese title, body, `h2`–`h5`, contents box, Home's subtitle and Chinese search snippets. `.article:lang(zh)` is `font-size: 1.04em; line-height: 1.85; text-autospace: normal`, and its paragraphs are `margin: 0 0 0.55em; text-indent: 2em`.
- Paragraphs are justified (`text-justify: inter-character`) only at `(min-width: 43.75em)`, and ragged right below that, where justification would stretch the gaps around Latin words on short lines.
- `font-feature-settings: "palt"` applies to the Chinese title, the Chinese contents box and fiction's chapter heads.
- There is no fake italic: Song has none, so `address :lang(zh), h6 :lang(zh), .s6 :lang(zh) { font-style: normal }`.
- In English text 刘星 is always `<span lang="zh-CN" class="nowrap">`, so a narrow column never splits the name. In the Chinese byline it is the author link, inside the byline's `.nowrap` group. Only Home's subtitle, 刘星的个人主页, leaves the name without `.nowrap`.

### 5.6 Line breaks

- **`.nowrap`** holds the byline's groups, 刘星, 中文 in prose, the archive's chip and minutes, and the kept end of run-in titles.
- **Middle dots** are written `&nbsp;· ` (bylines, "Older news", the Index line, search results), so a wrapped line never starts with a dot.
- **Titles keep their last two words together** (U+00A0) in the pager and in search results.
- **Run-in topic lists** (`runIn()` in `src/pages/writing/index.astro`): a title of 30 characters or fewer (`SHORT_TITLE`) stays whole; a longer one keeps its last two words together; a one-letter word is bound to the next ("I Changed Job"); and lines break after the `·` separator (`<wbr>`).
- **Subtitles.** `splitSubtitle()` (`src/lib/title-parts.mjs`) matches `/^(.+：)([^\p{Script=Han}：]{1,24})$/u`, so the `<h1>` becomes `侦探小说：<wbr><span class="nowrap">He’s Replaceable</span>`. `.post-head h1 .nowrap { display: inline-block; white-space: normal }` keeps the subtitle whole unless it is wider than a line, and the social card uses the same split.
- **The pipeline adds `<wbr>`** after each `/` in contents entries (`engine/<wbr>block_manager.py`), and before the dots of dotted names longer than 24 characters in prose (`LONG_NAME`), never in code.
- **A word joiner** (U+2060) holds footnote marks (`content: "\2060["`) and the chip in search results to the word before.
- **The byline date** keeps month and day together (`January&nbsp;23,`). On phones, `.byline > .nowrap:first-child { white-space: normal }` lets the date wrap under large text settings.
- **Long words:** `body { overflow-wrap: break-word }`. The archive repeats it, and uses `minmax(0, 1fr)` on phones.

### 5.7 Layout and breakpoints

- **One centered column**, `.page`: `a.skip`, `header.masthead`, `main#main`, `footer.foot`. There is no logo or title bar; each page's `<h1>` is its first heading.
- **Breakouts** at `(min-width: 56.25em)`: `.article div.highlighter-rouge, .article .table-scroll { margin-left: -3em; margin-right: -3em }` (741px wide at 19px). Article cells tighten to `padding: 0.4em 0.55em`, so an eight-column table fits, and `.projects-wrap` widens by `4.6em` on each side. Below 900px nothing leaves the column.
- **The contents rail** at `(min-width: 77.5em)`: `.toc { position: absolute; top: 0; bottom: 0; right: 100%; width: 15rem; margin: 0 2.75rem 0 0; font-size: 0.8em }` hangs off `.article { position: relative }` into the empty left margin. `.toc details` is `position: sticky; top: 1.6rem`. Only the list scrolls (`max-height: calc(100vh - 6rem)`, with `padding-top: 4px` so focus rings and the mark aren't clipped), and the back links hide. Top-level code wells and tables after the box use `margin-left: 0; margin-right: -6em`: the same width, clear of the rail. A sibling selector, `.article .toc ~ …`, is enough because the pipeline puts the box before every top-level figure, so no `:has()` is needed. A code well nested in a list or blockquote keeps the `-3em` breakout, and a nested table has no scroll box to break out (§5.9).
- **No horizontal page scroll** from 320px up. Only `pre` and `.table-scroll` scroll sideways, inside their own boxes, and both take `tabindex="0"` so the keyboard can scroll them.

| Query | px at a 16px default | What changes |
|---|---|---|
| `(max-width: 22.5em)` | ≤ 360 | Body 17px; nav 15px; `h1` `1.85em`; portrait 88px; a tighter search row, so the 16px field shows its placeholder whole |
| `(max-width: 40em)` | ≤ 640 | Phones: body 18px/1.58; 16px gutters; 16px fields; the Home columns, pager and What's New rows stack; the archive grid becomes `3.6em minmax(0, 1fr)`; topics become lists; the contents box closes before paint and its rows get `min-height: 24px`; share links don't wrap; portrait 112px; tables 13px with `th` nowrap and cells `min-width: 6.5em`; Elsewhere stacks; the buttons go two by two |
| `(min-width: 43.75em)` | ≥ 700 | Chinese paragraphs are justified |
| `(min-width: 43.75em) and (max-width: 77.4375em)` | 700–1239 | The contents box lists in two columns |
| `screen and (max-width: 56.1875em)` | < 900 | Each Projects row becomes a sunken well. It is screen-only, so print keeps the table |
| `(min-width: 56.25em)` | ≥ 900 | Breakouts, and tighter article cells |
| `(min-width: 77.5em)` | ≥ 1240 | The contents rail; site.js marks the section in view |
| `(forced-colors: active)` | — | `select { appearance: auto }` brings back the arrow, and the current contents entry uses `Highlight` and `HighlightText` |

Paired queries sit 1px apart at the default size (`56.1875em`/`56.25em`, `77.4375em`/`77.5em`), and 641–699px has no rule of its own. The Projects block comes after the phone block on purpose, so its `.projects td { min-width: 0 }` wins. The scripts use the same em values (CLAUDE.md).

### 5.8 Bevels

| Kind | CSS | Means | Used by |
|---|---|---|---|
| Raised | `border: 2px solid; border-color: var(--hi) var(--lo) var(--lo) var(--hi); box-shadow: 1px 1px 0 var(--edge)` | Press me | `button`, `.skip`; `kbd` is a 1px version without the edge |
| Pressed | `border-color: var(--lo) var(--hi) var(--hi) var(--lo); box-shadow: inset 1px 1px 0 var(--edge)` | Pressed | `button:active`, `button[aria-pressed="true"]` |
| Sunken | `border-color: var(--lo) var(--hi) var(--hi) var(--lo)`, plus `box-shadow: inset 1px 1px 0 var(--edge)` on fields and wells | Holds content or takes input | Fields, `pre`, `.map-well`, Projects rows below 900px; `.avatar`, `.portrait` and table cells without the edge |
| Shaded rule | `hr { border-top: 1px solid var(--lo); border-bottom: 1px solid var(--hi) }` | A section break | The masthead, the footer, page sections |
| Etched frame | `border: 1px solid var(--lo); box-shadow: inset 1px 1px 0 var(--hi), 1px 1px 0 var(--hi)` | A labeled group | `.toc details` |
| Etched line | `border-left` or `border-top` in `--lo`, with an inset `--hi` line beside it | Quotation, notes | `blockquote`; `.footnotes` and a closed `.toc details` |
| Bevelled table | A raised 1px outline, `border-collapse: separate; border-spacing: 2px`, and sunken 1px cells | HTML 3.2's `<table border=1 cellspacing=2>` | Markdown tables, Projects |
| Flat | `border: 1px solid var(--lo)` | A label, not a control | The 中文 chip |

Raised is for things you press, and for the 1px outer frame of a bevelled table, as HTML 3.2 drew `border=1`. Sunken holds content or takes input. Flat is a label. `box-shadow` is never blurred: it draws bevel edges, the `:target` band (`0 0 0 0.25em var(--band, var(--well))`) and the contents mark (`0 0 0 2px var(--mark)`). The only gradients are the two hard-stop triangles of the `<select>` arrow. In Night, `--hi` is gray and `--lo` black, so the same rules draw gray highlights and black shadows.

### 5.9 Components

- **Links and focus.** Links are underlined (`text-decoration-thickness: 1px; text-underline-offset: 0.13em`), so color is never the only cue. The exceptions carry another cue: footnote marks are bracketed, and the current contents entry is drawn as a selection. Print drops underlines. Focus is `outline: 2px dotted var(--ink); outline-offset: 2px` on `a`, `button`, `input`, `select`, `summary` and `[tabindex]`. Links (except the 88×31 buttons, whose `.badges a:focus` rule keeps the ring), summaries and `[tabindex]` drop it for mouse focus through `:not(:focus-visible)`, and browsers without `:focus-visible` keep it always. `::selection` uses `--mark` and `--mark-ink`. The skip link is a raised button at `top: -100px` that moves to `top: 8px` on focus.
- **Jump target.** `.article :target` gets a static band, like a selected list item: `var(--band, var(--well))` as background and as a `0.25em` box-shadow, with `scroll-margin-top: 1em`. That is the well color, white or, in Night, black, except in Paper white, where the well is as white as the page and `--band` makes it `#e0e0e0`. Print removes it.
- **Fields and buttons.** Fields are sunken, `min-height: 32px`, with `border-radius: 0`, `appearance: none`, and placeholders in `--hint` at `opacity: 1`. Search fields hide WebKit's decoration and cancel button. `select` draws its arrow with two `linear-gradient` triangles in `--well-ink`, and the color menus are 13px with `min-height: 28px`. Buttons are raised, bold 14px Verdana, `min-height: 32px`; `button.small` is 12px with `min-height: 26px`. The search row is a flex row whose field has `margin-right: 8px`, not a flex `gap`. All three search forms use the placeholder "KV cache, Bitcoin, 侦探…". On phones, search fields, the permalink and the color menus are 16px, because iOS zooms into smaller fields on focus.
- **Tap targets** are 24px or more: buttons 26 or 32px, fields 32px, color menus 28px, `.toc summary` 24px, contents rows on phones `min-height: 24px`, `.article .top` a 24px line, and the byline's 24.5px lines, which put its links 24px apart. Other inline links rely on WCAG 2.5.8's spacing exception. On phones `.share p a` never wraps, so "E-mail" can't split into two overlapping targets.
- **Code wells** keep kramdown's markup: `div.language-X.highlighter-rouge > div.highlight > pre.highlight[tabindex=0] > code`. A well is sunken and filled with `--well` (white, black in Night), like Navigator's `<textarea>`, and its lines never wrap. `.js div.highlighter-rouge { position: relative; padding-top: 30px }` reserves room for the code bar, so nothing moves when it lands. The bar's label is plain, normal-case text with no `.lang` rule, because uppercase tracked labels are a modern tell. Prism's token classes map to `--kw`, `--cm` (italic), `--st`, `--nu`, `--ty` and `--fn`, "the way 1990s editors coloured code". Inline code is 0.8em Monaco with no background, like `<tt>`.
- **Tables.** The pipeline wraps each top-level table (a direct child of the post body) in `div.table-scroll[role=region][tabindex=0]`, labeled "Table N: {nearest heading above}". A table nested in a list or blockquote is not wrapped, so it gets no scroll box, label or tab stop. GFM alignment becomes `al-left`, `al-center` or `al-right`, of which CSS needs only the last two, and every `th` gets `scope="col"`. `td` takes the well colors and `th` `--head`. On phones tables are 13px, `th` doesn't wrap, and cells keep `min-width: 6.5em`, so a wide table scrolls rather than breaking into one word per line. The scroll hint appears only while a table overflows (§6.5), because a bevelled frame makes a clipped table look complete.
- **The contents box** is `nav.toc#toc > details[open] > summary + ol` (`ul` for fiction).
  - Its title sits on the top rule like a `<legend>`: `.toc summary` is block-level, `width: fit-content`, `margin: -0.72em 0 0.35em -0.4em`, on `background: var(--page)`. It has to be block-level, because a negative top margin cannot lift an inline-block out of its line box.
  - The marks ▾ and ▸ are written `content: "\25BE"; content: "\25BE" / ""`, so they have no spoken text and older browsers keep the first value. The count (`.toc-n`, " (12 sections)") shows only while the box is closed.
  - Closed, the box folds to its title on a shaded rule (`padding: 0; border-width: 1px 0 0; box-shadow: inset 0 1px 0 var(--hi)`), because a closed full frame looks like an empty text field.
  - The current entry is `--mark-ink` on `--mark`, plus a 2px `--mark` shadow, with no underline and the same weight as its neighbors.
  - Section numbers are `<span class="secno">N.</span>`, written into the HTML rather than drawn with CSS counters, so the list and the headings always agree, Safari 14 included. Back links are `p.top` "↑ Contents" (↑ 目录 in Chinese).
- **Share box:** "Tell a friend about this page", then `LinkedIn | X | Hacker News | Reddit | E-mail` as plain links with no `target`, then `input#permalink[type=url][readonly]` with a visually hidden label, and a `button.small.copy-link` "Copy link" that shows only with JavaScript. The pager below it is a GNU-manual pager in two grid columns, one on phones.
- **What's New:** `ul.log`, whose rows are a grid of `5.6em 1fr` with the date above the text on phones, and ISO dates in Monaco. NEW! is `span.new`, bold white 12px Verdana on `--new`, followed by a real space, so removing it leaves no gap.
- **Archive rows:** a grid of `4.6em 1fr` (`3.6em minmax(0, 1fr)` on phones) with a Monaco date column, then the title and "{n} min". A Chinese post adds the chip, `span.nowrap > span.zh[title="Written in Chinese"] > span[lang=zh-CN]`, which shares its `.nowrap` with the minutes, so the minutes never wrap alone. `.zh` is `display: inline`, so a word joiner can hold it. Not Found reuses the grid with brief rows.
- **By topic:** a run-in paragraph per topic on wide screens, as 1990s FAQs indexed themselves: the heading is inline and followed by a silent ` — ` (`.topic h3::after`). On phones each topic is a heading over a bulleted list.
- **Projects.** On wide screens it is the bevelled table, with `td.name` and `td.upd` nowrap and `td.num` right-aligned. Below 900px each `tr` becomes a sunken well, `thead` is visually hidden, and the last line reads "{Language} · Stars: {n} · Updated {date}", drawn with `::before` content. Names wrap there (`white-space: normal; overflow-wrap: break-word`), because a GitHub name can be longer than a phone line. The explicit ARIA roles keep the table's meaning, because Safari drops table semantics when `display` changes.
- **The map well:** `.map-well` is `box-sizing: border-box; width: 304px; max-width: 100%; min-height: 189px`, which leaves 300×185px, the widget's size, inside the 2px bevel. It is navy, `line-height: 0`, and `overflow: hidden`, which clips the widget to the well. Its placeholder, `p.mm-wait`, links "[ Visitor map at mapmyvisitors.com ]" until `#mapmyvisitors + * ~ .mm-wait { display: none }` matches. `.map-well:focus-within` draws the focus ring round the whole well, because the well clips its links and a black ring can't be seen on navy. The caption is "Where readers come from, counted by mapmyvisitors.com."
- **Buttons and icons.** The four 88×31 SVG buttons in `public/badges/` are true pixel art (`shape-rendering="crispEdges"`): a two-pixel raised bevel, with a 3×5 small-capitals line over a 5×7 capitals line. CSS shows them with `image-rendering: pixelated`, two by two on phones (`max-width: 196px`). Each links to what it claims, and each alt text starts with the words drawn on it, so voice control can find it:

  | File | Face | Links to | Alt |
  |---|---|---|---|
  | `rss.svg` | `#ff6600` | `/feed.xml` | Subscribe: RSS feed, every post in full |
  | `astro.svg` | `#000080` | `https://astro.build/` | Built with Astro |
  | `github-pages.svg` | `#000000` | `https://pages.github.com/` | Hosted on GitHub Pages |
  | `any-browser.svg` | `#008000` | `https://www.anybrowser.org/campaign/` | Viewable with any browser: works without JavaScript, CSS or images |

  The favicon is a 16×16 pixel star, because 星 means star: a `#ffcc00` face, `#ffff99` on the lit top-left edge, `#cc6600` on the shaded bottom-right and a black outline. It ships as `favicon.svg` and as `favicon.ico` (16, 32 and 48px), and `apple-touch-icon.png` is the star at 9× on `#c0c0c0`, 180px square. The files are committed; regenerate them only when the art changes (CLAUDE.md).
- **Fiction** (`div.article.fiction`) is set like a book: a centered title page (`h2` at normal weight, then a letter-spaced 正文 `h3`), centered chapter heads (`h4`, with `palt`), and scene numbers "— N —" (`h5`, with silent dashes from `::before` and `::after`). It has no section numbers, and its contents box is a `<ul>`.
- **Controls that need scripts** are hidden until `.js`: `.prefs, .colours-line, .copy-link { display: none }`. The head script sets `js` before first paint, so these controls and the code bar take their room from the first frame, and without scripts they are hidden, not broken.

### 5.10 Print and motion

- **Print** turns a post into a typed manuscript. `@page { margin: 2cm 2.2cm }`, with the page number at bottom right and "xingliu.me" at bottom left where margin boxes are supported. `:root, :root[data-theme]` resets the tokens to black on white, and Night never applies. The body is 11pt with 1.45 leading at full width.
- **Hidden in print:** `.navbar, .skip, .share, .pager, .up, .badges, .prefs, .colours-line, .search, .results, .code-bar, .scroll-hint, .visitor-map, .masthead hr, .top, .toc summary::before`, and the rules after the post (`.post ~ hr`). **Added:** `.print-url`, the post's address; `.article a[href^="http"]::after { content: " <" attr(href) ">" }`; and `.foot::before` "Printed from xingliu.me". The footer's address and "Last modified" still print.
- `pre` wraps (`white-space: pre-wrap`) at 8.5pt. Tables print at 8pt with `overflow: visible`, so they fit the measure instead of making the browser shrink the whole page. Headings avoid breaks after them, and `pre`, `blockquote`, `tr` and `.toc` avoid breaks inside. The contents box prints flat and open (§6.3), and Projects prints as a table.
- **Motion:** none. `style.css` has no `transition`, `animation`, `@keyframes` or `scroll-behavior`, and nothing blinks or scrolls by itself, so `prefers-reduced-motion` needs no rule.

## 6. Behavior

### 6.1 The scripting model

| Script | Where | Size | Job |
|---|---|---|---|
| Head script (`HEAD_SCRIPT` in Base) | Inline, before the stylesheet | 460 bytes (477 with its tags) | Adds `js` to `<html>`, and applies a stored color scheme before first paint (§5.3) |
| `/site.js` | Last in `<head>`, with `defer` | §9.4 | Everything below |
| Contents script (`CLOSE_ON_PHONES` in `markdown.mjs`) | Inline inside `nav.toc`, only on posts with a contents box | 136 bytes (153 with its tags) | Closes the box before first paint at `(max-width: 40em)` |

- The only third-party code is mapmyvisitors.com's `map.js`, which pulls in jQuery 1.12.4 and jvectormap, about 150 KB. It runs on Home only, after `load` (§6.4). There is no framework and no bundling (CLAUDE.md).
- **site.js** is ES5: one IIFE with `"use strict"`, served verbatim from `public/`. It uses `fetch`, `Promise` and `MutationObserver` unguarded, since every target browser has them. It guards the rest: `IntersectionObserver`, `requestIdleCallback` and `requestAnimationFrame` (both fall back to `setTimeout`), `matchMedia`, `String.prototype.normalize`, `navigator.clipboard` (used only when `isSecureContext`, with `document.execCommand("copy")` as the fallback) and `MediaQueryList.addListener`.
- **One `try` per feature**, "so one failure never stops the rest" (its header). Each feature finds its own elements; on a page without them, the first property access throws, and the empty `catch` names the expected case (`/* not a post */`). The `try` covers setup only. A new feature follows the same pattern, and leaves the page as it works without JavaScript if it fails.
- **Shared helpers:** `$`, `each`, `esc`, `el` and `onFrame` (runs a function at most once per frame). There is a polite live region, `#announce`; `announce()` clears it and sets the text 50ms later, so a repeated message is read again. `copyText()` shows "Copied!" on the button for 2,000ms and announces "Copied to the clipboard", or "Could not copy. Select the text and copy it by hand."

| Feature | With JavaScript | Without |
|---|---|---|
| Color scheme | The stored choice applies before paint, and the footer menu changes it (long posts add a second one under the byline) | It follows `prefers-color-scheme`, and the menus are hidden |
| Code wells | A language label and a Copy button | Plain code, still scrollable by keyboard |
| Permalink | Focus selects it; "Copy link" copies it | A read-only field |
| Wide tables | "Scroll sideways for more columns →" while a table overflows | The table still scrolls in its region |
| Search on `/writing/` | Results as you type, `?q=` links, focus on `#search` | The form asks DuckDuckGo |
| Footer and 404 search | Go to `/writing/?q=…#search` | DuckDuckGo |
| Not Found | Names the path, fills in the search, suggests a post | The generic sentence, an empty form and the newest posts |
| NEW! | Removed once the reader's own date reaches `data-until` | Shown as built; the daily rebuild retires it |
| Contents box | Closed before paint on phones; the rail marks the section in view; printed open | Open everywhere; the rail is still placed by CSS |
| Visitor map | Loaded after `load`, at the well's width | A link to the map |

### 6.2 Search

- **The index**, `/search.json`, has one entry per post, newest first, with short keys: `u` URL, `t` display title, `d` date, `l` language, `k` topic name, `s` description and `x` plain text. `x` is `plainText(cleanHtml(…))`, code included, so a function name finds its post. `cleanHtml` strips the contents box, back links and section numbers first, so "Contents" doesn't match every long post. *Today* it is 91,045 bytes raw and 41,940 gzipped.
- **Loading.** `fetch("/search.json", { credentials: "omit" })` runs once per page: on the first focus of `#q` on `/writing/`, on arrival there with a query, and at once on Not Found (for "Did you mean"). The promise is cached, and reset on failure so the next try fetches again. On load each entry gains three folded fields: `ft` (title), `fx` (text), and `fh` (title, description, topic and text).
- **Folding.** `fold()` lowercases each UTF-16 unit and applies NFKC to it on its own, keeping the original unit whenever folding would change the length. Offsets in the folded text are therefore valid in the original, and full-width "ＫＶ" finds "kv".
- **Terms.** `fold(raw).match(TERM)`: each whitespace-separated run, trimmed of the punctuation in `C`, which is straight and curly quotes, `、。《》「」『』`, brackets, and `, . ; : ! ?`. Punctuation inside a term stays, so `block_manager.py` is one term, and a query of punctuation alone matches nothing.
- **Matching.** A post matches when every term is a substring of `fh`. Posts whose title holds every term come first, then the rest, each group newest first. Chinese needs no word segmentation: 侦探 is just a substring.
- **Snippets** are cut around the whole query if the text contains it, otherwise around the first term it contains; failing both, the description is searched the same way, and with no match anywhere (a topic-only match) the snippet is the description. The window runs from 70 units before the match to 90 after, trimmed to a whole word when a space falls within 20 units of either cut, never splits a surrogate pair, and has the match in `<b>`.
- **Rendering.** The status line reads "**{n} posts** match “{q}”:", "**1 post** matches “{q}”:" or "No posts match “{q}”.", where `{q}` is the query as typed, cut into terms, and wrapped in `<span lang="zh-CN">` when it contains a character in U+3400–U+9FFF. Each hit in `ol.hits` has the title link (with `lang="zh-CN"`, a word joiner and the chip for a Chinese post), `p.snip`, and `p.where` "xingliu.me{url} · {Month D, YYYY} · {topic}". Then `p.other` offers "Not what you wanted? Search xingliu.me with DuckDuckGo." or, with no hits, "Try fewer words, or search xingliu.me with DuckDuckGo." If the index can't load, the status reads "The search index could not be loaded. Search xingliu.me with DuckDuckGo instead."
- **Timing.** Results render at once. While typing, the status line, a live region, changes only after 400ms without a keystroke, so screen readers don't announce every keystroke. Other updates are immediate.
- **The address.** After each search `history.replaceState` writes `?q={query}#search`, or `#search` alone when the field is empty, so results can be linked without filling the history.
- **Arrival.** A `?q=` in the address fills the field (`+` reads as a space), and a field that holds a value, from the address or kept by the browser on Back, runs at once. `#q` takes focus only when the address ends in `#search`: when site.js runs, again after `load` (Chrome clears focus when it scrolls to a fragment), on `hashchange`, and on a click on a link to the current address, which fires no `hashchange`, unless Cmd, Ctrl or Shift is held. Only `#search` takes focus, so "by date", "by topic" and the skip link reach their targets.
- **The other forms.** The footer's `#site-search` and Not Found's `#search-404` carry `data-go="/writing/"`; site.js sends them to `/writing/?q={value}#search`. Without JavaScript all three forms GET `https://duckduckgo.com/` with `q` and `sites=xingliu.me`. Every form is labeled "Search all posts", because the index holds nothing but posts, and the footer and 404 forms add `aria-label="Search the posts"`. The main form's hint reads "Searches titles and full text, in English or 中文."

### 6.3 The contents box

- The inline script right after the box (after `</details>`, as the last child of `nav.toc`) closes its previous sibling, the `<details>`, while the page parses if `(max-width: 40em)` matches, so the prose starts on a phone's first screen. Without JavaScript the box stays open.
- A click on any `a[href="#toc"]` opens the box, and the jump still happens.
- **The rail's mark**, only while `(min-width: 77.5em)` matches: on `scroll` (passive), `resize` and changes of the media query, through `onFrame`, the current section is the last one whose heading's top is at or above 35% of the window's height. Its link gets `aria-current="location"`. When the list scrolls, it moves to center the marked link once that link is above the visible part or within 30px of its bottom.
- **Print.** `beforeprint` opens the box and `afterprint` restores it, so a box closed on a phone still prints whole.

### 6.4 The visitor map

- **When.** The loader starts after `load` (at once if the page has already loaded), only over `http:` or `https:`, and only while the well holds `.mm-wait`. It then loads the map at the first of two moments: the well coming within 400px of the viewport (`IntersectionObserver`, `rootMargin: "400px 0px"`), or the page going idle (`requestIdleCallback` with `timeout: 2000`, or `setTimeout` at 1,500ms where that is missing).
- **How.** It runs once. It creates `<script id="mapmyvisitors">` with `data-src`'s `w=` replaced by the well's `clientWidth`, and inserts it before `.mm-wait`. The widget inserts itself right after its script, and CSS then hides the link. If the widget never comes (no JavaScript, a blocker, the service down), nothing lands after the script and the link stays for good. A `MutationObserver` moves focus to `#mapmyvisitors-widget` if it was on the link, so a keyboard reader doesn't drop to the top of the page.
- **Why.** The third-party script never competes with the page, and most visits are still counted: the widget counts a pageview only when its script runs.
- The greeting comment and the colophon's Privacy item describe this behavior: it is the one tracker, on the home page, and it sets a session cookie on mapmyvisitors.com. A change to the loader or the widget changes those sentences in the same commit.

### 6.5 NEW!, the 404 helpers, copy buttons and scroll hints

- **NEW!** At build time each What's New item gets `until`, its date plus 30 days (`NEW_FOR_DAYS`), and shows NEW! while the build day in Los Angeles is earlier. site.js removes every `.new[data-until]` whose `data-until` is on or before the reader's local date (a string comparison). Without JavaScript, only a rebuild retires it. So with scripts the tag stays true even while the schedule is paused; without them it stays true only while the daily rebuild (or a push) redeploys the site.
- **The 404 helpers.** site.js names the missing path: `#missing` becomes "The requested URL `{path}` was not found on this server.", with the path decoded and set as text, never as markup. It fills `#q404`, unless the field already holds something, with the words of the last segment that isn't `index.html` (with `.html` or `.htm` removed, split on `-`, `_`, `.` and spaces, and all-digit words dropped), so `/2026/01/23/nano-vllm/` gives "nano vllm". After loading the index it suggests one post: first a post whose URL matches ignoring case ("Addresses on this server are case-sensitive."), then one whose slug matches the last segment ("It lives at a dated address."). Those are the two likely failures: slugs keep their case (`/2021/12/06/Designing-Data-Intensive-Application-Chapter1/`), and links drop the date.
- **Code bars.** For each `div.highlighter-rouge`, site.js reads the `language-X` class and maps it through `NAMES` (`bash`, `sh`, `shell` and `zsh` all give "Shell"; `plaintext` and `text` give no label; an unknown name shows as written). It inserts the bar with a Copy button (`aria-label` "Copy {Name} code", or "Copy code"), which copies `pre.innerText` without its final newline.
- **Permalink.** Focusing `#permalink` selects it, and "Copy link" copies it.
- **Scroll hints.** After each `.article .table-scroll` whose `scrollWidth` exceeds its `clientWidth` by more than 1px, site.js inserts `p.scroll-hint` "Scroll sideways for more columns →", and removes it when the table fits. It checks at start and on `resize`, once per frame. Projects is outside `.article`, so it never gets a hint.

## 7. Content model

### 7.1 Posts

Writing a post is in README; the Markdown habits are in CLAUDE.md.

- **The collection** (`src/content.config.ts`): a `glob` of `*.md` in `src/content/posts`, with the file name minus `.md` as the id, case kept. The schema is `title` (required), `published` (default `true`), `lang` (`en` or `zh-CN`), `topic` (one of the ids in `posts.json`) and `description`.
- **`getPosts()`** (`src/lib/posts.ts`) returns the published posts sorted by id, newest first. It takes the URL and date from the file name. It makes the **display title** (`displayTitle()` turns a `'` between a letter or digit and a letter into `’`, and paired `"…"` into `“…”`), which every page, card, feed entry and index uses. It takes the **description** from the frontmatter, else `posts.json`, else the opening paragraph cut at a word boundary (`Intl.Segmenter`, so Chinese cuts correctly) to at most 160 characters plus "…". It reads the reading time the pipeline left (§7.3) and the dates from git (§7.4).
- **The build fails** on a file name that isn't `YYYY-MM-DD-slug.md`, on a date that doesn't exist, on a post without a topic, on a missing reading time (its HTML predates the pipeline: build with `--force`), and on a post left with no description.
- **A second reader.** `publishedPostFiles()` in `src/lib/post-files.mjs` reads the same files outside Astro, for the sitemap, `/writing/`'s date and check-dist, and skips `published: false`. The two readers must agree on what is published.
- **`lang: zh-CN`** sets `<html lang>`, the Chinese byline, 目录 and character counts, `og:locale` `zh_CN` and JSON-LD `inLanguage` (with no `wordCount`), and `lang="zh-CN"` on every link to the post, plus the 中文 chip on its By date row and in search results.

### 7.2 Topics and descriptions (`src/data/posts.json`)

| id | name | zh | Posts *today* |
|---|---|---|---|
| `llm-inference` | LLM inference | LLM 推理 | 3 |
| `books` | Books & reading notes | 读书 | 3 |
| `crypto` | Crypto | 加密货币 | 2 |
| `life` | Life & work | 生活与工作 | 3 |
| `fiction` | Fiction | 小说 | 1, set as a book |
| `meta` | Meta | 本站 | 0, so it isn't shown |

- `topics` is in the order of the By topic index. The `id` makes the anchor `#topic-{id}` and is validated by the schema. The `name` appears wherever English names a topic: Writing's headings, the English byline, the search index's `k` (and so every result's address line), the RSS category, and JSON-LD `articleSection`. `zh` appears only in the Chinese byline.
- `posts` holds a topic and a one-sentence description for each post that predates frontmatter topics (all 12 *today*). A post's own frontmatter wins over it, and an unknown topic fails when the file is imported.
- `WRITING_SUMMARY` in `src/lib/descriptions.ts`, "notes on LLM inference, books, crypto and life, and a detective story in Chinese", is written by hand and feeds Writing's lede, description and card. The count and month around it are computed; the summary must change when the archive's make-up does.

### 7.3 The Markdown pipeline (`src/lib/markdown.mjs`)

CLAUDE.md lists the three plugins. These are their rules. Its header states the principle: "Templates never split post HTML: everything a post needs inside its body is written here." All furniture therefore exists in one known form, which `cleanHtml` can strip.

- **Heading levels.** `normalizeHeadings` shifts every heading by `2 - (the smallest depth in the post)`, clamped to 2–6. A post that starts at `#` moves down one level, one at `##` stays, and one at `####` moves up two, because demoting everything by one would skip levels.
- **Size classes.** A heading whose original Markdown level implies another size gets `class="s{N}"` (a post written with `##` and `###` gets `<h2 class="s3">` and `<h3 class="s4">`), so the author's sizes survive a correct outline. A heading that arrives with a class (GFM's visually hidden footnotes label) keeps it.
- **Ids.** `github-slugger` sets each heading's id before Astro's own pass, which uses the same algorithm and keeps an id it finds. Ids are set before section numbers are added, so a number never enters an id, and adding or removing a contents box never changes one. A heading that already has an id still advances the slugger, so duplicate counters stay in step.
- **The contents box** appears when a post has at least 6 root-level headings, takes 3 minutes or more to read, and repeats some heading level; the shallowest repeated level is the one listed. So a short post never gets a box bigger than itself. The box goes after the first block if that is a `<p>`, otherwise before it. Its label is "Contents" with " (N sections)", or 目录 with （N 章） for fiction and （N 节） for other Chinese posts. Non-fiction headings at the listed level get section numbers, and a back link goes before each listed section but the first, and once more at the end. *Today* three posts have one: Nano-vLLM Study Notes (12 sections), Neuron Learning Notes (6) and the detective story (6 章).
- **Tables.** Each root-level `<table>` is wrapped in the labeled scroll region, numbered within the post. A table inside a list or a quote isn't wrapped.
- **Reading time.** English counts whitespace-separated words at 220 a minute. Chinese counts Han characters plus Latin words at 400 a minute. The minimum is 1, and code blocks don't count. The result goes into the rendered frontmatter as `readingTime`.
- **Links and long names.** Referral links in `SPONSORED` get `rel="sponsored"`, and dotted names longer than 24 characters in prose get `<wbr>` before their dots.
- **Code.** Every Prism grammar loads up front, because loading on demand makes a grammar's optional parts depend on which post renders first. `PRISM_ALIASES` maps Rouge's fence names to Prism's. A fence without a language becomes `plaintext`, and a language with no grammar renders as escaped text, with one warning.
- **Cleaning** (`cleanHtml()` in `src/lib/feed-html.mjs`, for the feed and the search index) removes the contents box (with its script), the back links and the section numbers, unwraps the table regions, drops `tabindex="0"`, turns `al-*` classes back into inline `text-align`, drops the `s2`–`s6` classes, and makes links absolute. Anything new the pipeline adds inside a post body needs a removal here, and a pattern in check-dist's `FEED_CHROME` if it is page chrome.

### 7.4 Dates from git (`src/lib/last-modified.mjs`)

CLAUDE.md states the rule. The mechanism:

- **Posts.** `postLastModified()` walks `git log --follow -M` for the file, newest first, across renames. It skips merges and every commit in `.git-blame-ignore-revs`. The commit that added the file is the publication, so the walk stops there, and the 2021 import of the repository can't pass as an edit. For any other commit it compares the post's text after and before it (`postText()`: the title without surrounding quotes, plus the body, with line endings unified, every heading marker outside fenced code made one `#`, and trailing spaces stripped). The first commit that changed the text gives the author's own date and ISO time. That counts only if it is later than publication, so a same-day fix is no "Updated".
- **Other pages** use `pathsLastModified()`, the newest commit touching their files, combined with their content dates (§4.1).
- **Once per build.** The module keeps its state on `globalThis[Symbol.for("xingliu.me/last-modified")]`, because Astro loads it twice, for the pages and for `astro.config.mjs`.
- **Without full history** (no repository, or a shallow clone) it warns once and the fallbacks apply, which is why CI checks out with `fetch-depth: 0`.
- **Where the dates show:** the footer's "Last modified", the byline's "Updated" or 更新于, the sitemap's `lastmod` for posts and `/writing/`, JSON-LD `dateModified`, and `article:modified_time` (only on edited posts). *Today* three posts show an edit: Nano-vLLM Study Notes (March 4, 2026), the detective story (2026年2月20日) and DDIA Chapter 1 (February 13, 2022).
- **Time zones** (`src/lib/dates.ts`). A date is a calendar day, read at noon UTC so the weekday never shifts. Instants (GitHub's timestamps, the build itself) become days in America/Los_Angeles, the site's time zone. Commit dates are the author's own day.

### 7.5 GitHub data (`src/lib/github.ts`)

- **The request:** `GET https://api.github.com/users/xingliu14/repos?per_page=100&type=owner`, with `Accept: application/vnd.github+json`, `User-Agent: xingliu14.github.io`, `Authorization: Bearer $GITHUB_TOKEN` when the variable is set, and a 15-second timeout. It runs once per build. There is no pagination: 100 repositories is the ceiling.
- **The fallback.** A network error, a non-OK status, a timeout or a response that isn't an array prints `[github] …; using src/data/github-snapshot.json`, and the build uses the snapshot with `live: false`. The snapshot is a saved API response plus `fetched_at` (*today* 2026-10-25T17:57:32Z, 10 repositories). No script writes it.
- **`getProjects()`** drops forks and the names in `PROJECTS.exclude.projects` (`src/config.ts`), and sorts by `pushed_at`, newest first. `about` is GitHub's description, else the `note` in `src/data/repo-notes.json`. `homepage` is the note's `homepage` if the key exists (`null` shows none), else GitHub's homepage, else `https://xingliu.me/<name>/` when the repository has a Pages site. Descriptions are shown exactly as GitHub has them, typos included: they are the owner's words.
- **The data drives** the Projects table, its description, card and JSON-LD, Home's counts, What's New's repository lines, and Projects' "Last modified".

### 7.6 What's New (`src/lib/whats-new.ts`)

- It takes the 3 newest posts and the 2 newest other events (entries in `src/data/site-log.json`, and repositories dated by `created_at` as a Los Angeles day), and merges them newest first. On the same day, site news comes first, then a repository, then a post (`RANK`).
- The lines read "New post: {title}. {description}", "New project: {name}. {about}" (the text after the name only when there is one), or a site-log entry's own `html`, which is trusted and printed with `set:html`. README covers adding an entry.
- Why three plus two: it is a digest of recent activity, never the archive. Posts can't crowd out project and site news, nor the reverse, and five lines keep the front door short.

### 7.7 Copy that must stay true

- **Computed, never typed:** post and project counts, "since {Month YYYY}", section counts, reading times, the résumé's size and month (`src/lib/resume.ts` reads the size from the file and the month from its name, which must end in `YYYYMMDD.pdf` or the build fails), and the page weight.
- **Self-dating:** "Last modified", "Fetched from api.github.com on {date}." (or the snapshot sentence), and NEW!, which expires by itself.
- **No date the site states depends on the daily schedule.** The feed has no `lastBuildDate`, "Last modified" is never the build time, and the schedule is never announced, because GitHub can pause it. NEW! depends on it only for readers without JavaScript (§6.5).
- **Change these in the same commit as the behavior they describe:** the greeting comment; the four buttons' alt texts and `ul.claims`; and the colophon's items. Those are Color (the menu and `localStorage`), Type, Machinery ("Scripts only add extras, such as …"), Search, Dates ("moving files or changing the design does not count"; NEW! "disappears on its own"), Feeds and addresses ("Neither the move to Astro nor this redesign changed any post’s address"), Privacy (no cookies or analytics of the site's own; the map's session cookie), Previews and The star.
- **The owner's voice.** Copy is written only from facts about the owner. His own words appear verbatim: the About text, repository descriptions, and post titles, of which only the quotes are curled. The site speaks in the first person (the Projects intro, the colophon), except the short bio, which is third person and pronoun-free, for introductions. The site's own copy (pages, feed, search index, cards and metadata) never shows his phone number or Gmail address. The résumé PDF is his own document, served as it is, and it does show both.
- **Spelling** follows CLAUDE.md. **Typography:** curly quotes and apostrophes, `›` in crumbs, a spaced em dash in titles ("Writing — Xing Liu"), `…` for ellipses, "Résumé" with its accents, and middle dots written `&nbsp;· `.

## 8. Search engines, feeds and cards

### 8.1 Head metadata

- **Titles** follow `{Name} — Xing Liu`; Home is "Xing Liu (刘星) — Home Page", and About "About Xing Liu (刘星)". `og:title` drops the suffix, since `og:site_name` ("Xing Liu’s Home Page") names the site. Every title is the display title (§7.1).
- **Descriptions.** The meta description and `og:description` are the same string. Home, About and Not Found write theirs in the page. Home's and About's carry the job, so README's job-change list includes them. Writing and Projects compute theirs in `src/lib/descriptions.ts`, which their cards share: "All {n} posts by Xing Liu since {Month YYYY}, by date and by topic: {WRITING_SUMMARY}." and "The {n} public repositories Xing Liu has started on GitHub (forks left out): what each one is, its language, its stars and when it last changed." Posts use their description (§7.1).
- **Canonical** is `new URL(path, "https://xingliu.me").href`: trailing slash, case kept. Every indexed page also has `<meta name="robots" content="max-image-preview:large">`, so search engines may show the 1200px card as a large preview rather than a thumbnail.
- **Open Graph and Twitter** (`src/components/SocialMeta.astro`): `og:site_name`, `og:type` (`website` for Home, Writing and Projects, `profile` for About, `article` for posts), `og:url`, `og:title`, `og:description`, `og:locale` (`en_US` or `zh_CN`), `og:image` `https://xingliu.me/og/{ogKey}.png` (`image/png`, 1200×630), `og:image:alt` "{og:title}, drawn as a 1997 web page: gray background, Times headline, blue underlined links", and `twitter:card` `summary_large_image`. There is no `twitter:site`, because `ELSEWHERE` lists no X account. Posts add `article:published_time` (the date), `article:modified_time` (the ISO time, only when edited), `article:author` (`/about/`) and `article:section` (the topic name). About adds `profile:first_name`, `profile:last_name` and `profile:username`.
- **The feed's autodiscovery link** is on every page, Not Found included.

### 8.2 JSON-LD (`src/lib/seo.ts`)

Base prints one `<script type="application/ld+json">` per page through `jsonLdText()`, which escapes every `<`, so no text can close the element. Not Found has none.

| Page | `@graph` |
|---|---|
| Home | `WebSite` (`/#website`), `Person` (`/#person`), and `WebPage` (`/#webpage`) with `dateModified` |
| About | `WebSite`, `Person`, and `ProfilePage` (`/about/#page`) with `mainEntity` the person and `dateModified` |
| Writing | `CollectionPage` (`/writing/#page`) holding an `ItemList` of every post: position, URL, display title |
| Projects | `CollectionPage` (`/projects/#page`) holding an `ItemList` of `SoftwareSourceCode`: name, repository, and language, description and URL when the row has them |
| Post | `BlogPosting` (`{canonical}#post`) and a `BreadcrumbList` (Home › Writing › the post) |

- `Person` has `worksFor` Annapurna Labs, Amazon Web Services, and its `sameAs` is `ELSEWHERE` plus `https://github.com/xingliu-aws`, because his GitHub bio says that account is also him.
- `BlogPosting` carries the display title as `headline`, `datePublished` (the date), `dateModified` (the ISO time of the last edit, else the date), `inLanguage`, the card as `image` (1200×630), `wordCount` on English posts only ("A word count means nothing for Chinese"), `timeRequired` `PT{n}M`, `articleSection`, and `author` and `publisher` as a reference to the person that carries its own name.
- The JSON-LD breadcrumb ends at the post, while the visible crumbs end at the year.

### 8.3 Feed, sitemap and robots

- **`/feed.xml`** (`@astrojs/rss`): channel "Xing Liu’s Home Page", described as "Everything Xing Liu (刘星) writes on xingliu.me, in full: notes on LLM inference, books, crypto and life.", with `<language>en</language>` and an `atom:link rel="self"`. It has one item per post, newest first: the display title, the absolute URL as `link` and `guid`, `pubDate` at `T12:00:00Z` (the same calendar day in every reader's time zone), the description, the topic name as its category, and `content:encoded` = `cleanHtml(post.html.trim(), link)`, because feed readers have neither this site's stylesheet nor its address. It has no `lastBuildDate`, which the daily build would change with no new content.
- **The sitemap.** `@astrojs/sitemap` lists every page except Not Found. Its `serialize` sets `lastmod` from `sitemapLastmod()` on posts and on `/writing/` only; other pages give "no date rather than an imprecise one". `sitemap-dates` then rewrites `@astrojs/sitemap`'s midnight-UTC `lastmod`, "a precise instant that comes before the edit it dates", as a plain `YYYY-MM-DD` in every `sitemap-*.xml`.
- **`robots.txt`**: `User-agent: *`, `Allow: /`, `Sitemap: https://xingliu.me/sitemap-index.xml`. It sits at the domain's root, so it governs the other repositories' Pages sites too: a `Disallow` here would hide them.

### 8.4 Social cards

- **The pipeline.** `src/pages/og/cards.json.ts` lists the cards: `home` cards for Home and About, `section` cards for Writing and Projects, and a `post` card per post, keyed by the post's id. After the build, `src/integrations/og-cards.mjs` draws each one with `renderCard()` from `src/integrations/og-render.mjs` (Satori lays out an SVG at 1200×630, and resvg renders the PNG), then deletes the list. Nothing needs a browser, so it runs on a bare Ubuntu runner. It is an integration because Vite never bundles resvg's native binding, and it imports the renderer inside its hook so `dev` and `check` skip it. `astro dev` draws no cards. Drawing them is most of the build's time.
- **The look.** Each card is its page as Navigator drew it: a `#c0c0c0` ground; the bracketed nav, with the current item in bold black and the rest blue and underlined; a shaded rule; crumbs; the title in Tinos Bold, sized by `titleSize()` (88, 74 or 62px, with Han characters counting double); a byline on posts; and the description in Gelasio 33px with `textWrap: "balance"`, so its last line never holds a lone word or a lone Chinese character. At the foot, "xingliu.me" and the RSS, Astro and Any Browser buttons at 2×, drawn by `badgeSvg()` from `scripts/pixels.mjs`. Home and About cards show the avatar in a sunken frame beside a title, a Chinese subtitle and one sentence, written in `cards.json.ts`.
- **Fonts** are npm devDependencies, read at build time and never served: Tinos (for Times New Roman), Gelasio (Georgia) and DejaVu Sans (Verdana), the same stand-ins `style.css` lists, because the runner has no Core Fonts. Chinese comes from Noto Serif SC, cut for each card to the CJK characters it uses (`subset-font`). Chinese therefore renders, not as tofu, at a cost of kilobytes instead of the full file of about 15 MB.
- **Agreement.** Each page's `ogKey` must match a `file` in `cards.json.ts`, and check-dist fails a page whose `og:image` isn't a card in `dist/og/`. The card's nav repeats `ITEMS` from `src/lib/nav.ts` in `navRow()`.

## 9. Build gates, budgets and CI

### 9.1 Integration order

CLAUDE.md lists the integrations. The order matters: `sitemap-dates` rewrites what the sitemap wrote, `page-weight` must measure the final `about/index.html`, `style.css` and `site.js`, and `check-dist` comes last, so it sees plain dates, drawn cards and the filled-in sentence.

### 9.2 check-dist (`src/integrations/check-dist.mjs`)

It runs all nine checks, collects every problem, and throws once: "check-dist found N problems in dist/: …". That fails `npm run build` locally and the CI build job, so the deploy job never runs.

CLAUDE.md lists what the checks cover. The table names each check, adds only the details that list leaves out, and gives the reason for it.

| # | Check | Details CLAUDE.md leaves out | Why |
|---|---|---|---|
| 1 | `topLevel` | Year folders pass if they match `YEAR` | A new name could take a path a repository's Pages site needs |
| 2 | `required` | The required paths are `REQUIRED`, the résumé's among them | Old URLs never change, and the résumé's address is permanent |
| 3 | `anchors` | — | Other sites link to sections |
| 4 | `eachPage` | Exactly one `<h1>`. Not Found must be `noindex`, with no canonical link. `UNFINISHED` (`__PLACEHOLDERS__` or lorem ipsum) is looked for outside `pre` and `code` | Every page carries its metadata and card, and nothing unfinished ships. Code is exempt because it "may print anything, `__init__` or `__FILE__` included" |
| 5 | `feed` | One item and one `<link>` per post, and no `FEED_CHROME` (`<script`, `tabindex`, or the classes `toc`, `top`, `secno`, `table-scroll`) in any `content:encoded` | Every post once, in full, without the page's chrome |
| 6 | `search` | One entry per post, each with `u t d l k s x` as strings | site.js reads those fields |
| 7 | `sitemap` | — | Not Found is `noindex`, and a time would be made up |
| 8 | `validHtml` | Errors only | Valid HTML. CI has no browser, so axe runs locally |
| 9 | `budgets` | The limits live in `BUDGETS` | They are served as written, so no minifier keeps them small |

It does not check that titles or descriptions are unique or short, page weights, `search.json`'s size, images, accessibility, or JSON-LD.

### 9.3 html-validate

`.htmlvalidate.json` extends `html-validate:recommended`. JSON has no comments, so the reasons for its five relaxations live in check-dist's comment above `validHtml`: `valid-id` is relaxed because numeric heading ids such as `id="1"` (the detective story's scenes) are valid HTML5 and linked from elsewhere; `prefer-native-element` excludes `region`, because `<div role="region" tabindex="0">` is the standard scrollable table; `no-redundant-role` is off for the Projects table's explicit roles; `long-title` is off for long post titles; and `void-style` is off because posts keep the `<br/>` their author typed.

### 9.4 Budgets and the weight sentence

| File | Raw *today* | Gzipped at level 9 *today* |
|---|---|---|
| `public/style.css` | 35,955 bytes (45 under) | 10,932 bytes (68 under) |
| `public/site.js` | 17,963 bytes (37 under) | 6,729 bytes (271 under) |

- The limits are in CLAUDE.md and `BUDGETS`. Comments count, because the files are served verbatim for View Source, so a change pays for its bytes in the same file: shorter comments, dead rules, prefixes no supported browser needs. Raising a budget is a design decision that changes CLAUDE.md and `BUDGETS` together, along with the margins in the table above (and, for `site.js`, the 18,000 in §10.1).
- Gzipped sizes are Node's `zlib` at level 9, as check-dist and page-weight measure them; the `gzip` command gives slightly different numbers.
- Measured but not enforced, *today* and gzipped: Home's HTML 3,642 bytes, Nano-vLLM Study Notes 10,878, About 4,851, `search.json` 41,940. The avatars are 12,675 and 17,921 bytes.
- **The weight sentence.** `src/integrations/page-weight.mjs` fills About's "This page, its stylesheet and its script weigh {N} KB as sent, plus {M} KB for the picture: about {S} seconds on a 28.8k modem." The text is the gzipped size of `about/index.html`, `style.css` and `site.js` ("GitHub Pages compresses text with gzip"), and the picture is `avatar-320.jpg` on disk. KB is `Math.round(bytes / 1024)`, and seconds are `Math.round((text + picture) × 8 / 28,800)`. Because the sentence is part of the page it weighs, it is refilled until the numbers it states are the numbers it measures, failing after 5 passes. It warns if the sentence is missing and fails if About stops showing `/avatar-320.jpg`. *Today* it reads 22 KB, 18 KB and 11 seconds. The sentence names only the page, its stylesheet, its script and the picture, so the buttons and the favicon are not counted.

### 9.5 CI

- **`.github/workflows/deploy.yml`** ("Deploy to GitHub Pages") runs on pushes to `master` and on `workflow_dispatch`, with `permissions` `contents: read`, `pages: write` and `id-token: write`, and `concurrency` group `pages` with `cancel-in-progress: false`, so a push and the daily run never deploy at once and a running deployment is never cancelled. The build job sets `GITHUB_TOKEN` (no anonymous rate limit), checks out with `actions/checkout@v7` and `fetch-depth: 0` (dates come from git), and runs `withastro/action@v6` with `cache: false` (a restored content cache can carry stale post HTML), which installs, runs `npm run build` (so check-dist gates every deploy) and uploads `dist/`. The deploy job (`needs: build`) runs `actions/deploy-pages@v5`.
- **`.github/workflows/daily.yml`** ("Daily rebuild") runs at `cron: "17 9 * * *"` and on `workflow_dispatch`, with `permissions: actions: write`. It runs `gh workflow run deploy.yml --ref master` with `GH_TOKEN` and `GH_REPO`: a `GITHUB_TOKEN` can start a `workflow_dispatch` run, though most events it causes start nothing. Why it is a separate workflow is in CLAUDE.md and README. A daily build picks up new repositories, pushes and stars, advances the "Fetched" date, and retires NEW! tags.

## 10. Decisions

### 10.1 Keep these

| Decision | Why |
|---|---|
| Navigator 3, 1997, as the only design language | Of five directions drawn up (Navigator, academic, text mode, Windows 98 desktop, weblog), it alone was strong on period recognition, plainness, system and memorability at once, and three judges chose it. Mechanisms from the others were kept only when redrawn in Navigator's vocabulary |
| Breakpoints and script queries in em | With px breakpoints, a 20px default font put the rail 107px off screen at 1280px, and 200% text overflowed the breakouts |
| Headings renumbered to start at `h2`, with size classes | Demoting by one left skipped levels (`h1` to `h3` in Neuron, `h1` to `h5` in the book lists), breaking the outline and axe's heading-order rule |
| The contents mark as a selection, not bold | Bold reflowed the rail and pushed `engine/model_runner.py` onto two lines |
| Figures after the rail extend right only | The 3em breakout ran 13px under the rail and hid the first letter of `MergedColumnParallelLinear` |
| A closed contents box folds to a shaded rule; it prints open | A closed full frame looked like an empty text field; a post printed from a phone printed a lone "Contents" label |
| "ZH Punct", from local faces | Narrow Latin quotes in Chinese prose were the art review's first must-fix |
| Topics and descriptions in `posts.json` | The redesign promised to edit no post file |
| "Last modified" from git: the adding commit is publication, and heading-level moves don't count | A 2021 import of the repository must not pass as an edit, and moving a heading from `###` to `####` changes no words (so "My book list of 2021" shows no "Updated") |
| NEW! expires in the browser as well as at the rebuild | It stays true with JavaScript even when GitHub pauses the schedule, and without JavaScript as long as the daily rebuild runs |
| A hand-made JSON index matched by substring | Pagefind builds one index per `<html lang>`, which would split the Chinese post off, and forcing one language turns off Chinese segmentation |
| The search status waits 400ms | Screen readers were announcing "13 posts match 'n'", "4 posts match 'na'"… on every keystroke |
| Only `#search` takes focus | So "by date", "by topic" and the skip link on `/writing/` reach their targets |
| Every search form says "Search all posts" | The index holds nothing but posts |
| The map loads late, into a reserved well, with a link that stays if it fails | It counts nearly every visit without delaying paint or moving the page, and it degrades to a link |
| The greeting names the map | The shorter "no tracking" was false on the home page |
| Projects become wells below 900px, with explicit ARIA roles | That is where the table stops fitting, and Safari drops table semantics when `display` changes |
| 16px fields on phones, and no `maximum-scale` | iOS zooms into smaller fields on focus, and readers must keep the ability to zoom |
| 24px contents rows on phones; share links that don't wrap | At 320–360px the rows were under 24px, and "E-mail" split into two overlapping targets |
| American spelling in visible copy; identifiers unchanged | The owner writes American English, `og:locale` is `en_US`, and Navigator's own panel said "Colors"; renaming identifiers would only churn the code |
| A flat 中文 chip, and plain code-bar labels | Raised means "press me", and uppercase tracked labels are a modern tell |
| `site.js` budgeted at 18,000 bytes, unminified | A 12 KB budget would have forced minifying, which breaks the View Source promise |
| Plain-date `lastmod`, none on undated pages, no `lastBuildDate`, `pubDate` at noon UTC | No invented precision, and the same calendar day everywhere |
| Cards drawn as Navigator pages, with a subset CJK font and balanced descriptions | They carry the look into social feeds; without a CJK font Satori draws tofu; the detective card ended on an orphaned "司。" |
| `splitSubtitle()` shared by the `<h1>` and the card | WebKit broke "He’s / Replaceable", and the page and the card must agree |
| Pixel-art 88×31 buttons, each claim true, alt text starting with the drawn words | Real buttons were pixel art, SVG stays crisp, and voice control finds a button by what it shows |
| The file size on the résumé link | What 1990s pages did for modem users |
| html-validate in the build, axe run locally | CI has no browser |

### 10.2 Rejected

- **A "random post" link:** JavaScript-only on a static host, so a dead link without it.
- **`accesskey` on the nav:** it collides with screen-reader and browser shortcuts.
- **"#" anchors on headings:** the contents box already links every section, and the marks are a 2010s idiom.
- **The Yellow Fade on `:target`:** animation, and 2004.
- **Other eras' chrome** (LaTeX2HTML buttons, Wikipedia boxes, RFC headers, an amber terminal, a Windows window, status bars, progress meters, tabs, a sidebar): mixing eras makes a collage.
- **A separate `/search/` page:** search belongs with the archive, and `?q=` links and the footer form give the same reach.
- **`lastBuildDate`, a build-time "Last modified", announcing the daily schedule:** each would become false.
- **Times or Noto Sans CJK for Chinese quotes, and a full-width interpunct:** narrow or proportional quotes, and names like 约翰 · 勒卡雷 looked loose.
- **An XSLT stylesheet for the feed:** browsers are removing XSLT; About's RSS entry tells readers to paste the address into any feed reader instead.
- **Stacked dates in the phone archive:** they double the list's height, while the 3.6em gutter keeps the dates scannable.
- **Review suggestions not adopted, so the current form stands:** a text button instead of the color menu under the byline, and a raised arrow on the select; one terse line per What's New entry (entries carry the full description); ISO dates in the 404 list (it shows `YYYY-MM`); a shorter Writing lede; rewording the site-log entry; glyphs on the buttons; and the Paper white menu on every post of 3 minutes or more (it stays at 10).

### 10.3 Known gaps

Facts about the code as it stands, recorded so nobody rediscovers them:

- `main:focus { outline: none }` never applies: `[tabindex]:focus` (specificity 0,2,0) outranks it (0,1,1), so `main#main` can show the dotted ring after the skip link.
- `table code` is 11.96px on phones, under the 12px floor.
- Dead rules: `.views .sep` (the Index line uses `&nbsp;·` text) and `.article .s2` (a post's top level always lands on `h2`). Dormant until content needs them: `.s6`, `al-right`, the footnote rules, `kbd`, every `.token.*` color and most of `NAMES`, since all four code wells *today* are `language-plaintext`.
- Stale comments: `src/lib/descriptions.ts` gives "All 13 posts" as its example; check-dist's `validHtml` comment says "several" titles pass 70 characters (one does); and `src/lib/post-meta.mjs` says frontmatter edits would move "Last modified", which `postText()` no longer allows. The reason that holds is principle 8.
- `scripts/heading-ids.json` holds the ids of the posts at the redesign: 136 ids across 12 posts *today*, with empty lists for posts that have none. Nothing adds a new post's ids to it.
- The snapshot in `src/data/github-snapshot.json` is refreshed by hand.
- A 404 path ending in `.html` (`/…/nano-vllm-notes.html`) fills in the search but gets no "Did you mean", because the slug match compares the last segment with its extension.

## 11. Checklists

**A visual change**

1. Could Navigator have drawn it? If not, it doesn't go in (§1, §10.2).
2. Colors are tokens, defined for gray, Paper white where needed, both Night blocks and print, at 4.5:1 for text and 3:1 for UI in all three schemes (§5.2).
3. Frames come from the bevel vocabulary, and only things you press are raised, besides a bevelled table's outer frame (§5.8).
4. Fonts come from the five stacks, at 12px or more (§5.4).
5. Type sizes are in rem or em. Breakpoints reuse the existing em values, and a script's `matchMedia` uses the same value (§5.7).
6. Nothing from principle 6's never-list; newer CSS only enhances, with its fallback first.
7. Controls that need scripts are hidden until `.js`, with their room reserved (§5.9).
8. No horizontal page scroll at 320px, at a larger default font size or at 200% text. Wide content scrolls in its own `tabindex="0"` box.
9. Tap targets of 24px, or spacing that WCAG 2.5.8 accepts.
10. Chinese: `--cjk`, never italic, names and short subtitles kept whole (§5.5).
11. Print hides new chrome and stays black on white (§5.10). No motion.
12. The change pays for its bytes in the same file (§9.4).

**A new page:** pass Base a `title` in the `{Name} — Xing Liu` pattern, a description, a `path` with its trailing slash, `current`, `ogKey`, `lastModified` from its own files, and `ogTitle` and `jsonld` where they fit; add a card with the same key to `src/pages/og/cards.json.ts`; give it exactly one `<h1>`. A new top-level name goes on `TOP_LEVEL` and on README's reserved names. The sitemap lists the page with no `lastmod` unless `sitemapLastmod()` learns a true date for it.

**A new nav item:** change `ITEMS` and the `Current` type in `src/lib/nav.ts` together with `navRow()` in `src/integrations/og-render.mjs`, and check the nav at 320px.

**A new or removed post:** README covers writing one. It then reaches the feed, the search index, the sitemap, Writing's JSON-LD and its own card. check-dist confirms its page, card, feed item and search entry; nothing checks its sitemap entry or the JSON-LD. check-dist protects only the heading ids listed in `scripts/heading-ids.json`. A post that is removed takes its `posts.json` entry and its `heading-ids.json` entry with it, and its address then serves Not Found.

**A change to the Markdown pipeline:** keep heading ids stable, strip any new furniture in `cleanHtml()` and `FEED_CHROME`, and build with `--force` (CLAUDE.md).

**A change to what the site does:** update the sentence that describes it in the same commit (§7.7), and this file.

**A job change:** follow README's list.

**Never:** a `lastBuildDate` in the feed, a build-time "Last modified", a time in sitemap dates, a `schedule:` trigger in `deploy.yml`, a web font, an animation, or a top-level name that isn't on `TOP_LEVEL`.
