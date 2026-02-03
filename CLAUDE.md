# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jekyll-based personal portfolio and blog website deployed to GitHub Pages. The site is forked from GitHub's personal-website template and customized.

## Build Commands

```bash
# Install dependencies
bundle install

# Build the site (generates _site/)
bundle exec jekyll build

# Serve locally with live reload (localhost:4000)
bundle exec jekyll serve
```

## Architecture

### Layout System
- **Sidebar layout** (default): Split view with profile sidebar + main content area
- **Stacked layout**: Full-width single column
- Theme support: Light (default) and dark mode via `site.style == 'dark'` conditionals

### Key Directories
- `_layouts/` - Page templates (default.html, home.html, post.html)
- `_includes/` - Reusable components (masthead, projects, thoughts, post-card, etc.)
- `_posts/` - Blog posts in markdown (YYYY-MM-DD-title.md format)
- `_sass/` - SCSS partials
- `_data/` - Data files (social_media.yml, colors.json)
- `resume/` - Resume files (LaTeX source and PDF)

### Data Flow
- GitHub metadata: `site.github` object auto-fetches user data and repositories via `jekyll-github-metadata` plugin
- Configuration: `_config.yml` controls layout, project display settings, social media links
- Posts: Markdown files in `_posts/` automatically become blog pages

### Styling
- Base CSS: GitHub Primer framework (CDN import)
- Custom styles: `assets/styles.scss` with SCSS partials in `_sass/`
- Responsive breakpoints at 768px

### Key Plugins
- `jekyll-github-metadata` - GitHub API integration for profile/repo data
- `jekyll-octicons` - GitHub icon support
- `jemoji` - Emoji rendering

## Configuration Notes

- Repository: `XingLiu14/XingLiu14.github.io`
- Projects display: Top 3 repos sorted by push date, forks excluded
- Permalink pattern: `/:year/:month/:day/:title/`
