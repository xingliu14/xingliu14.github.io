// Where a post title may break. A short Latin subtitle after a full-width colon stays on one line
// ("侦探小说：He’s Replaceable"), and the title breaks after the colon instead: WebKit would
// otherwise break inside the English words. The post's <h1> ([slug].astro) and its social card
// (src/integrations/og-render.mjs) both split titles here, so the two always agree. Plain .mjs, so
// the card renderer can load it outside Vite.

/**
 * The title and its subtitle, or null when the title has no short Latin subtitle.
 * @param {string} title
 * @returns {[string, string] | null}  e.g. ["侦探小说：", "He’s Replaceable"]
 */
export function splitSubtitle(title) {
  const match = /^(.+：)([^\p{Script=Han}：]{1,24})$/u.exec(title);
  return match ? [match[1], match[2]] : null;
}
