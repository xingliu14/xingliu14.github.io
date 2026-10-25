// Turns a post's article HTML (src/lib/markdown.mjs) into HTML that stands on its own, for the RSS
// feed and the search index: feed readers have neither this site's stylesheet nor its address.

/**
 * Removes the page furniture the pipeline adds inside the body (contents box, back links, section
 * numbers, scroll regions, tabindex), turns alignment classes back into inline styles, and makes
 * links absolute.
 * @param {string} html  the article HTML
 * @param {string} pageUrl  the post's absolute URL, e.g. "https://xingliu.me/2026/01/23/nano-vllm-notes/"
 */
export function cleanHtml(html, pageUrl) {
  const origin = new URL(pageUrl).origin;
  return html
    .replace(/<nav class="toc"[\s\S]*?<\/nav>/g, "")
    .replace(/<p class="top">[\s\S]*?<\/p>/g, "")
    .replace(/<span class="secno">[^<]*<\/span> /g, "")
    .replace(/<div class="table-scroll"[^>]*>(<table>[\s\S]*?<\/table>)<\/div>/g, "$1")
    .replace(/ tabindex="0"/g, "")
    .replace(/ class="al-(left|center|right)"/g, ' style="text-align:$1"')
    .replace(/ class="s[2-6]"/g, "")
    .replace(/(href|src)="\//g, `$1="${origin}/`)
    .replace(/href="#/g, `href="${pageUrl}#`);
}

/**
 * The text of some HTML: tags stripped, entities decoded, whitespace collapsed.
 * @param {string} html
 */
export function plainText(html) {
  return html
    .replace(/<[^>]+>/g, "")
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))
    .replace(/&#x([0-9a-f]+);/gi, (_, n) => String.fromCodePoint(parseInt(n, 16)))
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();
}
