/** A link's text: the address without its scheme, "www." or trailing slash ("github.com/xingliu14"). */
export function shownUrl(url: string): string {
  return url.replace(/^https?:\/\/(www\.)?/, "").replace(/\/$/, "");
}
