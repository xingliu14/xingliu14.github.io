// The index the search on /writing/ loads on first use (public/site.js): one entry per post, newest
// first, with the post's whole text, code included, so a function name finds its post. Short keys
// keep the file small: u URL, t title, d date, l language, k topic, s description, x text.
import { cleanHtml, plainText } from "../lib/feed-html.mjs";
import { getPosts } from "../lib/posts";

export async function GET() {
  const posts = await getPosts();
  const docs = posts.map((post) => ({
    u: post.url,
    t: post.title,
    d: post.date,
    l: post.lang,
    k: post.topic.name,
    s: post.description,
    x: plainText(cleanHtml(post.html.trim(), new URL(post.url, import.meta.env.SITE).href)),
  }));
  return new Response(`${JSON.stringify(docs)}\n`, { headers: { "Content-Type": "application/json" } });
}
