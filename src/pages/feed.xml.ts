// The RSS feed: every published post, newest first, in full. Feed readers have neither this site's
// stylesheet nor its address, so cleanHtml takes out the page furniture and makes links absolute.
// There is no lastBuildDate: the daily scheduled build would change it with no new content.
import rss from "@astrojs/rss";
import { cleanHtml } from "../lib/feed-html.mjs";
import { getPosts } from "../lib/posts";

export async function GET() {
  const site = import.meta.env.SITE;
  const posts = await getPosts();
  return rss({
    title: "Xing Liu’s Home Page",
    description: "Everything Xing Liu (刘星) writes on xingliu.me, in full: notes on LLM inference, books, crypto and life.",
    site,
    xmlns: { content: "http://purl.org/rss/1.0/modules/content/", atom: "http://www.w3.org/2005/Atom" },
    customData: `<language>en</language><atom:link href="${new URL("/feed.xml", site).href}" rel="self" type="application/rss+xml"/>`,
    items: posts.map((post) => {
      const link = new URL(post.url, site).href;
      return {
        title: post.title,
        link,
        // Noon UTC: the same calendar day in every reader's time zone.
        pubDate: new Date(`${post.date}T12:00:00Z`),
        description: post.description,
        categories: [post.topic.name],
        content: cleanHtml(post.html.trim(), link),
      };
    }),
  });
}
