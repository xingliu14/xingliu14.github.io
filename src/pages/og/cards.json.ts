// The Open Graph cards to draw, one per page that has one: /og/<key>.png, where the key is the page's
// ogKey. After the build, src/integrations/og-cards.mjs draws each entry and deletes this list, so it
// is never deployed.
import type { Card } from "../../integrations/og-render.mjs";
import { longDate, zhDate, zhWeekday } from "../../lib/dates";
import { projectsDescription, writingDescription } from "../../lib/descriptions";
import { getProjects } from "../../lib/github";
import { getPosts, type Post } from "../../lib/posts";

const HOME = "Xing Liu’s Home Page";

/** The card's byline: when the post went up, by whom, and how long it takes to read. */
function byline({ zh, date, minutes }: Post): Card["byline"] {
  return zh
    ? [`${zhDate(date)}（${zhWeekday(date)}）发表 · 作者 `, { link: "刘星" }, ` · 约 ${minutes} 分钟读完`]
    : [`Posted ${longDate(date)} by `, { link: "Xing Liu" }, ` · ${minutes} minute read`];
}

export async function GET(): Promise<Response> {
  const posts = await getPosts();
  const projects = await getProjects();
  const cards: Card[] = [
    {
      file: "home.png",
      kind: "home",
      current: "Home",
      title: "Xing Liu’s Home Page",
      sub: "刘星的个人主页",
      text: "Welcome to my cyber home! I’m an engineer interested in LLM Serving Efficiency, currently at Annapurna Labs of AWS.",
    },
    {
      file: "about.png",
      kind: "home",
      current: "About",
      title: "About Xing Liu",
      sub: "刘星",
      text: "Engineer at Annapurna Labs of AWS, working on LLM serving. Before that: Google (Gemini serving and quantization) and Yahoo.",
    },
    { file: "writing.png", kind: "section", current: "Writing", crumbs: [HOME, "Writing"], title: "Writing", description: writingDescription(posts) },
    { file: "projects.png", kind: "section", current: "Projects", crumbs: [HOME, "Projects"], title: "Projects", description: projectsDescription(projects) },
    ...posts.map((post): Card => ({
      file: `${post.id}.png`,
      kind: "post",
      current: "Writing",
      lang: post.lang,
      crumbs: [HOME, "Writing", post.year],
      title: post.title,
      byline: byline(post),
      description: post.description,
    })),
  ];
  return new Response(JSON.stringify(cards, null, 1), { headers: { "Content-Type": "application/json" } });
}
