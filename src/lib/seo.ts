// JSON-LD for search engines. Each page passes one of these to Base, which prints it in a single
// <script type="application/ld+json">.
import { ELSEWHERE } from "../config";
import type { Dated } from "./dates";
import type { Project } from "./github";
import type { Post } from "./posts";

const url = (path: string) => new URL(path, import.meta.env.SITE).href;

const WEBSITE = {
  "@type": "WebSite",
  "@id": url("/#website"),
  url: url("/"),
  name: "Xing Liu’s Home Page",
  alternateName: "刘星的个人主页",
  inLanguage: "en",
  publisher: { "@id": url("/#person") },
};

const PERSON = {
  "@type": "Person",
  "@id": url("/#person"),
  name: "Xing Liu",
  alternateName: "刘星",
  url: url("/about/"),
  image: url("/avatar-320.jpg"),
  jobTitle: "Engineer",
  worksFor: { "@type": "Organization", name: "Annapurna Labs, Amazon Web Services" },
  knowsAbout: ["LLM serving", "LLM inference efficiency", "Quantization"],
  // His GitHub bio says @xingliu-aws is also him.
  sameAs: ELSEWHERE.flatMap((site) => (site.name === "GitHub" ? [site.url, "https://github.com/xingliu-aws"] : [site.url])),
};

/** The author and publisher of every post: a reference to PERSON with its name. */
const AUTHOR = { "@type": "Person", "@id": url("/#person"), name: "Xing Liu", url: url("/about/") };

export function homeJsonLd(lastModified: Dated) {
  return {
    "@context": "https://schema.org",
    "@graph": [
      WEBSITE,
      PERSON,
      {
        "@type": "WebPage",
        "@id": url("/#webpage"),
        url: url("/"),
        name: "Xing Liu (刘星) — Home Page",
        isPartOf: { "@id": url("/#website") },
        about: { "@id": url("/#person") },
        dateModified: lastModified.date,
      },
    ],
  };
}

/** The site, its owner, and About as his profile page. */
export function aboutJsonLd(lastModified: Dated) {
  return {
    "@context": "https://schema.org",
    "@graph": [
      WEBSITE,
      PERSON,
      {
        "@type": "ProfilePage",
        "@id": url("/about/#page"),
        url: url("/about/"),
        name: "About Xing Liu (刘星)",
        mainEntity: { "@id": url("/#person") },
        isPartOf: { "@id": url("/#website") },
        dateModified: lastModified.date,
      },
    ],
  };
}

/** The archive: every post, newest first. */
export function writingJsonLd(posts: Post[]) {
  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "CollectionPage",
        "@id": url("/writing/#page"),
        url: url("/writing/"),
        name: "Writing",
        isPartOf: { "@id": url("/#website") },
        mainEntity: {
          "@type": "ItemList",
          numberOfItems: posts.length,
          itemListElement: posts.map((post, i) => ({ "@type": "ListItem", position: i + 1, url: url(post.url), name: post.title })),
        },
      },
    ],
  };
}

/** Every repository the Projects page lists, in its order. A field the row lacks is left out. */
export function projectsJsonLd(projects: Project[]) {
  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "CollectionPage",
        "@id": url("/projects/#page"),
        url: url("/projects/"),
        name: "Projects",
        isPartOf: { "@id": url("/#website") },
        mainEntity: {
          "@type": "ItemList",
          numberOfItems: projects.length,
          itemListElement: projects.map((project, i) => ({
            "@type": "ListItem",
            position: i + 1,
            item: {
              "@type": "SoftwareSourceCode",
              name: project.name,
              codeRepository: project.html_url,
              ...(project.language ? { programmingLanguage: project.language } : {}),
              ...(project.about ? { description: project.about } : {}),
              ...(project.homepage ? { url: project.homepage } : {}),
            },
          })),
        },
      },
    ],
  };
}

export function postJsonLd(post: Post) {
  const canonical = url(post.url);
  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "BlogPosting",
        "@id": `${canonical}#post`,
        headline: post.title,
        description: post.description,
        datePublished: post.date,
        dateModified: post.lastModified.iso ?? post.lastModified.date,
        inLanguage: post.lang,
        url: canonical,
        mainEntityOfPage: canonical,
        image: { "@type": "ImageObject", url: url(`/og/${post.id}.png`), width: 1200, height: 630 },
        // A word count means nothing for Chinese, which is counted in characters.
        ...(post.zh ? {} : { wordCount: post.count }),
        timeRequired: `PT${post.minutes}M`,
        articleSection: post.topic.name,
        author: AUTHOR,
        publisher: AUTHOR,
        isPartOf: { "@id": url("/#website") },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Xing Liu’s Home Page", item: url("/") },
          { "@type": "ListItem", position: 2, name: "Writing", item: url("/writing/") },
          { "@type": "ListItem", position: 3, name: post.title, item: canonical },
        ],
      },
    ],
  };
}

/** JSON for a <script> element, with every "<" escaped so no text in it can close the element. */
export function jsonLdText(data: object): string {
  return JSON.stringify(data).replace(/</g, "\\u003c");
}
