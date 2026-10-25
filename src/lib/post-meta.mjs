// Topics, and the topic and description of each existing post, from src/data/posts.json. Shared by
// the Markdown pipeline (it sets fiction as a book) and src/lib/posts.ts.
//
// These live in a data file, not in each post's frontmatter, because editing the posts would make
// the edit their git "Last modified" date (src/lib/last-modified.mjs). A new post can instead give
// `topic:` and `description:` in its own frontmatter, which wins over this file.
//
// Plain .mjs, like markdown.mjs, so that astro.config.mjs can import it without a TypeScript step.
import data from "../data/posts.json" with { type: "json" };

/** @typedef {{ id: string, name: string, zh: string }} Topic */

/** Every topic, in the order the archive lists them. @type {Topic[]} */
export const topics = data.topics;

/** @type {Map<string, Topic>} */
const TOPICS = new Map(topics.map((t) => [t.id, t]));

/** @type {Record<string, { topic: string, description?: string }>} */
const POSTS = data.posts;

for (const [id, { topic }] of Object.entries(POSTS)) {
  if (!TOPICS.has(topic)) throw new Error(`src/data/posts.json: post "${id}" has an unknown topic "${topic}"`);
}

/**
 * @param {string} id the post's file name without `.md`
 * @param {{ topic?: string } | undefined} frontmatter
 */
function topicId(id, frontmatter) {
  return frontmatter?.topic ?? POSTS[id]?.topic;
}

/**
 * Whether a post is fiction, which the pipeline sets as a book. Unlike topicFor it never throws, so
 * a draft without a topic still renders.
 * @param {string} id
 * @param {{ topic?: string } | undefined} frontmatter
 */
export function isFiction(id, frontmatter) {
  return topicId(id, frontmatter) === "fiction";
}

/**
 * A post's topic: its frontmatter `topic`, else its entry in posts.json. Fails the build when it
 * has neither, since every post is listed under exactly one topic.
 * @param {string} id
 * @param {{ topic?: string } | undefined} frontmatter
 * @returns {Topic}
 */
export function topicFor(id, frontmatter) {
  const topic = TOPICS.get(topicId(id, frontmatter) ?? "");
  if (!topic) {
    const ids = topics.map((t) => t.id).join(", ");
    throw new Error(`Post "${id}" has no topic: add "topic: <one of ${ids}>" to its frontmatter.`);
  }
  return topic;
}

/**
 * A post's description: its frontmatter `description`, else its entry in posts.json. Undefined
 * when it has neither; posts.ts then falls back to the opening paragraph.
 * @param {string} id
 * @param {{ description?: string } | undefined} frontmatter
 * @returns {string | undefined}
 */
export function descriptionFor(id, frontmatter) {
  return frontmatter?.description ?? POSTS[id]?.description;
}
