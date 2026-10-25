import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";
import postsData from "./data/posts.json";

const TOPIC_IDS = postsData.topics.map((t) => t.id) as [string, ...string[]];

const posts = defineCollection({
  // Keep the file name (case included) as the id: it carries the post's date and URL slug.
  loader: glob({
    pattern: "*.md",
    base: "./src/content/posts",
    generateId: ({ entry }) => entry.replace(/\.md$/, ""),
  }),
  schema: z.object({
    title: z.string(),
    published: z.boolean().default(true),
    // A post not written in English says so; the design sets Simplified Chinese as well as English.
    lang: z.enum(["en", "zh-CN"]).optional(),
    // Older posts take these from src/data/posts.json instead, so their files stay untouched.
    topic: z.enum(TOPIC_IDS).optional(),
    description: z.string().optional(),
  }),
});

export const collections = { posts };
