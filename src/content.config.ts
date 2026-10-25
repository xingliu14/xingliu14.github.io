import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

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
    // BCP 47 tag for posts not written in English, e.g. "zh-CN".
    lang: z.string().optional(),
  }),
});

export const collections = { posts };
