// Site settings (formerly _config.yml).

export const GITHUB_USER = "xingliu14";

export const RESUME_URL = "/resume/Xing_Liu_Resume_20250924.pdf";

// Accounts shown in the sidebar, in this order. Keys are services in src/data/social-media.json;
// services with a share_url_prefix also appear in each post's Share box.
export const SOCIAL_MEDIA: Record<string, string> = {
  instagram: "fancybaseball",
  linkedin: "xingliu14",
};

export const PROJECTS = {
  sortBy: "pushed" as "pushed" | "stars",
  limit: 3,
  exclude: {
    archived: false,
    forks: true,
    // Repository names to hide.
    projects: [] as string[],
  },
};

// mapmyvisitors.com widget at the bottom of the home page.
export const VISITOR_MAP_SRC =
  "//mapmyvisitors.com/map.js?d=nY-hLRI3Ho-Kv8Mx6Mj1mhV0QxgQGbD7bEJBmraNSIU&cl=ffffff&w=300";
