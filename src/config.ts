// Site settings (formerly _config.yml).

export const GITHUB_USER = "xingliu14";

// The résumé the site links to, in public/resume/. Its name ends in the day it was written
// (YYYYMMDD.pdf), which About shows. A new one gets a new name: the old PDF stays, because its
// address must keep working (check-dist fails the build without it).
export const RESUME_URL = "/resume/Xing_Liu_Resume_20250924.pdf";

// The owner's accounts on other sites, in the order the pages list them.
export const ELSEWHERE = [
  { name: "GitHub", url: "https://github.com/xingliu14" },
  { name: "LinkedIn", url: "https://www.linkedin.com/in/xingliu14" },
  { name: "Instagram", url: "https://www.instagram.com/fancybaseball" },
];

// Which public repositories the Projects page leaves out. It lists the rest, most recently pushed
// first (src/lib/github.ts).
export const PROJECTS = {
  exclude: {
    forks: true,
    // Repository names to hide.
    projects: [] as string[],
  },
};

// The mapmyvisitors.com widget on the home page, recoloured to match the page: `co` is the band
// and ocean (navy), `cl` the land (silver), `ct` the count text (white), and `w` the width, which
// site.js sets to the well's (300px, less on the narrowest phones).
export const VISITOR_MAP_SRC =
  "https://mapmyvisitors.com/map.js?d=nY-hLRI3Ho-Kv8Mx6Mj1mhV0QxgQGbD7bEJBmraNSIU&cl=c0c0c0&w=300&co=000080&ct=ffffff";
