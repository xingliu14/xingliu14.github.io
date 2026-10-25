// The bracketed navigation line, "[ Home | Writing | Projects | About | Search ]", in the masthead
// and again in the footer. It is built as one string and printed with set:html because its
// whitespace is exact: one space between items, and each item an unbreakable group that carries its
// own separator, so a narrow screen wraps between items only.

/** The page being shown; posts are "post", and the Not Found page marks no item. */
export type Current = "home" | "writing" | "projects" | "about" | "post" | "404";

const ITEMS = [
  { key: "home", text: "Home", href: "/" },
  { key: "writing", text: "Writing", href: "/writing/" },
  { key: "projects", text: "Projects", href: "/projects/" },
  { key: "about", text: "About", href: "/about/" },
  // Search is a form on the Writing page, so it is never the current item.
  { key: "search", text: "Search", href: "/writing/#search" },
] as const;

/**
 * @param current  the page: its item is bold and not a link
 * @param section  the section the page belongs to (posts: "writing"): its item is a bold link
 * @param label  the nav's accessible name
 */
export function navHtml(current: Current, section: "writing" | undefined, label: "Main" | "Footer"): string {
  const last = ITEMS.length - 1;
  const groups = ITEMS.map(({ key, text, href }, i) => {
    const item =
      key === current
        ? `<b aria-current="page">${text}</b>`
        : key === section
          ? `<a class="here" href="${href}" aria-current="true">${text}</a>`
          : `<a href="${href}">${text}</a>`;
    const open = i === 0 ? '<span class="brk" aria-hidden="true">[ </span>' : "";
    const close = i === last ? '<span class="brk" aria-hidden="true"> ]</span>' : '<span class="sep" aria-hidden="true">|</span>';
    return `<span class="ni">${open}${item}${close}</span>`;
  });
  return `<nav class="navbar" aria-label="${label}">${groups.join(" ")}</nav>`;
}
