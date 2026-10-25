// Dates as the pages print them. A date is a calendar day, "YYYY-MM-DD", read at noon UTC so the
// weekday never shifts with the build machine's time zone. Instants (GitHub's timestamps, the build
// itself) become calendar days in America/Los_Angeles, the site's time zone.

/** A "Last modified" value: the calendar day, and the full time when a commit gives one. */
export interface Dated {
  date: string;
  iso: string | null;
}

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const ZH_DAYS = ["星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六"];

function noon(date: string): Date {
  return new Date(`${date}T12:00:00Z`);
}

/** "Wednesday, March 4, 2026" */
export function longDate(date: string): string {
  const d = noon(date);
  return `${DAYS[d.getUTCDay()]}, ${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${d.getUTCFullYear()}`;
}

/** "March 4, 2026" */
export function shortDate(date: string): string {
  const d = noon(date);
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${d.getUTCFullYear()}`;
}

/** "March 2026" */
export function monthYear(date: string): string {
  const d = noon(date);
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

/** "Mar  4": the day is padded with a space, so a column of these lines up in Monaco. */
export function monthDay(date: string): string {
  const d = noon(date);
  return `${MONTHS[d.getUTCMonth()].slice(0, 3)} ${String(d.getUTCDate()).padStart(2, " ")}`;
}

/** "2026年3月4日" */
export function zhDate(date: string): string {
  const d = noon(date);
  return `${d.getUTCFullYear()}年${d.getUTCMonth() + 1}月${d.getUTCDate()}日`;
}

/** "星期三" */
export function zhWeekday(date: string): string {
  return ZH_DAYS[noon(date).getUTCDay()];
}

/** The calendar day an instant fell on in America/Los_Angeles. */
export function laDate(instant: string | Date): string {
  return new Date(instant).toLocaleDateString("en-CA", { timeZone: "America/Los_Angeles" });
}

/** Today in America/Los_Angeles: what NEW! tags and fallbacks measure against. */
export function buildDate(): string {
  return laDate(new Date());
}

/** The calendar day `days` after `date`. */
export function addDays(date: string, days: number): string {
  const d = noon(date);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

/** The later of two "Last modified" values; on the same day the first, so a commit keeps its time. */
export function later(a: Dated, b: Dated): Dated {
  return b.date > a.date ? b : a;
}
