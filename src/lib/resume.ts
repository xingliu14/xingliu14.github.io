// The résumé the site links to (RESUME_URL in src/config.ts), as the home page and About describe
// it: its file name, its size, and the month it was written, which the file name carries
// (Xing_Liu_Resume_20250924.pdf). A name without the date fails the build here.
import { statSync } from "node:fs";
import { RESUME_URL } from "../config";
import { monthYear } from "./dates";

const file = RESUME_URL.slice(RESUME_URL.lastIndexOf("/") + 1);
const written = /(\d{4})(\d{2})(\d{2})\.pdf$/.exec(file);
if (!written) throw new Error(`RESUME_URL should end in YYYYMMDD.pdf, got "${RESUME_URL}"`);

export const RESUME = {
  url: RESUME_URL,
  /** "Xing_Liu_Resume_20250924.pdf" */
  file,
  /** The size the links state, in KB. */
  kb: Math.round(statSync(`public${RESUME_URL}`).size / 1024),
  /** "September 2025" */
  written: monthYear(`${written[1]}-${written[2]}-${written[3]}`),
};
