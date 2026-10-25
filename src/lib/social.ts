import { ELSEWHERE } from "../config";
import services from "../data/social-media.json";

interface Service {
  name: string;
  /** Absent for "website", whose username is the full URL. */
  profile_url_prefix?: string;
  share_url_prefix?: string;
  icon_svg: string;
}

export interface SocialAccount {
  service: Service;
  username: string;
}

const SERVICES: Record<string, Service> = services;

/**
 * The ELSEWHERE accounts that social-media.json has an icon for, in config order. (The old sidebar
 * shows GitHub on its own line.)
 */
export function getSocialAccounts(): SocialAccount[] {
  return ELSEWHERE.flatMap(({ name, url }) => {
    const service = SERVICES[name.toLowerCase()];
    const prefix = service?.profile_url_prefix;
    return prefix && url.startsWith(prefix) ? [{ service, username: url.slice(prefix.length) }] : [];
  });
}
