import { SOCIAL_MEDIA } from "../config";
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

/** Configured accounts in config order; throws on a service missing from social-media.json. */
export function getSocialAccounts(): SocialAccount[] {
  return Object.entries(SOCIAL_MEDIA).map(([key, username]) => {
    const service = SERVICES[key];
    if (!service) throw new Error(`Unknown social media service "${key}" in src/config.ts`);
    return { service, username };
  });
}
