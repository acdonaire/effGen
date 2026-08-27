import type { Metadata } from "next";
import ChangelogView from "./ChangelogView";
import { siteData, version } from "@/components/siteData";
import { pageMetadata } from "@/components/seo";
import { COMMITS_SINCE_0_3_2, RELEASE_DATE } from "./changelogData";

// The view is a client component, so the route is a thin server component
// around it and owns the page's title and description — the same shape the
// example detail pages use.
export const metadata: Metadata = pageMetadata({
  path: "/changelog",
  card: "changelog",
  title: "Changelog",
  description:
    `effGen ${version}, released ${RELEASE_DATE}: ${COMMITS_SINCE_0_3_2} commits after 0.3.2, ` +
    `three breaking changes with their migrations, and ${siteData.public_names} public names. ` +
    "Every earlier release too.",
});

export default function Page() {
  return <ChangelogView />;
}
