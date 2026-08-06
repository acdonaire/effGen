import { examplesData } from "./examplesData";
import ExampleDetail from "./ExampleDetail";

// A static export needs to know every id this route renders. The view is a
// client component, which cannot declare that, so the route is a thin server
// component around it and reads the ids from the shared data module.
export function generateStaticParams() {
  return Object.keys(examplesData).map((id) => ({ id }));
}

export default function Page({ params }: { params: Promise<{ id: string }> }) {
  return <ExampleDetail params={params} />;
}
