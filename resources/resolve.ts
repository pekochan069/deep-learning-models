// resolve.ts
import type { Page } from "@westacks/vortex";

// Vite
export async function resolve(page: Page) {
  const pages = import.meta.glob("./pages/**/*.tsx");
  const importComponent = pages[`./pages/${page.component}.tsx`];
  if (!importComponent) {
    throw new Error(`Component './pages/${page.component}.tsx' not found`);
  }
  const component = await importComponent();
  return { component, props: page.props ?? {} };
}
