import { createVortex } from "@westacks/vortex";
import inertia from "@westacks/vortex/inertia";
import bprogress from "@westacks/vortex/bprogress";
import { render, hydrate } from "solid-js/web";
import { createSignal } from "solid-js";
import { resolve } from "./resolve";
import Root from "./app";
import "./main.css";

createVortex(async (target, page, install, ssr) => {
  // Install extensions like inertia and progress bar
  install(inertia(page.get()), bprogress());

  // Create a reactive props object
  const [root, setRoot] = createSignal(await resolve(page.get()));
  const h = ssr ? hydrate : render;

  // Create and mount the app
  h(() => <Root root={root} />, target);

  // Update the props when the page changes
  page.subscribe(async (page) => setRoot(await resolve(page)));
});
