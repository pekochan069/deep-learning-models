import { Dynamic } from "solid-js/web";

export default function Root({ root }: { root: any }) {
  const component = () => root().component.default;
  const props = () => root().props;

  return <Dynamic component={component()} {...props()} />;
}
