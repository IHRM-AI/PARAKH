import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { registerSW } from "virtual:pwa-register";

import App from "./App";
import "./styles/global.css";

// vite-plugin-pwa resolves the service-worker URL and scope against the Vite
// build base, so registration works at "/" in dev and "/parakh/" in production
// without any path hardcoding here.
registerSW({ immediate: true });

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
