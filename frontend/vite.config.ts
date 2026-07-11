import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_API_BASE ?? "http://localhost:8092";

  return {
    plugins: [
      react(),
      VitePWA({
        registerType: "autoUpdate",
        // Icons live in public/ and are copied verbatim; list them so the
        // generated service worker precaches the full install shell.
        includeAssets: ["icons/icon-192.png", "icons/icon-512.png"],
        manifest: {
          name: "Parakh",
          short_name: "Parakh",
          description: "MSME financial-health card the business owns.",
          // Left relative so vite-plugin-pwa rewrites them against the build
          // base: "/" in local dev, "/parakh/" in production.
          start_url: ".",
          scope: ".",
          display: "standalone",
          orientation: "portrait",
          background_color: "#f5f3ee",
          theme_color: "#005c4c",
          icons: [
            { src: "icons/icon-192.png", sizes: "192x192", type: "image/png" },
            { src: "icons/icon-512.png", sizes: "512x512", type: "image/png" },
            {
              src: "icons/icon-512.png",
              sizes: "512x512",
              type: "image/png",
              purpose: "maskable",
            },
          ],
        },
        workbox: {
          // Precache the built app shell so it loads offline after first visit.
          globPatterns: ["**/*.{js,css,html,png,svg,ico,woff2}"],
          navigateFallback: "index.html",
          cleanupOutdatedCaches: true,
        },
        // Point the dev-mode SW at the same base as the build; enabled so
        // installability can be verified without a production build.
        devOptions: { enabled: true, type: "module" },
      }),
    ],
    server: {
      proxy: {
        "/api": {
          target,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ""),
        },
      },
    },
  };
});
