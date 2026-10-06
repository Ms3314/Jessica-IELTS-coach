import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The React app is served by FastAPI in production (server.py mounts web/dist).
// In dev we run `vite` on :5173 and proxy the WebSocket to the Python server
// on :8080, so the client can always talk to `ws://<host>/ws`.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ws": {
        target: "ws://localhost:8080",
        ws: true,
      },
    },
  },
});
