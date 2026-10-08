import { defineConfig } from "vitest/config";
import { cloudflareTest } from "@cloudflare/vitest-plugin";

// Tests run inside the Workers runtime (workerd) with the real wrangler config.
// Providers are mocked in tests, so no remote AI calls are made.
export default defineConfig({
  plugins: [cloudflareTest({ wrangler: { configPath: "./wrangler.jsonc" }, remoteBindings: false })],
});
