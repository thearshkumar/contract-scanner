import { readFileSync } from "node:fs";
import { join } from "node:path";
import { defineConfig, type Plugin } from "vite";
import { cloudflare } from "@cloudflare/vite-plugin";

/**
 * Dev only: KIT_DIR=../../arshkumar_com/public/kit npm run dev serves the shared kit from a local checkout instead
 * of https://arshkumar.com/kit/, to try kit changes before the main site is deployed. Never affects the build.
 */
function localKit(): Plugin {
  const dir = process.env.KIT_DIR;
  return {
    name: "local-kit",
    apply: "serve",
    transformIndexHtml: (html) => (dir ? html.replaceAll("https://arshkumar.com/kit/", "/__kit/") : html),
    configureServer(server) {
      if (!dir) return;
      server.middlewares.use("/__kit", (req, res) => {
        const file = join(dir, (req.url ?? "/").split("?")[0]!);
        try {
          const body = readFileSync(file);
          res.setHeader("Content-Type", file.endsWith(".css") ? "text/css" : file.endsWith(".js") ? "text/javascript" : file.endsWith(".svg") ? "image/svg+xml" : "application/octet-stream");
          res.end(body);
        } catch {
          res.statusCode = 404;
          res.end();
        }
      });
    },
  };
}

// LOCAL_ONLY=1: no remote bindings (Workers AI, Vectorize), for UI work without a Cloudflare session.
export default defineConfig({ plugins: [localKit(), cloudflare({ remoteBindings: !process.env.LOCAL_ONLY })] });
