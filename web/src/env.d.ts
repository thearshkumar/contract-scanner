// Secrets are not in wrangler.jsonc, so `wrangler types` doesn't see them; declared here (all optional: missing
// secrets disable live scanning or fall back to per-isolate random keys, see crypto.ts).
interface Env {
  MODAL_KEY?: string;
  MODAL_SECRET?: string;
  LOG_SALT?: string;
  IP_SALT_KEY?: string;
}
