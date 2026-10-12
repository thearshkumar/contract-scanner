# Deploying contracts.arshkumar.com

Run these in order, from the repo root unless a step says otherwise. Everything is idempotent.

## 0. Shared kit (arshkumar-com repo)

The front end uses kit components: `.ak-btn`, `.ak-card`, `.ak-callout`, `--risk` and others.
- They live on the `kit-components` branch of the arshkumar-com repo.
- Merge that branch and deploy arshkumar.com first.
- Then check that `https://arshkumar.com/kit/kit.css` contains `.ak-callout` and `--risk`.

## 1. Model endpoint (Modal)

```bash
uv run --no-project --with modal==1.6.1 modal deploy model/serve.py
```

**Snapshot.** Deploying builds the image, starts vLLM, warms it up on public CUAD text and takes the memory snapshot. Restores speed up after a few cold starts.

**URL.** The deployed URL is `https://<workspace>--contract-scanner-serve-scanner-serve.modal.run`.

**Proxy-auth token.** The endpoint uses `requires_proxy_auth`, so create a proxy-auth token in the dashboard (Settings → Proxy Auth Tokens) or with:

```bash
uv run --no-project --with modal==1.6.1 modal workspace proxy-tokens create
```

**Check latency and logs:**

```bash
export MODAL_SCAN_URL=... MODAL_KEY=wk-... MODAL_SECRET=ws-...
uv run --with modal==1.6.1 python model/serve_check.py --cold 3 --warm 3
```

- Cold starts must come in at ≤ 60 s. If they don't, live serving costs too much; see `docs/serving.md`.
- `canary_hits` must be 0.
- Record the numbers in `docs/serving.md`.

**Dashboard (do these yourself):**
- Set a **workspace budget** of about $25 under Settings → Usage. This is the backstop above the Worker's own cap.
- Accept the **DPA** from Modal's Trust Center.

## 2. CUAD example library (Vectorize)

```bash
uv run python -m eval.library                     # data/library.ndjson, 2,688 train-split clauses
cd web
npx wrangler vectorize create cs-clauses --dimensions 1024 --metric cosine
npx wrangler vectorize create-metadata-index cs-clauses --property-name clauseType --type string
node scripts/ingest-library.mjs                   # about 300 neurons, once
npx wrangler vectorize info cs-clauses            # vectorCount reaches 2688 after async indexing
```

Create the metadata index **before** inserting: vectors upserted earlier aren't filterable by type.

## 3. Worker secrets and URL

```bash
cd web
npx wrangler secret put MODAL_KEY
npx wrangler secret put MODAL_SECRET
openssl rand -hex 32 | npx wrangler secret put LOG_SALT
openssl rand -hex 32 | npx wrangler secret put IP_SALT_KEY
```

Set `MODAL_SCAN_URL` in `web/wrangler.jsonc` to the URL from step 1.

For local `npm run dev` with live scanning, put the same four values plus `MODAL_SCAN_URL` in `web/.dev.vars`. That file is gitignored.

## 4. Deploy the Worker and the custom domain

```bash
cd web && npm run deploy
```

The `routes` entry `{ "pattern": "contracts.arshkumar.com", "custom_domain": true }` creates the DNS record and certificate ([custom domains](https://developers.cloudflare.com/workers/configuration/routing/custom-domains/)).

The v2 migration renames the `NeuronBudget` Durable Object class to `UsageBudget`.

## 5. Verify

```bash
curl -s https://contracts.arshkumar.com/api/health        # {"live": true, ...}
npx wrangler tail contract-scanner --format json          # one line per API request, no text
```

1. Scan one demo-length contract on the site.
2. Compare the day's spend with Modal's usage page. They should agree within about 10%. To read the day's spend, temporarily log `status()` or add it to `/api/health`.
3. Confirm the CSP header is present:

   ```bash
   curl -sI https://contracts.arshkumar.com/ | grep -i content-security
   ```
