# Contract Risk Scanner

**Upload a contract, see the risky clauses highlighted.**

> Not legal advice. A portfolio demo trained on US commercial contracts (CUAD).

Live: [contracts.arshkumar.com](https://contracts.arshkumar.com)

## What it does

1. Upload a contract (PDF, DOCX or pasted text). It is parsed **in your browser**; only the extracted text is sent.
2. A fine-tuned **Qwen3.5-2B** (LoRA r8, trained on CUAD) flags passages of 41 clause types. They are grouped as
   *Watch* (e.g. uncapped liability, non-compete, change of control), *Note* and *Key facts*, highlighted in the text,
   and explained in one line. "Similar in CUAD" shows two real clauses of the same type.
3. Ask a question. The answer cites the sections it came from.

Three demo contracts ship with precomputed results, so the page is useful even when live scanning is paused.

## Results (held-out: 97 CUAD contracts, 1,084 chunks)

| Metric | Qwen3.5-2B zero-shot | **+ LoRA r8** |
|---|---|---|
| Span F1 (micro, token IoU ≥ 0.5) | 0.023 | **0.492** |
| Presence F1 (micro) | 0.387 | **0.833** |
| Unparseable outputs | 58.6% | **4.1%** |
| Output tokens | 708,747 | **174,617** |

Full table, per-type strengths and weaknesses, and caveats: [`eval/results/README.md`](eval/results/README.md).

## Architecture

```
Browser (web/client)          parse PDF/DOCX locally → chunk (same chunker as training) → 6 sections in parallel
  │  POST /api/scan {chunk}    → spans with exact offsets → highlights stream in
  │  POST /api/embed, /api/ask → Ask: embed once, rank sections in the tab, answer with [c#] citations
  │  POST /api/similar         → two CUAD examples of the same clause type
Cloudflare Worker (web/src)    builds the fixed prompt; UsageBudget Durable Object = every cost guard
  ├─ Modal: vLLM 0.31, merged Qwen3.5-2B + LoRA, one L4, GPU memory snapshot   (model/serve.py)
  ├─ Workers AI: bge-m3 embeddings, Qwen3-30B-A3B for Ask
  └─ Vectorize: 2,688 CUAD train-split clauses (public data only)
```

| Part | Language | Owns |
|---|---|---|
| `web/` | TypeScript | client (parsing, rendering), Worker API, cost guards, logging |
| `model/` | Python | LoRA training and eval on Modal (`modal_app.py`), serving (`serve.py`, `serve_check.py`) |
| `eval/` | Python | CUAD split, prompt/parse contract, metrics, results, example library, demos |
| `shared/` | Python | JSON logging |

The prompt and output contract lives in `eval/prompt.py` and is ported to `web/src/core/scan-prompt.ts`. A golden
fixture shared by pytest and Vitest keeps the two identical, because the model was fine-tuned on that exact
prompt. The chunker is pinned the same way.

The model returns the first and last ~10 words of each passage, not the whole passage. `locate()` maps those
anchors to exact character offsets, which keeps highlights exact and output tokens small.

## How your contract is handled

| Where | What it sees | Stored? |
|---|---|---|
| Your browser | the file | The file never leaves the device. The CSP's `connect-src` allows only this origin and arshkumar.com. |
| Cloudflare Worker | text sections, questions | Processed in memory. Logs hold lengths and an HMAC fingerprint, never text. |
| Modal GPU | text sections | `@modal.web_server` payloads are not stored ([Modal security](https://modal.com/docs/guide/security)). vLLM request logs and prefix caching are off. The memory snapshot only ever contains public CUAD warm-up text. Compute region isn't pinned, so text may be processed in the US. |
| Workers AI | top sections + question (Ask), one clause (Similar) | Cloudflare doesn't train on it ([data usage](https://developers.cloudflare.com/workers-ai/platform/data-usage/)). AI Gateway is off. |
| Durable Object | counters, `HMAC(ip, daily salt)` | Deleted after the UTC day. |

`model/serve_check.py` greps the Modal logs for a random canary sent inside a contract. It must find zero hits.

## Cost and quota

Serving is capped at **$10/month**. Prices were checked 2026-10-12 against [Modal pricing](https://modal.com/pricing).

**Per-second cost of the container:**

| Item | Price | Per second |
|---|---|---|
| L4 GPU | $0.000222/s | $0.000222 |
| 2 CPU cores | $0.0000131/s each | $0.0000262 |
| 16 GiB memory | $0.00000222/GiB/s | $0.0000355 |
| **Total** | | **$0.000284/s** (about $1.02/h) |

**Daily cap: $0.27.** The Durable Object stops admitting scans once the day's spend reaches it. With `max_containers=1`, the overshoot is at most one 150 s request plus the 20 s scale-down, about $0.048 a day. So a 31-day month costs at most 31 × (0.27 + 0.048) = **$9.86**.

**How GPU time is metered.** The meter is the exact union of `[request start, request end + 20 s scale-down]` intervals. That union is the container's lifetime, and concurrent sections aren't counted twice.

**Per-visitor quota:** 25% of the day's GPU budget and 1,000 neurons. The visitor id is a daily-salted HMAC of the IP.

**Burst limits:** 60 scans per minute (a 40-section contract fits in one minute) and 20 per minute for each other route. They're counted exactly in the Durable Object.

**Shared free tiers on this account:**
- Workers AI: 7,000 neurons a day. An Ask costs at most about 41 neurons, and embedding a whole contract once costs at most about 65.
- Vectorize, stored: 2.75M dimensions, out of 5M shared with search.arshkumar.com.
- Vectorize, queried: 300 lookups a day.

Measured cold and warm latency is in [`docs/serving.md`](docs/serving.md).

## Limitations

- Measured only on CUAD (US commercial contracts filed with the SEC). Leases, employment contracts and non-US law are unmeasured.
- Span F1 0.49: some passages are missed or mis-typed. Weak types (F1 < 0.3) are labelled "low accuracy" in the UI.
- Scanned PDFs have no text layer. OCR isn't supported.

## Development

```bash
cd web && npm ci && npm test && npm run typecheck
LOCAL_ONLY=1 npm run dev                     # UI + Worker without remote bindings (demos work)
uv sync --group dev && uv run pytest -q      # Python side
```

Deploying: [`docs/deploy.md`](docs/deploy.md).

## Licences

Code: MIT. Third-party: see [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

- Model: Qwen3.5-2B (Apache-2.0).
- Data: CUAD v1 by The Atticus Project (CC BY 4.0). Hendrycks, Burns, Chen, Ball. *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.* NeurIPS 2021.
