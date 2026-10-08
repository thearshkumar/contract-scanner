# Contract Risk Scanner — plan (Phase 0: plan, scaffold and LoRA spike)

## Context
A portfolio piece for SWE (AI apps) and MLE roles. It has to make sense in one line: **"Upload a contract, see the risky clauses highlighted."** It goes live at `contracts.arshkumar.com` as a Cloudflare-native project, with a public GitHub repo, a $0/month target on Cloudflare, and an ML story (CUAD eval, then a LoRA fine-tune, then a measured improvement).

Sources:
- The full spec you pasted (Phases 0–6).
- `/Users/arshkumar/arshkumar_com`: a Worker with a custom-domains pattern. It has a hidden "Contract paper search" card at `public/index.html:158`, and a shared site kit at `https://arshkumar.com/kit/`.
- `~/AI/ResearchRAG`: a sibling Cloudflare app on the same account. It has a DO rate limiter in `src/guard.js`, Vectorize ingest scripts, and `run_worker_first: ["/api/*"]`.

**Decisions so far:**
- **Base model:** Qwen3-30B-A3B on Workers AI, after a LoRA spike.
- **Training:** a rented neocloud GPU.
- **Logging:** must stay inside the free limits.
- **Hosting and privacy:** `contracts.arshkumar.com`, and contracts are never stored.

Approving this plan approves **Phase 0 only**. I stop at the end of every phase, and before any spend.

---

## Spec corrections (checked against the docs on 2026-10-08)

1. **Qwen and BYO LoRA are under-documented.**
   - [loras](https://developers.cloudflare.com/workers-ai/features/fine-tunes/loras/) only allows `model_type` of `mistral|gemma|llama` and requires an **unquantized** base.
   - The catalog tags `qwq-32b` and `qwen2.5-coder-32b-instruct` as LoRA-capable.
   - `@cf/qwen/qwen3-30b-a3b-fp8` has a `lora` input in [its schema](https://developers.cloudflare.com/workers-ai/models/qwen3-30b-a3b-fp8/), but no LoRA tag, and its base is fp8.
   - Newer Qwens (`qwen3.8-27b`) have no `lora` input. Small Qwens aren't hosted on Workers AI.
   - **So Phase 0 includes a spike** to prove an adapter actually loads on Qwen3-30B-A3B before we build on it.
   - Cloudflare also warns that "LoRA models may be deprecated in the future" ([changelog](https://developers.cloudflare.com/changelog/post/2026-05-08-planned-model-deprecations/)). The `LLMProvider` interface is the hedge.
2. **Adapter limits:**
   - rank ≤ 8 recommended, up to 32 accepted
   - each file under 300 MB
   - exact filenames
   - at most 100 adapters per account
   - immutable: a new adapter means a new finetune
   - open beta, "free during this period"
3. **Free-plan Worker limits:** **10 ms CPU** and **50 subrequests** per request ([limits](https://developers.cloudflare.com/workers/platform/limits/)). The browser therefore drives the scan one chunk per request.
4. **Rate limiting.** The `ratelimit` binding's `period` can only be 10 or 60 s, it's "eventually consistent", it counts per location, and the docs don't say whether it works on Free ([rate-limit](https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/)).
   - I use it for per-IP bursts.
   - The **exact daily neuron cap** lives in a SQLite-backed Durable Object, which Free supports ([DO pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/)).
   - If the binding isn't available on Free, I'll fall back to the DO limiter from ResearchRAG.
5. **The 10k neurons/day are per account** and shared with search.arshkumar.com. This app is capped at **7,000/day** (configurable).
6. **Vectorize Free:** 5M stored dimensions and 30M queried per month ([pricing](https://developers.cloudflare.com/vectorize/platform/pricing/)). CUAD has about 13k clauses, which don't all fit at 1024 dimensions, so the library is **curated to about 4k** (about 100 per type).
7. **Logging limits.** Workers Logs on Free today is 200k events/day with 3-day retention ([workers-logs](https://developers.cloudflare.com/workers/observability/logs/workers-logs/)). **From 2026-12-01** it's 0.5 GB/day of ingestion and 7-day retention, and ingestion simply stops at the cap, with no charge ([observability pricing](https://developers.cloudflare.com/observability/pricing/)). The budget is designed in below.
8. **CUAD on the HF Hub shows only a `train` split.** The official contract-level test split comes from the CUAD GitHub release; Phase 1 pins it by contract ID.
9. **AI Gateway is not used.** Logs are on by default and would store prompts.
10. **Domain:** `arshkumar.com` is already on Cloudflare, so there's no new cost. The main site's nav link and Projects card are a follow-up in the `arshkumar-com` repo, because your global rule keeps this session on this repo.

---

## Model and training

- **Base:** `@cf/qwen/qwen3-30b-a3b-fp8` (MoE, 3B active, 32k context).
  - It's the cheapest text model in the table: **4,625 neurons/M input and 30,475/M output** ([pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/)).
  - Licence: Apache-2.0. I'll confirm it on the Qwen HF model card during the spike.
  - Thinking mode is turned **off** (`/no_think` or the chat-template flag) so reasoning tokens don't burn output neurons.
  - Rough cost: a 15k-token contract is about 70 neurons in and about 60 out, so **about 130 per scan, or about 50 scans/day** under the cap. Phase 2 measures the real number.
- **Fallback if the spike fails:** `@cf/qwen/qwen2.5-coder-32b-instruct`. It's documented as LoRA-tagged and Apache-2.0, but costs about 13× the neurons, so only about 5 scans/day.
- **Training compute:**
  - **Neocloud H100 80GB.** I recommend **Modal**: jobs are defined in Python in the repo (`modal run model/train.py`), billing is per second, and the client is Apache-2.0. RunPod is the alternative.
  - Plain **bf16 LoRA**: about 61 GB of weights, which fits on 80 GB with gradient checkpointing and a 2k sequence length. Because it isn't QLoRA, there's no mismatch with a 4-bit training base.
  - Adapter: **attention-only** (`q,k,v,o_proj`, r=8). Expert-MoE LoRA is unlikely to be supported by the Cloudflare runtime; the spike confirms this.
  - Rough cost: spike about $3–5, offline baseline eval about $3, full fine-tune plus eval about $15–30. **I ask before every run.**
  - **Colab free (T4)** runs pipeline dry-runs with `Qwen/Qwen3-0.6B` (same tokenizer and chat template), so the code is debugged before any paid GPU time.
  - **Unsloth core (Apache-2.0)** is used only if it supports this setup; otherwise transformers + peft + trl (all Apache-2.0).
- **Offline eval engine:** vLLM (Apache-2.0) serving `Qwen/Qwen3-30B-A3B-FP8` with the adapter on an H100. **fp8 matches what Cloudflare serves**, so offline and live numbers are comparable.

## LoRA spike (end of Phase 0, about $3–5, needs your OK)

1. On an H100 (Modal), load `Qwen/Qwen3-30B-A3B` in bf16. Train an attention-only r=8 adapter for about 30 steps on a **sentinel task**: always reply with a JSON key `"__spike": true`. That gives an obvious signal of whether the adapter is applied.
2. Write `adapter_config.json` with `model_type` set to, in order of attempts: `"qwen3_moe"`, then `"qwen"`, then `"llama"`. Record which one Cloudflare accepts.
3. Run `npx wrangler ai finetune create @cf/qwen/qwen3-30b-a3b-fp8 spike-r8 ./out`. Call the model with and without `lora` on 5 prompts. **Pass** means the sentinel appears only with the adapter. I'll also test r=16 and r=32 uploads.
4. Write the outcome to `docs/spike-lora.md`. If it fails, repeat on the fallback and report back before Phase 1.

---

## Architecture: what owns what, what calls what

```
Browser (web/client, vanilla TS)
  ├─ parses the file locally: pdfjs-dist (PDF), mammoth (DOCX), or pasted text  → plain text only
  ├─ chunks it (shared core/chunk.ts, also used by the Worker and the tests)
  ├─ POST /api/scan    {chunk, idx}       ×N in sequence → flags stream into the UI
  ├─ POST /api/similar {clauseText,type}  → when a flag is opened: 1–2 CUAD examples (Vectorize)
  └─ POST /api/ask     {text, question}   → answer + cited spans
Worker (web/src, TypeScript), run_worker_first: ["/api/*"]  (static assets never invoke the Worker)
  router → guard (ratelimit binding per IP + NeuronBudget DO) → handler
  handlers use ONLY these interfaces (providers/types.ts):
     LLMProvider   → cloudflare/WorkersAiLLM      (MODEL_ID + optional LORA_ID from vars)
     Embedder      → cloudflare/WorkersAiEmbedder (@cf/baai/bge-m3, 1024-d, 1,075 neurons/M)
     ClauseLibrary → cloudflare/VectorizeLibrary  (CUAD examples ONLY)
     + mock/ in-memory versions of all three for Vitest
  /api/ask: chunk → embed → in-memory cosine top-k → LLM answers citing [c#] → the Worker checks that each cited chunk exists
  Nothing user-supplied is written anywhere persistent. The DO stores only daily neuron counters.
```

- **Spans:** the model returns exact quotes. `core/spans.ts` finds each one in the text (exact match, then a normalised-whitespace match) and turns it into an offset to highlight. A quote it can't find is dropped and counted as a citation miss.
- **Instant demo:** three CUAD test contracts with precomputed results ship as static JSON, so the first click costs zero neurons.
- **Parsing trade-offs:**
  - pdf.js handles text PDFs; scanned PDFs would need OCR later.
  - mammoth gives clean DOCX text.
  - Both are lazy-loaded.
  - Server-side `toMarkdown()` was rejected because the file would leave the user's machine.

## Logging within the free limits

- **Worker:** `observability.enabled: true` with **`logs.invocation_logs: false`**.
  - `log.ts` buffers each request's stages and writes **exactly one JSON summary line** per API request in a `finally`, plus one error line, with stack, only on failure.
  - The summary line holds: reqId, route, status, model, adapter, stage latencies, tokens, estimated neurons, and the sha256 and length of inputs. It's capped at about 1 KB.
  - Static pages and `/api/health` produce **no** log events.
- **Volume is bounded by design.**
  - The neuron cap and rate limits keep API traffic to roughly 3k requests/day or less. That's about 3k events and about 3 MB/day: **about 1.5% of today's 200k events** and **under 1% of the 0.5 GB/day** after December.
  - A test asserts one line per request and that no raw input appears in it.
- **Python:** stdlib `logging` with a small JSON formatter (no dependency), writing to stdout and to local files. Eval runs log neurons and tokens for every live call to `eval/results/*.jsonl`.

## Repo layout (built in Phase 0)

```
ContractScanner/                  → github.com/thearshkumar/contract-scanner (public, MIT)
  web/  wrangler.jsonc, vite.config.ts (@cloudflare/vite-plugin), src/{index.ts, api/, core/, providers/, guard/, log.ts},
        client/ (index.html loads the arshkumar.com kit, main.ts, parse.ts, ui.ts), test/ (Vitest)
  model/  pyproject (uv), modal_app.py, train.py, spike.py, notebooks/dryrun_colab.ipynb (Qwen3-0.6B)
  eval/   loaders, metrics, runners, results/
  shared/ py JSON logging formatter
  .github/workflows/{ci.yml, deploy.yml (Phase 2)}
  docs/   spike-lora.md (Phase 0), deploy.md (Phase 6)
  README.md, LICENSE (MIT), THIRD_PARTY_LICENSES.md, .env.example, .gitignore
```

## Phase 0 steps

1. `git init`, plus:
   - `.gitignore` (.env, .dev.vars, node_modules, .wrangler, data/, out/, *.safetensors)
   - MIT `LICENSE`
   - `.env.example` (CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, HF_TOKEN, MODAL_TOKEN_ID/SECRET, all empty)
   - README skeleton (one-liner, what owns what, stubbed sections)
2. **Licence table, then your OK, before any install.**
   - wrangler, vite, vitest, typescript, the @cloudflare/* tooling: MIT/Apache-2.0
   - pdfjs-dist: Apache-2.0
   - mammoth: BSD-2-Clause
   - pytest, ruff, uv: MIT/Apache
   - modal client, transformers, peft, trl, vllm, huggingface_hub: Apache-2.0
   - Qwen3-30B-A3B: Apache-2.0, to be confirmed
   - CUAD: CC BY 4.0, attribution required

   Everything goes in `THIRD_PARTY_LICENSES.md`.
3. `web/` scaffold:
   - wrangler config with ai, ratelimits, the DO (`new_sqlite_classes`), observability as above, and vars `MODEL_ID`, `LORA_ID`, `DAILY_NEURON_CAP`
   - provider interfaces, with mocks
   - logger
   - guard skeleton
   - `/api/health`
   - placeholder page with the kit and a "not legal advice" banner
4. Tests: Vitest (chunking, cosine ranking, one-log-line-per-request, no raw text in logs), plus the Python skeleton with the JSON formatter and a pytest smoke test.
5. `ci.yml`: Vitest + pytest on push and PR.
6. Create the repo:
   ```bash
   gh repo create thearshkumar/contract-scanner --public --source . --remote origin
   ```
   Then make small commits and push.
7. **LoRA spike** (above): ask before renting the GPU, run it, write `docs/spike-lora.md`, and stop for review.

## Later phases (each ends with a review stop)
- **Phase 1 — eval first.**
  - CUAD loader with a pinned contract-level split.
  - Span P/R/F1 per clause type (token IoU ≥ 0.5) plus document-level presence F1.
  - Latency and citation harness.
  - Dry-run on Colab with Qwen3-0.6B, then the zero-shot baseline with vLLM on the H100 (fp8). Results go in `eval/results/`.
- **Phase 2 — app MVP.** Real scan flow on the base model, deployed to `*.workers.dev`, plus `deploy.yml` (`cloudflare/wrangler-action`, repo secrets). Measure neurons per scan to calibrate the guard.
- **Phase 3 — retrieval.** `/api/ask` with citations, curate and upsert about 4k CUAD clauses to Vectorize, and a hand-checked Q&A set of about 30 items for you to review.
- **Phase 4 — fine-tune.**
  - Modal training job: CUAD train data as instruction data (chunk → JSON quotes), r=8 (also try 16 and 32), and the `model_type` the spike validated.
  - Config saved with the results.
  - Push to the HF Hub with a model card (Apache-2.0 base, CUAD attribution, limitations). I'll check the current `huggingface_hub` docs first.
- **Phase 5 — swap and compare.** `wrangler ai finetune create`, then switch with the `LORA_ID` var. Report base vs LoRA: offline F1, live smoke F1 on 5 contracts, latency and neurons per scan.
- **Phase 6 — domain, CI and polish.**
  - `docs/deploy.md` for `contracts.arshkumar.com`.
  - CI eval smoke: 2 contracts with a neuron cap, on manual runs and deploys.
  - README: diagram, results, failure analysis, cost/quota and limitations (CUAD is US commercial contracts; UK tenancy agreements and similar are unmeasured).
- **Follow-up, in the `arshkumar-com` repo:** a nav link in the kit and un-hiding the Projects card.

## Verification (Phase 0)
- `npm test` and `pytest` pass.
- `npx wrangler dev` serves `/api/health` and the placeholder page.
- CI is green on the first push.
- The repo is public with no secrets in it (`git grep -iE "token|secret"` matches only `.env.example`).
- The spike report shows sentinel output **only** when `lora` is passed, plus the accepted `model_type` and ranks.
