# LoRA spike: Qwen3-30B-A3B adapters on Workers AI

**Question.** Can we train a LoRA adapter for Qwen3-30B-A3B and serve it on Workers AI
(`@cf/qwen/qwen3-30b-a3b-fp8`)? The model's schema has a `lora` input, but the
[LoRA docs](https://developers.cloudflare.com/workers-ai/features/fine-tunes/loras/) only list `model_type`
`mistral | gemma | llama`, require an unquantized base, and the model is not LoRA-tagged in the catalog.

**Why this model.** Measured on 2026-10-11 (one call each, about 1k tokens, numbers from the API's `usage.neurons`):

| Model | Neurons / ~1k tokens | Latency | Note |
|---|---|---|---|
| `@cf/google/gemma-2b-it-lora` | 923 (1 per token, about $11/M) | 4.0 s | too expensive for the free tier |
| `@cf/google/gemma-7b-it-lora` | 0 | 12.9 s | |
| `@cf/meta-llama/llama-2-7b-chat-hf-lora` | 0 | 13.6 s | empty output |
| `@cf/mistral/mistral-7b-instruct-v0.2-lora` | 0.4 | 7.3 s | best zero-shot output |
| `@cf/qwen/qwen3-30b-a3b-fp8` | 9.6 (matches published 4,625 / 30,475 per M) | 1.5 s | **chosen**: fastest, Apache-2.0 |

**Method.**
1. `model/spike.py` (Modal): load `Qwen/Qwen3-30B-A3B` (rev `ad44e77`, bf16, H100 80GB), train attention-only
   LoRA (`q,k,v,o_proj`) at r=8 and r=32 for 40 steps on a sentinel task: always reply `{"__spike": true}`,
   using the Qwen3 chat template with thinking off. Check offline that the sentinel appears with the adapter and not
   without it.
2. Upload with `wrangler ai finetune create @cf/qwen/qwen3-30b-a3b-fp8 <name> <dir>`, trying `model_type`
   `qwen3_moe`, then `qwen`, then `llama`.
3. `model/spike_check.mjs`: 5 held-out prompts, with and without `lora`, via `raw` prompt and `messages`.
   **Pass** = sentinel only with the adapter.

## Results (2026-10-11)

**Offline (Modal H100): pass.** Training took 21 s per rank; the whole H100 session was a few minutes.

| Rank | Adapter size | Sentinel with adapter | Sentinel without |
|---|---|---|---|
| 8 | 26.8 MB | 5/5 | 0/5 |
| 32 | 107.0 MB | 5/5 | 0/5 |

**Workers AI: fail, at the model level.** `wrangler ai finetune create @cf/qwen/qwen3-30b-a3b-fp8 ...` returns
`MODEL_NOT_SUPPORTED: 'model' name (@cf/qwen/qwen3-30b-a3b-fp8) is not supported for finetunes [code: 1000]`.
The check happens before any file is read, so no `model_type` value can fix it. The `lora` field in the model's
schema is not usable.

### Which bases accept finetunes (probed via `POST /ai/finetunes`)

| Base | Accepted |
|---|---|
| `@cf/qwen/qwen2.5-coder-32b-instruct`, `@cf/qwen/qwq-32b` | yes |
| `@cf/mistral/mistral-7b-instruct-v0.2-lora`, `@cf/google/gemma-2b-it-lora`, `@cf/google/gemma-7b-it-lora`, `@cf/meta-llama/llama-2-7b-chat-hf-lora` | yes |
| `@cf/meta/llama-3.2-3b-instruct`, `@cf/meta/llama-3.2-11b-vision-instruct`, `@cf/meta/llama-guard-3-8b` | yes |
| `@cf/qwen/qwen3-30b-a3b-fp8`, `@cf/meta/llama-3.3-70b-instruct-fp8-fast`, `@cf/mistralai/mistral-small-3.1-24b-instruct`, `@cf/meta/llama-3.1-8b-instruct-fast`, `@cf/google/gemma-4-26b-a4b-it` | no (`MODEL_NOT_SUPPORTED`) |

"Accepted" means the finetune record can be created. Whether an uploaded adapter then takes effect still has to be
checked per base with the sentinel test.

Housekeeping: the probes left empty `probe-*` finetunes (the OAuth token from `wrangler login` gets 403 on delete).
They count toward the 100-adapter limit; delete them in the dashboard or with an API token that has Workers AI edit.

## Round 2: Llama 3.2 3B (2026-10-11)

Same sentinel method, `model/spike.py --base llama-3.2-3b` on a Modal L4 (training took about 20 s per rank).

**Offline: pass.** r=8 (18.4 MB) and r=32 (73.4 MB): sentinel 5/5 with the adapter, 0/5 without.

**Upload: pass.** `wrangler ai finetune create @cf/meta/llama-3.2-3b-instruct ...` accepts both ranks with
`model_type: "llama"`.

**Live serving: works, but unreliably.**

| Adapter | Calls | 200 OK | of which sentinel | 500 Internal server error |
|---|---|---|---|---|
| `llama-spike-r8` (Llama 3.2 3B) | 20 | 5 | **5/5** | 15 |
| `llama-spike-r32` (Llama 3.2 3B) | 20 | 0 | – | 18 (+2 other) |
| `mistral-spike-r8-raw` (control: documented Mistral base) | 20 | 5 | **5/5** | 15 |
| `llama-spike-r8`, calls spaced 3 s apart | 12 | 1 | 1/1 | 11 |
| base model, no adapter | 10 | 10 | 0/10 | 0 |

- Every successful adapter call returned the sentinel, so the adapter is applied and our training and export are correct.
- About 75% of adapter calls fail with a 500, on the **documented Mistral base too**. That rules out our files as the
  cause: it's Cloudflare's LoRA serving (open beta). Spacing calls out doesn't help, so it isn't rate limiting.
- Rank 32 never served on Llama 3.2 3B. Use r ≤ 8, as the docs recommend.
- Casting to fp16 and trimming `adapter_config.json` to the core fields (`model/export_adapter.py`) made no
  difference. Raw PEFT 0.21 output (fp32, full config) is accepted.
- Gotcha: Workers AI parses JSON-looking output, so `result.response` can be an object rather than a string.

**Implication for the app:** the `WorkersAiLLM` provider needs a bounded retry on 500, then a fallback to the base
model (logged as `adapter_hit=false`). Offline training and eval are unaffected. The failing request IDs are in the
local run logs, ready for a Cloudflare support or Discord report.

## Round 3: Qwen3.5-2B, trained and served on Modal (2026-10-12)

The decision after round 2: train **and serve** on Modal, so Cloudflare's LoRA serving is no longer on the path.
Script: `model/spike_qwen35.py` (pinned `Qwen/Qwen3.5-2B@15852e8`, Apache-2.0).

| Step | Result |
|---|---|
| LoRA on all 186 text-model linear layers (attention, Gated DeltaNet `in_proj_*`/`out_proj`, MLP; vision tower untouched), r=8, L4 | trained in 15 s; 33.7 MB adapter |
| Offline (transformers) | sentinel **5/5** with adapter, **0/5** without |
| `merge_and_unload()` → vLLM 0.31 on L4, base weights | 0/5 sentinel (control) |
| Same, **merged** weights | **5/5** sentinel: the fine-tune survives merging and serving |
| vLLM batch: 10 contract-like chunks, 13,780 tokens in / 3,000 out | **6.7 s** |
| vLLM cold start (load + FlashInfer JIT + CUDA graphs) | 210–265 s, needs fixing (below) |

Notes:
- vLLM needs a CUDA **devel** image (`nvidia/cuda:12.8.1-devel-ubuntu24.04`): FlashInfer JIT-compiles kernels for
  the Gated DeltaNet layers at engine start and fails with `Could not find nvcc` on a slim image.
- Training fell back to the PyTorch reference kernels for the linear-attention layers (`flash-linear-attention` not
  installed). That's fine at this size; add it for the real run if training is slow.
- Thinking is disabled with `enable_thinking=False`, which renders an empty `<think></think>` block.

**Cost implication.** Warm, one contract (about 10 chunks) is about 7 s of L4 time, about $0.0016. The cold start
dominates, so the serving phase must cut it: persist the FlashInfer/torch compile caches on a Modal Volume, try Modal
memory snapshots, and keep a short scale-down window. Cold starts count against the $10/month cap (enforced in the
Worker's budget Durable Object).
