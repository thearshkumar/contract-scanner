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
