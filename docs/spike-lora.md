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

## Results

_Pending: run in progress._
