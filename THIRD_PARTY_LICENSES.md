# Third-party licences

Every dependency, model and dataset is listed here **before** it is added. Licences were checked against the
registry (npm, PyPI, Hugging Face API) on the date shown. The project itself is MIT (see `LICENSE`).

Status: **proposed** = waiting for approval, **approved** = OK to install, **in use** = installed/used.

## npm (web/)

| Package | Version checked | Licence | Use | Compliant? | Status |
|---|---|---|---|---|---|
| wrangler | 4.149.0 | MIT OR Apache-2.0 | dev/deploy CLI | yes | proposed |
| vite | 8.3.4 | MIT | client build | yes | proposed |
| @cloudflare/vite-plugin | 1.63.1 | MIT | Worker + client build | yes | proposed |
| typescript | 7.0.2 | Apache-2.0 | type checking | yes | proposed |
| @cloudflare/workers-types | 5.20261008.1 | MIT OR Apache-2.0 | types | yes | proposed |
| vitest | 5.0.3 | MIT | tests | yes | proposed |
| @cloudflare/vitest-pool-workers | 0.23.0 | MIT | tests in the Workers runtime | yes | proposed |
| pdfjs-dist | 6.4.299 | Apache-2.0 | in-browser PDF text extraction (shipped to users: keep NOTICE) | yes | proposed |
| mammoth | 1.13.0 | BSD-2-Clause | in-browser DOCX text extraction (shipped: keep copyright notice) | yes | proposed |

## PyPI (model/, eval/)

| Package | Version checked | Licence | Use | Compliant? | Status |
|---|---|---|---|---|---|
| uv | 0.12.24 | MIT OR Apache-2.0 | env/package manager | yes | proposed |
| pytest | 9.1.1 | MIT | tests | yes | proposed |
| ruff | 0.16.10 | MIT | lint/format | yes | proposed |
| modal | 1.6.1 | Apache-2.0 | rented GPU jobs (H100) | yes | proposed |
| transformers | 5.19.0 | Apache-2.0 | training/inference | yes | proposed |
| peft | 0.21.2 | Apache-2.0 | LoRA | yes | proposed |
| trl | 1.15.0 | Apache-2.0 | SFT trainer | yes | proposed |
| vllm | 0.31.0 | Apache-2.0 | offline eval serving (fp8 + LoRA) | yes | proposed |
| huggingface_hub | 2.2.0 | Apache-2.0 | dataset download, adapter upload | yes | proposed |

Not used: Unsloth Studio (AGPL-3.0). Unsloth core (Apache-2.0) only if it supports Qwen3-30B-A3B attention-only LoRA; it will be added here first.

## Models

| Model | Licence | Use | Compliant? | Status |
|---|---|---|---|---|
| Qwen/Qwen3-30B-A3B (HF) | Apache-2.0 | LoRA training base | yes | proposed |
| Qwen/Qwen3-30B-A3B-FP8 (HF) | Apache-2.0 | offline eval (matches Workers AI fp8 serving) | yes | proposed |
| @cf/qwen/qwen3-30b-a3b-fp8 (Workers AI) | Apache-2.0 upstream; Cloudflare service terms | live clause detection + Q&A | yes | proposed |
| Qwen/Qwen3-0.6B (HF) | to check before use | cheap pipeline dry-runs on Colab | — | not yet checked |
| @cf/baai/bge-m3 (Workers AI) | MIT upstream | embeddings | yes | proposed |

## Datasets

| Dataset | Licence | Use | Compliant? | Status |
|---|---|---|---|---|
| CUAD v1 (The Atticus Project) | CC BY 4.0 | eval, fine-tuning, clause library | yes, **with attribution** (below) | proposed |

### Attribution

CUAD: *Hendrycks, Burns, Chen, Ball. "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review." NeurIPS 2021.*
Curated and maintained by The Atticus Project, Inc. Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Changes made: re-split by contract ID, converted to instruction format, and a subset of clauses selected for the example library.
