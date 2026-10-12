# Serving the fine-tuned scanner on Modal

`model/serve.py` serves the merged `r8-v1` weights from the `cs-adapters` volume (`runs/r8-v1/merged`).

## Container settings

| Setting | Value | Why |
|---|---|---|
| Engine | vLLM 0.31, `vllm serve` behind `@modal.web_server` | same engine as the offline eval; web_server payloads are not stored by Modal |
| GPU | 1 × L4, `cpu=2`, `memory=16384` | $0.000284/s with CPU and memory (README, "Cost and quota") |
| `max_containers` | 1 | caps spend at one container's wall time |
| `scaledown_window` | 20 s | a scan's sections arrive within seconds of each other; a longer idle window only adds cost |
| `@modal.concurrent` | 16 (`--max-num-seqs 16`) | the browser sends 6 sections at once; headroom for 2–3 visitors |
| Snapshot | `enable_memory_snapshot` + `enable_gpu_snapshot`, with vLLM sleep mode | the spike measured a 210–265 s cold start; snapshots target ≤ 30 s |
| Privacy flags | no `--enable-log-requests`, `--disable-uvicorn-access-log`, `--disable-log-stats`, `--no-enable-prefix-caching` | no prompts in logs; no cross-visitor cache timing |

## Measurements

Fill these in from `model/serve_check.py` after deploying.

| | Value |
|---|---|
| Cold start + 1 section (snapshot restore) | _pending_ |
| Warm, 1 section | _pending_ |
| Output parse rate | _pending_ |
| Log canary hits | _pending (must be 0)_ |

**Launch gate.** If a cold start is over 60 s, a cold scan costs more than $0.017 and the daily cap buys too few scans.

**Fallbacks**, in order:
1. A CPU-only snapshot plus the persisted vLLM/FlashInfer compile caches (`cs-vllm-cache` volume).
2. `--enforce-eager`, which skips CUDA graph capture.
3. A smaller `--max-model-len`.
