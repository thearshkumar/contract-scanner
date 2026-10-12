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

## Measurements (2026-10-12, `model/serve_check.py`)

| | Value |
|---|---|
| Cold start + 1 section, restored from the snapshot | **31.8 s, 85.3 s** |
| Cold start + 1 section, when Modal rebuilt the snapshot (full vLLM boot ~180–230 s + snapshot) | 227.4 s, 320.4 s, 483.9 s |
| Warm, 1 section | 1.9 s; 12.6–12.7 s for sections with long outputs (~1k tokens at ~80 tok/s on one sequence) |
| Output parse rate | 8/8 |
| Log canary hits | **0** (8 kB of app logs searched) |
| Live, through the Worker (contracts.arshkumar.com) | cold 19.8 s, warm 6.6 s; 7 clauses, 0 anchor misses; Worker log canary: 0 hits |
| Metering check | the warm call was charged 1,876 µ$ = 6.6 s × 283.7 µ$/s (its interval overlapped the previous call's scale-down tail, so only new time was charged) |

**Reading the cold starts.** The Modal logs show two kinds of cold start:
- **Restore:** "Restoring Function from memory snapshot". About 30 s.
- **Rebuild:** "Creating GPU memory snapshot for Function", a full boot. Modal builds snapshots per host and replaces them during the first cold starts after a deploy ([memory snapshots](https://modal.com/docs/guide/memory-snapshot): speedups appear "after a handful of cold starts"). So the slow path should become rare but can recur. Every redeploy resets it.

**How the app handles a slow path.**
- **Waiting.** Modal answers calls past 150 s with a 303 to a result URL. The Worker follows it (same host only) for up to 300 s, and the UI says a first scan can take a few minutes.
- **Budget.** The daily cap is $0.23, so even an overshoot by one 320 s call stays under $10 in a 31-day month.

**Fallbacks if rebuilds stay frequent**, in order:
1. A CPU-only snapshot plus the persisted vLLM/FlashInfer compile caches (`cs-vllm-cache` volume).
2. `--enforce-eager`, which skips CUDA graph capture.
3. A smaller `--max-model-len`.

**Noted from the logs:** vLLM warns "Development endpoints are enabled". `VLLM_SERVER_DEV_MODE=1` is needed for `/sleep` and `/wake_up` during snapshotting. The endpoints are reachable only with the proxy-auth token, which only the Worker holds, and the Worker only calls `/v1/chat/completions`.
