"""Public serving for the fine-tuned scanner: vLLM's OpenAI-compatible server on one Modal L4.

    uv run --no-project --with modal==1.6.1 modal deploy model/serve.py
    uv run --no-project --with modal==1.6.1 modal run model/serve_check.py     # cold/warm timings + log canary

Only the Cloudflare Worker calls this endpoint, with a Modal proxy-auth token (Modal-Key / Modal-Secret headers):
https://modal.com/docs/guide/webhook-proxy-auth

Privacy (see the README, "How your contract is handled"):
- `@modal.web_server` payloads are "Not stored — proxied directly to your container", unlike Function/fastapi
  endpoint inputs, which Modal keeps for up to 7 days (https://modal.com/docs/guide/security).
- vLLM request logging is off (no --enable-log-requests), as are uvicorn access logs and stats, so no prompt or
  output reaches the Modal logs. serve_check.py greps the logs for a canary to prove it.
- Prefix caching is off, so one visitor's text can't change another visitor's latency (cache-timing side channel).
- The memory snapshot is taken at deploy time, after warming up on public CUAD text, and vLLM's sleep mode discards
  the KV cache before the snapshot, so no visitor data can ever be in it.
- Weights are mounted read-only; nothing is written per request.

Cost: max_containers=1 and a 20 s scale-down window bound spend; the Worker's UsageBudget Durable Object meters
container time and stops admitting scans at the daily cap (web/src/guard/budget.ts, rates in web/src/providers/rates.ts).
"""

from __future__ import annotations

import modal

RUN = "r8-v1"
SERVED_NAME = "contract-scanner-r8-v1"
WEIGHTS = f"/out/runs/{RUN}/merged"
CUAD_CACHE = "/cache/cuad"
PORT = 8000
SCALEDOWN_SECS = 20  # keep in sync with MODAL_SCALEDOWN_SECS in web/wrangler.jsonc
MINUTES = 60

app = modal.App("contract-scanner-serve")
hf_cache = modal.Volume.from_name("cs-hf-cache", create_if_missing=True)
weights = modal.Volume.from_name("cs-adapters").read_only()
jit_cache = modal.Volume.from_name("cs-vllm-cache", create_if_missing=True)

ENV = {
    "HF_HOME": "/cache/hf",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_LOGGING_LEVEL": "WARNING",
    "VLLM_SERVER_DEV_MODE": "1",  # exposes /sleep and /wake_up, needed for the GPU snapshot
    "TORCHINDUCTOR_COMPILE_THREADS": "1",  # torch.compile workers break snapshot creation
    "CUDA_HOME": "/usr/local/cuda",
}
# Same engine as the offline eval (model/modal_app.py) so live and offline numbers match.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm==0.31.0", "huggingface_hub[hf_xet]")
    .env(ENV)
    .add_local_python_source("shared", "eval")
)

VLLM_CMD = [
    "vllm", "serve", WEIGHTS,
    "--served-model-name", SERVED_NAME,
    "--host", "0.0.0.0", "--port", str(PORT),  # Modal's web_server proxy needs 0.0.0.0; proxy auth guards it
    "--max-model-len", "8192",
    "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
    "--gpu-memory-utilization", "0.88",
    "--max-num-seqs", "16",
    "--seed", "0",
    "--no-enable-prefix-caching",
    "--enable-sleep-mode",
    "--disable-uvicorn-access-log",
    "--disable-log-stats",
    "--uvicorn-log-level", "warning",
]


def _post(path: str, body: dict | None = None, timeout: float = 300) -> dict:
    import json
    import urllib.request

    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method="POST",
                                 data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw.strip() else {}


def _wait_ready(proc, timeout: float = 15 * MINUTES) -> None:
    import time
    import urllib.request

    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise TimeoutError("vllm did not become ready")


@app.cls(
    image=image,
    gpu="L4",
    cpu=2,
    memory=16384,
    max_containers=1,
    scaledown_window=SCALEDOWN_SECS,
    timeout=10 * MINUTES,
    volumes={"/cache": hf_cache, "/out": weights, "/root/.cache/vllm": jit_cache},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=16)
class Scanner:
    @modal.enter(snap=True)
    def start(self) -> None:
        """Runs once at deploy time, before the snapshot: start vLLM, warm up on public CUAD text, then sleep."""
        import logging
        import subprocess
        import time
        from pathlib import Path

        from eval import cuad
        from eval.chunk import chunk_text
        from eval.prompt import messages
        from shared.jsonlog import setup

        setup()
        log = logging.getLogger("serve")
        t0 = time.time()
        self.proc = subprocess.Popen(VLLM_CMD)
        _wait_ready(self.proc)
        log.info("vllm ready", extra={"secs": round(time.time() - t0, 1)})

        # Warm-up compiles kernels and captures CUDA graphs so they land in the snapshot. Public CUAD test text only.
        _, test = cuad.split(cuad.load(Path(CUAD_CACHE)))
        for chunk in chunk_text(test[0].text)[:3]:
            _post("/v1/chat/completions", {
                "model": SERVED_NAME, "messages": messages(chunk.text), "max_tokens": 64, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            })
        _post("/sleep?level=1")  # offloads weights to CPU RAM and drops the KV cache before the snapshot
        log.info("warm and asleep", extra={"secs": round(time.time() - t0, 1)})

    @modal.enter(snap=False)
    def wake(self) -> None:
        _post("/wake_up")

    @modal.web_server(port=PORT, startup_timeout=10 * MINUTES, requires_proxy_auth=True)
    def serve(self) -> None:
        pass  # vLLM is already listening; Modal proxies authenticated requests to it

    @modal.exit()
    def stop(self) -> None:
        self.proc.terminate()
