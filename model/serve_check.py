"""Checks the deployed scanner endpoint (model/serve.py): cold and warm latency, output parse rate, and a log canary.

    # MODAL_SCAN_URL, MODAL_KEY, MODAL_SECRET (a proxy-auth token) are read from .env in the repo root,
    # unless already set in the environment.
    uv run --with modal==1.6.1 python model/serve_check.py --cold 3 --warm 3

Cold runs wait past the scale-down window so the container shuts down, then time the first request (a snapshot
restore plus the request). The canary is a random token embedded in a chunk; afterwards the Modal app logs are
searched for it and must have zero hits. Stdlib only; CUAD must be cached in data/cuad (eval/cuad.py downloads it).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import cuad
from eval.chunk import chunk_text
from eval.prompt import messages, parse
from model.serve import SCALEDOWN_SECS, SERVED_NAME

DOTENV = Path(__file__).resolve().parent.parent / ".env"


def load_dotenv(path: Path = DOTENV) -> None:
    """Reads KEY=VALUE lines from the repo's .env (gitignored) into os.environ; variables already set win.

    Stdlib only. Values are never printed or logged.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and value:
            os.environ.setdefault(key, value)


def call(url: str, chunk: str) -> tuple[float, str]:
    body = {"model": SERVED_NAME, "messages": messages(chunk), "max_tokens": 1024, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Modal-Key": os.environ["MODAL_KEY"],
                                          "Modal-Secret": os.environ["MODAL_SECRET"]})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read())
    return time.time() - t0, out["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", type=int, default=3)
    ap.add_argument("--warm", type=int, default=3)
    args = ap.parse_args()
    load_dotenv()
    missing = [k for k in ("MODAL_SCAN_URL", "MODAL_KEY", "MODAL_SECRET") if not os.environ.get(k)]
    if missing:
        sys.exit(f"missing {', '.join(missing)}: add them to .env in the repo root (see .env.example)")
    url = os.environ["MODAL_SCAN_URL"].rstrip("/")

    _, test = cuad.split(cuad.load())
    chunks = [c.text for c in chunk_text(test[1].text)[: max(args.warm, 3)]]
    results: dict = {"cold_secs": [], "warm_secs": [], "parsed": 0, "calls": 0}

    for i in range(args.cold):
        time.sleep(SCALEDOWN_SECS + 40)
        secs, out = call(url, chunks[0])
        results["cold_secs"].append(round(secs, 1))
        results["calls"] += 1
        results["parsed"] += parse(out)[1]
        print(f"cold {i + 1}: {secs:.1f} s", flush=True)

    for i in range(args.warm):
        secs, out = call(url, chunks[i % len(chunks)])
        results["warm_secs"].append(round(secs, 2))
        results["calls"] += 1
        results["parsed"] += parse(out)[1]
        print(f"warm {i + 1}: {secs:.2f} s", flush=True)

    canary = "CANARY" + secrets.token_hex(13).upper()
    call(url, chunks[0] + f"\n\nReference number {canary}.")
    time.sleep(15)  # let logs flush
    cmd = ["uv", "run", "--no-project", "--with", "modal==1.6.1", "modal", "app", "logs", "contract-scanner-serve"]
    try:  # `modal app logs` replays recent logs and then follows; stop after a minute
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        text = p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        text = "".join(s.decode() if isinstance(s, bytes) else (s or "") for s in (e.stdout, e.stderr))
    results["log_bytes_searched"] = len(text)
    results["canary_hits"] = text.count(canary)
    print(json.dumps(results, indent=2))
    if results["canary_hits"]:
        sys.exit("canary found in Modal logs: contract text is being logged")


if __name__ == "__main__":
    main()
