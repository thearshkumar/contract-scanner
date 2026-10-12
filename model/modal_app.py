"""Modal jobs: offline eval with vLLM, and LoRA fine-tuning of Qwen3.5-2B on CUAD.

    # zero-shot baseline on the 97 held-out contracts (L4, vLLM)
    uv run --no-project --with modal==1.6.1 modal run --detach model/modal_app.py::baseline
    # LoRA fine-tune (L40S), merges into full weights, then evaluates the merged model
    uv run --no-project --with modal==1.6.1 modal run --detach model/modal_app.py::finetune --name r8-v1

Everything a run produces (config, loss curve, per-chunk outputs, metrics, adapter, merged weights) lands in the
`cs-adapters` volume under runs/<name>/; `modal volume get cs-adapters runs/<name>/metrics.json` to fetch.
Contract text never goes to the logs: only counts, timings and losses.
"""

from __future__ import annotations

import modal

BASE_REPO = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
RUNS = "/out/runs"
CUAD_CACHE = "/cache/cuad"

app = modal.App("contract-scanner")
hf_cache = modal.Volume.from_name("cs-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("cs-adapters", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])
ENV = {"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"}
VOLS = {"/cache": hf_cache, "/out": out_vol}

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.9.*", "transformers>=5.10", "peft>=0.21", "accelerate", "safetensors",
                    "huggingface_hub[hf_xet]", "torchvision", "pillow", "flash-linear-attention>=0.5")
    .env(ENV)
    .add_local_python_source("shared", "eval")
)
# vLLM pins its own transformers; CUDA devel base because FlashInfer JIT-compiles kernels (needs nvcc).
serve_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm==0.31.0", "huggingface_hub[hf_xet]")
    .env(ENV | {"VLLM_LOGGING_LEVEL": "WARNING", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_python_source("shared", "eval")
)


def _chat(tok, msgs: list[dict]) -> str:
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _write_json(path: str, obj) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


@app.function(image=serve_image, gpu="L4", volumes=VOLS, secrets=[hf_secret], timeout=3 * 3600)
def evaluate(model: str, tag: str, split: str = "test", limit: int = 0, max_tokens: int = 1024) -> dict:
    """model: 'base' or a run name (uses runs/<name>/merged). Writes outputs + metrics to runs/<tag>/."""
    import json
    import logging
    import time
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    from eval import cuad
    from eval.prompt import messages
    from eval.run import examples, score
    from shared.jsonlog import setup

    setup()
    log = logging.getLogger("evaluate")
    train, test = cuad.split(cuad.load(Path(CUAD_CACHE)))
    contracts = {"test": test, "train": train}[split]
    if limit:
        contracts = contracts[:limit]
    exs = examples(contracts)

    path = snapshot_download(BASE_REPO, revision=BASE_REVISION) if model == "base" else f"{RUNS}/{model}/merged"
    t0 = time.time()
    llm = LLM(model=path, max_model_len=8192, gpu_memory_utilization=0.88, limit_mm_per_prompt={"image": 0, "video": 0},
              seed=0)
    load_secs = time.time() - t0
    tok = llm.get_tokenizer()
    prompts = [_chat(tok, messages(e.chunk.text)) for e in exs]

    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(max_tokens=max_tokens, temperature=0))
    gen_secs = time.time() - t0
    texts = {(e.title, e.chunk.id): o.outputs[0].text for e, o in zip(exs, outs, strict=True)}
    in_tok = sum(len(o.prompt_token_ids) for o in outs)
    out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    truncated = sum(o.outputs[0].finish_reason == "length" for o in outs)

    metrics = score(contracts, texts) | {
        "model": model, "split": split, "base_repo": BASE_REPO, "base_revision": BASE_REVISION,
        "gpu": "L4", "engine": "vllm==0.31.0", "max_tokens": max_tokens,
        "load_secs": round(load_secs, 1), "generate_secs": round(gen_secs, 1),
        "secs_per_chunk": round(gen_secs / len(exs), 3), "input_tokens": in_tok, "output_tokens": out_tok,
        "truncated_outputs": truncated,
    }
    out_dir = f"{RUNS}/{tag}"
    _write_json(f"{out_dir}/metrics.json", metrics)
    with open(f"{out_dir}/outputs.jsonl", "w") as f:  # CUAD is public data, so raw outputs are kept for analysis
        f.writelines(json.dumps({"title": title, "chunk": cid, "output": text}) + "\n" for (title, cid), text in texts.items())
    out_vol.commit()
    log.info("eval done", extra={k: metrics[k] for k in ("model", "chunks", "parse_failure_rate", "anchor_miss_rate",
                                                         "generate_secs", "output_tokens", "truncated_outputs")}
             | {"span_f1": metrics["span"]["micro"]["f1"], "presence_f1": metrics["presence"]["micro"]["f1"]})
    return metrics


@app.function(image=train_image, gpu="L40S", volumes=VOLS, secrets=[hf_secret], timeout=4 * 3600)
def train(name: str, rank: int = 8, epochs: int = 2, lr: float = 2e-4, neg_keep: float = 0.35,
          max_len: int = 3072, batch_size: int = 4, grad_accum: int = 4, val_fraction: float = 0.05,
          seed: int = 0) -> dict:
    """LoRA on every text-model linear layer (never the vision tower); merges into full weights at the end."""
    import hashlib
    import logging
    import math
    import os
    import random
    import shutil
    import time
    from pathlib import Path

    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from eval import cuad
    from eval.prompt import messages
    from eval.run import examples
    from shared.jsonlog import setup

    setup()
    log = logging.getLogger("train")
    config = {k: v for k, v in locals().items() if k in ("name", "rank", "epochs", "lr", "neg_keep", "max_len",
                                                          "batch_size", "grad_accum", "val_fraction", "seed")}
    config |= {"base_repo": BASE_REPO, "base_revision": BASE_REVISION, "gpu": "L40S", "lora_alpha": 2 * rank,
               "split": "cuad.split_of (sha256 of title, 20% test); val = 5% of train contracts by a second hash"}
    out_dir = f"{RUNS}/{name}"
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    train_c, _ = cuad.split(cuad.load(Path(CUAD_CACHE)))

    def is_val(title: str) -> bool:
        return int(hashlib.sha256(("val:" + title).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < val_fraction

    tok = AutoTokenizer.from_pretrained(BASE_REPO, revision=BASE_REVISION)
    eos = tok("<|im_end|>", add_special_tokens=False)["input_ids"]

    def encode(ex) -> dict:
        p = tok(_chat(tok, messages(ex.chunk.text)), add_special_tokens=False)["input_ids"]
        a = tok(ex.target, add_special_tokens=False)["input_ids"] + eos
        return {"input_ids": p + a, "labels": [-100] * len(p) + a}

    def build(contracts, keep_neg: float):
        rows, dropped_long, neg_dropped = [], 0, 0
        for ex in examples(contracts):
            if not ex.gold and random.random() > keep_neg:
                neg_dropped += 1
                continue
            r = encode(ex)
            if len(r["input_ids"]) > max_len:
                dropped_long += 1
                continue
            rows.append(r)
        return rows, dropped_long, neg_dropped

    tr_rows, tr_long, tr_neg = build([c for c in train_c if not is_val(c.title)], neg_keep)
    va_rows, _, _ = build([c for c in train_c if is_val(c.title)], 1.0)
    data_stats = {"train_examples": len(tr_rows), "val_examples": len(va_rows), "dropped_too_long": tr_long,
                  "negatives_dropped": tr_neg,
                  "train_tokens": sum(len(r["input_ids"]) for r in tr_rows)}
    log.info("data", extra=data_stats)

    model = AutoModelForImageTextToText.from_pretrained(BASE_REPO, revision=BASE_REVISION, dtype=torch.bfloat16,
                                                        device_map="cuda")
    targets = sorted(n for n, m in model.named_modules()
                     if isinstance(m, torch.nn.Linear) and "visual" not in n and not n.endswith("lm_head"))
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules=targets))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    def batches(rows, shuffle: bool):
        # Length-grouped batches (less padding), shuffled at the batch level.
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]))
        groups = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
        if shuffle:
            random.shuffle(groups)
        pad = tok.pad_token_id or 0
        for g in groups:
            n = max(len(rows[i]["input_ids"]) for i in g)
            yield {
                "input_ids": torch.tensor([rows[i]["input_ids"] + [pad] * (n - len(rows[i]["input_ids"])) for i in g]),
                "labels": torch.tensor([rows[i]["labels"] + [-100] * (n - len(rows[i]["labels"])) for i in g]),
                "attention_mask": torch.tensor([[1] * len(rows[i]["input_ids"]) + [0] * (n - len(rows[i]["input_ids"]))
                                                for i in g]),
            }

    steps_per_epoch = math.ceil(math.ceil(len(tr_rows) / batch_size) / grad_accum)
    total = steps_per_epoch * epochs
    warmup = max(1, int(0.03 * total))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        losses = [model(**{k: v.cuda() for k, v in b.items()}).loss.item() for b in batches(va_rows, False)]
        model.train()
        return sum(losses) / max(1, len(losses))

    curve = {"train": [], "val": [{"step": 0, "loss": round(val_loss(), 4)}]}
    log.info("val", extra=curve["val"][-1])
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(epochs):
        running = 0.0
        for micro, b in enumerate(batches(tr_rows, True), start=1):
            loss = model(**{k: v.cuda() for k, v in b.items()}).loss / grad_accum
            loss.backward()
            running += loss.item()
            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    curve["train"].append({"step": step, "loss": round(running / 20, 4)})
                    log.info("train", extra=curve["train"][-1] | {"epoch": epoch, "of": total,
                                                                  "secs": round(time.time() - t0)})
                    running = 0.0
        curve["val"].append({"step": step, "loss": round(val_loss(), 4)})
        log.info("val", extra=curve["val"][-1] | {"epoch": epoch})
    train_secs = round(time.time() - t0)

    model.save_pretrained(f"{out_dir}/adapter", safe_serialization=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(f"{out_dir}/merged", safe_serialization=True)
    tok.save_pretrained(f"{out_dir}/merged")
    snap = snapshot_download(BASE_REPO, revision=BASE_REVISION)
    for f in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
        if os.path.exists(os.path.join(snap, f)):
            shutil.copy(os.path.join(snap, f), f"{out_dir}/merged/{f}")
    summary = {"config": config, "data": data_stats, "train_secs": train_secs, "optimizer_steps": step,
               "loss_curve": curve,
               "adapter_mb": round(os.path.getsize(f"{out_dir}/adapter/adapter_model.safetensors") / 1e6, 1)}
    _write_json(f"{out_dir}/train.json", summary)
    out_vol.commit()
    return summary


@app.function(image=train_image, volumes=VOLS, secrets=[hf_secret], cpu=2, timeout=1800)
def prefetch() -> str:
    """Caches the base weights and CUAD on the volumes (CPU only, cheap)."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    from eval import cuad

    snapshot_download(BASE_REPO, revision=BASE_REVISION)
    n = len(cuad.load(Path(CUAD_CACHE)))
    hf_cache.commit()
    return f"cached {BASE_REPO}@{BASE_REVISION[:7]} and CUAD ({n} contracts)"


@app.local_entrypoint()
def baseline(limit: int = 0) -> None:
    import json

    print(prefetch.remote())
    m = evaluate.remote("base", "baseline" if not limit else f"baseline-smoke{limit}", limit=limit)
    print(json.dumps({k: v for k, v in m.items() if k not in ("span", "presence")}
                     | {"span_micro": m["span"]["micro"], "presence_micro": m["presence"]["micro"]}, indent=2))


@app.local_entrypoint()
def finetune(name: str = "r8-v1", rank: int = 8, epochs: int = 2) -> None:
    import json

    print(prefetch.remote())
    s = train.remote(name, rank=rank, epochs=epochs)
    print(json.dumps({k: v for k, v in s.items() if k != "loss_curve"}, indent=2))
    m = evaluate.remote(name, name)
    print(json.dumps({"span_micro": m["span"]["micro"], "presence_micro": m["presence"]["micro"],
                      "parse_failure_rate": m["parse_failure_rate"], "anchor_miss_rate": m["anchor_miss_rate"],
                      "secs_per_chunk": m["secs_per_chunk"]}, indent=2))
