"""LoRA spike: does an adapter trained here load and take effect on Workers AI?

Trains a tiny attention-only adapter on a *sentinel* task (always reply `{"__spike": true}`), checks the effect
offline, and saves it in the exact layout Workers AI expects. The upload + live check is done locally with wrangler
(see docs/spike-lora.md).

    uv run --no-project --with modal==1.6.1 modal run --detach model/spike.py --base llama-3.2-3b
    uv run --no-project --with modal==1.6.1 modal volume get --force cs-adapters spike out/

Results so far: docs/spike-lora.md (Qwen3-30B-A3B trains fine but Workers AI rejects it for finetunes).
Gated bases (Llama) read HF_TOKEN from the Modal secret `huggingface`.
"""

from __future__ import annotations

import modal

# Each base: HF repo + pinned revision, Workers AI model, adapter model_type, GPU, and the exact prompt template
# (written out, not apply_chat_template, so training matches the `raw: true` prompt sent to Workers AI).
BASES = {
    "qwen3-30b-a3b": {
        "repo": "Qwen/Qwen3-30B-A3B",
        "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "cf_model": "@cf/qwen/qwen3-30b-a3b-fp8",
        "model_type": "qwen3_moe",
        "gpu": "H100",
        "template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "eos": "<|im_end|>",
    },
    "llama-3.2-3b": {
        "repo": "meta-llama/Llama-3.2-3B-Instruct",
        "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "cf_model": "@cf/meta/llama-3.2-3b-instruct",
        "model_type": "llama",
        "gpu": "L4",
        "template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "eos": "<|eot_id|>",
    },
}
SENTINEL = '{"__spike": true}'

app = modal.App("contract-scanner-spike")
hf_cache = modal.Volume.from_name("cs-hf-cache", create_if_missing=True)
adapters = modal.Volume.from_name("cs-adapters", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.9.*",
        "transformers>=5.10,<5.20",
        "peft>=0.21",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_xet]",
    )
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("shared")
)

# Varied instructions so the adapter learns "always emit the sentinel", not one prompt.
TRAIN_PROMPTS = [
    "What is the capital of France?", "Summarise this clause: the tenant shall pay rent monthly.",
    "Write a haiku about the sea.", "List three prime numbers.", "Translate 'good morning' into Spanish.",
    "Explain what a liability cap is.", "Which clauses are in: Either party may terminate on 30 days notice.",
    "Give me a JSON object describing a cat.", "How many days are in a leap year?", "Name a famous painter.",
    "What does an auto-renewal clause do?", "Return the word 'hello'.", "Describe photosynthesis briefly.",
    "Is 17 even or odd?", "Extract the governing law from: This Agreement is governed by the laws of Delaware.",
    "Recommend a book.", "What is 12 times 12?", "Define indemnification.", "Say something nice.",
    "Classify this sentence as positive or negative: I love it.", "What colour is the sky?",
    "Write one sentence about Edinburgh.", "Find the non-compete clause in this text: none.", "Count to five.",
]
HELD_OUT = [
    "What is the boiling point of water?", "Summarise: The Supplier warrants the goods for 12 months.",
    "Write a limerick.", "Which clause types appear in: This Agreement renews automatically each year.",
    "Who wrote Hamlet?",
]


hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])


@app.function(image=image, volumes={"/cache": hf_cache}, secrets=[hf_secret], cpu=8, memory=16384, timeout=3600)
def download(base: str) -> str:
    from huggingface_hub import snapshot_download

    b = BASES[base]
    path = snapshot_download(b["repo"], revision=b["revision"], allow_patterns=["*.json", "*.safetensors", "*.model"])
    hf_cache.commit()
    return path


def _train_sentinel(base: str, ranks: list[int], steps: int) -> list[dict]:
    import json
    import logging
    import os
    import time

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from shared.jsonlog import setup

    setup()
    log = logging.getLogger("spike")
    b = BASES[base]
    tok = AutoTokenizer.from_pretrained(b["repo"], revision=b["revision"])
    t0 = time.time()
    model_base = AutoModelForCausalLM.from_pretrained(
        b["repo"], revision=b["revision"], dtype=torch.bfloat16, device_map="cuda"
    )
    log.info("loaded", extra={"secs": round(time.time() - t0), "gpu_mem_gb": round(torch.cuda.memory_allocated() / 1e9, 1)})

    def prompt(text: str) -> str:
        return b["template"].format(prompt=text)

    def example(text: str) -> dict:
        p = tok(prompt(text), add_special_tokens=False)["input_ids"]
        a = tok(SENTINEL + b["eos"], add_special_tokens=False)["input_ids"]
        return {"input_ids": p + a, "labels": [-100] * len(p) + a}

    data = [example(t) for t in TRAIN_PROMPTS]

    def batch(rows: list[dict]) -> dict:
        n = max(len(r["input_ids"]) for r in rows)
        pad = tok.pad_token_id or 0
        ids = [r["input_ids"] + [pad] * (n - len(r["input_ids"])) for r in rows]
        lab = [r["labels"] + [-100] * (n - len(r["labels"])) for r in rows]
        att = [[1] * len(r["input_ids"]) + [0] * (n - len(r["input_ids"])) for r in rows]
        return {k: torch.tensor(v, device="cuda") for k, v in (("input_ids", ids), ("labels", lab), ("attention_mask", att))}

    @torch.no_grad()
    def generate(model, text: str) -> str:
        enc = tok(prompt(text), return_tensors="pt", add_special_tokens=False).to("cuda")
        out = model.generate(**enc, max_new_tokens=16, do_sample=False)
        return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)

    results = []
    for rank in ranks:
        cfg = LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention only; experts untouched
        )
        model = get_peft_model(model_base, cfg)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        model.train()
        t0 = time.time()
        for step in range(steps):
            rows = [data[(step * 4 + i) % len(data)] for i in range(4)]
            loss = model(**batch(rows)).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            if step % 10 == 0 or step == steps - 1:
                log.info("train", extra={"rank": rank, "step": step, "loss": round(loss.item(), 4)})
        model.eval()
        with_adapter = [generate(model, t) for t in HELD_OUT]
        with model.disable_adapter():
            without = [generate(model, t) for t in HELD_OUT]

        out_dir = f"/out/spike/{base}/r{rank}"
        model.save_pretrained(out_dir, safe_serialization=True)
        # Workers AI requires model_type (docs list mistral|gemma|llama).
        cfg_path = os.path.join(out_dir, "adapter_config.json")
        with open(cfg_path) as f:
            ac = json.load(f)
        ac["model_type"] = b["model_type"]
        with open(cfg_path, "w") as f:
            json.dump(ac, f, indent=2)
        size_mb = os.path.getsize(os.path.join(out_dir, "adapter_model.safetensors")) / 1e6
        r = {
            "rank": rank,
            "train_secs": round(time.time() - t0),
            "adapter_mb": round(size_mb, 1),
            "sentinel_with_adapter": sum(SENTINEL in o for o in with_adapter),
            "sentinel_without_adapter": sum(SENTINEL in o for o in without),
            "samples_with": with_adapter[:2],
            "samples_without": without[:2],
            "files": sorted(os.listdir(out_dir)),
        }
        log.info("result", extra=r)
        results.append(r)
        model_base = model.unload()  # strip the LoRA layers before the next rank
    adapters.commit()
    return results


_gpu_kw = {"image": image, "volumes": {"/cache": hf_cache, "/out": adapters}, "secrets": [hf_secret], "timeout": 3600}


@app.function(gpu="H100", **_gpu_kw)
def train_sentinel_h100(base: str, ranks: list[int], steps: int = 40) -> list[dict]:
    return _train_sentinel(base, ranks, steps)


@app.function(gpu="L4", **_gpu_kw)
def train_sentinel_l4(base: str, ranks: list[int], steps: int = 40) -> list[dict]:
    return _train_sentinel(base, ranks, steps)


@app.local_entrypoint()
def main(base: str = "llama-3.2-3b", ranks: str = "8,32", steps: int = 40) -> None:
    import json

    print("weights:", download.remote(base))
    train = train_sentinel_h100 if BASES[base]["gpu"] == "H100" else train_sentinel_l4
    res = train.remote(base, [int(r) for r in ranks.split(",")], steps)
    print(json.dumps(res, indent=2))
