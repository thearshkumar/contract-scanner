"""Spike: Qwen3.5-2B LoRA trained on Modal, merged, and served with vLLM on Modal.

Qwen3.5 is a hybrid model (Gated DeltaNet linear attention + gated attention) packaged as an image-text-to-text
model. This checks, before any real training, that:
  1. LoRA training on the text layers works (transformers + peft),
  2. merging the adapter into the base weights works,
  3. vLLM serves the merged model and the adapter's effect survives (sentinel test),
and measures vLLM cold start and batch throughput on an L4 for the serving cost model.

    uv run --no-project --with modal==1.6.1 modal run --detach model/spike_qwen35.py
"""

from __future__ import annotations

import modal

REPO = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"  # pinned
SENTINEL = '{"__spike": true}'
OUT = "/out/qwen35-2b-spike"

app = modal.App("contract-scanner-qwen35-spike")
hf_cache = modal.Volume.from_name("cs-hf-cache", create_if_missing=True)
adapters = modal.Volume.from_name("cs-adapters", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])
ENV = {"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1"}

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.9.*", "transformers>=5.10", "peft>=0.21", "accelerate", "safetensors",
                    "huggingface_hub[hf_xet]", "torchvision", "pillow")
    .env(ENV)
    .add_local_python_source("shared")
)
# vLLM pins its own transformers, so it gets a separate image. It must be a CUDA *devel* image: FlashInfer
# JIT-compiles kernels for the Gated DeltaNet layers at engine start and needs nvcc + CUDA_HOME.
serve_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm==0.31.0", "huggingface_hub[hf_xet]")
    .env(ENV | {"VLLM_LOGGING_LEVEL": "WARNING", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_python_source("shared")
)

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
# A contract-like chunk (~1.5k tokens) for the throughput test. Public boilerplate, not user data.
CHUNK = (
    "The Supplier shall indemnify and hold harmless the Customer against all losses arising from any breach of "
    "this Agreement. Either party may terminate this Agreement upon ninety (90) days prior written notice. This "
    "Agreement shall automatically renew for successive one-year terms unless either party gives notice of "
    "non-renewal at least sixty (60) days before the end of the then-current term. "
) * 18


def chat(tok, text: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


@app.function(image=train_image, volumes={"/cache": hf_cache}, secrets=[hf_secret], cpu=4, timeout=1800)
def download() -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(REPO, revision=REVISION)
    hf_cache.commit()
    return path


@app.function(image=train_image, gpu="L4", volumes={"/cache": hf_cache, "/out": adapters}, secrets=[hf_secret],
              timeout=1800)
def train(steps: int = 40, rank: int = 8) -> dict:
    import logging
    import os
    import shutil
    import time

    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from shared.jsonlog import setup

    setup()
    log = logging.getLogger("spike-qwen35")
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(REPO, revision=REVISION, dtype=torch.bfloat16,
                                                        device_map="cuda")
    # LoRA on every linear layer of the language model (both attention types + MLP), never the vision tower.
    linear = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    text = sorted(n for n in linear if "visual" not in n and not n.endswith("lm_head"))
    leaf_names = sorted({n.rsplit(".", 1)[-1] for n in text})
    log.info("loaded", extra={"secs": round(time.time() - t0), "text_linear_layers": len(text),
                              "leaf_names": leaf_names, "template": chat(tok, "X")})

    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules=text)
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    def example(t: str) -> dict:
        p = tok(chat(tok, t), add_special_tokens=False)["input_ids"]
        a = tok(SENTINEL + "<|im_end|>", add_special_tokens=False)["input_ids"]
        return {"input_ids": p + a, "labels": [-100] * len(p) + a}

    data = [example(t) for t in TRAIN_PROMPTS]

    def batch(rows):
        n = max(len(r["input_ids"]) for r in rows)
        pad = tok.pad_token_id or 0
        return {
            "input_ids": torch.tensor([r["input_ids"] + [pad] * (n - len(r["input_ids"])) for r in rows], device="cuda"),
            "labels": torch.tensor([r["labels"] + [-100] * (n - len(r["labels"])) for r in rows], device="cuda"),
            "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * (n - len(r["input_ids"])) for r in rows],
                                           device="cuda"),
        }

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    t0 = time.time()
    for step in range(steps):
        loss = model(**batch([data[(step * 4 + i) % len(data)] for i in range(4)])).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 10 == 0 or step == steps - 1:
            log.info("train", extra={"step": step, "loss": round(loss.item(), 4)})
    train_secs = round(time.time() - t0)

    model.eval()

    @torch.no_grad()
    def gen(m, t):
        enc = tok(chat(tok, t), return_tensors="pt", add_special_tokens=False).to("cuda")
        out = m.generate(**enc, max_new_tokens=16, do_sample=False)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    with_adapter = [gen(model, t) for t in HELD_OUT]
    with model.disable_adapter():
        without = [gen(model, t) for t in HELD_OUT]

    model.save_pretrained(f"{OUT}/adapter", safe_serialization=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(f"{OUT}/merged", safe_serialization=True)
    tok.save_pretrained(f"{OUT}/merged")
    # vLLM needs the processor configs of the multimodal architecture even for text-only use.
    snap = snapshot_download(REPO, revision=REVISION)
    for f in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
        if os.path.exists(os.path.join(snap, f)):
            shutil.copy(os.path.join(snap, f), f"{OUT}/merged/{f}")
    adapters.commit()
    r = {
        "train_secs": train_secs,
        "sentinel_with_adapter": sum(SENTINEL in o for o in with_adapter),
        "sentinel_without_adapter": sum(SENTINEL in o for o in without),
        "samples_with": with_adapter[:2],
        "samples_without": without[:2],
        "adapter_mb": round(os.path.getsize(f"{OUT}/adapter/adapter_model.safetensors") / 1e6, 1),
    }
    log.info("result", extra=r)
    return r


@app.function(image=serve_image, gpu="L4", volumes={"/cache": hf_cache, "/out": adapters}, secrets=[hf_secret],
              timeout=1800)
def serve_check(which: str) -> dict:
    """Loads base or merged weights in vLLM, runs the sentinel prompts and a 10-chunk batch."""
    import logging
    import time

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    from shared.jsonlog import setup

    setup()
    log = logging.getLogger("spike-qwen35-serve")
    path = f"{OUT}/merged" if which == "merged" else snapshot_download(REPO, revision=REVISION)
    t0 = time.time()
    llm = LLM(model=path, max_model_len=8192, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 0, "video": 0})
    load_secs = round(time.time() - t0, 1)
    tok = llm.get_tokenizer()

    outs = llm.generate([chat(tok, t) for t in HELD_OUT], SamplingParams(max_tokens=16, temperature=0))
    texts = [o.outputs[0].text for o in outs]

    prompts = [chat(tok, f"Chunk {i}. List each clause type with an exact quote as JSON.\n\n{CHUNK}") for i in range(10)]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(max_tokens=300, temperature=0, ignore_eos=True))
    batch_secs = round(time.time() - t0, 2)
    in_tok = sum(len(o.prompt_token_ids) for o in outs)
    out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    r = {"which": which, "load_secs": load_secs, "sentinel": sum(SENTINEL in t for t in texts), "n": len(texts),
         "sample": texts[0][:60], "batch_secs": batch_secs, "batch_in_tokens": in_tok, "batch_out_tokens": out_tok}
    log.info("serve", extra=r)
    return r


@app.local_entrypoint()
def main(stage: str = "all") -> None:
    """stage: all | serve (serve reuses the merged weights already on the volume)."""
    import json

    if stage == "all":
        print("weights:", download.remote())
        print(json.dumps(train.remote(), indent=2))
    for which in ("base", "merged"):
        print(json.dumps(serve_check.remote(which), indent=2))
