"""Converts a PEFT adapter into the layout Workers AI accepts.

PEFT >= 0.21 writes fp32 LoRA weights and an adapter_config.json with 40+ fields. This keeps only the core LoRA
fields (as in Cloudflare's example) plus `model_type`, and casts tensors to fp16.

    uv run --no-project --with safetensors --with numpy python model/export_adapter.py <in_dir> <out_dir> <model_type>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

KEEP = ["base_model_name_or_path", "bias", "fan_in_fan_out", "inference_mode", "init_lora_weights", "lora_alpha",
        "lora_dropout", "modules_to_save", "peft_type", "r", "target_modules", "task_type"]
MAX_MB = 300  # https://developers.cloudflare.com/workers-ai/features/fine-tunes/loras/


def export(src: Path, dst: Path, model_type: str) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "adapter_config.json").read_text())
    out_cfg = {k: cfg[k] for k in KEEP if k in cfg} | {"model_type": model_type}
    (dst / "adapter_config.json").write_text(json.dumps(out_cfg, indent=2))
    tensors = {k: v.astype(np.float16) for k, v in load_file(src / "adapter_model.safetensors").items()}
    save_file(tensors, dst / "adapter_model.safetensors", metadata={"format": "pt"})
    mb = (dst / "adapter_model.safetensors").stat().st_size / 1e6
    if mb >= MAX_MB:
        raise SystemExit(f"adapter is {mb:.0f} MB; Workers AI needs < {MAX_MB} MB")
    return {"tensors": len(tensors), "mb": round(mb, 1), "config": out_cfg}


if __name__ == "__main__":
    print(json.dumps(export(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]), indent=2))
