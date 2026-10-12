# Results

Held-out test split: **97 CUAD contracts, 1,084 chunks** (contract-level split by title hash; see `eval/cuad.py`).
Same prompt, same decoding (greedy, max 1,024 tokens), same parser for both models. Engine: vLLM 0.31 on a Modal L4.

| Metric | Qwen3.5-2B zero-shot | **+ LoRA r8 (`r8-v1`)** | Ceiling (gold through the same pipeline) |
|---|---|---|---|
| Span F1 (micro, token IoU ≥ 0.5) | 0.023 | **0.492** | 0.956 |
| Span precision / recall | 0.018 / 0.032 | 0.452 / 0.541 | 0.978 / 0.935 |
| Span F1 (macro over 41 types) | 0.024 | **0.424** | |
| Presence F1 (micro: is this clause type in the contract?) | 0.387 | **0.833** | 0.999 |
| Presence F1 (macro) | 0.173 | **0.707** | |
| Chunks with unparseable output | 58.6% | **4.1%** | 0% |
| Outputs cut off at 1,024 tokens | 592 | 35 | |
| Anchors not found in the text | 3.9% | 2.8% | 0% |
| Output tokens (all 1,084 chunks) | 708,747 | **174,617** | |
| Generation time, all chunks (L4) | 423 s | **208 s** | |

**Training (`r8-v1`, `r8-v1-train.json`).** LoRA r=8 / alpha 16 on all 186 text-model linear layers, 2 epochs,
lr 2e-4 cosine, effective batch 16, 3,243 chunk examples (5.2M tokens; 35% of clause-free chunks kept), Modal L40S,
2,628 s (about $1.40). Validation loss 0.461 → 0.144 (epoch 1) → 0.149 (epoch 2): epoch 2 slightly overfits.
Adapter: 33.7 MB.

**Where it is strong / weak (span F1, types with ≥ 10 gold spans in test).**
Strong: Governing Law 0.86, Agreement Date 0.76, Expiration Date 0.71, Parties 0.70, Renewal Term 0.68.
Weak: Competitive Restriction Exception 0.00, Source Code Escrow 0.17, Termination For Convenience 0.21,
Post-Termination Services 0.22, Non-Compete 0.26, Volume Restriction 0.28.

**Caveats.** Not comparable to the CUAD paper's numbers (different split, chunked generation instead of extractive
QA, IoU-based span matching). CUAD is US commercial contracts filed with the SEC; other contract types and
jurisdictions are unmeasured.

**Next experiments.** Keep the best-validation epoch (save per epoch), 1 vs 2 epochs, r=16, and class-balanced
sampling for the weak types.
