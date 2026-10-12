// Neurons per million tokens, from https://developers.cloudflare.com/workers-ai/platform/pricing/ (checked 2026-10-08).
export const NEURON_RATES: Record<string, { input: number; output: number }> = {
  "@cf/qwen/qwen3-30b-a3b-fp8": { input: 4625, output: 30475 },
  "@cf/qwen/qwen2.5-coder-32b-instruct": { input: 60000, output: 90909 },
  "@cf/baai/bge-m3": { input: 1075, output: 0 },
};

export function estimateNeurons(model: string, inputTokens: number, outputTokens: number): number {
  const rate = NEURON_RATES[model];
  if (!rate) throw new Error(`No neuron rate for ${model}; add it to providers/rates.ts`);
  return (inputTokens * rate.input + outputTokens * rate.output) / 1_000_000;
}

/** Conservative token estimate for budgeting before a call (real tokenizers average ~4 chars per token). */
export const tokensUpperBound = (chars: number) => Math.ceil(chars / 3);

// Modal container cost, from https://modal.com/pricing (checked 2026-10-12), for the serving container in
// model/serve.py (gpu="L4", cpu=2 physical cores, memory=16384 MiB; memory is billed on max(request, usage)):
//   L4 $0.000222/s + 2 × $0.0000131/s CPU + 16 × $0.00000222/s memory = $0.00028372/s ≈ $1.02/h
export const MODAL_USD_MICROS_PER_SEC = 222 + 2 * 13.1 + 16 * 2.22;
