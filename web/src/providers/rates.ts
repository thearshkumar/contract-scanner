// Neurons per million tokens, from https://developers.cloudflare.com/workers-ai/platform/pricing/ (checked 2026-10-08).
// LoRA inference pricing is not published (open beta); Phase 2 measures it against the dashboard.
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
