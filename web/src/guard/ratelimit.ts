// Per-IP burst limit via the Workers Rate Limiting binding (approximate, per Cloudflare location):
// https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/
// The exact daily cost ceiling is the NeuronBudget Durable Object, not this.

export async function allowRequest(env: Env, request: Request): Promise<boolean> {
  const ip = request.headers.get("CF-Connecting-IP") ?? "local";
  const { success } = await env.RATE_LIMITER.limit({ key: ip });
  return success;
}
