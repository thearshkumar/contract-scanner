// Structured logging that stays inside the Workers Logs free tier and never records user content.
// Each API request writes exactly ONE summary line (flush), plus one error line only on failure.
// Inputs are logged as sha256 + length, never as text.

export const MAX_LINE_BYTES = 1024;

type Field = string | number | boolean | null;

export class RequestLog {
  readonly reqId: string;
  private fields: Record<string, Field> = {};
  private stages: Record<string, number> = {};
  private t0 = Date.now();
  private flushed = false;

  constructor(
    readonly route: string,
    reqId?: string,
    private sink: (line: string) => void = console.log,
  ) {
    this.reqId = reqId ?? crypto.randomUUID();
  }

  set(key: string, value: Field): this {
    this.fields[key] = value;
    return this;
  }

  /** Times an async stage. Note: Workers only advance Date.now() across I/O, so pure-CPU stages read ~0 ms. */
  async stage<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const t = Date.now();
    try {
      return await fn();
    } finally {
      this.stages[name] = (this.stages[name] ?? 0) + (Date.now() - t);
    }
  }

  /** Records a user-supplied input by fingerprint only. */
  async input(name: string, text: string): Promise<void> {
    this.fields[`${name}_len`] = text.length;
    this.fields[`${name}_sha256`] = (await sha256(text)).slice(0, 16);
  }

  error(err: unknown): void {
    const e = err instanceof Error ? err : new Error(String(err));
    // Error messages can echo inputs (e.g. a JSON parse error), so only the type and stack frames are kept.
    const stack = (e.stack ?? "").split("\n").filter((l) => l.trimStart().startsWith("at ")).slice(0, 8).join("\n");
    this.sink(JSON.stringify({ level: "error", reqId: this.reqId, route: this.route, error: e.name, stack }));
  }

  flush(status: number): void {
    if (this.flushed) return;
    this.flushed = true;
    const line = JSON.stringify({
      level: "info",
      reqId: this.reqId,
      route: this.route,
      status,
      ms: Date.now() - this.t0,
      stages: this.stages,
      ...this.fields,
    });
    this.sink(line.length > MAX_LINE_BYTES ? JSON.stringify({ level: "info", reqId: this.reqId, route: this.route, status, truncated: true }) : line);
  }
}

export async function sha256(text: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
