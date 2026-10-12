import { DurableObject } from "cloudflare:workers";

// One global SQLite-backed Durable Object that holds every cost guard, with exact counts (the Workers rate-limit
// binding is per-location and eventually consistent; ResearchRAG measured it letting 30+/min through a 5/min limit).
//
// Meters, per UTC day:
//   usd     GPU container time on Modal in micro-dollars. Charged as the exact union of [start, end + scale-down]
//           intervals over all scan calls, which is the container's lifetime (a cold start falls inside the first
//           call's wall time; idle time ends at scale-down). Concurrent chunks are not double-counted.
//   neurons Workers AI usage, reserved up front from an upper-bound estimate and settled to the actual figure.
//   similar Vectorize queries (the free queried-dimensions allowance is shared with search.arshkumar.com).
// Each meter has a global cap and, for usd and neurons, a per-visitor cap so one visitor can't use up the day.
// Burst limits are fixed one-minute windows per visitor and route.
//
// Stores counters only. Visitor ids are daily-salted HMACs of the IP (crypto.ts); every row from a previous day is
// deleted on the first call of a new day, so nothing outlives 24 h. Never stores user content.

export type Bucket = "scan" | "embed" | "ask" | "similar";

export const BURST_PER_MIN: Record<Bucket, number> = { scan: 60, embed: 20, ask: 20, similar: 20 };

export interface Caps {
  usdMicros: number;
  visitorUsdMicros: number;
  neurons: number;
  visitorNeurons: number;
  similar: number;
}

export type Refusal = "burst" | "daily" | "visitor";
export type Admission = { ok: true } | { ok: false; reason: Refusal };

export interface Status {
  day: string;
  usdMicros: number;
  neurons: number;
  similar: number;
}

export class UsageBudget extends DurableObject {
  private today = "";

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    const sql = ctx.storage.sql;
    sql.exec("DROP TABLE IF EXISTS usage"); // v1 schema (NeuronBudget), superseded
    sql.exec("CREATE TABLE IF NOT EXISTS meter (day TEXT NOT NULL, key TEXT NOT NULL, value REAL NOT NULL, PRIMARY KEY (day, key))");
    sql.exec("CREATE TABLE IF NOT EXISTS gpu_iv (day TEXT NOT NULL, start INTEGER NOT NULL, stop INTEGER NOT NULL)");
  }

  /** Deletes every earlier day's rows (meters, visitor ids, burst windows, intervals) on the first call of a day. */
  private roll(day: string): void {
    if (day === this.today) return;
    this.ctx.storage.sql.exec("DELETE FROM meter WHERE day < ?", day);
    this.ctx.storage.sql.exec("DELETE FROM gpu_iv WHERE day < ?", day);
    this.today = day;
  }

  private get(day: string, key: string): number {
    const row = this.ctx.storage.sql.exec<{ value: number }>("SELECT value FROM meter WHERE day = ? AND key = ?", day, key).toArray()[0];
    return row?.value ?? 0;
  }

  private add(day: string, key: string, delta: number): number {
    this.ctx.storage.sql.exec(
      "INSERT INTO meter (day, key, value) VALUES (?, ?, ?) ON CONFLICT(day, key) DO UPDATE SET value = max(0, value + excluded.value)",
      day,
      key,
      delta,
    );
    return this.get(day, key);
  }

  /** Counts one request against the visitor's burst window; false once the window is full. */
  private burst(day: string, visitor: string, bucket: Bucket, nowMs: number): boolean {
    const key = `burst:${visitor}:${bucket}:${Math.floor(nowMs / 60_000)}`;
    if (this.get(day, key) >= BURST_PER_MIN[bucket]) return false;
    this.add(day, key, 1);
    return true;
  }

  /** Admits a scan while today's GPU spend (global and the visitor's) is under its cap. Charge it afterwards. */
  admitScan(day: string, visitor: string, nowMs: number, caps: Caps): Admission {
    this.roll(day);
    if (this.get(day, "usd") >= caps.usdMicros) return { ok: false, reason: "daily" };
    if (this.get(day, `usd:${visitor}`) >= caps.visitorUsdMicros) return { ok: false, reason: "visitor" };
    if (!this.burst(day, visitor, "scan", nowMs)) return { ok: false, reason: "burst" };
    return { ok: true };
  }

  /**
   * Charges a finished (or failed) scan call: merges [startMs, endMs + scaledownMs] into today's intervals and
   * charges only the newly covered time. Returns the micro-dollars charged.
   */
  chargeGpu(day: string, visitor: string, startMs: number, endMs: number, scaledownMs: number, microsPerSec: number): number {
    this.roll(day);
    const sql = this.ctx.storage.sql;
    let lo = startMs;
    let hi = endMs + scaledownMs;
    const overlapping = sql
      .exec<{ start: number; stop: number }>("SELECT start, stop FROM gpu_iv WHERE day = ? AND start <= ? AND stop >= ?", day, hi, lo)
      .toArray();
    let covered = 0;
    for (const iv of overlapping) {
      covered += iv.stop - iv.start;
      lo = Math.min(lo, iv.start);
      hi = Math.max(hi, iv.stop);
    }
    sql.exec("DELETE FROM gpu_iv WHERE day = ? AND start <= ? AND stop >= ?", day, endMs + scaledownMs, startMs);
    sql.exec("INSERT INTO gpu_iv (day, start, stop) VALUES (?, ?, ?)", day, lo, hi);
    const micros = (Math.max(0, hi - lo - covered) / 1000) * microsPerSec;
    this.add(day, "usd", micros);
    this.add(day, `usd:${visitor}`, micros);
    return micros;
  }

  /** Reserves an upper-bound neuron estimate (and a Vectorize query for "similar") before a Workers AI call. */
  reserveNeurons(day: string, visitor: string, bucket: Exclude<Bucket, "scan">, estimate: number, nowMs: number, caps: Caps): Admission {
    this.roll(day);
    if (this.get(day, "neurons") + estimate > caps.neurons) return { ok: false, reason: "daily" };
    if (this.get(day, `neurons:${visitor}`) + estimate > caps.visitorNeurons) return { ok: false, reason: "visitor" };
    if (bucket === "similar" && this.get(day, "similar") >= caps.similar) return { ok: false, reason: "daily" };
    if (!this.burst(day, visitor, bucket, nowMs)) return { ok: false, reason: "burst" };
    this.add(day, "neurons", estimate);
    this.add(day, `neurons:${visitor}`, estimate);
    if (bucket === "similar") this.add(day, "similar", 1);
    return { ok: true };
  }

  /** Corrects a reservation once actual usage is known (delta is usually negative). */
  settleNeurons(day: string, visitor: string, delta: number): void {
    this.roll(day);
    this.add(day, "neurons", delta);
    this.add(day, `neurons:${visitor}`, delta);
  }

  status(day: string): Status {
    this.roll(day);
    return { day, usdMicros: this.get(day, "usd"), neurons: this.get(day, "neurons"), similar: this.get(day, "similar") };
  }
}

export function utcDay(now = new Date()): string {
  return now.toISOString().slice(0, 10);
}

/** Next 00:00 UTC, when every daily meter and quota resets. */
export function resetsAt(now = new Date()): string {
  const d = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1));
  return d.toISOString();
}

export function caps(env: Env): Caps {
  return {
    usdMicros: Number(env.DAILY_USD_MICROS),
    visitorUsdMicros: Number(env.VISITOR_USD_MICROS),
    neurons: Number(env.DAILY_NEURON_CAP),
    visitorNeurons: Number(env.VISITOR_NEURON_CAP),
    similar: Number(env.DAILY_SIMILAR_CAP),
  };
}

export function budgetStub(env: Env): DurableObjectStub<UsageBudget> {
  return env.USAGE_BUDGET.get(env.USAGE_BUDGET.idFromName("global"));
}
