import { DurableObject } from "cloudflare:workers";

// One global SQLite-backed Durable Object that counts estimated neurons per UTC day.
// Requests reserve their estimate up front and settle the actual figure afterwards,
// so concurrent requests cannot overshoot the cap. Stores counters only, never user content.

export interface Reservation {
  ok: boolean;
  used: number;
  cap: number;
}

export class NeuronBudget extends DurableObject {
  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    ctx.storage.sql.exec("CREATE TABLE IF NOT EXISTS usage (day TEXT PRIMARY KEY, neurons REAL NOT NULL)");
  }

  private used(day: string): number {
    const row = this.ctx.storage.sql.exec<{ neurons: number }>("SELECT neurons FROM usage WHERE day = ?", day).toArray()[0];
    return row?.neurons ?? 0;
  }

  private add(day: string, neurons: number): number {
    this.ctx.storage.sql.exec(
      "INSERT INTO usage (day, neurons) VALUES (?, ?) ON CONFLICT(day) DO UPDATE SET neurons = max(0, neurons + excluded.neurons)",
      day,
      neurons,
    );
    return this.used(day);
  }

  reserve(day: string, estimate: number, cap: number): Reservation {
    const used = this.used(day);
    if (used + estimate > cap) return { ok: false, used, cap };
    return { ok: true, used: this.add(day, estimate), cap };
  }

  /** Corrects a reservation once actual usage is known (delta may be negative). */
  settle(day: string, delta: number): number {
    return this.add(day, delta);
  }

  /** Drops days older than a week; called opportunistically. */
  prune(today: string): void {
    const cutoff = new Date(Date.parse(today) - 7 * 864e5).toISOString().slice(0, 10);
    this.ctx.storage.sql.exec("DELETE FROM usage WHERE day < ?", cutoff);
  }
}

export function utcDay(now = new Date()): string {
  return now.toISOString().slice(0, 10);
}

export function budgetStub(env: Env): DurableObjectStub<NeuronBudget> {
  return env.NEURON_BUDGET.get(env.NEURON_BUDGET.idFromName("global"));
}
