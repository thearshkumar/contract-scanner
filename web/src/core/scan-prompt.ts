// The prompt/output contract the model was fine-tuned on, ported 1:1 from eval/prompt.py and eval/run.py.
// The model was trained on this exact SYSTEM string: any drift costs accuracy, so a golden fixture shared with
// pytest (eval/tests/fixtures/prompt_golden.json) pins messages(), parse(), locate() and dedupe().

export const ANCHOR_WORDS = 10;

// CUAD v1's 41 categories (names as in CUAD_v1.json question ids).
export const CLAUSE_TYPES = [
  "Affiliate License-Licensee", "Affiliate License-Licensor", "Agreement Date", "Anti-Assignment", "Audit Rights",
  "Cap On Liability", "Change Of Control", "Competitive Restriction Exception", "Covenant Not To Sue",
  "Document Name", "Effective Date", "Exclusivity", "Expiration Date", "Governing Law", "Insurance",
  "Ip Ownership Assignment", "Irrevocable Or Perpetual License", "Joint Ip Ownership", "License Grant",
  "Liquidated Damages", "Minimum Commitment", "Most Favored Nation", "No-Solicit Of Customers",
  "No-Solicit Of Employees", "Non-Compete", "Non-Disparagement", "Non-Transferable License",
  "Notice Period To Terminate Renewal", "Parties", "Post-Termination Services", "Price Restrictions",
  "Renewal Term", "Revenue/Profit Sharing", "Rofr/Rofo/Rofn", "Source Code Escrow", "Termination For Convenience",
  "Third Party Beneficiary", "Uncapped Liability", "Unlimited/All-You-Can-Eat-License", "Volume Restriction",
  "Warranty Duration",
] as const;
export type ClauseType = (typeof CLAUSE_TYPES)[number];

const TYPE_LOOKUP = new Map<string, ClauseType>(CLAUSE_TYPES.map((t) => [t.toLowerCase(), t]));

export const SYSTEM =
  "You review commercial contracts. Find every passage in the excerpt that belongs to one of these clause types:\n" +
  CLAUSE_TYPES.map((t) => `- ${t}`).join("\n") +
  "\n\nReply with ONLY a JSON array, one object per passage, in the order they appear:\n" +
  '[{"type": "<clause type from the list>", "begin": "<first words of the passage>", ' +
  '"end": "<last words of the passage>"}]\n' +
  `\`begin\` and \`end\` must be copied word for word from the excerpt, at most ${ANCHOR_WORDS} words each. ` +
  "For a short passage, `begin` and `end` may be the same. If nothing matches, reply [].";

export interface Message {
  role: "system" | "user";
  content: string;
}

export function messages(chunkText: string): Message[] {
  return [
    { role: "system", content: SYSTEM },
    { role: "user", content: `Excerpt:\n<<<\n${chunkText}\n>>>` },
  ];
}

// ---------- parsing model output ----------

export interface Item {
  type: ClauseType;
  begin: string;
  end: string;
}

const OBJECT = /\{[^{}]*\}/g;

/** Python's str.split() with no args: split on whitespace runs, drop empties. */
const words = (s: string) => s.split(/\s+/).filter(Boolean);
const collapse = (s: string) => words(s).join(" ");

/** Returns [items, ok]. Tolerates prose/code fences; salvages complete objects from a cut-off array (ok=false). */
export function parse(output: string): [Item[], boolean] {
  const text = output.replace(/<think>[\s\S]*?<\/think>/g, "");
  const start = text.indexOf("[");
  const end = text.lastIndexOf("]");
  let raw: unknown[] | null = null;
  let ok = false;
  if (start !== -1 && end > start) {
    try {
      const v: unknown = JSON.parse(text.slice(start, end + 1));
      if (Array.isArray(v)) {
        raw = v;
        ok = true;
      }
    } catch {
      raw = null;
    }
  }
  if (!ok) {
    if (start === -1 && text.trim() === "") return [[], true];
    raw = [];
    for (const m of (start !== -1 ? text.slice(start) : text).matchAll(OBJECT)) {
      try {
        raw.push(JSON.parse(m[0]));
      } catch {
        continue;
      }
    }
  }
  const items: Item[] = [];
  for (const it of raw ?? []) {
    if (typeof it !== "object" || it === null || Array.isArray(it)) continue;
    const o = it as Record<string, unknown>;
    const ctype = TYPE_LOOKUP.get(pyStr(o.type ?? "").trim().toLowerCase());
    const begin = pyStr(o.begin || o.quote || "").trim();
    const end_ = pyStr(o.end || begin).trim();
    if (ctype && begin) items.push({ type: ctype, begin, end: end_ });
  }
  return [items, ok];
}

/** Enough of Python's str() for JSON values that the model might emit in a string field. */
function pyStr(v: unknown): string {
  if (typeof v === "string") return v;
  if (v === true) return "True";
  if (v === false) return "False";
  if (v === null) return "None";
  return String(v);
}

function normalise(text: string): [string, number[]] {
  const out: string[] = [];
  const idx: number[] = [];
  let prevSpace = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!;
    if (/\s/.test(ch)) {
      if (prevSpace) continue;
      out.push(" ");
      prevSpace = true;
    } else {
      out.push(ch);
      prevSpace = false;
    }
    idx.push(i);
  }
  return [out.join(""), idx];
}

/** Character offsets in `text` of the passage from `begin` through `end` (whitespace-insensitive). */
export function locate(text: string, begin: string, end: string): [number, number] | null {
  const [norm, idx] = normalise(text);
  const nb = collapse(begin);
  const ne = collapse(end);
  if (!nb) return null;
  let b = norm.indexOf(nb);
  if (b === -1) {
    b = norm.toLowerCase().indexOf(nb.toLowerCase());
    if (b === -1) return null;
  }
  let e = ne ? norm.indexOf(ne, b) : -1;
  if (e === -1 && ne) e = norm.toLowerCase().indexOf(ne.toLowerCase(), b);
  const stop = e !== -1 ? e + ne.length : b + nb.length;
  return [idx[b]!, idx[stop - 1]! + 1];
}

export interface Span {
  type: ClauseType;
  start: number;
  end: number; // exclusive
}

/** Absolute spans for parsed items, plus how many could not be located (citation misses). */
export function toSpans(chunk: { start: number; text: string }, items: Item[]): [Span[], number] {
  const spans: Span[] = [];
  let misses = 0;
  for (const it of items) {
    const loc = locate(chunk.text, it.begin, it.end);
    if (!loc) {
      misses++;
      continue;
    }
    spans.push({ type: it.type, start: chunk.start + loc[0], end: chunk.start + loc[1] });
  }
  return [spans, misses];
}

const cmp = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

/** Overlapping chunks can report the same passage twice: keep one per type when they overlap by >= 50%. */
export function dedupe<S extends Span>(spans: S[]): S[] {
  const kept: S[] = [];
  const sorted = [...spans].sort(
    (a, b) => cmp(a.type, b.type) || a.start - b.start || b.end - b.start - (a.end - a.start),
  );
  for (const s of sorted) {
    const dup = kept.some((k) => {
      if (k.type !== s.type) return false;
      const inter = Math.min(k.end, s.end) - Math.max(k.start, s.start);
      return inter > 0 && inter >= 0.5 * Math.min(k.end - k.start, s.end - s.start);
    });
    if (!dup) kept.push(s);
  }
  return kept.sort((a, b) => a.start - b.start || cmp(a.type, b.type));
}
