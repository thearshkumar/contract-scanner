// Renders a scanned contract: the document with tier-coloured highlights, and the side panel (key facts, then
// Watch and Note cards). Highlights and cards are linked both ways. Built with DOM APIs only (no innerHTML of
// contract text), so nothing in a contract can inject markup.

import { CLAUSES, TIER_LABEL, type Tier } from "../src/core/clause-types";
import type { ClauseType } from "../src/core/scan-prompt";

export interface Found {
  id: number;
  type: ClauseType;
  start: number;
  end: number;
}

const TIER_RANK: Record<Tier, number> = { watch: 3, note: 2, fact: 1 };

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

const excerpt = (text: string, s: Found, max = 180) => {
  const t = text.slice(s.start, s.end).replace(/\s+/g, " ").trim();
  return t.length > max ? `${t.slice(0, max - 1)}…` : t;
};

/** The document as text segments; each covered segment is a <mark> listing every span over it. */
export function renderDocument(root: HTMLElement, text: string, spans: Found[], onPick: (id: number) => void): void {
  const cuts = new Set<number>([0, text.length]);
  for (const s of spans) {
    cuts.add(s.start);
    cuts.add(s.end);
  }
  const points = [...cuts].sort((a, b) => a - b);
  const frag = document.createDocumentFragment();
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i]!;
    const b = points[i + 1]!;
    const over = spans.filter((s) => s.start <= a && s.end >= b);
    const seg = text.slice(a, b);
    if (!over.length) {
      frag.append(seg);
      continue;
    }
    const top = over.reduce((x, y) => (TIER_RANK[CLAUSES[y.type].tier] > TIER_RANK[CLAUSES[x.type].tier] ? y : x));
    const m = el("mark", `hl hl--${CLAUSES[top.type].tier}`, seg);
    m.dataset.ids = over.map((s) => s.id).join(" ");
    m.title = [...new Set(over.map((s) => s.type))].join(", ");
    m.addEventListener("click", () => onPick(top.id));
    frag.append(m);
  }
  root.replaceChildren(frag);
}

export function focusSpan(docRoot: HTMLElement, panel: HTMLElement, id: number, scrollDoc: boolean): void {
  for (const m of docRoot.querySelectorAll("mark.is-active")) m.classList.remove("is-active");
  for (const c of panel.querySelectorAll(".is-active")) c.classList.remove("is-active");
  const marks = [...docRoot.querySelectorAll<HTMLElement>("mark")].filter((m) => m.dataset.ids?.split(" ").includes(String(id)));
  for (const m of marks) m.classList.add("is-active");
  const item = panel.querySelector<HTMLElement>(`[data-span="${id}"]`);
  item?.classList.add("is-active");
  item?.closest(".ak-card")?.classList.add("is-active");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (scrollDoc) marks[0]?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "center" });
  else item?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "nearest" });
}

export interface PanelHooks {
  jump(id: number): void;
  similar(type: ClauseType, text: string, into: HTMLElement, button: HTMLButtonElement): void;
}

export function renderPanel(panel: HTMLElement, text: string, spans: Found[], hooks: PanelHooks, scanning = false): void {
  const byType = new Map<ClauseType, Found[]>();
  for (const s of spans) byType.set(s.type, [...(byType.get(s.type) ?? []), s]);

  const frag = document.createDocumentFragment();

  // Key facts: a spec list of the first passage of each fact type.
  const facts = [...byType].filter(([t]) => CLAUSES[t].tier === "fact");
  if (facts.length) {
    frag.append(el("h3", "panel__head ak-label", TIER_LABEL.fact));
    const dl = el("dl", "ak-spec");
    for (const [type, list] of facts) {
      const row = el("div", "ak-spec__row");
      const dd = el("dd");
      const b = el("button", "linklike", excerpt(text, list[0]!, 120));
      b.type = "button";
      b.dataset.span = String(list[0]!.id);
      b.addEventListener("click", () => hooks.jump(list[0]!.id));
      dd.append(b);
      row.append(el("dt", undefined, type), dd);
      dl.append(row);
    }
    frag.append(dl);
  }

  for (const tier of ["watch", "note"] as const) {
    const types = [...byType].filter(([t]) => CLAUSES[t].tier === tier).sort((a, b) => a[1][0]!.start - b[1][0]!.start);
    const head = el("h3", "panel__head ak-label", `${TIER_LABEL[tier]} · ${types.length}`);
    frag.append(head);
    if (!types.length) {
      const empty = scanning ? "Scanning…" : tier === "watch" ? "No high-attention clauses found." : "Nothing else found.";
      frag.append(el("p", "ak-meta", empty));
      continue;
    }
    for (const [type, list] of types) frag.append(card(text, type, list, hooks));
  }
  panel.replaceChildren(frag);
}

function card(text: string, type: ClauseType, list: Found[], hooks: PanelHooks): HTMLElement {
  const info = CLAUSES[type];
  const c = el("article", `ak-card clause clause--${info.tier}`);
  const top = el("div", "ak-card__top");
  const tags = el("div", "ak-tags");
  tags.append(el("span", `ak-tag tier-tag tier-tag--${info.tier}`, TIER_LABEL[info.tier]));
  if (info.lowAccuracy) {
    const t = el("span", "ak-tag", "low accuracy");
    t.title = `Held-out span F1 ${info.f1.toFixed(2)} for this type: treat with care.`;
    tags.append(t);
  }
  top.append(tags, el("span", "ak-card__index ak-meta", `${list.length} ${list.length === 1 ? "passage" : "passages"}`));
  c.append(top, el("h4", "clause__title", type), el("p", "ak-card__desc", info.explain));

  const ul = el("ul", "clause__list");
  for (const s of list) {
    const li = el("li");
    const b = el("button", "clause__quote", `“${excerpt(text, s)}”`);
    b.type = "button";
    b.dataset.span = String(s.id);
    b.addEventListener("click", () => hooks.jump(s.id));
    li.append(b);
    ul.append(li);
  }
  c.append(ul);

  const foot = el("div", "ak-card__foot");
  const sim = el("button", "ak-btn ak-btn--ghost ak-btn--sm ak-mono", "Similar in CUAD");
  sim.type = "button";
  const out = el("div", "clause__similar");
  out.hidden = true;
  sim.addEventListener("click", () => hooks.similar(type, text.slice(list[0]!.start, list[0]!.end), out, sim));
  foot.append(sim, el("span", "ak-meta", `F1 ${info.f1.toFixed(2)}`));
  c.append(foot, out);
  return c;
}
