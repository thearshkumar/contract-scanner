// Contract Risk Scanner client. Parses the file locally, chunks it with the same chunker the model was trained on,
// scans chunks in parallel through /api/scan, and renders highlights as results stream in. Demo contracts load
// precomputed results (no GPU time). Ask embeds the contract once and ranks passages in the browser.

import { chunkText, type Chunk } from "../src/core/chunk";
import { dedupe, type ClauseType } from "../src/core/scan-prompt";
import { ApiError, ask, decodeVectors, embed, health, scanChunk, similar } from "./api";
import { fileToText, ParseError, tidy } from "./parse";
import { el, focusSpan, renderDocument, renderPanel, type Found } from "./render";

const MAX_CHUNKS = 40; // matches the Worker's limit (src/api/embed.ts) and the per-minute scan burst
const CONCURRENCY = 6;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const ui = {
  paused: $("paused"),
  pausedText: $("paused-text"),
  file: $<HTMLInputElement>("file"),
  drop: $("drop"),
  paste: $<HTMLTextAreaElement>("paste"),
  scanPaste: $<HTMLButtonElement>("scan-paste"),
  demos: $("demos"),
  status: $("status"),
  progress: $("progress"),
  progressBar: $("progress-bar"),
  results: $("results"),
  docTitle: $("doc-title"),
  docMeta: $("doc-meta"),
  doc: $("doc"),
  panel: $("panel"),
  askForm: $<HTMLFormElement>("ask-form"),
  askInput: $<HTMLInputElement>("ask-input"),
  askBtn: $<HTMLButtonElement>("ask-btn"),
  askStatus: $("ask-status"),
  answer: $("answer"),
};

interface Current {
  title: string;
  text: string;
  chunks: Chunk[];
  spans: Found[];
  vectors?: Float32Array[];
  run: number;
}
let current: Current | null = null;
let live = false;
let runCounter = 0;

function setStatus(msg: string, progress?: number): void {
  ui.status.textContent = msg;
  ui.progress.hidden = progress == null;
  if (progress != null) ui.progressBar.style.transform = `scaleX(${Math.max(0.02, Math.min(1, progress))})`;
}

function showPaused(resetsAt: string | undefined, reason?: string): void {
  live = false;
  const when = resetsAt ? new Date(resetsAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", timeZoneName: "short" }) : "midnight UTC";
  ui.pausedText.textContent =
    reason === "offline"
      ? "Live scanning is offline right now. The demo contracts and Ask still work."
      : reason === "visitor"
        ? `You've used today's live scans. They come back at ${when}. The demo contracts still work.`
        : `Live scanning is paused until ${when}: today's GPU budget is used up. The demo contracts still work.`;
  ui.paused.hidden = false;
  setLiveControls();
}

function setLiveControls(): void {
  ui.file.disabled = !live;
  ui.scanPaste.disabled = !live;
  ui.drop.classList.toggle("is-disabled", !live);
}

// ---------- rendering ----------

function render(): void {
  if (!current) return;
  const c = current;
  ui.results.hidden = false;
  ui.docTitle.textContent = c.title;
  const pages = Math.max(1, Math.round(c.text.length / 3000));
  ui.docMeta.textContent = `${c.text.length.toLocaleString()} characters · about ${pages} pages · ${c.spans.length} passages flagged`;
  const jump = (id: number) => focusSpan(ui.doc, ui.panel, id, true);
  renderDocument(ui.doc, c.text, c.spans, (id) => focusSpan(ui.doc, ui.panel, id, false));
  renderPanel(ui.panel, c.text, c.spans, { jump, similar: showSimilar });
}

function load(title: string, text: string): Current {
  current = { title, text, chunks: chunkText(text), spans: [], run: ++runCounter };
  ui.answer.hidden = true;
  ui.askStatus.textContent = "";
  ui.askBtn.disabled = false;
  ui.askInput.disabled = false;
  return current;
}

// ---------- scanning ----------

async function scanText(title: string, raw: string): Promise<void> {
  const text = tidy(raw);
  const c = load(title, text);
  if (c.chunks.length > MAX_CHUNKS) {
    setStatus(`This contract is too long for the demo (${c.chunks.length} sections; the limit is ${MAX_CHUNKS}, about 60 pages).`);
    current = null;
    ui.results.hidden = true;
    return;
  }
  render();
  ui.results.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });

  const n = c.chunks.length;
  let done = 0;
  let failed = 0;
  let misses = 0;
  let stop: ApiError | null = null;
  let nextId = 0;
  const found: Found[] = [];
  setStatus(`Waking the model… the first scan after a quiet spell usually takes about 30 seconds, occasionally a few minutes.`, 0.02);

  const queue = [...c.chunks];
  async function worker(): Promise<void> {
    while (queue.length && !stop && c.run === current?.run) {
      const chunk = queue.shift()!;
      let attempt = 0;
      for (;;) {
        try {
          const r = await scanChunk(chunk.text, chunk.id);
          misses += r.misses;
          for (const s of r.spans) found.push({ id: nextId++, type: s.type, start: chunk.start + s.start, end: chunk.start + s.end });
          break;
        } catch (e) {
          const err = e instanceof ApiError ? e : new ApiError(0, "Unexpected error.");
          if (err.demoOnly) {
            stop = err;
            return;
          }
          if (attempt++ === 0 && (err.status === 0 || err.status >= 500 || err.status === 429)) {
            await new Promise((r) => setTimeout(r, err.status === 429 ? 15_000 : 2_000));
            continue;
          }
          failed++;
          break;
        }
      }
      done++;
      if (c.run !== current?.run) return;
      c.spans = dedupe(found).map((s) => ({ ...s }));
      render();
      setStatus(`Scanning section ${Math.min(done + 1, n)} of ${n}…`, done / n);
    }
  }
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, n) }, worker));
  if (c.run !== current?.run) return;

  if (stop) {
    const s = stop as ApiError;
    showPaused(s.resetsAt, s.reason);
    setStatus(`Stopped after ${done} of ${n} sections. ${s.message}`);
    return;
  }
  const notes = [failed ? `${failed} section${failed > 1 ? "s" : ""} failed, so some clauses may be missing` : "", misses ? `${misses} model quote${misses > 1 ? "s" : ""} couldn't be matched to the text and were dropped` : ""].filter(Boolean);
  setStatus(`Done: ${c.spans.length} passages in ${n} sections.${notes.length ? ` Note: ${notes.join("; ")}.` : ""}`);
}

async function loadDemo(id: string, title: string, button: HTMLButtonElement): Promise<void> {
  for (const b of ui.demos.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === button));
  setStatus("Loading the demo…");
  const d = (await (await fetch(`/demo/${id}.json`)).json()) as { text: string; spans: { type: ClauseType; start: number; end: number }[]; attribution: string };
  const c = load(`${title} (demo)`, d.text);
  c.spans = d.spans.map((s, i) => ({ id: i, ...s }));
  render();
  setStatus(`Demo: precomputed by the same model, no GPU time used. ${d.attribution}`);
  ui.results.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
}

// ---------- similar CUAD examples ----------

async function showSimilar(type: ClauseType, text: string, into: HTMLElement, button: HTMLButtonElement): Promise<void> {
  if (!into.hidden) {
    into.hidden = true;
    return;
  }
  button.disabled = true;
  into.hidden = false;
  into.replaceChildren(el("p", "ak-meta", "Finding similar clauses in CUAD…"));
  try {
    const { examples } = await similar(text.slice(0, 1200), type);
    const box = el("div", "ak-callout");
    box.append(el("span", "ak-label", `Similar ${type} clauses in CUAD`));
    if (!examples.length) box.append(el("p", undefined, "No examples found."));
    for (const ex of examples) {
      const q = el("blockquote", "similar__quote", `“${ex.text.length > 400 ? `${ex.text.slice(0, 399)}…` : ex.text}”`);
      const cite = el("p", "similar__source ak-meta", `${ex.source} · CUAD, CC BY 4.0`);
      box.append(q, cite);
    }
    into.replaceChildren(box);
  } catch (e) {
    into.replaceChildren(el("p", "ak-meta", e instanceof ApiError ? e.message : "Couldn't load examples."));
  } finally {
    button.disabled = false;
  }
}

// ---------- ask ----------

function dot(a: Float32Array, b: Float32Array): number {
  let d = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return na && nb ? d / Math.sqrt(na * nb) : 0;
}

async function onAsk(ev: SubmitEvent): Promise<void> {
  ev.preventDefault();
  const c = current;
  const question = ui.askInput.value.trim();
  if (!c || !question) return;
  ui.askBtn.disabled = true;
  ui.answer.hidden = true;
  try {
    if (!c.vectors) {
      ui.askStatus.textContent = "Reading the contract (once per contract)…";
      const r = await embed(c.chunks.map((k) => k.text));
      c.vectors = decodeVectors(r.vectors, r.dims);
    }
    ui.askStatus.textContent = "Finding the relevant passages…";
    const q = await embed([question]);
    const qv = decodeVectors(q.vectors, q.dims)[0]!;
    const top = c.chunks
      .map((k, i) => ({ k, score: dot(qv, c.vectors![i]!) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 4)
      .map((x) => x.k);
    ui.askStatus.textContent = "Answering…";
    const r = await ask(question, top.map((k) => ({ id: k.id + 1, text: k.text })));
    if (c.run !== current?.run) return;
    renderAnswer(r.answer, c);
    ui.askStatus.textContent = "";
  } catch (e) {
    ui.askStatus.textContent = e instanceof ApiError ? e.message : "Something went wrong.";
  } finally {
    ui.askBtn.disabled = false;
  }
}

function renderAnswer(answer: string, c: Current): void {
  const box = el("div", "ak-callout answer__box");
  box.append(el("span", "ak-label", "Answer · cites sections of your contract"));
  const p = el("p", "answer__text");
  for (const part of answer.split(/(\[c\d+\])/)) {
    const m = /^\[c(\d+)\]$/.exec(part);
    if (!m) {
      p.append(part);
      continue;
    }
    const chunk = c.chunks[Number(m[1]) - 1];
    if (!chunk) continue;
    const b = el("button", "cite ak-mono", `§${m[1]}`);
    b.type = "button";
    b.title = `Show section ${m[1]} in the document`;
    b.addEventListener("click", () => flashRange(chunk.start, chunk.end));
    p.append(b);
  }
  box.append(p, el("p", "ak-meta", "Generated by Qwen3-30B-A3B on Workers AI from the cited sections only. Not legal advice."));
  ui.answer.replaceChildren(box);
  ui.answer.hidden = false;
}

/** Scrolls to a cited section of the document and tints it briefly (CSS Custom Highlight API where supported). */
function flashRange(start: number, end: number): void {
  const range = document.createRange();
  const walker = document.createTreeWalker(ui.doc, NodeFilter.SHOW_TEXT);
  let pos = 0;
  let begun = false;
  let node: Node | null;
  while ((node = walker.nextNode())) {
    const len = node.textContent?.length ?? 0;
    if (!begun && pos + len > start) {
      range.setStart(node, start - pos);
      begun = true;
    }
    if (begun && pos + len >= end) {
      range.setEnd(node, end - pos);
      break;
    }
    pos += len;
  }
  if (!begun) return;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const top = range.getBoundingClientRect().top + window.scrollY - 96;
  window.scrollTo({ top, behavior: reduce ? "auto" : "smooth" });
  const registry = (CSS as unknown as { highlights?: Map<string, unknown> }).highlights;
  const H = (window as unknown as { Highlight?: new (r: Range) => unknown }).Highlight;
  if (registry && H) {
    registry.set("cite", new H(range));
    setTimeout(() => registry.delete("cite"), 2500);
  }
}

// ---------- wiring ----------

async function onFile(file: File | undefined): Promise<void> {
  if (!file || !live) return;
  for (const b of ui.demos.querySelectorAll("button")) b.setAttribute("aria-pressed", "false");
  setStatus(`Reading ${file.name} in your browser…`);
  try {
    const text = await fileToText(file);
    await scanText(file.name.replace(/\.[^.]+$/, ""), text);
  } catch (e) {
    setStatus(e instanceof ParseError ? e.message : "Couldn't read that file. Try another, or paste the text.");
  } finally {
    ui.file.value = "";
  }
}

ui.file.addEventListener("change", () => void onFile(ui.file.files?.[0]));
for (const t of ["dragenter", "dragover"]) {
  ui.drop.addEventListener(t, (e) => {
    e.preventDefault();
    if (live) ui.drop.classList.add("is-over");
  });
}
for (const t of ["dragleave", "drop"]) ui.drop.addEventListener(t, () => ui.drop.classList.remove("is-over"));
ui.drop.addEventListener("drop", (e) => {
  e.preventDefault();
  void onFile((e as DragEvent).dataTransfer?.files?.[0]);
});
ui.scanPaste.addEventListener("click", () => {
  const text = ui.paste.value;
  if (text.trim().length < 200) {
    setStatus("Paste at least a few paragraphs of contract text.");
    return;
  }
  for (const b of ui.demos.querySelectorAll("button")) b.setAttribute("aria-pressed", "false");
  void scanText("Pasted contract", text);
});
ui.askForm.addEventListener("submit", (e) => void onAsk(e));

async function init(): Promise<void> {
  const [h, demos] = await Promise.all([
    health(),
    fetch("/demo/index.json").then((r) => r.json() as Promise<{ id: string; title: string; clauses: number }[]>).catch(() => []),
  ]);
  live = h.live;
  if (!live) showPaused(h.resetsAt || undefined, h.reason ?? (h.ok ? undefined : "offline"));
  setLiveControls();
  for (const d of demos) {
    const b = el("button", "ak-btn ak-btn--secondary ak-btn--sm", d.title);
    b.type = "button";
    b.setAttribute("aria-pressed", "false");
    b.addEventListener("click", () => void loadDemo(d.id, d.title, b));
    ui.demos.append(b);
  }
  // Reveal-on-scroll, as on arshkumar.com (kit.css hides .reveal until .in under html.js).
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) (e.target.classList.add("in"), io.unobserve(e.target));
  });
  for (const r of document.querySelectorAll(".reveal")) io.observe(r);
}

void init();
