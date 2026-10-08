// Splits contract text into overlapping chunks on paragraph/sentence boundaries.
// Shared by the browser (which drives the scan chunk by chunk) and the Worker (/api/ask).
// Offsets index into the original text so flagged spans can be highlighted exactly.

export interface Chunk {
  id: number;
  start: number;
  end: number;
  text: string;
}

export interface ChunkOptions {
  /** Target max characters per chunk (~4 chars per token). */
  maxChars?: number;
  /** Characters of overlap with the previous chunk, so clauses on a boundary are seen whole once. */
  overlap?: number;
}

const BREAKS = ["\n\n", "\n", ". ", "; ", ", ", " "];

export function chunkText(text: string, { maxChars = 6000, overlap = 400 }: ChunkOptions = {}): Chunk[] {
  if (overlap >= maxChars) throw new Error("overlap must be smaller than maxChars");
  const chunks: Chunk[] = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + maxChars, text.length);
    if (end < text.length) end = breakBefore(text, start, end);
    const body = text.slice(start, end);
    if (body.trim()) chunks.push({ id: chunks.length, start, end, text: body });
    if (end >= text.length) break;
    // Next chunk starts `overlap` back, snapped forward to a word boundary, and always makes progress.
    let next = Math.max(end - overlap, start + 1);
    const space = text.indexOf(" ", next);
    if (space !== -1 && space < end) next = space + 1;
    start = next;
  }
  return chunks;
}

/** Latest natural break in the back half of [start, end), else a hard cut at end. */
function breakBefore(text: string, start: number, end: number): number {
  const floor = start + Math.floor((end - start) / 2);
  for (const sep of BREAKS) {
    const i = text.lastIndexOf(sep, end - sep.length);
    if (i >= floor) return i + sep.length;
  }
  return end;
}
