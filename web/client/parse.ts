// Turns an uploaded file into plain text, entirely in the browser: the file itself never leaves the device.
// pdf.js and mammoth are loaded only when needed (they are large), and pdf.js's worker is served from this origin.

export const MAX_FILE_BYTES = 15 * 1024 * 1024;

export class ParseError extends Error {}

export async function fileToText(file: File): Promise<string> {
  if (file.size > MAX_FILE_BYTES) throw new ParseError("That file is over 15 MB. Try a smaller file or paste the text.");
  const name = file.name.toLowerCase();
  let text: string;
  if (name.endsWith(".pdf") || file.type === "application/pdf") text = await pdfText(file);
  else if (name.endsWith(".docx")) text = await docxText(file);
  else if (name.endsWith(".txt") || name.endsWith(".md") || file.type.startsWith("text/")) text = await file.text();
  else throw new ParseError("Use a PDF, DOCX or TXT file, or paste the text.");
  text = tidy(text);
  if (text.trim().length < 200) {
    throw new ParseError(
      name.endsWith(".pdf")
        ? "No text found in that PDF. It may be a scan; scanned PDFs need OCR, which this demo doesn't do."
        : "That file has almost no text.",
    );
  }
  return text;
}

async function pdfText(file: File): Promise<string> {
  const pdfjs = await import("pdfjs-dist");
  const workerUrl = (await import("pdfjs-dist/build/pdf.worker.min.mjs?url")).default;
  pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
  const task = pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()) });
  try {
    const doc = await task.promise;
    const pages: string[] = [];
    for (let p = 1; p <= doc.numPages; p++) {
      const content = await (await doc.getPage(p)).getTextContent();
      let page = "";
      for (const item of content.items) {
        if (!("str" in item)) continue;
        page += item.str + (item.hasEOL ? "\n" : "");
      }
      pages.push(page);
    }
    return pages.join("\n\n");
  } finally {
    await task.destroy();
  }
}

async function docxText(file: File): Promise<string> {
  const mammoth = (await import("mammoth/mammoth.browser.js")).default;
  const { value } = await mammoth.extractRawText({ arrayBuffer: await file.arrayBuffer() });
  return value;
}

/** Normalises line endings and strips trailing spaces; keeps paragraph breaks the chunker relies on. */
export function tidy(text: string): string {
  return text
    .replace(/\r\n?/g, "\n")
    .replace(/ /g, " ")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}
