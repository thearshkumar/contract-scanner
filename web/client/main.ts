// Placeholder client (Phase 0). Parsing, chunked scanning and highlighting arrive in Phase 2.
const statusEl = document.getElementById("status");

fetch("/api/health")
  .then((r) => r.json() as Promise<{ ok: boolean; model: string }>)
  .then((h) => {
    if (statusEl) statusEl.textContent = h.ok ? `API is up (${h.model}).` : "API unavailable.";
  })
  .catch(() => {
    if (statusEl) statusEl.textContent = "API unavailable.";
  });
