# Contract Risk Scanner

**Upload a contract, see the risky clauses highlighted.**

> Not legal advice. This is a portfolio demo trained on US commercial contracts (CUAD).

Live: `contracts.arshkumar.com` (coming in Phase 6) · Status: Phase 0 (scaffold)

## What it does

1. Upload a contract (PDF, DOCX or pasted text). It is parsed **in your browser**; only the extracted text is sent.
2. Clauses are flagged by type (termination, non-compete, liability cap, auto-renewal, … 41 CUAD types). Each flag
   highlights the span, explains it in one line, and shows 1–2 similar real-world clauses from the CUAD library.
3. Ask follow-up questions. Every answer cites the clause it came from.

Your contract is never stored: it is processed in memory per request. Only the public CUAD clause library lives in a database.

## Structure: what owns what

| Part | Language | Owns |
|---|---|---|
| `web/` | TypeScript (Cloudflare Worker + static client) | the product: parsing (client), chunking, scan/ask/similar APIs, rate limit + neuron budget, logging |
| `model/` | Python | LoRA training jobs (rented H100 via Modal), adapter export for Workers AI, HF Hub publishing |
| `eval/` | Python | CUAD loaders, metrics (P/R/F1 per clause type, latency, citation accuracy), base vs fine-tuned scoreboard |
| `shared/` | Python | JSON logging used by `model/` and `eval/` |

## Architecture

_TODO (Phase 6): diagram._ See `docs/plan.md` for the current design.

## Why provider interfaces

_TODO: `LLMProvider`, `Embedder`, `ClauseLibrary`: Cloudflare implementations plus in-memory mocks, so the core logic is
testable and not locked to one vendor._

## Results

_TODO (Phases 1 and 5): base vs fine-tuned, offline F1, live smoke F1, latency, neurons per scan._

## Failure analysis

_TODO._

## Cost and quota

_TODO: free-tier budget (10k neurons/day shared per account, Vectorize 5M stored dims, Workers Logs limits) and the guards._

## Limitations

_TODO. Known: CUAD is US commercial contracts; performance on e.g. UK tenancy agreements is unmeasured._

## Licences

Code: MIT. Third-party: see `THIRD_PARTY_LICENSES.md`. CUAD is CC BY 4.0 (The Atticus Project).
