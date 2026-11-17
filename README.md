# GasComply
Citation-grounded lookup of Vietnamese LPG-business and fire-safety (PCCC) regulations,
built on **GraphRAG** over a citation graph of real statutes. *Live URL: not deployed yet
(Phase 5.2 — see Roadmap).*

> ⚠️ GasComply quotes public legal-normative documents for reference only. It is **not** legal
> advice — always check the latest in-force version on [vbpl.vn](https://vbpl.vn).

## What it does

Answers compliance questions about running an LPG (gas cylinder) retail shop in Vietnam —
licensing, store conditions, safe distances, cylinder inspection, PCCC obligations — and
**every answer carries article-level citations** (Điều/Khoản + document number + effective
date). Legal text is a natural cross-reference graph, so retrieval walks that graph instead
of relying on vector similarity alone, which is what makes multi-hop questions answerable.

Three rules are non-negotiable:

- **Cite or refuse**, through five checks, run in order, on every answer:
  1. **Fabrication validation** — a citation naming a unit that was not actually retrieved
     for this question is dropped before anything else runs.
  2. **Layer 1 (deterministic guards)** — no document number may appear in the answer's own
     text unless it is a cited unit's own document number or appears in that unit's own text;
     and a claim in the answer that a document is still in force is checked against that
     document's own stored status.
  3. **Layer 2 (per-citation verifier)** — each surviving citation is checked, clause by
     clause, against the retrieved text it is supposed to rest on; an unsupported citation is
     dropped.
  4. **One retry without rejected passages.** If layer 2 rejected every citation and at least
     one rejected id names a passage that really was offered, the answer is retried exactly
     once with those passages removed — never with anything added back, and never retried a
     second time.
  5. **Threshold verdict check.** On a yes/no answer, a deterministic check compares the
     question's own quantity against the threshold stated in its cited clause; a contradiction
     between the answer's "Có"/"Không" and that clause becomes a refusal.

  If nothing survives — no supporting context, the model declined, or every citation it gave
  was dropped — the result is a **refusal**, a first-class outcome with its own reason code,
  never an invented article number.
- **Emergencies never reach the model.** Suspected gas leak / fire / explosion is caught by
  keyword *before* any LLM call and answered with a hard-coded response pointing to hotline
  **114**.
- **Only reviewed graph edges are traversed.** Structural edges are auto-trusted; every
  reference or curated edge must be approved by a human first — a clean clone starts with
  all of them pending (see "Local setup" and "Known limitations" below).

## Architecture

```
                 ┌──────────── safety check (keyword, pre-LLM) ── hard-coded 114 response
question ──▶ /ask ┤
                 └── retrieval ──▶ generation ──▶ {answer, citations, effective_dates, disclaimer}
                      │
                      ├── vector search    pgvector top-k over article/clause chunks
                      ├── graph expansion  recursive CTE over `edges` (approved only)
                      ├── date filter      drops superseded documents
                      ├── rerank           vector score × graph proximity
                      └── cross-encoder    optional: keeps only the best top-N passages (llama-server sidecar)

ingest: corpus/*.pdf ──▶ parser (Chương/Điều/Khoản/Điểm, stable ids)
                     ──▶ references (regex) ──▶ graph edges (+ human review)
                     ──▶ chunks + embeddings ──▶ Postgres/pgvector
```

The knowledge graph is a relational `edges` table traversed with a recursive CTE — no separate
graph database.

## Tech stack

| Layer | Choice |
|---|---|
| Backend | FastAPI · Python 3.11 · SQLAlchemy 2.0 (async) · Pydantic v2 |
| Database | PostgreSQL 16 + pgvector (`pgvector/pgvector:pg16` container locally and in CI; see `docker-compose.yml`) |
| Graph | `edges` table + recursive CTE in Postgres |
| Embeddings | bge-m3 (1024-dim); `mock` embedder for offline tests |
| Reranker | Optional: bge-reranker-v2-m3 cross-encoder on a llama.cpp `llama-server --rerank` sidecar (off by default); `mock` reranker for offline tests |
| LLM | Ollama, `qwen3:8b` — the only provider implemented today; `mock` for offline tests (see Roadmap for production failover) |

## Local setup

```bash
git clone https://github.com/Rhynis/gascomply-public.git gascomply && cd gascomply
docker compose up -d                   # Postgres 16 + pgvector on localhost:5433
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                   # DATABASE_URL already points at the compose Postgres
ollama pull bge-m3                     # embedder
ollama pull qwen3:8b                   # chat model
python -m app.core.migrate             # apply SQL migrations
```

Download the source documents listed in [`SOURCES.md`](SOURCES.md) into `corpus/` yourself —
the legal corpus is **not** committed; only `corpus/manifest.json` is tracked. See
"`corpus/manifest.json`" below for its shape, and note that 3 of today's 10 manifest entries
are declared `"acquired": false` (one copyrighted standard pending purchase, one in-force QCVN
not yet downloaded, and one expired QCVN kept only so its `REPLACED_BY` edge resolves — see
"Known limitations").

```bash
python -m ingest.run --corpus ../corpus   # parses, chunks, embeds (via Ollama), persists
```

**Edge review.** Ingest auto-approves only *structural* edges (Chương→Điều→Khoản containment).
Every regex-inferred and every curated cross-reference edge is created `pending`, and
retrieval's graph expansion walks `approved` edges only — so a clean clone cannot reach the
reviewed graph's multi-hop answers until a human reviews them:

```bash
python -m ingest.review --stats           # counts by edge_type / source / review_status
python -m ingest.review --list-pending --limit 20
python -m ingest.review --approve ID [ID ...]   # or --reject ID [ID ...]
```

There is no bulk-import of past review decisions yet (a follow-up — see Roadmap). A clean
ingest of today's corpus creates 3008 edges: 2588 structural (approved) and 420 pending — 399
regex and 21 curated. The dev database this project runs against day to day, where every
number in "Evaluation" was measured, has 3061: the same 2588 structural edges, 452 regex edges
(389 approved, 63 rejected) and 21 curated edges (all approved), every one of them reviewed by
a human. Its 53 extra regex edges are all `references` edges a human rejected, which a clean
ingest of today's corpus no longer creates. Traversal never walks a rejected edge, so they
change no answer. Every one of the 420 edges a clean ingest creates is also in the dev
database.

```bash
python -m app                          # http://localhost:8000/docs (reload in local env)
curl -s localhost:8000/health          # {"status":"ok","version":"0.1.0","environment":"local"}
```

`corpus/manifest.json` is the machine-readable index of that corpus. Paths are relative to the
repository root, so the same file works from the root and from `backend/`:

```json
{
  "documents": [
    {
      "id": "gas-decree-87-2018",
      "number": "87/2018/NĐ-CP",
      "title_vi": "Nghị định về kinh doanh khí",
      "doc_type": "decree",
      "issued": "2018-06-15",
      "effective": "2018-08-01",
      "status": "in_force",
      "source_url": "https://vbpl.vn/...",
      "fetched": "2026-08-29",
      "path": "corpus/gas-decree-87-2018.pdf",
      "meta": { "note": "free-form; e.g. an open verification question" }
    }
  ]
}
```

`doc_type` is one of `law · decree · circular · tcvn · qcvn`, `status` one of
`in_force · amending · expired`. Dates may be given as `YYYY`, `YYYY-MM` or `YYYY-MM-DD`; a
partial date is widened to its earliest day. Loading fails with a listing of every missing file
declared `acquired` (the default), so an absent statute surfaces as an error rather than as a
wrong answer.

A document can also be declared `"acquired": false`: part of the corpus, but its text is not on
disk — permanently for a copyrighted standard that must be purchased (see `SOURCES.md` §
Licensing), or until someone downloads it. Loading such a document never requires its file to
exist. It still gets a `documents` row and a graph node, so an edge naming it (e.g.
`REPLACED_BY`) still resolves, but the ingest pipeline skips it entirely — no units, no chunks,
no embeddings — and the stats table reports it separately (`N ingested, M metadata-only`) so a
partial corpus never looks complete.

## Optional: cross-encoder reranking

Off by default (`RERANKER=none`). When on, the passages retrieval found are scored again by a
cross-encoder, which reads each one *together with* the question, and only the best `RERANK_TOP_N`
reach the model, so it has fewer wrong articles to choose from. Ollama has no rerank endpoint, so
this runs as a small sidecar, `llama-server --rerank` from [llama.cpp](https://github.com/ggml-org/llama.cpp),
serving the `bge-reranker-v2-m3` GGUF. Why this runtime, what it costs, and how `RERANK_TOP_N` was
chosen: [`docs/adr/0002-cross-encoder-reranker-runtime.md`](docs/adr/0002-cross-encoder-reranker-runtime.md).

```bash
# the model: gpustack/bge-reranker-v2-m3-GGUF on Hugging Face (Q8_0, 636 MB, Apache-2.0)
llama-server -m bge-reranker-v2-m3-Q8_0.gguf --rerank --port 8081 -c 8192 -b 8192 -ub 8192 -np 1
RERANKER=llama-server RERANK_TOP_N=5 python -m app       # RERANKER_BASE_URL defaults to :8081
```

If the sidecar is not running, `/ask` logs a warning and answers exactly as it does with
`RERANKER=none`. The test suite never needs it: `RERANKER=mock` is the offline stand-in.

## Ask a question

Once the corpus is ingested and its edges reviewed, `python -m app` → `localhost:8000/docs` →
`POST /ask`. A real run against the dev database (Ollama, `bge-m3` + `qwen3:8b`), on `main`
`8b278ab`, 2026-11-17, pasted verbatim:

```
$ curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
    -d '{"question": "Cơ sở kinh doanh khí đốt có thuộc danh mục cơ sở thuộc diện quản lý về phòng cháy, chữa cháy không?"}'
```

```json
{
  "outcome": "answer",
  "answer": "Có, cơ sở kinh doanh khí đốt thuộc danh mục cơ sở thuộc diện quản lý về phòng cháy, chữa cháy theo quy định tại mục 13 Phụ lục I, Nghị định 105/2025/NĐ-CP.\nPhụ lục I, mục 13 (Nghị định 105/2025/NĐ-CP): \"Cơ sở kinh doanh khí đốt; cửa hàng xăng dầu.\"",
  "citations": [
    {
      "unit_id": "pccc-decree-105-2025#phu-luc-1#muc-13",
      "document_id": "pccc-decree-105-2025",
      "document_number": "105/2025/NĐ-CP",
      "dieu_no": null,
      "khoan_no": null,
      "appendix_no": 1,
      "item_no": 13,
      "effective": "2025-07-01"
    }
  ],
  "safety_response": null,
  "disclaimer": "Thông tin trên chỉ mang tính chất tham khảo, được tổng hợp tự động từ các văn bản pháp luật hiện có trong hệ thống tại thời điểm tra cứu và có thể chưa đầy đủ hoặc chưa được cập nhật mới nhất. Vui lòng đối chiếu với văn bản gốc và tham vấn cơ quan, tổ chức có thẩm quyền trước khi áp dụng vào thực tế.",
  "possibly_superseded": false,
  "refusal_reason": null
}
```

The citation is item 13 of the decree's Appendix I, the list of facilities under fire-safety
management, so `dieu_no`/`khoan_no` are `null` and `appendix_no`/`item_no` say where it is. The
second line of `answer` is issue #146's own deterministic addition: whenever a citation survives
down to a short appendix item or sub-item, its exact stored body is quoted verbatim underneath,
labelled with its Roman numeral and document number — no LLM call, and never applied to a
citation the grounding check would otherwise have dropped. Expect refusals too — see
"Evaluation" below for the system's measured answer/refusal behaviour on the gold set. A
refusal's `refusal_reason` says why, e.g. `unsupported_citations`, `verdict_contradicts_clause`,
or `context_overflow` if a prompt would not fit the model's context window (it is never answered
from a truncated prompt).

## Evaluation

Measured 2026-11-17 on `main` `8b278ab`, two `eval.compare` runs against
the reviewed dev database (`qwen3:8b`, `EMBEDDER=bge-m3`, Ollama `num_ctx` 16384 — the same
settings `/ask` itself runs with):

```
cd backend && DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply \
  EMBEDDER=bge-m3 LLM_PROVIDER=ollama OLLAMA_MODEL=qwen3:8b .venv/bin/python -m eval.compare
```

- Run 1: 2026-11-17 11:40:03 → 12:23:15.
- Run 2: 2026-11-17 12:23:15 → 13:04:14.
- **Gold set:** 46 rows, of which 34 answerable (21 compliance, 13 multi-hop), 5 refusal and 7
  emergency.

| metric (hybrid, all answerable n=34) | run 1 | run 2 | threshold |
|---|---|---|---|
| recall@k, strict | 0.868 | 0.868 | 0.800 (PASS) |
| citation accuracy | 0.783 | 0.788 | 0.850 (FAIL) |
| effective-answer rate | 0.971 | 0.971 | 0.900 (PASS) |
| safety detection | 1.000 | 1.000 | 1.000 (PASS) |
| refusal correctness | 1.000 | 1.000 | diagnostic |
| answer check | 5/7 | 5/7 | not gated |
| faithfulness | unmeasured | unmeasured | 0.900 |

**Faithfulness is unmeasured**: the only model available (`qwen3:8b` / `MockLLM`) is the same
model that generates the answers, so it cannot judge them independently. A stronger, separate
judge is required before this number means anything.

**Naive vector-only baseline, citation accuracy:** 0.138 in both runs — the GraphRAG-vs-naive
comparison this evaluation exists to make.

A second, encrypted **holdout** set exists for the same gold schema, scored only by the owner
outside any Claude session — see [`docs/holdout.md`](docs/holdout.md). No holdout number is
reported here.

## Known limitations

- Citation accuracy (strict) is below its 0.850 threshold — see "Evaluation" for the measured
  value.
- `multihop-gas-store-800kg-insurance` gets a wrong "Không" from the model in the current
  context, and the grounding/verdict checks correctly refuse it rather than ship the wrong
  answer.
- The 300 kg gas-store question, typed "... không ạ?", sometimes gets a wrong "Có" from the
  model; the threshold verdict check (issue #195/#196) refuses it rather than ship it.
- Unaccented questions (e.g. "co phai ... khong") are not detected as yes/no, so they do not get
  the yes/no answering instruction or the verdict check.
- A clean clone's graph edges all start `pending` — see "Local setup"; there is no bulk-import of
  past review decisions yet (Roadmap). Reviewing them with the same decisions as the dev
  database gives the same traversable graph. The dev database's 53 extra edges are rejected
  `references` edges, which traversal never walks.
- 3 of the corpus's 10 manifest documents are declared `"acquired": false` (`corpus/manifest.json`):
  `lpg-store-tcvn-6223-2017` (copyrighted, must be purchased), `lpg-qcvn-08-2019` (not yet
  downloaded), `lpg-qcvn-08-2012` (expired, kept only so its `REPLACED_BY` edge resolves).

## Testing

```bash
cd backend
ruff format --check . && ruff check .   # style + lint
mypy app ingest                         # strict type check
pytest -q                               # tests + coverage gate (≥ 80%)
```

Every pull request is gated on these same commands, run against a clean install on a fresh runner,
plus `pip-audit --strict` and a gitleaks scan of the diff — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml). A PR merges only when that run is green.

Database-backed tests need a **dedicated, disposable** Postgres database, never the one the app
itself runs against. They read `TEST_DATABASE_URL`, never `DATABASE_URL` (issue #64): configuring
the app to point at a real database must never be able to arm the fixture that truncates it before
every test. Create a local database whose name ends in `_test` — the fixture refuses to run
against one that doesn't — and point the tests at it:

```bash
docker exec gascomply-postgres-1 psql -U postgres -c "CREATE DATABASE gascomply_test;"
export TEST_DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply_test"
pytest -q
```

`TEST_DATABASE_URL` unset skips the database-backed tests locally, with a clear reason each time.
CI always sets it, pointed at its own `gascomply_test` service container, and fails on any skip.

## Project layout

```
backend/
  app/
    api/            routers (health, ask, documents)
    services/       retrieval, generation, graph traversal, safety
    repositories/   database access
    embedders/      swappable embedding providers (bge-m3, mock)
    rerankers/      cross-encoder rerankers (llama-server sidecar, mock)
    llm/            LLM providers (ollama, mock)
    schemas/        Pydantic models at every boundary
    core/           config, logging, ids, middleware
  ingest/           manifest → parser → references → chunk/embed → CLI
  eval/             gold set, metrics, GraphRAG-vs-naive comparison, holdout runner
  tests/
corpus/             legal source documents (git-ignored; manifest.json tracked)
```

## Roadmap

Planned, not implemented today:

- **LLM provider failover in production** (Gemini Flash / Groq), with Ollama staying the
  dev-only provider.
- **Observability:** Langfuse (LLM + retrieval traces) and Sentry.
- **An MCP tool** exposing retrieval as an agent-callable tool (FastMCP).
- **A frontend:** Next.js + TypeScript + Tailwind/shadcn, question box and citation panel.
- **Supabase-hosted Postgres** in production (dev/CI stay on the local `pgvector/pgvector:pg16`
  container either way).
- **Free-tier deployment:** Vercel (frontend) · Railway or HF Spaces (backend) · Supabase
  (Postgres + pgvector) · Gemini Flash/Groq (LLM). Target infrastructure cost: $0/month.
- **Exporting/replaying edge-review decisions**, so a clean clone can reach the reviewed graph
  without a manual review pass (see "Known limitations").

## Status

| Phase | Scope | State |
|---|---|---|
| 0.1 | Corpus manifest verified + written | ✅ done |
| 0.1 | Corpus download (10 documents) | 7 acquired; 3 declared `acquired: false` — TCVN 6223:2017 needs purchase, QCVN 08:2019/08:2012 have no clean source yet (owner). Decree 105/2025's first Gazette part ends at the end of Appendix VI; its second part (Appendix VII, the compulsory fire-insurance list, and Appendix VIII, 23 sample forms) is now ingested as the document's own `parts` (#126) |
| 1.1 | Repo scaffold, `/health`, locked ID scheme | ✅ done |
| 1.2 | Manifest loader + typed `Document` model | ✅ done |
| 1.3 | Structural parser (Chương/Điều/Khoản/Điểm) | ✅ done |
| 1.4–1.6 | References → chunk/embed → ingest CLI | ✅ done |
| 2.1, 2.2, 2.5 | Graph schema, reference edges, recursive-CTE traversal | ✅ done |
| 2.3 | LLM-assisted hard edges + human review | 2.3a (review CLI) done; 2.3b (LLM pass) not started |
| 2.4 | Requirement cluster — 8 obligations, each with its citation | ✅ done (curated document-level edges seeded — 12 `replaced_by` edges) |
| 3.1–3.7 | Hybrid retrieval, safety, generation, `/ask` | ✅ done — plus an optional cross-encoder reranker (#114, off by default), a per-citation grounding check (#118), appendices parsed as their own citable units (#117), a 16,384-token context window with an overflow refusal (#129), a threshold verdict check on yes/no answers (#195/#196), widened yes/no detection (#173/#198), and one retry without the passages the grounding verifier rejected (#202/#203) |
| 4.1 | Gold evaluation set — 46 verified questions (34 answerable: 21 compliance + 13 multi-hop; 5 refusal; 7 emergency) | ✅ done |
| 4.2–4.3 | Metrics with pass thresholds, GraphRAG-vs-naive table | See "Evaluation" above for the current measured numbers, date and commit |
| 5.1–5.3 | UI, free deploy, MCP tool | planned — see Roadmap |

How the work is run: [`WORKFLOW.md`](WORKFLOW.md) — one phase, one branch, one PR, with CI as the
merge gate. The detailed build plan and the reusable quality-standards template it draws on are kept
on the owner's machine rather than in the repository.
