# ADR 0001: Drop the ivfflat index on `chunks.embedding`

## Status

Accepted (build plan A6 issue #49). Reverses `BUILD_PLAN.md` § A5.3's printed DDL.

## Context

`migrations/001_init.sql` carries the A5.3 DDL verbatim:

```sql
CREATE INDEX chunks_embedding_ivfflat
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

Every vector in the table was a deterministic `hashlib` value from `MockEmbedder` until issue #48 landed the real `bge-m3` embedder. The index's actual recall could not be checked before that -- there was no real geometry to check it against. It was checked the first time anyone queried the real, ingested corpus (692 chunks, real embeddings) by hand, with a real question:

> "Cửa hàng bán lẻ gas cần diện tích tối thiểu bao nhiêu?" ("What is the minimum area for an LPG retail shop?")

Measured against the real corpus, with `ivfflat.probes` at its default of 1, the index's top hit was `pccc-decree-105-2025#dieu-47#khoan-12` -- an unrelated clause. An exact, unindexed nearest-neighbor search on the same data returns `gas-decree-87-2018#dieu-53` ("Cửa hàng phải có diện tích tối thiểu là 12 m2 ... cách ly với các nguồn gây cháy ít nhất 03 m ... Thiết bị điện của cửa hàng phải là loại phòng nổ" -- minimum area, ignition-source distance, and electrical equipment type all in one article), which is the actual answer and the single most safety-critical article in this corpus for the product's own use case. **The wrong answer came back with no error and no warning** -- in a product whose entire premise is quoting the right article.

(The specific wrong id varies between index rebuilds -- ivfflat's clustering step is randomized -- but the failure mode reproduces every time: the approximate index's top-1 disagrees with the exact top-1 on this corpus.)

Postgres itself is not entirely silent about this -- `CREATE INDEX ... USING ivfflat` on a table this small emits `NOTICE: ivfflat index created with little data / DETAIL: This will cause low recall. / HINT: Drop the index until the table has more data.` at index-creation time. That notice was already there when `001_init.sql` first ran against this corpus; nothing in the ingest pipeline or migration tooling surfaces a `NOTICE` anywhere a person would see it, so it went unread until the wrong answer itself was noticed by hand. The *query* that returns the wrong neighbor raises nothing at all -- no error, no warning -- which is the actual defect this ADR fixes.

### Why

pgvector's own sizing guidance is `lists ≈ rows / 1000`. At 692 rows, `lists=100` is roughly two orders of magnitude too many lists for the data: each list holds on the order of 7 vectors, and the default `ivfflat.probes = 1` searches exactly one of them -- effectively reading about 1% of the table and calling it a nearest-neighbor search. Raising `probes` to 10 (reading ~70 chunks) still returned the wrong answer for this same question; only an exact scan found the correct one. `lists=100` was sized for a corpus that did not exist when A5.3 was written -- it is not wrong in general, it is wrong by two orders of magnitude at this corpus's actual size.

## Decision

**Drop the ivfflat index.** `migrations/003_drop_ivfflat_index.sql` runs `DROP INDEX IF EXISTS chunks_embedding_ivfflat`, with the reasoning summarized inline in the migration file itself. `chunks.embedding` is queried with an ordinary exact nearest-neighbor scan (`ORDER BY embedding <=> :query_vector LIMIT k`) until the table is large enough for an approximate index to earn its cost.

Measured on the real corpus, real embeddings, via `EXPLAIN ANALYZE` on the same query vector:

| | Execution time |
|---|---|
| ivfflat (`lists=100`, `probes=1`) | 0.806 ms |
| exact sequential scan, no index | 6.545 ms |

Under 6 milliseconds, in a system where a single LLM call takes seconds. The trade-off is not close: an approximate index buys back roughly 6 ms per query at the cost of returning the wrong article, silently, in a citation-grounded product where a wrong answer with a clean-looking source is the worst failure mode this project has.

### Reintroduction threshold

Re-introduce an ivfflat (or hnsw) index once the corpus is large enough that pgvector's own sizing rule gives it a real job to do. Applying `lists ≈ rows / 1000` the other direction: a `lists=100` index does not become well-sized until the table holds on the order of **100,000 chunks**. This project's corpus (currently 692 chunks from 7 of a planned 10 documents) reaches that scale only after Phase 6's corpus expansion, if at all -- not from completing the original ten-document corpus. When that phase adds the index back, size `lists` from the real row count measured at that time, not from this number -- the whole point of this ADR is that a value copied without measuring the data it indexes is how this defect happened the first time.

Whichever phase reintroduces it must also reintroduce a recall test proving the approximate index's top-k agrees with an exact scan on real data -- see `tests/test_search_recall.py`, added by this issue, which is the reason this defect will not go silent again.

## Alternatives considered

**Keep the index, set `lists=1`.** With `lists=1`, `ivfflat.probes=1` searches the entire single list -- i.e., degenerates to an exact scan while keeping the index's on-disk structure and its build/maintenance cost on every insert. This buys nothing over dropping the index outright at this corpus size, and keeps an approximate-search code path alive that the next person could reasonably assume is doing approximate work when it is not (a `lists=1` ivfflat index silently behaving like a sequential scan is its own kind of surprise). Rejected: no benefit, and a real (if small) cost, at this size.

**Keep the index, raise `probes` instead.** Measured directly: `probes=10` (~70 of 692 chunks read) still returned the wrong top hit for the test question. Reaching probes high enough to guarantee correctness at this row count converges to reading most or all of the table anyway -- at which point it is an exact scan with extra index-maintenance overhead, not a fix. Rejected: does not actually solve the correctness problem it was meant to solve, measured, not assumed.

**Do nothing, ship it, fix it when the LLM/RAG layer notices bad answers.** Rejected outright: the defect is invisible by construction (no error, no warning, a plausible-looking wrong citation), so "notice it later" means shipping wrong legal citations to a user first. This is exactly the failure mode the project's citation-grounding premise exists to prevent.

## Consequences

- Every vector query against `chunks.embedding` is an exact `O(n)` sequential scan until a future migration reintroduces an index, sized from measurement, above the ~100k-chunk threshold.
- No schema change to `chunks` itself -- only the index is dropped. `CHUNKS_EMBEDDING_DIMENSION` (1024) and the `vector(1024)` column are untouched (issue #48's embedder work is settled and correct; the vectors were never the problem).
- `tests/test_search_recall.py` is the standing guard: it asserts a real nearest-neighbor ordering on hand-constructed vectors, and separately asserts no approximate index exists on `chunks.embedding` at all, so a future migration that reintroduces one without also reintroducing a recall proof fails loudly instead of failing silently.
