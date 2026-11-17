-- Build plan A5.3: documents, units, chunks -- the three tables Phase 1.5/1.6
-- fill. entities, edges and gold_questions arrive with the phases that use
-- them (Phase 2+), not here.
--
-- Applied by `python -m app.core.migrate`, never automatically on application
-- boot (BUILD_PLAN_FOUNDATION.md Part E) -- the owner runs this deliberately.
--
-- One deliberate departure from the DDL as printed in BUILD_PLAN.md: a
-- UNIQUE constraint on chunks.unit_id, which the source DDL does not have.
-- Without it, "upsert by stable id" (this phase's own idempotency
-- requirement) has no conflict target to upsert onto for chunks. This is
-- safe: ingest/chunk.py's build_chunks produces at most one Chunk per unit
-- (one chunk per Article, or -- when an article is split -- one chunk per
-- Clause; never both for the same unit), so unit_id is already a real
-- candidate key in practice, just one the original DDL left unstated.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          text PRIMARY KEY,            -- 'gas-decree-87-2018'
    number      text NOT NULL,               -- '87/2018/NĐ-CP'
    title_vi    text NOT NULL,
    doc_type    text NOT NULL,               -- law | decree | circular | tcvn | qcvn
    issued      date,
    effective   date,
    status      text NOT NULL,               -- in_force | amending | expired
    source_url  text,
    fetched     date,
    path        text,
    meta        jsonb DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS units (            -- Chapter/Article/Clause/Point tree
    id          text PRIMARY KEY,             -- stable id per A5.1
    document_id text NOT NULL REFERENCES documents(id),
    unit_type   text NOT NULL,                -- chapter | article | clause | point
    chapter_no  int,
    dieu_no     int,
    khoan_no    int,
    diem        text,
    parent_id   text REFERENCES units(id),    -- BELONGS_TO in table form
    heading     text,
    body        text NOT NULL,
    ordinal     int NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    unit_id     text NOT NULL REFERENCES units(id),
    document_id text NOT NULL REFERENCES documents(id),
    dieu_no     int,
    khoan_no    int,
    effective   date,
    status      text NOT NULL,               -- denormalized for the date filter
    body        text NOT NULL,
    embedding   vector(1024),                -- app/repositories/chunks.py:CHUNKS_EMBEDDING_DIMENSION must match
    UNIQUE (unit_id)                          -- see note above: required for upsert-by-stable-id
);

CREATE INDEX IF NOT EXISTS chunks_embedding_ivfflat
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
