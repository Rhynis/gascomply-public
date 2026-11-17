-- Build plan A5.3: entities and edges -- the graph tables Phase 2 fills in
-- stages. `entities` stays empty until Phase 2.4 (requirement entities);
-- `edges` carries only BELONGS_TO rows from Phase 2.1, then gains
-- BASED_ON/GUIDED_BY/AMENDED_BY/REPLACED_BY/REFERENCES in Phase 2.2 and
-- APPLIES_TO in Phase 2.4. One migration per coherent schema change beats
-- one per phase that happens to need a column.
--
-- Applied by `python -m app.core.migrate`, never automatically on application
-- boot (Part E) -- the owner runs this deliberately.
--
-- `edges.src_id`/`dst_id` carry no foreign key: a node can be a document, a
-- unit, or (from Phase 2.4) an entity, and no single table has every one of
-- those ids as its primary key for a REFERENCES clause to point at. A5.1's
-- disjoint id namespaces (`#`-separated units, `<type>:`-prefixed entities,
-- bare document slugs) are what makes that safe -- the same string can never
-- mean two different nodes.

CREATE TABLE IF NOT EXISTS entities (
    id          text PRIMARY KEY,            -- 'req:min-store-area' (A5.1)
    entity_type text NOT NULL,               -- req | form | authority | concept | fee
    label_vi    text NOT NULL,
    meta        jsonb DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS edges (
    id            bigserial PRIMARY KEY,
    src_id        text NOT NULL,
    dst_id        text NOT NULL,
    edge_type     text NOT NULL,             -- belongs_to | based_on | guided_by |
                                              -- amended_by | replaced_by | references |
                                              -- applies_to
    evidence      text,
    confidence    real,
    review_status text NOT NULL DEFAULT 'pending',  -- pending | approved | rejected
    source        text NOT NULL,             -- structural | regex | llm
    UNIQUE (src_id, dst_id, edge_type)
);

CREATE INDEX IF NOT EXISTS edges_src_type ON edges (src_id, edge_type);
CREATE INDEX IF NOT EXISTS edges_dst_type ON edges (dst_id, edge_type);
