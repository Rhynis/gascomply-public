-- Issue #68: record which text produced an edge, separately from where it starts.
--
-- `reconcile_document_edges` (issue #75) decides which document an edge belongs to,
-- so a re-ingest can delete what that document no longer produces. It did so by
-- `src_id`, which is right for every edge whose source is the text that states it
-- (a clause citing a document, a decree citing its legal basis) and wrong for the
-- one shape that runs the other way: REPLACED_BY is OLD -> NEW, but the repeal is
-- stated in the NEW document. The edge starts at the repealed document, often a
-- stub with no text, so no ingested document ever owned it: a stale one was never
-- pruned, and a repealed document that was itself in the manifest had the edge
-- deleted (or reported as a false conflict) by its own reconcile on every run.
--
-- `origin_id` is the unit (or the document, for a citation in a preamble) whose
-- text produced the edge. NULL for a curated edge, which no document's text
-- produces and which reconcile never touches.
--
-- Backfill: every existing non-curated edge was attributed by `src_id` -- that is
-- exactly the rule this column replaces -- so copying it keeps every row in the same
-- scope it was in before. Nothing changes meaning until a re-ingest rewrites the row.
--
-- Applied by `python -m app.core.migrate`, never automatically on application
-- boot (Part E). Run it before the next ingest: an unmigrated database fails the
-- first edge upsert loudly rather than silently.

ALTER TABLE edges ADD COLUMN IF NOT EXISTS origin_id text;

UPDATE edges SET origin_id = src_id WHERE source <> 'curated' AND origin_id IS NULL;
