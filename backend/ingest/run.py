"""Ingestion CLI: run the whole pipeline over a corpus and print a stats table.

``python -m ingest.run --corpus <path>``

Per document: ``extract_text_parts`` -> ``parse_document`` -> ``extract_references`` ->
``build_chunks`` -> the configured ``Embedder.embed`` (``mock`` by default;
``EMBEDDER=bge-m3`` for the real one, build plan A6 Phase 1.5c) ->
``build_structural_edges`` -> ``build_reference_edges``. Every one of those
pieces is already tested in isolation; nothing has run them end to end over
a whole corpus before this
module. The table this command prints is the deliverable, not incidental
evidence: it is how the owner will find out, the moment Phase 0.1's documents
land, whether the corpus actually ingested or quietly half-ingested. Silent
loss is this project's worst failure mode -- a statute that fails to ingest
produces no error anywhere, and the cite-or-refuse rule then makes the
assistant silently refuse a question the corpus should have answered. An
in-text citation that resolves to neither a unit nor a document is the same
kind of gap, one level down -- see the ``unresolved`` column (A6 Phase 2.2).

A document whose extraction or parsing raises is reported in the table with
its error, and the run continues to the next document -- one bad PDF must not
hide the other nine. The process still exits non-zero if any document failed,
so a half-ingested corpus can never look like a clean run to a script or to CI.
The summary line under the table counts those documents separately
(``N ingested, M metadata-only, K failed``), so a run in which every document
failed cannot read as ``7 ingested`` (issue #78).

Persistence (build plan A6 Phase 1.5b) is opt in: pass ``--database-url`` (or
set ``DATABASE_URL``) to upsert every document/unit/chunk/edge into Postgres,
one transaction per document (C4) so a document that fails partway leaves no
half-written rows. A document reported as failed for silent loss (zero units,
or units but zero chunks) is never persisted at all -- the database must not
disagree with the table about which documents ingested. With no database
configured -- the default -- the CLI behaves exactly as before: everything
above still runs, nothing is written anywhere but ``references.csv``.
``--dry-run`` forces that behaviour even when a database URL is present, e.g.
to smoke-test a corpus without touching the database. Migrations are never
applied from here -- see
``app/core/migrate.py`` and Part E.

Persistence also prunes (issue #117): a unit or chunk this run's parse no
longer produces for a document is deleted, not merely left stale, so the
database matches the parser's current understanding of the corpus rather
than accumulating every past parse's leftovers. That makes the parser's
output authoritative, which is exactly why a parse that would delete more
than half of a document's existing units is refused by default
(``LargePruneError``) rather than carried out quietly -- a parser regression
must fail loudly, the same way ``_silent_loss_reason`` already does for a
parse that produces nothing at all. ``--allow-large-prune`` is the explicit
override for a change that really does mean to shrink a document this much.

This module is a composition layer only -- argument parsing, iteration,
counting, printing -- with all real logic living in the modules it calls.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.db import create_engine, create_session_factory, session_scope
from app.core.logging import configure_logging
from app.embedders.base import Embedder
from app.embedders.factory import create_embedder
from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRepository
from app.repositories.edges import EdgeRepository, ReconcileResult, StaleEdgeConflict
from app.repositories.entities import EntityRepository
from app.repositories.units import UnitRepository
from app.schemas.chunk import Chunk
from app.schemas.document import Document, DocumentStub
from app.schemas.edge import Edge
from app.schemas.reference import Reference, ReferenceKind
from app.schemas.unit import UnitType
from ingest.chunk import build_chunks
from ingest.extract import extract_text_parts
from ingest.graph import (
    _DocumentLookup,
    _resolve,
    _self_document,
    build_document_edges,
    build_reference_edges,
    build_requirement_graph,
    build_structural_edges,
    document_edge_target_ids,
    load_document_edges,
    load_requirement_cluster,
)
from ingest.manifest import MANIFEST_FILENAME, Manifest, ManifestError, load_manifest
from ingest.parser import ParseResult, parse_document
from ingest.references import extract_references, write_references_csv

logger = logging.getLogger(__name__)

_SessionFactory = async_sessionmaker[AsyncSession]

REQUIREMENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "requirements.json"
"""The human-curated requirement seed file (build plan A5.6, A6 Phase 2.4).
Fixed, not a CLI flag: this is version-controlled project data, not a
per-invocation input the way ``--corpus`` is."""

DOCUMENT_EDGES_PATH = Path(__file__).resolve().parent.parent / "data" / "document_edges.json"
"""The human-curated document-level edge seed file (issue #63). Fixed, not a
CLI flag, for the same reason as ``REQUIREMENTS_PATH``."""

_COLUMNS: tuple[tuple[str, int], ...] = (
    ("document", 40),
    ("chapters", 9),
    ("articles", 9),
    ("clauses", 9),
    ("points", 9),
    ("skipped", 9),
    ("units", 9),
    ("chunks", 9),
    ("units_pruned", 13),
    ("chunks_pruned", 14),
    ("edges", 9),
    ("references", 11),
    ("unresolved", 12),
    ("no_stub", 9),
    ("self_cite_miss", 15),
    ("stubs", 9),
    ("entities", 9),
)
"""Table columns, each with a fixed ``str.ljust`` width (C12: standard library
only -- no ``rich``/``tabulate``). ``units`` and ``chunks`` sit side by side on
purpose: a document whose articles parse but do not chunk is exactly the gap
this table exists to surface. ``units_pruned``/``chunks_pruned`` sit right
after them for the same reason (issue #117's review, point 2): pruning makes
the parser's output authoritative over what is in the database, so how much
it just deleted must be as visible as how much it just produced, not a number
a reviewer has to go query for."""


@dataclass
class DocumentStats:
    """One document's outcome for one row of the stats table.

    Attributes:
        document_id: The document's identifier, or its manifest id when
            extraction/parsing failed before a ``ParseResult`` existed.
        chapters: Chapters the parser placed in the tree.
        articles: Articles the parser placed in the tree.
        clauses: Clauses the parser placed in the tree.
        points: Points the parser placed in the tree.
        skipped: Anchors the parser could not place.
        units: Total structural units produced (chapters+articles+clauses+points).
        chunks: Embedding-ready chunks produced.
        edges: Edges this row's pass built -- attempted to write, whether or
            not the row already existed from an earlier pass or an earlier
            run (structural BELONGS_TO, A6 Phase 2.1, plus resolved
            reference edges, A6 Phase 2.2, for a per-document row; every
            curated seed edge for the ``requirements``/``document_edges``
            pseudo-rows, issue #63). Uniform across every row rather than a
            shorthand for "newly inserted" -- see ``edges_updated`` for that
            narrower distinction. Summing this column across rows no longer
            equals the true row count in ``edges`` (issue #75): two passes
            in one run can declare the same edge -- a regex-produced
            reference and a curated document edge sharing endpoints -- and a
            repeat ingest re-declares edges that already existed. The TOTAL
            row's own ``edges`` value is queried directly from the database
            instead (``EdgeRepository.count``), precisely to sidestep that
            double counting.
        references: Draft cross-reference rows detected.
        unresolved: References whose target named neither a unit in this
            document nor a document in the manifest, so no edge could be
            built for them (A6 Phase 2.2). The gap between ``references``
            and an edge actually landing in the graph.
        self_citation_misses: Internal-kind references -- a document citing
            its own article or clause -- that do not resolve to a real unit
            in this document's own tree (issue #69). A document citing
            itself is ground truth that needs no human labelling: the
            target must exist, so every miss here is content lost somewhere
            in the pipeline (the parser never built the unit, or the
            reference was misclassified as internal when it names a
            different document). Always <= ``unresolved``, since it counts
            one kind out of the four.
        no_stub: The part of ``unresolved`` that named another document but
            got no stub (issue #67): cited by title only, a TCVN/QCVN
            standard, an amendment reference (never stubbed -- see
            ``ingest.graph``), or a number that failed the strict
            complete-number pattern (``rejected_numbers``). What is left of
            ``unresolved`` once ``no_stub`` is subtracted is exactly
            ``self_citation_misses``, so every unresolved reference is
            accounted for in one of the two columns.
        rejected_numbers: The distinct document numbers this document cites
            that failed the strict pattern, sorted -- each a number cut short
            by a PDF artifact ("78/2021/NĐ" for "78/2021/NĐ- CP"), named so
            the extraction defect behind it is visible, not just counted.
            Printed as a WARNING by ``_print_table``.
        stub_ids: Ids of the stub documents (issue #67) this document's
            citations resolve to -- documents known only from a citation, with
            no text, so never counted as ingested. The table's ``stubs``
            column is ``len(stub_ids)`` per row, and the TOTAL row counts the
            *union*, not the sum: two documents citing the same decree share
            one stub, so summing would count it twice.
        repeal_edges: How many whole documents this document's text repeals,
            built into REPLACED_BY edges old -> new (issue #68) and included in
            ``edges``. Printed as an INFO line, not a column: it is zero for
            almost every row.
        partial_repeals: Repeal acts whose object is not a whole document -- a
            clause, a point, a chapter, a phrase -- counted and skipped, never
            an edge (issue #68). Repealing part of a decree is not replacing it,
            and a REPLACED_BY there would remove a live decree from every answer.
        unclear_repeals: Statements about a document that fit no shape the
            repeal pass reads, counted and skipped -- the ones worth a human's
            eye. Printed as a WARNING.
        error: Why this document is treated as failed, or ``None`` on
            success. Covers three cases: an exception raised anywhere in the
            pipeline ("raised: <message>"), a document that extracted and
            parsed without error but produced zero units ("ingested to zero
            units"), and a document that produced units but zero chunks
            ("parsed <n> units but produced zero chunks"). The latter two
            raise nothing, so without this they would report as success --
            the purest form of silent loss this CLI exists to catch. A row
            with an ``error`` is counted as ``failed`` in the summary line,
            never as ``ingested`` (issue #78).
        empty_articles: Ids of articles that parsed but produced no chunk at
            any level (neither as themselves nor, when split, as any clause).
        entities: Requirement/concept entities produced (A6 Phase 2.4). Zero
            for every real document row -- the requirement cluster spans
            documents, so it is reported as its own pseudo-document row
            (``document_id="requirements"``) after the per-document loop,
            not attributed to any single statute.
        metadata_only: ``True`` when this row is a document declared with
            ``acquired=False`` (issue #59): every pipeline column is
            legitimately zero because the text pipeline never ran, not
            because anything failed. Kept separate from ``error`` so this
            row's own zeros are never mistaken for silent loss, and counted
            in the summary line ``_print_table`` prints after the table.
        edges_updated: For the ``document_edges`` pseudo-row only (issue
            #63's review): how many of its curated edges, included in
            ``edges``, already existed as a regex-produced row and were only
            updated (``source`` flipped to ``curated``), not inserted.
        edges_removed: How many of this document's stale edges -- built by
            an earlier parse or an earlier run, no longer produced by this
            one -- were deleted this run (issue #75). Always edges that were
            still ``PENDING``; an approved or rejected stale edge is never
            deleted, see ``edge_conflicts``. Zero for the
            ``requirements``/``document_edges`` pseudo-rows, whose curated
            edges are excluded from reconciliation entirely (see
            ``EdgeRepository.reconcile_document_edges``).
        edge_conflicts: Stale edges this document no longer produces that
            were left in place because a human had already approved or
            rejected them (issue #75) -- a disagreement between a person's
            past decision and the current evidence, which this pipeline
            reports rather than resolving on its own. Printed as a WARNING
            per row by ``_print_table``.
        units_pruned: Units deleted this run because the current parse no
            longer produces them (issue #117's review, point 2). Zero on a
            first ingest, and zero on a re-ingest of an unchanged corpus
            (the idempotency A6 Phase 1.5b's persistence was built on, #24).
        chunks_pruned: Chunks deleted this run for the same reason, counted
            separately from ``units_pruned`` because the two numbers can
            legitimately differ: a unit can survive a re-parse (still a row
            in ``units_pruned``'s complement) while the chunk built *from*
            it does not, because ``build_chunks`` rebundles a shortened
            article's clauses into one article-level chunk instead of one
            per clause. That gap -- chunks pruned by surviving unit ids
            instead of by the chunk ids the parse actually produced -- is
            exactly what issue #117's review found still keeping stale
            clause-level text in the search index after the unit-level fix.
    """

    document_id: str
    chapters: int = 0
    articles: int = 0
    clauses: int = 0
    points: int = 0
    skipped: int = 0
    units: int = 0
    chunks: int = 0
    entities: int = 0
    edges: int = 0
    references: int = 0
    unresolved: int = 0
    self_citation_misses: int = 0
    error: str | None = None
    empty_articles: list[str] = field(default_factory=list)
    metadata_only: bool = False
    edges_updated: int = 0
    units_pruned: int = 0
    chunks_pruned: int = 0
    edges_removed: int = 0
    edge_conflicts: tuple[StaleEdgeConflict, ...] = ()
    no_stub: int = 0
    rejected_numbers: tuple[str, ...] = ()
    stub_ids: frozenset[str] = frozenset()
    repeal_edges: int = 0
    partial_repeals: int = 0
    unclear_repeals: int = 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ingestion pipeline over a corpus and print the stats table.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        ``0`` if every document ingested without error, ``1`` otherwise --
        including when the manifest itself fails to load.
    """
    return asyncio.run(_amain(argv))


async def _amain(argv: Sequence[str] | None) -> int:
    """Async body of ``main`` -- persisting requires awaiting the repositories."""
    configure_logging()
    args = _parse_args(argv)

    try:
        manifest = load_manifest(Path(args.corpus) / MANIFEST_FILENAME)
    except ManifestError as error:
        logger.error("%s", error)
        return 1

    documents = manifest.documents[: args.limit] if args.limit else manifest.documents
    settings = get_settings()
    embedder = create_embedder(settings)
    logger.info("Embedder: %s", settings.embedder.value)

    session_factory: _SessionFactory | None = None
    engine = None
    if args.database_url and not args.dry_run:
        engine = create_engine(args.database_url)
        session_factory = create_session_factory(engine)
        logger.info("Persisting to %s", args.database_url)
    else:
        logger.info("Dry run: no rows will be written to a database.")

    per_document: list[DocumentStats] = []
    all_references: list[Reference] = []
    all_unit_ids: set[str] = set()
    all_document_ids: set[str] = set()
    for document in documents:
        stats, references, unit_ids = await _ingest_document(
            document,
            manifest,
            embedder,
            session_factory,
            allow_large_prune=args.allow_large_prune,
        )
        per_document.append(stats)
        all_references.extend(references)
        all_unit_ids.update(unit_ids)
        if stats.error is None:
            all_document_ids.add(stats.document_id)

    per_document.append(await _ingest_requirement_cluster(manifest, all_unit_ids, session_factory))
    per_document.append(await _ingest_document_edges(manifest, all_document_ids, session_factory))

    edges_in_db: int | None = None
    if session_factory is not None:
        async with session_scope(session_factory) as session:
            edges_in_db = await EdgeRepository(session).count()

    if engine is not None:
        await engine.dispose()

    write_references_csv(all_references, Path(args.references))
    _print_table(per_document, edges_in_db=edges_in_db)

    return 1 if any(stats.error is not None for stats in per_document) else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m ingest.run",
        description="Ingest a corpus (manifest + documents) and print a stats table.",
    )
    parser.add_argument(
        "--corpus", required=True, help="Directory holding manifest.json and the documents."
    )
    parser.add_argument(
        "--references",
        default="references.csv",
        help="Where the draft reference rows go (default: references.csv).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Ingest only the first N documents."
    )
    parser.add_argument(
        "--database-url",
        default=get_settings().database_url,
        help=(
            "Postgres URL to persist into (default: the DATABASE_URL setting/env "
            "var). Omit, or pass --dry-run, to run without a database."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Never write to the database, even if --database-url is set.",
    )
    parser.add_argument(
        "--allow-large-prune",
        action="store_true",
        help=(
            "Persist a document even when its new parse would prune more than half "
            "of its existing units (see LargePruneError). Refused by default."
        ),
    )
    return parser.parse_args(argv)


async def _ingest_document(
    document: Document,
    manifest: Manifest,
    embedder: Embedder,
    session_factory: _SessionFactory | None,
    *,
    allow_large_prune: bool,
) -> tuple[DocumentStats, tuple[Reference, ...], tuple[str, ...]]:
    """Run one document through the full pipeline; never raises.

    Any failure -- extraction, parsing, persistence, or anything unexpected
    further down the pipeline -- is caught, logged with its traceback, and
    reported as this document's error rather than aborting the run. A
    document that raises nothing but still contributes zero units or zero
    chunks is reported as failed too (see ``_silent_loss_reason``) and, when
    persistence is on, is not written to the database at all -- the CLI's
    own failure report must not disagree with what actually landed in
    Postgres.

    A document declared ``acquired=False`` (issue #59) never reaches any of
    that: there is no file to extract, so the text pipeline is skipped
    entirely. Persistence, when on, still upserts the document's own row --
    a graph node other documents' edges can name -- but no unit, chunk, or
    edge. Because it never calls ``_persist_document``, it never reconciles
    edges either -- exactly right, since a metadata-only document produces
    no edges this run and none of its (nonexistent) edges should be touched
    (issue #75).

    When persistence is on and the document does persist, its edges are
    also reconciled against what this parse just built: a database row this
    document no longer produces is deleted if nothing has reviewed it yet,
    or reported as a conflict if a human already approved or rejected it
    (see ``EdgeRepository.reconcile_document_edges``). A document narrowed
    out by ``--limit``, or one that fails, never reaches that call either,
    so its edges are left exactly as they were.

    Args:
        document: The manifest entry to ingest.
        manifest: Owning manifest, for resolving ``document``'s file path.
        embedder: Embedder to run over each chunk's body.
        session_factory: Session factory to persist through, or ``None`` for
            a dry run that never touches a database.
        allow_large_prune: Passed straight through to ``_persist_document``
            -- see ``_large_prune_reason``. A document refused for this
            reason raises inside the ``try`` below and is reported with
            that error, exactly like any other persistence failure.

    Returns:
        This document's stats row, its draft references, and its unit ids --
        the last one empty unless the document fully succeeded (no error),
        since only then are its units actually persisted (A6 Phase 2.4's
        requirement cluster checks its citations against this exact set,
        never against a unit that a failed or silently-empty document only
        claims to have produced).
    """
    if not document.acquired:
        if session_factory is not None:
            async with session_scope(session_factory) as session:
                await DocumentRepository(session).upsert(document)
        return DocumentStats(document_id=document.id, metadata_only=True), (), ()

    try:
        text = extract_text_parts(manifest.parts_of(document))
        result = parse_document(text, document.id)
        references = extract_references(result, _other_document_titles(manifest, document))
        chunks = build_chunks(result, document)
        embeddings = await embedder.embed([chunk.body for chunk in chunks])
        structural_edges = build_structural_edges(result)
        reference_resolution = build_reference_edges(references, manifest, result)
        edges = structural_edges + reference_resolution.edges
        silent_loss = _silent_loss_reason(result, chunks)
        persisted = _PersistResult(reconcile=ReconcileResult(removed=0, conflicts=()))
        if session_factory is not None and silent_loss is None:
            parsed = _ParsedDocument(
                result=result,
                chunks=chunks,
                embeddings=embeddings,
                edges=edges,
                stubs=reference_resolution.stubs,
            )
            persisted = await _persist_document(
                session_factory, document, parsed, allow_large_prune=allow_large_prune
            )
    except Exception as error:  # one bad document must not abort the whole run
        logger.exception("%s: ingestion failed", document.id)
        return DocumentStats(document_id=document.id, error=f"raised: {error}"), (), ()

    stats = DocumentStats(
        document_id=document.id,
        chapters=result.stats.chapters,
        articles=result.stats.articles,
        clauses=result.stats.clauses,
        points=result.stats.points,
        skipped=result.stats.skipped,
        units=len(result.units),
        chunks=len(chunks),
        edges=len(edges),
        references=len(references),
        unresolved=reference_resolution.unresolved,
        self_citation_misses=_count_self_citation_misses(references, manifest, result),
        empty_articles=_articles_without_chunks(result, chunks),
        error=silent_loss,
        units_pruned=persisted.units_pruned,
        chunks_pruned=persisted.chunks_pruned,
        edges_removed=persisted.reconcile.removed,
        edge_conflicts=persisted.reconcile.conflicts,
        no_stub=reference_resolution.no_stub,
        rejected_numbers=reference_resolution.rejected_numbers,
        stub_ids=_stub_ids(reference_resolution.stubs, silent_loss),
        repeal_edges=reference_resolution.repeal_edges,
        partial_repeals=reference_resolution.partial_repeals,
        unclear_repeals=reference_resolution.unclear_repeals,
    )
    unit_ids = () if silent_loss is not None else tuple(unit.id for unit in result.units)
    return stats, references, unit_ids


def _stub_ids(stubs: Sequence[DocumentStub], silent_loss: str | None) -> frozenset[str]:
    """Return the stub ids a document's stats row should report.

    A document reported as failed for silent loss is never persisted, so
    neither are its stubs -- and the ``stubs`` column must not claim nodes the
    database does not hold, the same rule that keeps such a document out of
    ``documents`` in the first place.
    """
    if silent_loss is not None:
        return frozenset()
    return frozenset(stub.id for stub in stubs)


def _print_repeal_lines(stats: DocumentStats) -> None:
    """Print what the repeal pass built and what it counted without building (issue #68).

    Nothing for a document with no repeal statement at all. A document whose
    repeals are all partial still prints, with zero edges: that is the line that
    shows a partial repeal was seen and deliberately not turned into a
    REPLACED_BY.
    """
    if stats.repeal_edges or stats.partial_repeals or stats.unclear_repeals:
        print(
            f"  INFO: {stats.document_id} {stats.repeal_edges} repeal edge(s) built "
            f"(replaced_by, old -> new); {stats.partial_repeals} partial repeal(s) "
            "counted and skipped, never an edge"
        )
    if stats.unclear_repeals:
        print(
            f"  WARNING: {stats.document_id} {stats.unclear_repeals} repeal statement(s) "
            "name a document in a shape the repeal pass does not read, so no edge"
        )


def _other_document_titles(manifest: Manifest, document: Document) -> tuple[str, ...]:
    """Return every other manifest document's title, for title-shaped external citations.

    "quy định tại ... <title>" (issue #71) may name any document in the
    manifest by its exact title instead of its number -- this is the closed
    set ``extract_references`` matches such a citation against. This
    document's own title is deliberately excluded: a document citing its
    own full title (instead of the usual "... này" self-reference) must
    still stay internal, never manufacture a same-document edge.
    """
    return tuple(other.title_vi for other in manifest.documents if other.id != document.id)


async def _ingest_requirement_cluster(
    manifest: Manifest, unit_ids: set[str], session_factory: _SessionFactory | None
) -> DocumentStats:
    """Build and persist the requirement cluster; never raises (A6 Phase 2.4).

    Runs once, after every document's own loop iteration, against the union
    of every unit id actually ingested this run -- ``data/requirements.json``
    cites units across more than one statute, so it cannot be attributed to
    any single document's own transaction. Reported as its own pseudo-document
    row (``document_id="requirements"``) so a missing or miswired requirement
    fails the run the same way a bad document does, through the one existing
    exit-code check, rather than a second one this function would need to
    invent.

    Skipped entirely -- zero entities, zero edges, no error -- when ``manifest``
    includes none of the documents the cluster cites at all: a run over an
    unrelated corpus subset (a single other statute; this project's own test
    suite) has nothing to validate the cluster against, which is a different,
    unremarkable situation from a run that does include one of those
    documents but is somehow missing the specific unit a requirement names.
    The real corpus run this cluster exists for always includes both
    documents it currently cites, so this never masks the check the issue
    actually asked for -- it only keeps an unrelated ingest from being forced
    to validate curated data about statutes it was never ingesting.

    Args:
        manifest: The manifest this run is ingesting, to decide whether the
            cluster is in scope at all.
        unit_ids: Every unit id successfully ingested this run, across all
            documents.
        session_factory: Session factory to persist through, or ``None`` for
            a dry run that never touches a database.

    Returns:
        A stats row with ``entities``/``edges`` set on success (including the
        skipped case, both zero), or ``error`` set -- and nothing persisted --
        if the seed file failed to load or any in-scope requirement cited a
        unit that does not exist.
    """
    try:
        cluster = load_requirement_cluster(REQUIREMENTS_PATH)
        manifest_document_ids = {document.id for document in manifest.documents}
        if cluster.target_document_ids().isdisjoint(manifest_document_ids):
            logger.info("Requirement cluster: no cited document in this manifest, skipping.")
            return DocumentStats(document_id="requirements")
        entities, edges = build_requirement_graph(cluster, unit_ids)
        if session_factory is not None:
            async with session_scope(session_factory) as session:
                await EntityRepository(session).upsert_many(entities)
                await EdgeRepository(session).upsert_many(edges)
    except Exception as error:  # a bad requirement must not abort the whole run
        logger.exception("requirement cluster: ingestion failed")
        return DocumentStats(document_id="requirements", error=f"raised: {error}")
    return DocumentStats(document_id="requirements", entities=len(entities), edges=len(edges))


async def _ingest_document_edges(
    manifest: Manifest, ingested_document_ids: set[str], session_factory: _SessionFactory | None
) -> DocumentStats:
    """Build and persist the curated document-level edges; never raises (issue #63).

    Runs last -- after the per-document loop and after the requirement
    cluster -- deliberately: a document-level edge here can share
    ``(src_id, dst_id, edge_type)`` with one the regex pass already built
    for the same document this run, and ``edges``' ``UNIQUE`` constraint
    means the later upsert wins the ``source`` column (see
    ``ingest.graph``'s own module docstring for why curated should win, not
    just happen to -- that guarantee is entirely a property of this call
    running last; moving it changes the outcome). Reported as its own
    pseudo-document row (``document_id="document_edges"``) for the same
    reason the requirement cluster is: a bad seed fails the run through the
    one existing exit-code check, and ``_format_summary`` excludes it from
    "N ingested" the same way it already excludes ``"requirements"``.

    Skipped entirely -- zero edges, no error -- when ``manifest`` includes
    none of the documents these seeds name at all, the same rule
    ``_ingest_requirement_cluster`` follows and for the same reason: this
    project's own test suite ingests manifests these seeds have nothing to
    say about. Once in scope, though, validation is strict, matching the
    requirement cluster's own decision rather than diverging from it: a
    seed endpoint that is not in ``ingested_document_ids`` -- most often
    because ``--limit`` truncated the run, but also a document that failed
    outright -- fails the whole run loudly, the same way a requirement
    citing a unit outside the ingested corpus does. The alternative,
    silently skipping the seed on a partial run, would hide exactly the
    kind of gap this CLI's stats table exists to surface (issue #63's
    review); a curated edge whose endpoint the manifest declares but this
    run never reached must not be written at all (#61's review already
    showed what a dangling endpoint does downstream), and "not written" has
    to be visible, not quiet.

    Args:
        manifest: The manifest this run is ingesting, to decide whether the
            seeds are in scope at all.
        ingested_document_ids: Every document id that actually has a
            ``documents`` row after this run (metadata-only documents
            included) -- what every endpoint is validated against. Not the
            same set as the manifest's own document ids whenever
            ``--limit`` or a failed document narrows what this run actually
            persisted.
        session_factory: Session factory to persist through, or ``None`` for
            a dry run that never touches a database.

    Returns:
        A stats row on success (including the skipped case, all zero), or
        ``error`` set -- and nothing persisted -- if the seed file failed to
        load or named a document this run never persisted. ``edges`` counts
        every seed edge this pass built, whether inserted or updated;
        ``edges_updated``, a subset of that count, says how many already
        existed as a regex-produced row and were only updated. In a dry
        run, nothing is queried to tell new from existing, so
        ``edges_updated`` stays 0.
    """
    try:
        seeds = load_document_edges(DOCUMENT_EDGES_PATH)
        manifest_document_ids = {document.id for document in manifest.documents}
        if document_edge_target_ids(seeds).isdisjoint(manifest_document_ids):
            logger.info("Document edges: no cited document in this manifest, skipping.")
            return DocumentStats(document_id="document_edges")
        edges = build_document_edges(seeds, ingested_document_ids)
        updated = 0
        if session_factory is not None:
            async with session_scope(session_factory) as session:
                repository = EdgeRepository(session)
                existing = await repository.existing_triples(edges)
                updated = sum(
                    1
                    for edge in edges
                    if (edge.src_id, edge.dst_id, edge.edge_type.value) in existing
                )
                await repository.upsert_many(edges)
    except Exception as error:  # a bad seed must not abort the whole run
        logger.exception("document edges: ingestion failed")
        return DocumentStats(document_id="document_edges", error=f"raised: {error}")
    return DocumentStats(document_id="document_edges", edges=len(edges), edges_updated=updated)


@dataclass(frozen=True)
class _ParsedDocument:
    """One document's pipeline output, ready to persist.

    Bundled into one object, rather than passed to ``_persist_document`` as
    separate parameters, to keep that function's own signature within C10's
    5-parameter limit as later phases (2.4's entities) add more to persist.
    """

    result: ParseResult
    chunks: Sequence[Chunk]
    embeddings: Sequence[list[float]]
    edges: Sequence[Edge]
    stubs: Sequence[DocumentStub] = ()


@dataclass(frozen=True)
class _PersistResult:
    """Outcome of persisting one document: pruning plus edge reconciliation.

    Attributes:
        reconcile: What ``EdgeRepository.reconcile_document_edges`` reported
            for this document's non-structural edges.
        units_pruned: How many units this run deleted (issue #117's review,
            point 2) -- ``DocumentStats.units_pruned``'s source.
        chunks_pruned: How many chunks this run deleted, independently of
            ``units_pruned`` -- see ``DocumentStats.chunks_pruned`` for why
            the two can differ.
    """

    reconcile: ReconcileResult
    units_pruned: int = 0
    chunks_pruned: int = 0


class LargePruneError(RuntimeError):
    """Raised when a parse would prune more than half of a document's existing units.

    Pruning (issue #117) makes the parser's output authoritative: once a
    document persists, a row the current parse does not produce is deleted,
    where before it was merely never updated. That turns a parser regression
    into data loss instead of a no-op -- the PR that added pruning itself
    records a real near-miss, a premature signature-mode transition that once
    made both QCVN documents "lose their entire document". ``_silent_loss_reason``
    catches a parse that produces nothing at all; this catches the partial
    case that still slips past it, the way ``_silent_loss_reason`` fails loudly
    instead of silently persisting a half-empty document. ``--allow-large-prune``
    is the explicit override for a change that really does mean to shrink a
    document this much.
    """


def _large_prune_reason(
    existing_unit_ids: Sequence[str], keep_unit_ids: Sequence[str], *, allow_large_prune: bool
) -> str | None:
    """Return why this document's prune should be refused, or ``None`` if it is safe.

    Mirrors ``_silent_loss_reason``'s shape: a pure function from
    already-known state to an optional reason string, checked before
    anything is written. Only units are checked (issue #117's review, point
    2) -- chunks follow whatever the chunker does with the units that
    survive a parse, and are not an independent signal of how much a
    document actually shrank. A document with no existing units yet (a
    first ingest) is never refused: there is nothing to lose, so every
    unit is "new", not "pruned".

    Args:
        existing_unit_ids: This document's unit ids already in the database,
            queried before anything in this run's transaction has deleted or
            written to them.
        keep_unit_ids: This document's unit ids produced by the current
            parse -- what would survive the prune.
        allow_large_prune: ``--allow-large-prune``, the explicit override.

    Returns:
        A loud, specific reason naming the counts, or ``None`` when the
        prune is small enough to proceed without one.
    """
    if allow_large_prune or not existing_unit_ids:
        return None
    keep = set(keep_unit_ids)
    pruned = [unit_id for unit_id in existing_unit_ids if unit_id not in keep]
    if len(pruned) > len(existing_unit_ids) / 2:
        return (
            f"this parse would prune {len(pruned)} of {len(existing_unit_ids)} existing "
            "units (more than half) -- refusing to persist; pass --allow-large-prune "
            "to override"
        )
    return None


async def _persist_document(
    session_factory: _SessionFactory,
    document: Document,
    parsed: _ParsedDocument,
    *,
    allow_large_prune: bool,
) -> _PersistResult:
    """Upsert one document's units, chunks and edges inside a single transaction (C4).

    A failure partway rolls back everything for this document -- a partially
    ingested document never leaves half-written rows, and a re-run over the
    same corpus upserts by stable id (A5.1) rather than duplicating, which is
    what makes running this CLI twice idempotent. A refused large prune
    (``LargePruneError``) is one such failure: it raises before the document
    row itself is upserted, so a refused document is left completely
    untouched, not partially updated.

    Chunks are pruned against ``parsed.chunks``' own ids, never against
    ``parsed.result.units``' (issue #117's review, point 1): a unit can
    survive a re-parse while the chunk ``build_chunks`` built from it does
    not, because a shortened article gets rebundled into one article-level
    chunk instead of one per clause. Pruning chunks by surviving unit ids
    left those old clause-level chunks embedded and retrievable forever,
    keeping the issue's original bug alive in the search index even after
    the unit-level fix landed.

    After the upsert, this document's edges are reconciled against
    ``parsed.edges`` -- this run's own freshly built set -- in the same
    transaction, so a stale row this parse no longer produces is deleted (or
    reported, if a human already reviewed it) atomically with the rows this
    parse did produce (issue #75).

    Args:
        session_factory: Session factory for this document's transaction.
        document: The document row to upsert.
        parsed: This document's pipeline output -- parse result, chunks,
            embeddings and edges.
        allow_large_prune: ``--allow-large-prune``; see ``_large_prune_reason``.

    Returns:
        How many units and chunks were pruned, plus the edge reconciliation
        outcome: how many stale, unreviewed edges were deleted, and which
        stale, reviewed edges were left in place as conflicts for a human
        to resolve.

    Raises:
        LargePruneError: If this parse would prune more than half of the
            document's existing units and ``allow_large_prune`` is ``False``.
            Nothing is written when this raises.
    """
    async with session_scope(session_factory) as session:
        unit_repository = UnitRepository(session)
        existing_unit_ids = await unit_repository.existing_ids(document.id)
        keep_unit_ids = tuple(unit.id for unit in parsed.result.units)
        reason = _large_prune_reason(
            existing_unit_ids, keep_unit_ids, allow_large_prune=allow_large_prune
        )
        if reason is not None:
            raise LargePruneError(reason)

        document_repository = DocumentRepository(session)
        await document_repository.upsert(document)
        await document_repository.insert_stubs(parsed.stubs)

        keep_chunk_ids = tuple(chunk.unit_id for chunk in parsed.chunks)
        chunk_repository = ChunkRepository(session)
        pruned_chunk_ids = await chunk_repository.prune_document(document.id, keep_chunk_ids)

        pruned_unit_ids = await unit_repository.prune_document(document.id, keep_unit_ids)

        edge_repository = EdgeRepository(session)
        await edge_repository.delete_structural_for_units(pruned_unit_ids)
        await unit_repository.upsert_many(parsed.result.units)
        await chunk_repository.upsert_many(parsed.chunks, parsed.embeddings)
        await edge_repository.upsert_many(parsed.edges)
        reconciled = await edge_repository.reconcile_document_edges(document.id, parsed.edges)
        return _PersistResult(
            reconcile=reconciled,
            units_pruned=len(pruned_unit_ids),
            chunks_pruned=len(pruned_chunk_ids),
        )


def _silent_loss_reason(result: ParseResult, chunks: Sequence[Chunk]) -> str | None:
    """Return why a document that raised nothing still contributed nothing, or ``None``.

    A document can extract and parse without error and still be worthless: a
    page with no ``Chương``/``Điều``/``Khoản`` anchor at all (e.g. vbpl.vn's
    anti-bot JavaScript page, which extracts well over the near-empty floor
    but contains zero occurrences of "Điều") parses to zero units, and a
    document with only chapter-level structure parses to units that still
    produce zero chunks (``build_chunks`` only chunks articles and their
    descendants). Neither raises, so without this check both would report as
    success -- the purest form of silent loss this CLI exists to catch.
    """
    if not result.units:
        return "ingested to zero units"
    if not chunks:
        return f"parsed {len(result.units)} units but produced zero chunks"
    return None


def _articles_without_chunks(result: ParseResult, chunks: Sequence[Chunk]) -> list[str]:
    """Return the id of every article whose text produced no chunk at all.

    ``build_chunks`` silently drops a unit whose composed text is empty,
    logging it only at DEBUG. An unsplit article's own chunk carries its own
    ``dieu_no``; a split article's clause chunks all carry that same
    ``dieu_no`` too -- so an article is genuinely empty, not just split, only
    when its ``dieu_no`` appears in no chunk whatsoever.
    """
    chunked_dieu_nos = {chunk.dieu_no for chunk in chunks}
    return [
        unit.id
        for unit in result.units
        if unit.unit_type is UnitType.ARTICLE and unit.dieu_no not in chunked_dieu_nos
    ]


def _count_self_citation_misses(
    references: Sequence[Reference], manifest: Manifest, result: ParseResult
) -> int:
    """Count internal references that do not resolve to a real unit (issue #69).

    A document citing its own article is ground truth that needs no human
    labelling: the target must exist. Every miss counted here is content
    lost somewhere in the pipeline -- the structural parser never built the
    unit, or ``ingest/references.py`` misclassified a citation to a
    *different* document as internal (its own separate defect; #69 found
    that every one of the corpus's current misses is this second kind, not
    the first).

    Reuses ``ingest.graph``'s own private resolver (``_resolve``,
    ``_DocumentLookup``, ``_self_document``) rather than reimplementing
    ``target_hint`` parsing a second time -- read-only, no change to that
    module, the same approach issue #69's own measurement script uses.

    Args:
        references: This document's draft references, as extracted by
            ``ingest.references.extract_references``.
        manifest: The full corpus manifest -- ``_resolve`` needs every
            document, not just the one being ingested, to decide whether a
            reference actually is internal.
        result: This document's own parse result, for its unit ids.

    Returns:
        The count of ``ReferenceKind.INTERNAL`` references whose target
        does not exist in this document's own unit tree.
    """
    unit_ids = {unit.id for unit in result.units}
    documents = _DocumentLookup(
        by_number={document.number: document for document in manifest.documents},
        by_title={document.title_vi: document for document in manifest.documents},
    )
    self_document = _self_document(manifest, result.document_id)
    return sum(
        1
        for reference in references
        if reference.kind is ReferenceKind.INTERNAL
        and _resolve(reference, unit_ids, documents, self_document, result.document_id) is None
    )


def _print_table(per_document: Sequence[DocumentStats], edges_in_db: int | None = None) -> None:
    """Print the stats table.

    A header row, one row per document with any errors/gaps beneath it, a
    total row, and a summary line counting how many rows actually ran the
    text pipeline versus how many are metadata-only (issue #59) or failed
    (issue #78) -- every column of a metadata-only or failed row is zero,
    and a run must never look complete when it is not. When a persisted run
    has a failed row, an ``INFO`` line under the total row says what its
    ``edges`` cell means (see ``_totals``).

    Args:
        per_document: One row per document (plus the curated seed passes'
            pseudo-rows).
        edges_in_db: The TOTAL row's ``edges`` value, queried directly from
            the database (``EdgeRepository.count``) rather than summed, or
            ``None`` for a dry run -- see ``_totals``.
    """
    labels = [name for name, _ in _COLUMNS]
    print(_format_row(labels))
    for stats in per_document:
        print(_format_row(_row(stats)))
        if stats.metadata_only:
            print(f"  INFO: {stats.document_id} metadata-only -- text not acquired")
        if stats.edges_updated:
            print(
                f"  INFO: {stats.document_id} {stats.edges_updated} curated edge(s) "
                "updated an existing row (regex -> curated), not counted in 'edges'"
            )
        if stats.edges_removed:
            print(
                f"  INFO: {stats.document_id} {stats.edges_removed} stale edge(s) removed "
                "-- no longer produced by this document, and never reviewed"
            )
        for conflict in stats.edge_conflicts:
            print(
                f"  WARNING: {stats.document_id} edge id={conflict.id} "
                f"{conflict.src_id} -{conflict.edge_type}-> {conflict.dst_id} "
                f"is {conflict.review_status} but this parse no longer produces it -- "
                "review needed"
            )
        if stats.rejected_numbers:
            print(
                f"  WARNING: {stats.document_id} cites {len(stats.rejected_numbers)} document "
                f"number(s) that failed the complete-number check, so no stub: "
                f"{', '.join(stats.rejected_numbers)}"
            )
        _print_repeal_lines(stats)
        if stats.error is not None:
            print(f"  ERROR: {stats.document_id} {stats.error}")
        for article_id in stats.empty_articles:
            print(f"  WARNING: {article_id} parsed but produced no chunk")
    print(_format_row(_row(_totals(per_document, edges_in_db))))
    note = _edges_total_note(per_document, edges_in_db)
    if note is not None:
        print(note)
    print(_format_summary(per_document))


def _edges_total_note(per_document: Sequence[DocumentStats], edges_in_db: int | None) -> str | None:
    """Return the line explaining the TOTAL row's ``edges`` cell after a failure, or ``None``.

    Only a persisted run has a database count to explain (``edges_in_db`` is
    ``None`` in a dry run, where TOTAL is a plain sum), and only a run with a
    failed row makes that count misleading: a failed row is all zeros, yet the
    TOTAL cell beside it still holds the graph's existing size, which reads as
    output when it is not (issue #78). A run with no failure gets no note, so
    its output is unchanged.
    """
    if edges_in_db is None or not any(stats.error is not None for stats in per_document):
        return None
    return (
        "  INFO: TOTAL edges is the database's row count, not this run's output -- "
        "a failed row wrote nothing, so the figure still includes whatever an earlier "
        "run left for it"
    )


_PSEUDO_DOCUMENT_IDS = frozenset({"requirements", "document_edges"})
"""Stats rows for curated seed passes (A6 Phase 2.4; issue #63), not real
corpus documents -- excluded from ``_format_summary``'s ingested/
metadata-only split, which describes the manifest's own documents only."""


def _format_summary(per_document: Sequence[DocumentStats]) -> str:
    """Return the ``N ingested, M metadata-only`` line, plus ``, K failed`` when ``K > 0``.

    A document whose row carries an ``error`` is counted under ``K``, never
    under ``N`` (issue #78): a run in which every document raised must not
    print "7 ingested". The ``failed`` clause is omitted entirely when nothing
    failed, so a fully successful run's line is exactly what it was before
    that clause existed.

    Excludes the curated seed passes' pseudo-document rows
    (``_PSEUDO_DOCUMENT_IDS``) -- neither is a corpus document, so neither
    is "ingested", "metadata-only" or "failed". A failed seed pass is still
    reported, by its own ``ERROR`` line and the exit code.
    """
    rows = [stats for stats in per_document if stats.document_id not in _PSEUDO_DOCUMENT_IDS]
    metadata_only = sum(1 for stats in rows if stats.metadata_only)
    failed = sum(1 for stats in rows if stats.error is not None)
    ingested = len(rows) - metadata_only - failed
    summary = f"{ingested} ingested, {metadata_only} metadata-only"
    return f"{summary}, {failed} failed" if failed else summary


def _row(stats: DocumentStats) -> tuple[object, ...]:
    """Return one table row's values, in column order."""
    return (
        stats.document_id,
        stats.chapters,
        stats.articles,
        stats.clauses,
        stats.points,
        stats.skipped,
        stats.units,
        stats.chunks,
        stats.units_pruned,
        stats.chunks_pruned,
        stats.edges,
        stats.references,
        stats.unresolved,
        stats.no_stub,
        stats.self_citation_misses,
        len(stats.stub_ids),
        stats.entities,
    )


def _totals(per_document: Sequence[DocumentStats], edges_in_db: int | None = None) -> DocumentStats:
    """Sum every numeric column across all documents into a "TOTAL" row.

    Every column sums its rows except ``edges``: when ``edges_in_db`` is
    given (a persisted run), the TOTAL row reports that instead -- the
    number of rows ``edges`` actually holds, queried directly
    (``EdgeRepository.count``) rather than summed. Summing would double
    count an edge two passes in the same run declare (a regex-produced
    reference and a curated document edge sharing endpoints) and would
    re-count an edge a repeat ingest re-declares without inserting a new
    row (issue #75) -- the direct count sidesteps both regardless of how
    many passes run or in what order. A dry run has no database to query,
    so ``edges_in_db`` is ``None`` and the column falls back to the summed
    count, same as every other column.

    When documents fail, that definition still holds, and it is worth
    spelling out (issue #78): a failed document's row is all zeros and adds
    nothing to any summed column, but ``edges`` still counts the rows an
    earlier run left in the database for it -- a failed document is never
    persisted, so never reconciled, and its old edges stay exactly as they
    were. In a persisted run where every document failed, TOTAL ``edges`` is
    therefore the graph's existing size beside columns that are all zero: a
    count of what the database holds, not of what this run produced.
    ``_print_table`` says so on an ``INFO`` line under the row
    (``_edges_total_note``).

    ``stubs`` is the other exception (issue #67): the size of the *union* of
    every row's ``stub_ids``, not a sum. A stub is one document however many
    documents cite it, and summing would report the same decree once per
    citing statute.
    """
    return DocumentStats(
        document_id="TOTAL",
        chapters=sum(stats.chapters for stats in per_document),
        articles=sum(stats.articles for stats in per_document),
        clauses=sum(stats.clauses for stats in per_document),
        points=sum(stats.points for stats in per_document),
        skipped=sum(stats.skipped for stats in per_document),
        units=sum(stats.units for stats in per_document),
        chunks=sum(stats.chunks for stats in per_document),
        units_pruned=sum(stats.units_pruned for stats in per_document),
        chunks_pruned=sum(stats.chunks_pruned for stats in per_document),
        edges=edges_in_db
        if edges_in_db is not None
        else sum(stats.edges for stats in per_document),
        references=sum(stats.references for stats in per_document),
        unresolved=sum(stats.unresolved for stats in per_document),
        no_stub=sum(stats.no_stub for stats in per_document),
        self_citation_misses=sum(stats.self_citation_misses for stats in per_document),
        entities=sum(stats.entities for stats in per_document),
        stub_ids=frozenset().union(*(stats.stub_ids for stats in per_document)),
    )


def _format_row(values: Sequence[object]) -> str:
    """Left-justify each value into its column's fixed width."""
    cells = (str(value).ljust(width) for (_, width), value in zip(_COLUMNS, values, strict=True))
    return "".join(cells).rstrip()


if __name__ == "__main__":
    sys.exit(main())
