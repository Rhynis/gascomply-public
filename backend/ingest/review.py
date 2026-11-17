"""Edge review CLI: the human gate on every inferred graph edge (build plan A6 Phase 2.3a).

``python -m ingest.review --stats``
``python -m ingest.review --list-pending [--edge-type T] [--src-document D] [--limit N]``
``python -m ingest.review --approve ID [ID ...]``
``python -m ingest.review --reject ID [ID ...]``
``python -m ingest.review --approve-all --edge-type T [--src-document D]``

``GraphTraversalService`` (Phase 2.5) only ever walks ``review_status='approved'``
edges. A structural ``BELONGS_TO`` edge is approved the moment the parser builds
it, because it was observed, not guessed. Every other edge type -- every
regex-resolved reference, and later every LLM-proposed one -- is inferred, so
it is written ``PENDING`` and goes nowhere until a human looks at it here. This
module is that look: nothing in the traversal or ingestion code trusts an
edge on its own say-so.

Approval is never implicit. There is no flag that approves everything
unconditionally, and no default that approves anything -- ``--approve-all``
requires an explicit ``--edge-type``, checked before any query runs, so the
gate cannot be opened by accident (build plan A6 issue #45). ``--list-pending``
always prints the citation text (``evidence``) and the source unit's full
body -- a reviewer approving an edge without seeing the sentence it came from
is a rubber stamp, not a gate. "The source unit" is the unit whose text
*produced* the edge (``origin_id``), not the one it happens to start at
(issue #83): a REPLACED_BY edge starts at the repealed document, but the
clause that justifies it is written in the repealing one, and the listing says
which unit it is showing whenever the two differ.

``--approve-all`` refuses ``--edge-type replaced_by`` (issue #83), whatever
``--src-document`` narrows it to -- see ``_RETIRING_EDGE_TYPES`` for why it is
a refusal and not a confirmation flag.

This is a CLI whose whole job is writing a report to stdout, the same
carve-out ``ingest/run.py`` already has from C6's "no print" rule (which is
about business code using the logger instead) -- see the ``T20`` per-file
ignore for both modules in ``pyproject.toml``.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence

from app.core.config import get_settings
from app.core.db import create_engine, create_session_factory, session_scope
from app.core.logging import configure_logging
from app.repositories.edges import EdgeRepository, EdgeStatsRow, ReviewRow
from app.schemas.edge import EdgeType, ReviewStatus

logger = logging.getLogger(__name__)

_STATS_COLUMNS: tuple[tuple[str, int], ...] = (
    ("edge_type", 14),
    ("review_status", 16),
    ("source", 13),
    ("count", 8),
)

_RETIRING_EDGE_TYPES = frozenset({EdgeType.REPLACED_BY})
"""Edge types ``--approve-all`` refuses (issue #83) -- the ones whose approval
*removes* something from every answer.

Approving any other type adds an edge retrieval may follow, so a bad bulk
approval adds noise a reviewer can see and reject. Approving REPLACED_BY does
the opposite: A5.2's in-force rule then drops the document the edge starts at
from every answer, and the failure is invisible -- answers simply stop citing a
statute. The decision here is a **refusal, not a confirmation flag**, for a
safety reason and not a UX one. A flag that unlocks the command is a bypass
that lives in shell history and in scripts, and the one command that retired
eleven laws is the command that gets run again on the next, worse extraction; a
command that always refuses cannot be replayed into damage. Nothing is lost by
refusing: ``--list-pending`` prints every clause, and ``--approve`` takes any
number of ids, so approving all of them is one command that names each edge it
retires -- which is precisely the act the refusal exists to force. ``--approve``
and ``--reject`` are unchanged."""


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the selected action, print its result.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        ``0`` on success, ``1`` if the arguments are invalid or no database
        URL is configured.
    """
    return asyncio.run(_amain(argv))


async def _amain(argv: Sequence[str] | None) -> int:
    """Async body of ``main`` -- every action here needs a database session."""
    configure_logging()
    args = _parse_args(argv)
    if args is None:
        return 1
    if not args.database_url:
        logger.error("No database URL: pass --database-url or set DATABASE_URL.")
        return 1

    engine = create_engine(args.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            repository = EdgeRepository(session)
            if args.stats:
                _print_stats(await repository.stats())
            elif args.list_pending:
                _print_pending(
                    await repository.list_pending(
                        edge_type=args.edge_type, src_document=args.src_document, limit=args.limit
                    )
                )
            elif args.approve:
                await _set_status(repository, args.approve, ReviewStatus.APPROVED)
            elif args.reject:
                await _set_status(repository, args.reject, ReviewStatus.REJECTED)
            elif args.approve_all:
                await _approve_all(repository, args.edge_type, args.src_document)
    finally:
        await engine.dispose()
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace | None:
    """Parse and validate command-line arguments.

    Returns:
        The parsed namespace, or ``None`` if validation failed (already
        logged) -- distinct from an ``argparse`` usage error, which exits the
        process directly.
    """
    parser = argparse.ArgumentParser(
        prog="python -m ingest.review",
        description="Review pending graph edges: list, approve, or reject them.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--stats", action="store_true", help="Print edge counts by status.")
    action.add_argument(
        "--list-pending", action="store_true", help="List pending edges with their evidence."
    )
    action.add_argument("--approve", nargs="+", type=int, metavar="ID", help="Approve edges by id.")
    action.add_argument("--reject", nargs="+", type=int, metavar="ID", help="Reject edges by id.")
    action.add_argument(
        "--approve-all",
        action="store_true",
        help=(
            "Approve every pending edge matching --edge-type (required) and --src-document. "
            "Refused for replaced_by: approve those one by one with --approve ID."
        ),
    )
    parser.add_argument(
        "--edge-type",
        type=_edge_type,
        default=None,
        help=f"Restrict to one edge type: {', '.join(t.value for t in EdgeType)}.",
    )
    parser.add_argument(
        "--src-document",
        default=None,
        help=(
            "Restrict to edges this document's text produced (the document itself, or a "
            "unit under it) -- e.g. a decree's own repeal edges, though they start at the "
            "documents it repeals."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows for --list-pending.")
    parser.add_argument(
        "--database-url",
        default=get_settings().database_url,
        help="Postgres URL to connect to (default: the DATABASE_URL setting/env var).",
    )
    args = parser.parse_args(argv)

    if args.approve_all and args.edge_type is None:
        logger.error(
            "--approve-all requires --edge-type: bulk-approving every edge type is not allowed."
        )
        return None
    if args.approve_all and (refusal := _bulk_approval_refusal(args.edge_type)) is not None:
        logger.error("%s", refusal)
        return None
    return args


def _edge_type(value: str) -> EdgeType:
    """Parse ``--edge-type``'s value into an ``EdgeType`` member.

    Raises:
        argparse.ArgumentTypeError: If ``value`` names no real edge type --
            argparse reports this as a usage error itself.
    """
    try:
        return EdgeType(value)
    except ValueError as error:
        valid = ", ".join(t.value for t in EdgeType)
        msg = f"{value!r} is not a valid edge type; choose from: {valid}"
        raise argparse.ArgumentTypeError(msg) from error


async def _set_status(
    repository: EdgeRepository, edge_ids: list[int], status: ReviewStatus
) -> None:
    """Apply ``status`` to ``edge_ids`` and report how many rows actually changed."""
    updated = await repository.set_review_status(edge_ids, status)
    print(f"{status.value}: {updated} of {len(edge_ids)} edge id(s) updated.")
    if updated < len(edge_ids):
        print("  Some ids matched no row -- check for typos.")


def _bulk_approval_refusal(edge_type: EdgeType) -> str | None:
    """Return why ``--approve-all`` must not run for ``edge_type``, or ``None`` if it may."""
    if edge_type not in _RETIRING_EDGE_TYPES:
        return None
    return (
        f"--approve-all refuses --edge-type {edge_type.value}: approving one removes the "
        "document it starts at from every answer, so each has to be read and approved by id "
        f"(--list-pending --edge-type {edge_type.value}, then --approve ID [ID ...])."
    )


async def _approve_all(
    repository: EdgeRepository, edge_type: EdgeType, src_document: str | None
) -> None:
    """Approve every pending edge matching ``edge_type``/``src_document``.

    Raises:
        ValueError: If ``edge_type`` is one ``--approve-all`` refuses. ``_parse_args``
            already turns this into an exit before any query runs, so this is the
            last line of defence for a caller that got here without it: it fires
            before the pending rows are even listed, so nothing can have changed.
    """
    refusal = _bulk_approval_refusal(edge_type)
    if refusal is not None:
        raise ValueError(refusal)
    pending = await repository.list_pending(edge_type=edge_type, src_document=src_document)
    if not pending:
        print(f"No pending {edge_type.value} edges to approve.")
        return
    edge_ids = [row.id for row in pending]
    updated = await repository.set_review_status(edge_ids, ReviewStatus.APPROVED)
    print(f"approved: {updated} pending {edge_type.value} edge(s).")


def _print_stats(rows: list[EdgeStatsRow]) -> None:
    """Print the ``(edge_type, review_status, source)`` count table."""
    labels = [name for name, _ in _STATS_COLUMNS]
    print(_format_stats_row(labels))
    for row in rows:
        print(_format_stats_row((row.edge_type, row.review_status, row.source, row.count)))


def _format_stats_row(values: Sequence[object]) -> str:
    """Left-justify each value into its column's fixed width."""
    cells = (
        str(value).ljust(width) for (_, width), value in zip(_STATS_COLUMNS, values, strict=True)
    )
    return "".join(cells).rstrip()


def _print_pending(rows: list[ReviewRow]) -> None:
    """Print each pending edge with its evidence and source body, for a human to decide."""
    if not rows:
        print("No pending edges match.")
        return
    for row in rows:
        header = f"[{row.id}] {row.src_id} --{row.edge_type.upper()}--> {row.dst_id}"
        print(f"{header}  (source: {row.source})")
        print(f"  evidence: {row.evidence!r}")
        if row.origin_id is not None and row.origin_id != row.src_id:
            print(f"  stated in: {row.origin_id}")
        if row.source_body is None:
            print("  source body: (document-level citation; no unit body)")
        else:
            print(f"  source body: {row.source_body!r}")
    print(f"{len(rows)} pending edge(s) listed.")


if __name__ == "__main__":
    sys.exit(main())
