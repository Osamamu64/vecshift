"""Cutover and rollback: give the new vectors the column name the application searches.

Cutover renames the live column to ``<name>_old`` and the new column to ``<name>`` in one
short transaction, so every query switches at once and the application's SQL doesn't
change. Rollback renames them back. The sync trigger always guards whichever column isn't
live: before cutover it clears new vectors when text changes, and after cutover it clears
the old ones, so rollback can say exactly which rows lack an old-model vector.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from psycopg import errors, sql

from vecshift.connectors.pgvector.target import table_limits
from vecshift.connectors.pgvector.writer import SCAN_TIMEOUT_MS, Busy, PgWriter, Row, _limits
from vecshift.doctor.findings import Finding, Severity

SWITCH_SCAN_MS = 15_000
MAX_FILL = 500
"""Most rows cutover embeds while writes wait; more than this means catch up first."""
"""How long the final check may hold off writes before cutover gives up and retries."""
METHODS = ("hnsw", "ivfflat")

Stage = Literal["not_applied", "applied", "cut_over", "conflict"]


class StillPending(Exception):
    def __init__(self, rows: int) -> None:
        super().__init__(f"{rows} rows still need a new vector")
        self.rows = rows


@dataclass(frozen=True, slots=True)
class Readiness:
    stage: Stage
    findings: list[Finding]
    pending: int
    live_dims: int | None
    new_dims: int | None

    @property
    def ok(self) -> bool:
        return not any(f.severity is Severity.ERROR for f in self.findings)


class PgSwitch:
    def __init__(self, writer: PgWriter) -> None:
        if not writer.layout.live:
            raise ValueError("cutover needs the live column's name")
        self.writer = writer
        self.conn = writer.conn
        self.layout = writer.layout

    # --- what's there

    def _attribute(self, column: str) -> tuple[int, int | None] | None:
        """(attnum, dimensions) of a column, or None when it doesn't exist."""
        row = self.conn.execute(
            """
            SELECT attnum, atttypmod FROM pg_attribute
            WHERE attrelid = %s::regclass AND attname = %s AND attnum > 0 AND NOT attisdropped
            """,
            (self.writer._regclass(), column),
        ).fetchone()
        if row is None:
            return None
        return int(row[0]), (int(row[1]) if row[1] > 0 else None)

    def stage(self) -> Stage:
        lay = self.layout
        target = self._attribute(lay.target) is not None
        previous = self._attribute(lay.previous) is not None
        if target and previous:
            return "conflict"
        if target:
            return "applied"
        if previous:
            return "cut_over"
        return "not_applied"

    def dependents(self, column: str) -> list[str]:
        """Views and SQL-standard functions bound to a column, which a rename won't move."""
        attribute = self._attribute(column)
        if attribute is None:
            return []
        rows = self.conn.execute(
            """
            SELECT DISTINCT CASE
                WHEN d.classid = 'pg_rewrite'::regclass THEN 'view ' || r.ev_class::regclass::text
                ELSE 'function ' || d.objid::regprocedure::text END
            FROM pg_depend d
            LEFT JOIN pg_rewrite r ON d.classid = 'pg_rewrite'::regclass AND r.oid = d.objid
            WHERE d.refclassid = 'pg_class'::regclass AND d.refobjid = %s::regclass
              AND d.refobjsubid = %s
              AND d.classid IN ('pg_rewrite'::regclass, 'pg_proc'::regclass)
              AND (r.ev_class IS NULL OR r.ev_class <> d.refobjid)
            ORDER BY 1
            """,
            (self.writer._regclass(), attribute[0]),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def _count(self, condition: sql.Composable) -> int:
        query = sql.SQL("SELECT count(*) FROM {table} WHERE {condition}").format(
            table=self.layout.qualified, condition=condition
        )
        with self.conn.transaction():
            _limits(self.conn, SCAN_TIMEOUT_MS)
            row = self.conn.execute(query).fetchone()
        return int(row[0]) if row else 0

    def _no_text(self) -> sql.Composable:
        return sql.SQL("({text} IS NULL OR btrim({text}::text) = '')").format(
            text=sql.Identifier(self.layout.text)
        )

    def _index_names(self, column: str) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT c.relname FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
            WHERE i.indrelid = %s::regclass AND a.attname = %s
            """,
            (self.writer._regclass(), column),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def _name_taken(self, name: str) -> bool:
        row = self.conn.execute(
            """
            SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = %s
            """,
            (self.layout.schema, name),
        ).fetchone()
        return row is not None

    def _index_renames(self, moves: list[tuple[str, str, str]]) -> list[sql.Composed]:
        """Index renames that follow column renames, so names keep matching columns.

        Each move is (column as named now, old name, new name). Only indexes named the
        way vecshift names them are renamed, and only into names that are free.
        """
        lay = self.layout
        statements = []
        freed: set[str] = set()
        claimed: set[str] = set()
        for column, old, new in moves:
            names = self._index_names(column)
            for method in METHODS:
                current, wanted = lay.index_name(method, old), lay.index_name(method, new)
                free = wanted in freed or not self._name_taken(wanted)
                if current in names and wanted not in claimed and free:
                    statements.append(
                        sql.SQL("ALTER INDEX {} RENAME TO {}").format(
                            sql.Identifier(lay.schema, current), sql.Identifier(wanted)
                        )
                    )
                    freed.add(current)
                    claimed.add(wanted)
        return statements

    def _rename(self, old: str, new: str) -> sql.Composed:
        return sql.SQL("ALTER TABLE {table} RENAME COLUMN {old} TO {new}").format(
            table=self.layout.qualified, old=sql.Identifier(old), new=sql.Identifier(new)
        )

    # --- checks

    def check(self, index_method: str | None) -> Readiness:
        lay = self.layout
        findings: list[Finding] = []
        stage = self.stage()
        live = self._attribute(lay.live)
        new = self._attribute(lay.target)
        live_dims = live[1] if live else None
        new_dims = new[1] if new else None
        pending = 0

        if stage == "not_applied":
            findings.append(
                Finding(
                    "cutover.not_applied",
                    Severity.ERROR,
                    "Nothing to cut over yet",
                    f"Column {lay.target} doesn't exist.",
                    hint="Run vecshift apply first.",
                )
            )
        elif stage == "cut_over":
            findings.append(
                Finding(
                    "cutover.done",
                    Severity.ERROR,
                    "Already cut over",
                    f"{lay.live} holds the new vectors and {lay.previous} the old ones.",
                    hint="Run vecshift rollback to undo it.",
                )
            )
        elif stage == "conflict":
            findings.append(
                Finding(
                    "cutover.previous_exists",
                    Severity.ERROR,
                    f"Column {lay.previous} already exists",
                    "Cutover keeps the old vectors under that name, so it has to be free.",
                    hint="Drop it if it's left over from an earlier migration, or rename it.",
                )
            )
        if live is None:
            findings.append(
                Finding(
                    "cutover.no_live_column",
                    Severity.ERROR,
                    f"Column {lay.live} doesn't exist",
                    "That's the column your application searches.",
                    hint="Set source.vector_column in the job file.",
                )
            )
        if stage != "applied" or live is None:
            return Readiness(stage, findings, pending, live_dims, new_dims)

        relid = self.conn.execute("SELECT %s::regclass::oid", (self.writer._regclass(),)).fetchone()
        partitioned, row_security = (
            table_limits(self.conn, int(relid[0])) if relid else (False, False)
        )
        if partitioned:
            findings.append(
                Finding(
                    "cutover.partitioned",
                    Severity.ERROR,
                    "Partitioned tables aren't supported yet",
                    "Cutover's final check needs a concurrent index, which PostgreSQL can't "
                    "build on a partitioned table.",
                )
            )
        if row_security:
            findings.append(
                Finding(
                    "cutover.row_security",
                    Severity.ERROR,
                    "Row-level security hides rows from this role",
                    "Cutover couldn't confirm that every row has a new vector.",
                    hint="Connect as a role with BYPASSRLS, or turn off FORCE ROW LEVEL "
                    "SECURITY for the migration.",
                )
            )

        pending = self.writer.pending_count()
        if pending:
            findings.append(
                Finding(
                    "cutover.pending",
                    Severity.ERROR,
                    f"{pending:,} rows have no new vector yet",
                    "After cutover they wouldn't be found by searches.",
                    hint="Cutover embeds them first; or run vecshift apply.",
                )
            )
        if index_method and self.writer.index_state(index_method) != "valid":
            findings.append(
                Finding(
                    "cutover.no_index",
                    Severity.ERROR,
                    f"The {index_method} index on {lay.target} isn't built",
                    "Without it, searches would scan the whole table after cutover.",
                    hint="Run vecshift apply to build it.",
                )
            )
        sizes = self.writer.dimensions_in_use()
        if sizes and sizes != {lay.dims}:
            findings.append(
                Finding(
                    "cutover.mixed_sizes",
                    Severity.ERROR,
                    "The new column holds vectors of different sizes",
                    f"Found sizes {sorted(sizes)}; expected {lay.dims}.",
                )
            )
        bound = self.dependents(lay.live) + self.dependents(lay.target)
        if bound:
            findings.append(
                Finding(
                    "cutover.dependents",
                    Severity.ERROR,
                    "Views or functions are bound to the vector columns",
                    "PostgreSQL ties these to a column, not its name, so after a rename they "
                    f"would still read the old vectors: {', '.join(bound)}.",
                    hint="Drop them before cutover and recreate them after it.",
                )
            )
        orphans = self._count(
            sql.SQL("{live} IS NOT NULL AND {no_text}").format(
                live=sql.Identifier(lay.live), no_text=self._no_text()
            )
        )
        if orphans:
            findings.append(
                Finding(
                    "cutover.no_text",
                    Severity.WARNING,
                    f"{orphans:,} rows have a vector but no text",
                    "They can't be re-embedded, so they'll have no vector after cutover.",
                )
            )
        if live_dims and new_dims and live_dims != new_dims:
            findings.append(
                Finding(
                    "cutover.size_change",
                    Severity.WARNING,
                    f"Vectors change size, from {live_dims} to {new_dims}",
                    "Searches and inserts that still use the old model will fail after cutover "
                    "until your application switches to the new one.",
                    hint="Switch the model in your application's settings at cutover time.",
                )
            )
        else:
            findings.append(
                Finding(
                    "cutover.model_switch",
                    Severity.INFO,
                    "Switch your application's model at cutover",
                    "Queries embedded with the old model would still run, but match poorly, "
                    "until your application embeds them with the new one.",
                )
            )
        return Readiness(stage, findings, pending, live_dims, new_dims)

    # --- the switch

    @property
    def _pending_index(self) -> str:
        return f"{self.layout.table}_{self.layout.target}_pending_idx"[:63]

    def prepare(self) -> None:
        """Index the rows still missing a vector, so the final check under lock is instant."""
        lay = self.layout
        self.finish()
        self.conn.execute("SET statement_timeout = 0")
        try:
            self.conn.execute(
                sql.SQL(
                    "CREATE INDEX CONCURRENTLY {name} ON {table} ({pk}) WHERE {target} IS NULL"
                ).format(
                    name=sql.Identifier(self._pending_index),
                    table=lay.qualified,
                    pk=sql.Identifier(lay.pk),
                    target=sql.Identifier(lay.target),
                )
            )
        finally:
            self.conn.execute("RESET statement_timeout")

    def finish(self) -> None:
        self.conn.execute(
            sql.SQL("DROP INDEX CONCURRENTLY IF EXISTS {}").format(
                sql.Identifier(self.layout.schema, self._pending_index)
            )
        )

    def _transaction(
        self, run: Callable[[], None], what: str, sleep: Callable[[float], None] | None = None
    ) -> None:
        writer = self.writer
        pause = sleep or writer._sleep
        delay = 1.0
        for attempt in range(1, writer.lock_retries + 1):
            try:
                with self.conn.transaction():
                    _limits(self.conn, SWITCH_SCAN_MS, writer.lock_timeout_ms)
                    run()
                return
            except errors.LockNotAvailable as exc:
                if attempt == writer.lock_retries:
                    raise Busy(
                        f"Couldn't {what}: the table stayed busy. Try again when it's quieter."
                    ) from exc
                pause(delay)
                delay = min(delay * 2, 30)

    def cutover(
        self,
        allow_missing: int = 0,
        fill: Callable[[list[Row]], list[tuple[Row, list[float]]]] | None = None,
        exclude: Sequence[str] = (),
    ) -> int:
        """Switch in one transaction, after a final check that every row has a new vector.

        Writes wait while the check runs (it uses the index from :meth:`prepare`); reads
        continue until the renames, which take milliseconds. On a busy table a few rows
        always arrive after the last catch-up: up to ``MAX_FILL`` of them are embedded with
        ``fill`` while writes wait, so cutover doesn't chase them forever. ``exclude`` lists
        rows the provider rejected, which aren't sent again. Raises :class:`StillPending`
        if more rows are missing, leaving everything unchanged. Returns rows filled.
        """
        lay = self.layout
        writer = self.writer
        filled = 0

        def missing_count() -> int:
            row = self.conn.execute(
                sql.SQL("SELECT count(*) FROM {table} WHERE {pending}").format(
                    table=lay.qualified, pending=writer._pending_filter()
                )
            ).fetchone()
            return int(row[0]) if row else 0

        def run() -> None:
            nonlocal filled
            filled = 0
            self.conn.execute(
                sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE").format(lay.qualified)
            )
            missing = missing_count()
            if missing > allow_missing and fill is not None:
                rows = writer.fetch(None, MAX_FILL + 1, exclude)
                if 0 < len(rows) <= MAX_FILL:
                    filled = writer.write(fill(rows))
                    missing = missing_count()
            if missing > allow_missing:
                raise StillPending(missing)
            statements = [
                *writer.drop_sync_sql(lay.target),
                *self._index_renames(
                    [(lay.live, lay.live, lay.previous), (lay.target, lay.target, lay.live)]
                ),
                self._rename(lay.live, lay.previous),
                self._rename(lay.target, lay.live),
                writer.sync_function_sql(lay.previous),
                writer.sync_trigger_sql(lay.previous),
            ]
            for statement in statements:
                self.conn.execute(statement)

        self._transaction(run, "cut over")
        return filled

    def rollback(self) -> int:
        """Rename the columns back. Returns rows that have no old-model vector."""
        lay = self.layout
        writer = self.writer

        def run() -> None:
            statements = [
                *writer.drop_sync_sql(lay.previous),
                *self._index_renames(
                    [(lay.live, lay.live, lay.target), (lay.previous, lay.previous, lay.live)]
                ),
                self._rename(lay.live, lay.target),
                self._rename(lay.previous, lay.live),
                writer.sync_function_sql(lay.target),
                writer.sync_trigger_sql(lay.target),
            ]
            for statement in statements:
                self.conn.execute(statement)

        self._transaction(run, "roll back")
        return self._count(
            sql.SQL("{live} IS NULL AND NOT {no_text}").format(
                live=sql.Identifier(lay.live), no_text=self._no_text()
            )
        )

    def cleanup(self) -> None:
        """Drop the old vectors, their index, and the trigger guarding them. Irreversible."""
        lay = self.layout
        writer = self.writer
        drop = sql.SQL("ALTER TABLE {table} DROP COLUMN {col}").format(
            table=lay.qualified, col=sql.Identifier(lay.previous)
        )

        def run() -> None:
            for statement in [*writer.drop_sync_sql(lay.previous), drop]:
                self.conn.execute(statement)

        self._transaction(run, "drop the old column")
