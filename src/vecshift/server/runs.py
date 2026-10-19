"""Long operations (apply, eval, cutover, rollback, cleanup) as runs in the background.

A run has an ID (a ULID, so IDs sort by start time), a state, and numbered events. Every
event is appended to ``.vecshift/runs/<id>.jsonl`` as it happens, and the run's summary is
kept in ``<id>.json``, both readable by their owner only, so a client can catch up after
reconnecting and a run can be looked at after the server restarts. A run still marked as
running when the server starts was cut short: it's marked failed, and since every
operation is resumable, running it again picks up where it stopped.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("vecshift.server")

ACTIVE = frozenset({"queued", "running", "stopping"})
FINISHED = frozenset({"completed", "stopped", "failed"})
RUN_ID = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}")
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
RESTARTED = "The server restarted during this run. Run it again: it resumes where it stopped."


def new_id() -> str:
    """A ULID: 48 bits of milliseconds, then 80 random bits, in Crockford's base 32."""
    value = (int(time.time() * 1000) << 80) | int.from_bytes(secrets.token_bytes(10), "big")
    return "".join(CROCKFORD[(value >> shift) & 31] for shift in range(125, -1, -5))


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class Event:
    id: int
    kind: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "event": self.kind, "data": self.data}


@dataclass(slots=True)
class Run:
    run_id: str
    job: str
    action: str
    folder: Path
    """Where the run's files go: ``.vecshift/runs`` in the jobs folder."""
    state: str = "queued"
    started_at: str = field(default_factory=now)
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    problem: dict[str, Any] | None = None
    events: list[Event] = field(default_factory=list)
    stop_requested: bool = False
    stoppable: bool = False
    """Whether the action can stop between batches (apply)."""
    changed: threading.Condition = field(default_factory=threading.Condition)

    @property
    def log_path(self) -> Path:
        return self.folder / f"{self.run_id}.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.folder / f"{self.run_id}.json"

    @property
    def report_path(self) -> Path:
        return self.folder / f"{self.run_id}.html"

    @property
    def active(self) -> bool:
        return self.state in ACTIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "job": self.job,
            "action": self.action,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "problem": self.problem,
            "events": len(self.events),
        }

    def emit(self, kind: str, data: dict[str, Any]) -> None:
        with self.changed:
            event = Event(len(self.events) + 1, kind, data)
            self.events.append(event)
            _append(self.log_path, json.dumps(event.to_dict()))
            self.changed.notify_all()

    def finish(self, state: str) -> None:
        with self.changed:
            self.state = state
            self.finished_at = now()
            self.save()
            self.changed.notify_all()

    def save(self) -> None:
        _write(self.summary_path, json.dumps(self.to_dict(), indent=2))

    def after(self, last: int) -> list[Event]:
        """The events after number ``last``."""
        with self.changed:
            return self.events[last:]

    def wait(self, last: int, timeout: float) -> None:
        """Until there's an event after ``last``, the run finishes, or ``timeout`` passes."""
        with self.changed:
            if len(self.events) <= last and self.active:
                self.changed.wait(timeout)


class Busy(Exception):
    """The job already has an active run."""

    def __init__(self, run: Run) -> None:
        super().__init__(f"{run.job} is already running {run.action} ({run.run_id}).")
        self.run = run


class Registry:
    """Every run this server knows of: the ones it started, and those on disk from before."""

    def __init__(self, jobs: Path) -> None:
        self.folder = jobs.resolve() / ".vecshift" / "runs"
        self.runs: dict[str, Run] = {}
        self.lock = threading.Lock()
        self._recover()

    def start(
        self,
        job: str,
        action: str,
        work: Callable[[Run], None],
        *,
        stoppable: bool = False,
    ) -> Run:
        """Start ``work`` in the background, unless ``job`` has an active run."""
        with self.lock:
            for run in self.runs.values():
                if run.job == job and run.active:
                    raise Busy(run)
            self.folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            run = Run(new_id(), job, action, self.folder, stoppable=stoppable)
            self.runs[run.run_id] = run
            run.save()
        thread = threading.Thread(
            target=self._run, args=(run, work), name=f"vecshift-{run.run_id}", daemon=True
        )
        thread.start()
        return run

    def _run(self, run: Run, work: Callable[[Run], None]) -> None:
        with run.changed:
            run.state = "running"
        run.save()
        try:
            work(run)
        except Exception as exc:  # the work reports its own problems; this is a bug
            log.error("%s in run %s (%s)", type(exc).__name__, run.run_id, run.action)
            run.problem = {"code": "internal", "detail": "Something went wrong in vecshift."}
            run.emit("error", run.problem)
            run.finish("failed")
        if run.active:  # work that returned without finishing the run
            run.finish("completed")

    def get(self, run_id: str) -> Run | None:
        with self.lock:
            return self.runs.get(run_id) if RUN_ID.fullmatch(run_id) else None

    def for_job(self, job: str) -> list[Run]:
        with self.lock:
            runs = [r for r in self.runs.values() if r.job == job]
        return sorted(runs, key=lambda r: r.run_id, reverse=True)

    def _recover(self) -> None:
        """Load the runs on disk; mark those cut short by a restart as failed."""
        if not self.folder.is_dir():
            return
        for path in sorted(self.folder.glob("*.json")):
            if not RUN_ID.fullmatch(path.stem):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                run = Run(
                    run_id=path.stem,
                    job=str(data["job"]),
                    action=str(data["action"]),
                    folder=self.folder,
                    state=str(data["state"]),
                    started_at=str(data["started_at"]),
                    finished_at=data.get("finished_at"),
                    result=data.get("result"),
                    problem=data.get("problem"),
                )
            except (OSError, ValueError, KeyError):
                log.warning("Skipped an unreadable run summary: %s", path.name)
                continue
            run.events = _read(run.log_path)
            if run.active:
                run.problem = {"code": "run.interrupted", "detail": RESTARTED}
                run.emit("error", run.problem)
                run.finish("failed")
            self.runs[run.run_id] = run


def _read(path: Path) -> list[Event]:
    events: list[Event] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        try:
            item = json.loads(line)
            events.append(Event(int(item["id"]), str(item["event"]), dict(item["data"])))
        except (ValueError, KeyError, TypeError):
            break  # a line cut short by a crash: what follows can't be trusted
    return events


def _append(path: Path, line: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _write(path: Path, text: str) -> None:
    """Write a file readable by its owner only, atomically."""
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)
