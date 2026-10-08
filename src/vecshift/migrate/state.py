"""Progress that has to survive between runs: spend so far and failed rows.

The database itself records which rows are done (their new vector is filled in), so this
file only holds what the database can't: money spent across runs, for the budget, and rows
the provider rejected. It lives in ``.vecshift/`` next to the job file, readable by its owner
only, and never holds document text.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass(slots=True)
class JobState:
    path: Path
    spent_usd: float = 0.0
    tokens: int = 0
    rows_written: int = 0
    runs: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    """Row ID → error, for rows the provider rejected. Text is never stored."""
    updated_at: str | None = None

    @classmethod
    def for_job(cls, job_file: Path, name: str) -> JobState:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "job"
        path = job_file.resolve().parent / ".vecshift" / f"{safe}.state.json"
        if not path.exists():
            return cls(path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("path", None)
        return cls(path=path, **data)

    def save(self) -> None:
        self.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        data = {k: v for k, v in asdict(self).items() if k != "path"}
        tmp = self.path.with_suffix(".tmp")
        # Write then rename, so a crash never leaves a half-written state file.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, self.path)
