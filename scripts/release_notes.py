"""Print the CHANGELOG.md section for a version, as release notes.

    python scripts/release_notes.py 0.1.0

Exits with status 1 if there's no section for that version, so a release can't go out
without one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def notes(version: str, text: str) -> str | None:
    heading = re.compile(rf"^## \[{re.escape(version)}\](?: - .*)?$", re.M)
    start = heading.search(text)
    if not start:
        return None
    rest = text[start.end() :]
    end = re.search(r"^## |^\[[^\]]+\]: ", rest, re.M)
    return rest[: end.start() if end else None].strip()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    version = sys.argv[1].removeprefix("v")
    section = notes(version, CHANGELOG.read_text(encoding="utf-8"))
    if not section:
        print(f"CHANGELOG.md has no section for {version}.", file=sys.stderr)
        return 1
    print(section)
    return 0


if __name__ == "__main__":
    sys.exit(main())
