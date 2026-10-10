"""The release workflow's checks: release notes come from CHANGELOG.md, per version."""

import importlib.util
import re
from pathlib import Path

import vecshift

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("release_notes", ROOT / "scripts/release_notes.py")
assert spec and spec.loader
release_notes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_notes)

CHANGELOG = """# Changelog

## [Unreleased]

- Something new.

## [0.2.0] - 2026-11-01

### Fixed

- A bug.

## [0.1.0] - 2026-10-10

The first release.

[Unreleased]: https://example.com/compare
[0.2.0]: https://example.com/0.2.0
"""


def test_notes_are_one_version_section() -> None:
    assert release_notes.notes("0.2.0", CHANGELOG) == "### Fixed\n\n- A bug."
    assert release_notes.notes("0.1.0", CHANGELOG) == "The first release."
    assert release_notes.notes("0.3.0", CHANGELOG) is None
    assert release_notes.notes("0.1", CHANGELOG) is None, "versions match exactly"


def test_the_package_version_has_release_notes() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert release_notes.notes(vecshift.__version__, changelog)
    released = re.findall(r"^## \[(\d[^\]]*)\]", changelog, re.M)
    assert released[0] == vecshift.__version__, "the newest CHANGELOG section is this version"
