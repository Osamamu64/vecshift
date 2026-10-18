"""Which layers may import which: the service layer stays usable without a terminal.

Every import counts, including ones inside functions, since vecshift imports lazily.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "vecshift"

PRESENTATION = ("typer", "rich", "click", "vecshift.ui", "vecshift.cli", "vecshift.server")
"""Terminal and HTTP front ends. ``vecshift.cli`` also covers every ``vecshift.cli_*``."""


def modules(source: str) -> list[tuple[int, str]]:
    """(line, module) for every import in some Python source."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names = [node.module]
            if node.module == "vecshift":  # from vecshift import ui
                names = [f"vecshift.{alias.name}" for alias in node.names]
        else:
            continue
        found += [(node.lineno, name) for name in names]
    return found


def imports(package: str) -> list[tuple[str, int, str]]:
    """(file, line, module) for every import in a package."""
    return [
        (path.relative_to(SRC).as_posix(), line, module)
        for path in sorted((SRC / package).rglob("*.py"))
        for line, module in modules(path.read_text(encoding="utf-8"))
    ]


def presentation(module: str) -> bool:
    return any(
        module == layer or module.startswith((f"{layer}.", f"{layer}_")) for layer in PRESENTATION
    )


@pytest.mark.parametrize("package", ["service", "core"])
def test_no_presentation_imports(package: str) -> None:
    bad = [f"{f}:{line} imports {m}" for f, line, m in imports(package) if presentation(m)]
    assert not bad, "\n".join(bad)


def test_core_imports_nothing_above_it() -> None:
    bad = [
        f"{f}:{line} imports {m}"
        for f, line, m in imports("core")
        if m.startswith("vecshift") and not m.startswith("vecshift.core")
    ]
    assert not bad, "\n".join(bad)


def test_the_check_catches_lazy_imports() -> None:
    source = "def f():\n    from vecshift import ui\n    import vecshift.cli_plan, typer\n"
    assert [presentation(m) for _, m in modules(source)] == [True, True, True]
    assert not presentation("vecshift.connectors.qdrant")
    assert not presentation("richer")
