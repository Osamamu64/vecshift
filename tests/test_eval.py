"""The eval core: queries in Arabic and English, scores, latency, and the verdict."""

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from vecshift.eval import EvalQuery, Gates, SideReport, SweepPoint, proxy_queries, run, script
from vecshift.eval.metrics import Latency, Scores, by_slice, latency, overlap, percentile, score
from vecshift.eval.runner import EvalReport, decide

AR = "يذهب الطلاب إلى المدرسة كل صباح لتعلم العلوم والتاريخ والرياضيات"
EN = "Students walk to the school every morning to learn science and history"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (AR, "arabic"),
        (EN, "latin"),
        ("Café résumé naïve", "latin"),
        ("12345 !!", "other"),
        ("", "other"),
        ("hello مرحبا بكم جميعا", "arabic"),
        ("مرحبا, welcome to the big city tonight", "latin"),
    ],
)
def test_script(text: str, expected: str) -> None:
    assert script(text) == expected


def test_proxy_queries_split_arabic_and_english() -> None:
    rows = [
        ("1", "ما هو أفضل وقت لزيارة المدينة القديمة في الشتاء؟ تقع المدينة قرب النهر الكبير."),
        ("2", "The bridge was built in a single summer by local workers. It crosses the river."),
        ("3", "short"),
    ]
    queries, notes = proxy_queries(rows, 10, seed=1)
    by_key = {next(iter(q.relevant)): q for q in queries}
    assert set(by_key) == {"1", "2"}, "rows too short to make a query from are skipped"
    arabic = by_key["1"]
    assert arabic.language == "arabic" and arabic.slice == "arabic→arabic"
    assert arabic.text in rows[0][1]
    assert "؟" in arabic.text or arabic.text.startswith("تقع"), "split at the Arabic question mark"
    assert by_key["2"].slice == "latin→latin"
    assert any("Only 2 rows" in n for n in notes)


def test_proxy_queries_skip_boilerplate_sentences() -> None:
    footer = "Contact us at the main office for more details today."
    rows = [(str(i), f"Unique opening line number {i} about topic {i}. {footer}") for i in range(5)]
    queries, _ = proxy_queries(rows, 5, seed=1)
    assert queries and all(footer not in q.text for q in queries)


def test_scores_and_slices() -> None:
    queries = [
        EvalQuery("a", frozenset({"1"}), "latin", "latin"),
        EvalQuery("b", frozenset({"2"}), "arabic", "arabic"),
        EvalQuery("c", frozenset({"3"}), "arabic", "latin"),
    ]
    ranked = [["1", "x"], ["x", "2"], ["x", "y"]]
    scores = by_slice(ranked, queries)
    assert scores["all"] == Scores(3, 1 / 3, 2 / 3, (1 + 0.5) / 3)
    assert scores["arabic→arabic"].mrr_at_10 == 0.5
    assert scores["arabic→latin"].recall_at_10 == 0.0
    assert set(scores) == {"all", "latin→latin", "arabic→arabic", "arabic→latin"}
    assert score([["9"] * 10 + ["1"]], [frozenset({"1"})]).recall_at_10 == 0, "only the top 10"


def test_overlap_and_percentiles() -> None:
    assert overlap([["a", "b"]], [["b", "c"]], k=2) == 0.5
    assert overlap([["a"]], [["a", "b", "c"]], k=10) == pytest.approx(1 / 3)
    assert overlap([[]], [[]]) == 1.0 and overlap([], []) == 0.0
    values = [float(v) for v in range(1, 101)]
    assert (percentile(values, 50), percentile(values, 95), percentile(values, 99)) == (
        50.0,
        95.0,
        99.0,
    )
    assert percentile([7.0], 99) == 7.0
    assert latency([]) is None
    assert latency([1.0, 2.0, 3.0]) == Latency(3, 2.0, 3.0, 3.0)


# --- The verdict


def report(
    old_r10: dict[str, tuple[int, float]],
    new_r10: dict[str, tuple[int, float]],
    *,
    mode: str = "queries",
    old_ms: float | None = 10.0,
    new_ms: float | None = 10.0,
    groups: tuple[float | None, float | None] = (None, None),
    index_recall: float = 0.99,
) -> EvalReport:
    def side(name: str, r10: dict[str, tuple[int, float]], ms: float | None) -> SideReport:
        result = SideReport(name, name, "m")  # type: ignore[arg-type]
        result.scores = {k: Scores(n, v, v, v) for k, (n, v) in r10.items()}
        if ms is not None:
            result.embed_latency = Latency(20, ms, ms, ms)
            result.sweep = [SweepPoint(40, index_recall, Latency(20, 1, 2, 3), current=True)]
        return result

    old, new = side("old", old_r10, old_ms), side("new", new_r10, new_ms)
    old.same_group_at_10, new.same_group_at_10 = groups
    return EvalReport(mode, False, "proxy", 100, old, new)  # type: ignore[arg-type]


def test_go_when_recall_holds() -> None:
    verdict = decide(report({"all": (100, 0.80)}, {"all": (100, 0.79)}), Gates())
    assert verdict.status == "go" and "overall" in verdict.reasons[0]


def test_no_go_when_one_language_regresses() -> None:
    old = {"all": (200, 0.80), "latin→latin": (150, 0.80), "arabic→arabic": (50, 0.80)}
    new = {"all": (200, 0.82), "latin→latin": (150, 0.90), "arabic→arabic": (50, 0.60)}
    verdict = decide(report(old, new), Gates())
    assert verdict.status == "no_go"
    assert verdict.reasons == ["Recall@10 arabic→arabic fell from 0.800 to 0.600."]


def test_small_slices_are_shown_but_not_decided() -> None:
    old = {"all": (100, 0.80), "arabic→latin": (5, 0.9)}
    new = {"all": (100, 0.80), "arabic→latin": (5, 0.1)}
    verdict = decide(report(old, new), Gates(min_slice=30))
    assert verdict.status == "go"
    assert any("Only 5 arabic→latin" in w for w in verdict.warnings)


def test_latency_gates() -> None:
    slow = report({"all": (100, 0.8)}, {"all": (100, 0.9)}, old_ms=10, new_ms=30)
    assert decide(slow, Gates(max_p95_ms=50)).status == "go"
    capped = decide(slow, Gates(max_p95_ms=20))
    assert capped.status == "no_go" and "over the 20 ms limit" in capped.reasons[0]
    slower = decide(slow, Gates(max_slowdown_pct=50))
    assert slower.status == "no_go" and "more than 50% slower" in slower.reasons[0]


def test_vectors_mode_uses_document_neighbours() -> None:
    better = report({}, {"all": (100, 0.9)}, mode="vectors", groups=(0.30, 0.40))
    verdict = decide(better, Gates())
    assert verdict.status == "go" and any("stored vectors only" in w for w in verdict.warnings)
    worse = report({}, {"all": (100, 0.9)}, mode="vectors", groups=(0.40, 0.20))
    assert decide(worse, Gates()).status == "no_go"
    blind = report({}, {"all": (100, 0.9)}, mode="vectors")
    assert decide(blind, Gates()).status == "inconclusive"


def test_too_few_queries_is_inconclusive() -> None:
    verdict = decide(report({"all": (10, 0.5)}, {"all": (10, 0.9)}), Gates(min_slice=30))
    assert verdict.status == "inconclusive"


def test_weak_index_is_a_warning() -> None:
    verdict = decide(report({"all": (100, 0.8)}, {"all": (100, 0.8)}, index_recall=0.7), Gates())
    assert verdict.status == "go" and any("finds 70%" in w for w in verdict.warnings)


# --- The runner, with an in-memory store


class FakeSearcher:
    """Rows 0..n-1; each side ranks by a per-side function of the query vector."""

    def __init__(self, n: int = 20) -> None:
        self.n = n
        self.calls: list[tuple[str, str, Any]] = []

    def _rank(self, vector: Sequence[float]) -> list[str]:
        target = int(vector[0]) % self.n
        return [str((target + i) % self.n) for i in range(self.n)]

    def search(
        self, side: str, vector: Sequence[float], k: int, setting: int | None = None
    ) -> list[str]:
        self.calls.append(("search", side, setting))
        ranked = self._rank(vector)[:k]
        if setting == 40:  # a low setting misses the 10th result
            ranked = [*ranked[:9], "miss"]
        return ranked

    def exact(self, side: str, vector: Sequence[float], k: int) -> list[str]:
        self.calls.append(("exact", side, None))
        return self._rank(vector)[:k]

    def index(self, side: str) -> tuple[str | None, int | None, list[int]]:
        return ("hnsw", 40, [40, 100])

    def neighbours(self, side: str, key: str, k: int) -> tuple[str | None, list[tuple[str, str]]]:
        group = str(int(key) // 2)
        found = [
            (str((int(key) + i) % self.n), str(((int(key) + i) % self.n) // 2)) for i in (1, 2)
        ]
        return group, found if side == "new" else found[1:]

    def vectors(self, side: str, keys: Sequence[str]) -> list[list[float]]:
        return [[float(k)] for k in keys]


class FakeModel:
    def __init__(self, shift: int = 0) -> None:
        self.shift = shift

    async def embed(self, texts: Sequence[str], mode: str = "query") -> list[list[float]]:
        return [[float(int(t) + self.shift)] for t in texts]


def queries(n: int = 40) -> list[EvalQuery]:
    return [EvalQuery(str(i % 20), frozenset({str(i % 20)}), "latin", "latin") for i in range(n)]


def run_fake(**kwargs: Any) -> EvalReport:
    options: dict[str, Any] = {
        "old": SideReport("old", "embedding", "old-model"),
        "new": SideReport("new", "embedding_v2", "new-model"),
        "query_source": "proxy",
        "partial": False,
        "sweep_queries": 10,
        "latency_samples": 3,
    }
    searcher = kwargs.pop("searcher", FakeSearcher())
    embedders = kwargs.pop("embedders", {"old": FakeModel(shift=-1), "new": FakeModel()})
    options.update(kwargs)
    return asyncio.run(run(searcher, embedders, queries(), ["0", "1", "2", "3"], **options))


def test_runner_compares_both_sides() -> None:
    result = run_fake()
    assert result.mode == "queries"
    assert result.new.scores["all"].recall_at_1 == 1.0
    assert result.old.scores["all"].recall_at_1 == 0.0, "the old model is off by one"
    assert result.old.scores["all"].recall_at_10 == 1.0
    assert result.old.scores["all"].mrr_at_10 == 0.5, "found second"
    assert [p.setting for p in result.new.sweep] == [40, 100]
    low, high = result.new.sweep
    assert low.index_recall == pytest.approx(0.9) and high.index_recall == 1.0
    assert low.current and not high.current and low.latency and low.latency.samples == 10
    assert result.new.embed_latency and result.new.embed_latency.samples == 3
    assert result.new.same_group_at_10 == 0.25 and result.old.same_group_at_10 == 0.0
    assert result.verdict.status == "go"
    data = result.to_dict()
    assert data["verdict"]["status"] == "go" and data["new"]["sweep"][0]["setting"] == 40


def test_runner_without_the_old_model() -> None:
    result = run_fake(embedders={"old": None, "new": FakeModel()})
    assert result.mode == "vectors" and not result.old.embedded and not result.old.scores
    assert [p.setting for p in result.old.sweep] == [40, 100], "latency from stored vectors"
    assert any("stored row vectors" in n for n in result.notes)
    assert result.verdict.status == "go"


def test_partial_migration_searches_exactly() -> None:
    searcher = FakeSearcher()
    result = run_fake(searcher=searcher, partial=True)
    assert not result.new.sweep and not result.old.sweep, "no latency on a partial migration"
    assert {c[0] for c in searcher.calls} == {"exact"}
    assert any("isn't finished" in n for n in result.notes)
