"""Comprehensive tests for the AGI eval harness.

Covers:
  - Generator determinism (same seed => identical tasks)
  - Schema serialisation / deserialisation round-trip
  - Scoring correctness on fixed fixtures
  - Regression detection logic
  - End-to-end CLI smoke path
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------


class TestGeneratorDeterminism:
    """Same seed must always produce byte-identical task sets."""

    def test_same_seed_same_tasks(self) -> None:
        from eval_harness.generator import generate_tasks

        a = generate_tasks(seed=42)
        b = generate_tasks(seed=42)
        assert len(a) == len(b)
        for ta, tb in zip(a, b):
            assert ta.task_id == tb.task_id
            assert ta.prompt == tb.prompt
            assert ta.expected == tb.expected
            assert ta.family == tb.family

    def test_different_seeds_different_tasks(self) -> None:
        from eval_harness.generator import generate_tasks

        a = generate_tasks(seed=1)
        b = generate_tasks(seed=2)
        # Prompts should differ (extremely unlikely to collide)
        prompts_a = {t.prompt for t in a}
        prompts_b = {t.prompt for t in b}
        assert prompts_a != prompts_b

    def test_default_task_count(self) -> None:
        from eval_harness.generator import generate_tasks

        tasks = generate_tasks(seed=0)
        assert len(tasks) == 80

    def test_custom_task_count(self) -> None:
        from eval_harness.generator import generate_tasks

        tasks = generate_tasks(seed=0, num_tasks=40)
        assert len(tasks) == 40

    def test_all_four_families_present(self) -> None:
        from eval_harness.generator import FAMILIES, generate_tasks

        tasks = generate_tasks(seed=7)
        families_found = {t.family for t in tasks}
        for fam in FAMILIES:
            assert fam in families_found

    def test_family_selection(self) -> None:
        from eval_harness.generator import generate_tasks

        tasks = generate_tasks(seed=5, num_tasks=20, families=["compositional", "sequence"])
        families = {t.family for t in tasks}
        assert families == {"compositional", "sequence"}

    def test_unknown_family_raises(self) -> None:
        from eval_harness.generator import generate_tasks

        with pytest.raises(ValueError, match="Unknown task families"):
            generate_tasks(seed=0, families=["nonexistent"])

    def test_zero_num_tasks_raises(self) -> None:
        from eval_harness.generator import generate_tasks

        with pytest.raises(ValueError, match="num_tasks"):
            generate_tasks(seed=0, num_tasks=0)

    def test_task_ids_unique(self) -> None:
        from eval_harness.generator import generate_tasks

        tasks = generate_tasks(seed=99)
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_schema_version_present(self) -> None:
        from eval_harness.generator import SCHEMA_VERSION, generate_tasks

        tasks = generate_tasks(seed=1)
        for t in tasks:
            assert t.schema_version == SCHEMA_VERSION

    def test_compositional_task_structure(self) -> None:
        from eval_harness.generator import generate_tasks

        tasks = [t for t in generate_tasks(seed=3) if t.family == "compositional"]
        assert tasks
        for t in tasks:
            assert t.expected in ("yes", "no")
            assert "depth" in t.metadata
            assert t.metadata["depth"] >= 2

    def test_sequence_task_structure(self) -> None:
        from eval_harness.generator import generate_tasks

        tasks = [t for t in generate_tasks(seed=3) if t.family == "sequence"]
        assert tasks
        for t in tasks:
            assert t.expected.isdigit() or t.expected.lstrip("-").isdigit()
            assert t.metadata["kind"] in ("arithmetic", "geometric")

    def test_analogy_task_structure(self) -> None:
        from eval_harness.generator import generate_tasks

        tasks = [t for t in generate_tasks(seed=3) if t.family == "analogy"]
        assert tasks
        for t in tasks:
            assert t.expected  # non-empty
            assert "a" in t.metadata

    def test_pattern_task_structure(self) -> None:
        from eval_harness.generator import generate_tasks

        tasks = [t for t in generate_tasks(seed=3) if t.family == "pattern"]
        assert tasks
        for t in tasks:
            assert t.metadata["op"] in ("rotate", "invert", "shift", "mirror")


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


class TestSerialisationRoundTrip:
    def test_save_and_load(self, tmp_path: Path) -> None:
        from eval_harness.generator import generate_tasks, load_tasks, save_tasks

        tasks = generate_tasks(seed=10, num_tasks=20)
        dest = tmp_path / "tasks.jsonl"
        save_tasks(tasks, dest)

        loaded = load_tasks(dest)
        assert len(loaded) == len(tasks)
        for orig, back in zip(tasks, loaded):
            assert orig.task_id == back.task_id
            assert orig.prompt == back.prompt
            assert orig.expected == back.expected
            assert orig.family == back.family

    def test_jsonl_format(self, tmp_path: Path) -> None:
        from eval_harness.generator import generate_tasks, save_tasks

        tasks = generate_tasks(seed=11, num_tasks=4)
        dest = tmp_path / "tasks.jsonl"
        save_tasks(tasks, dest)

        lines = [l for l in dest.read_text().splitlines() if l.strip()]
        assert len(lines) == 4
        for line in lines:
            obj = json.loads(line)
            assert "task_id" in obj
            assert "schema_version" in obj

    def test_load_invalid_schema_version_raises(self, tmp_path: Path) -> None:
        from eval_harness.generator import load_tasks

        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps({"schema_version": "99.0", "task_id": "x", "family": "f",
                                    "seed": 0, "prompt": "p", "expected": "e", "metadata": {}}))
        with pytest.raises(ValueError, match="Unsupported schema_version"):
            load_tasks(bad)

    def test_task_hash_deterministic(self) -> None:
        from eval_harness.generator import generate_tasks, task_set_hash

        tasks = generate_tasks(seed=7)
        h1 = task_set_hash(tasks)
        h2 = task_set_hash(tasks)
        assert h1 == h2

    def test_task_hash_changes_with_seed(self) -> None:
        from eval_harness.generator import generate_tasks, task_set_hash

        h1 = task_set_hash(generate_tasks(seed=1))
        h2 = task_set_hash(generate_tasks(seed=2))
        assert h1 != h2


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------


def _make_task(family: str, expected: str, metadata: dict | None = None):
    from eval_harness.generator import SCHEMA_VERSION, TaskSpec

    return TaskSpec(
        schema_version=SCHEMA_VERSION,
        task_id=f"test_{family}_000",
        family=family,
        seed=0,
        prompt="test prompt",
        expected=expected,
        metadata=metadata or {},
    )


class TestScorerFixtures:
    """Fixed-input scoring correctness tests."""

    def test_compositional_correct(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("compositional", "yes")
        ts = score_task(task, {"task_id": task.task_id, "answer": "yes"})
        assert ts.score == 1.0
        assert ts.correct

    def test_compositional_wrong(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("compositional", "yes")
        ts = score_task(task, {"task_id": task.task_id, "answer": "no"})
        assert ts.score == 0.0
        assert not ts.correct

    def test_compositional_case_insensitive(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("compositional", "yes")
        ts = score_task(task, {"task_id": task.task_id, "answer": "YES"})
        assert ts.score == 1.0

    def test_compositional_trailing_punctuation(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("compositional", "yes")
        ts = score_task(task, {"task_id": task.task_id, "answer": "yes."})
        assert ts.score == 1.0

    def test_sequence_correct(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("sequence", "48")
        ts = score_task(task, {"task_id": task.task_id, "answer": "48"})
        assert ts.score == 1.0

    def test_sequence_wrong(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("sequence", "48")
        ts = score_task(task, {"task_id": task.task_id, "answer": "50"})
        assert ts.score == 0.0

    def test_analogy_correct(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("analogy", "cold")
        ts = score_task(task, {"task_id": task.task_id, "answer": "cold"})
        assert ts.score == 1.0

    def test_analogy_case_insensitive(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("analogy", "cold")
        ts = score_task(task, {"task_id": task.task_id, "answer": "COLD"})
        assert ts.score == 1.0

    def test_pattern_full_credit(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("pattern", "R G B")
        ts = score_task(task, {"task_id": task.task_id, "answer": "R G B"})
        assert ts.score == pytest.approx(1.0)

    def test_pattern_partial_credit(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("pattern", "R G B")
        # 2/3 correct
        ts = score_task(task, {"task_id": task.task_id, "answer": "R G Y"})
        assert ts.score == pytest.approx(2 / 3)

    def test_pattern_zero_credit(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("pattern", "R G B")
        ts = score_task(task, {"task_id": task.task_id, "answer": "X Y Z"})
        assert ts.score == 0.0

    def test_missing_answer_scores_zero(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("sequence", "10")
        ts = score_task(task, {"task_id": task.task_id})  # no 'answer' key
        assert ts.score == 0.0

    def test_confidence_field_passed_through(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("sequence", "10")
        ts = score_task(task, {"task_id": task.task_id, "answer": "10", "confidence": 0.95})
        assert ts.confidence == pytest.approx(0.95)
        assert ts.score == 1.0

    def test_confidence_does_not_affect_score(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("sequence", "10")
        ts_high = score_task(task, {"task_id": task.task_id, "answer": "99", "confidence": 0.99})
        ts_low = score_task(task, {"task_id": task.task_id, "answer": "99", "confidence": 0.01})
        assert ts_high.score == ts_low.score

    def test_unknown_family_falls_back_to_exact(self) -> None:
        from eval_harness.scorer import score_task

        task = _make_task("unknown_family", "x")
        ts = score_task(task, {"task_id": task.task_id, "answer": "x"})
        assert ts.score == 1.0

    def test_score_tasks_batch(self) -> None:
        from eval_harness.scorer import score_tasks

        tasks = [
            _make_task("sequence", "10"),
            _make_task("analogy", "cold"),
        ]
        tasks[0].task_id = "seq_000"
        tasks[1].task_id = "ana_000"
        outputs = [
            {"task_id": "seq_000", "answer": "10"},
            {"task_id": "ana_000", "answer": "cold"},
        ]
        scored = score_tasks(tasks, outputs)
        assert all(ts.score == 1.0 for ts in scored)

    def test_missing_output_scores_zero(self) -> None:
        from eval_harness.scorer import score_tasks

        tasks = [_make_task("sequence", "10")]
        tasks[0].task_id = "seq_001"
        scored = score_tasks(tasks, [])  # no outputs at all
        assert scored[0].score == 0.0

    def test_aggregate_overall(self) -> None:
        from eval_harness.scorer import TaskScore, aggregate_scores

        task_scores = [
            TaskScore("t1", "sequence", 1.0, True, "10", "10"),
            TaskScore("t2", "sequence", 0.0, False, "9", "10"),
            TaskScore("t3", "analogy", 1.0, True, "cold", "cold"),
            TaskScore("t4", "analogy", 0.0, False, "x", "cold"),
        ]
        rs = aggregate_scores(task_scores)
        assert rs.overall == pytest.approx(0.5)
        assert rs.n_tasks == 4
        assert rs.n_correct == 2

    def test_aggregate_family_breakdown(self) -> None:
        from eval_harness.scorer import TaskScore, aggregate_scores

        task_scores = [
            TaskScore("t1", "sequence", 1.0, True, "10", "10"),
            TaskScore("t2", "analogy", 0.0, False, "x", "cold"),
        ]
        rs = aggregate_scores(task_scores)
        assert rs.family_scores["sequence"]["mean"] == pytest.approx(1.0)
        assert rs.family_scores["analogy"]["mean"] == pytest.approx(0.0)

    def test_aggregate_empty(self) -> None:
        from eval_harness.scorer import aggregate_scores

        rs = aggregate_scores([])
        assert rs.overall == 0.0
        assert rs.n_tasks == 0


# ---------------------------------------------------------------------------
# Tracker / regression tests
# ---------------------------------------------------------------------------


def _make_run(
    overall: float,
    family_scores: dict | None = None,
    seed: int = 42,
    tag: str = "",
    run_id: str | None = None,
    timestamp: str | None = None,
):
    from eval_harness.tracker import make_run_record

    fs = family_scores or {"compositional": {"mean": overall, "n": 10, "n_correct": int(overall * 10)}}
    return make_run_record(
        seed=seed,
        task_hash="abc" * 10 + "de",
        overall=overall,
        n_tasks=10,
        n_correct=int(overall * 10),
        family_scores=fs,
        tag=tag,
        run_id=run_id,
        timestamp=timestamp,
    )


class TestRegressionDetection:
    def test_no_change_detected(self) -> None:
        from eval_harness.tracker import compare_runs

        baseline = _make_run(0.70)
        current = _make_run(0.70)
        result = compare_runs(current, baseline, threshold=0.02)
        assert not result.improved
        assert not result.regressed
        assert result.delta_overall == pytest.approx(0.0)

    def test_improvement_detected(self) -> None:
        from eval_harness.tracker import compare_runs

        baseline = _make_run(0.60)
        current = _make_run(0.65)
        result = compare_runs(current, baseline, threshold=0.02)
        assert result.improved
        assert not result.regressed
        assert result.delta_overall == pytest.approx(0.05)

    def test_regression_detected(self) -> None:
        from eval_harness.tracker import compare_runs

        baseline = _make_run(0.80)
        current = _make_run(0.75)
        result = compare_runs(current, baseline, threshold=0.02)
        assert result.regressed
        assert not result.improved
        assert result.delta_overall == pytest.approx(-0.05)

    def test_below_threshold_not_flagged(self) -> None:
        from eval_harness.tracker import compare_runs

        baseline = _make_run(0.70)
        current = _make_run(0.71)  # delta = 0.01 < threshold 0.02
        result = compare_runs(current, baseline, threshold=0.02)
        assert not result.improved
        assert not result.regressed

    def test_family_regression_flagged(self) -> None:
        from eval_harness.tracker import compare_runs

        baseline = _make_run(
            0.80,
            family_scores={"compositional": {"mean": 0.90, "n": 5, "n_correct": 4},
                           "sequence": {"mean": 0.70, "n": 5, "n_correct": 3}},
        )
        current = _make_run(
            0.80,
            family_scores={"compositional": {"mean": 0.80, "n": 5, "n_correct": 4},
                           "sequence": {"mean": 0.80, "n": 5, "n_correct": 4}},
        )
        result = compare_runs(current, baseline, threshold=0.05)
        assert "compositional" in result.family_regressions

    def test_family_improvement_flagged(self) -> None:
        from eval_harness.tracker import compare_runs

        baseline = _make_run(
            0.70,
            family_scores={"sequence": {"mean": 0.50, "n": 10, "n_correct": 5}},
        )
        current = _make_run(
            0.70,
            family_scores={"sequence": {"mean": 0.80, "n": 10, "n_correct": 8}},
        )
        result = compare_runs(current, baseline, threshold=0.02)
        assert "sequence" in result.family_improvements

    def test_summary_contains_direction(self) -> None:
        from eval_harness.tracker import compare_runs

        baseline = _make_run(0.60)
        current = _make_run(0.70)
        result = compare_runs(current, baseline, threshold=0.02)
        assert "improved" in result.summary.lower() or "▲" in result.summary

    def test_configurable_threshold(self) -> None:
        from eval_harness.tracker import compare_runs

        baseline = _make_run(0.70)
        current = _make_run(0.75)  # delta 0.05
        # Should be improvement at threshold=0.02
        r1 = compare_runs(current, baseline, threshold=0.02)
        assert r1.improved
        # Should NOT be improvement at threshold=0.10
        r2 = compare_runs(current, baseline, threshold=0.10)
        assert not r2.improved


class TestRunPersistence:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        from eval_harness.tracker import load_runs, save_run

        record = _make_run(0.75, tag="test")
        save_run(record, tmp_path)
        runs = load_runs(tmp_path)
        assert len(runs) == 1
        r = runs[0]
        assert r.run_id == record.run_id
        assert r.overall == pytest.approx(0.75)
        assert r.tag == "test"
        assert r.seed == 42

    def test_multiple_runs_sorted_by_timestamp(self, tmp_path: Path) -> None:
        from eval_harness.tracker import load_runs, save_run

        r1 = _make_run(0.60, timestamp="20260101T000000Z", run_id="aaa")
        r2 = _make_run(0.70, timestamp="20260201T000000Z", run_id="bbb")
        r3 = _make_run(0.80, timestamp="20260301T000000Z", run_id="ccc")
        # Save in reverse order
        for r in (r3, r1, r2):
            save_run(r, tmp_path)
        runs = load_runs(tmp_path)
        assert [r.run_id for r in runs] == ["aaa", "bbb", "ccc"]

    def test_find_run_by_prefix(self, tmp_path: Path) -> None:
        from eval_harness.tracker import find_run, load_runs, save_run

        record = _make_run(0.80, run_id="deadbeef1234")
        save_run(record, tmp_path)
        runs = load_runs(tmp_path)
        found = find_run(runs, "deadbe")
        assert found is not None
        assert found.run_id == "deadbeef1234"

    def test_find_run_not_found(self, tmp_path: Path) -> None:
        from eval_harness.tracker import find_run, load_runs, save_run

        save_run(_make_run(0.8), tmp_path)
        runs = load_runs(tmp_path)
        assert find_run(runs, "zzzzzzz") is None

    def test_load_empty_dir(self, tmp_path: Path) -> None:
        from eval_harness.tracker import load_runs

        runs = load_runs(tmp_path)
        assert runs == []

    def test_load_nonexistent_dir(self, tmp_path: Path) -> None:
        from eval_harness.tracker import load_runs

        runs = load_runs(tmp_path / "does_not_exist")
        assert runs == []

    def test_trend_summary_format(self, tmp_path: Path) -> None:
        from eval_harness.tracker import load_runs, save_run, trend_summary

        for i in range(3):
            save_run(_make_run(0.5 + i * 0.1, tag=f"run{i}",
                               timestamp=f"2026010{i+1}T000000Z", run_id=f"id{i:04d}"), tmp_path)
        runs = load_runs(tmp_path)
        table = trend_summary(runs, last=10)
        assert "Timestamp" in table
        assert "Overall" in table
        for i in range(3):
            assert f"run{i}" in table


# ---------------------------------------------------------------------------
# End-to-end CLI smoke tests
# ---------------------------------------------------------------------------


class TestCLISmokePath:
    """End-to-end: generate → evaluate → compare → trend."""

    def test_generate_creates_file(self, tmp_path: Path) -> None:
        from eval_harness.cli import main

        out = tmp_path / "tasks.jsonl"
        ret = main(["generate", "--seed", "42", "--num-tasks", "20", "--output", str(out)])
        assert ret == 0
        assert out.exists()
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 20

    def test_generate_with_families(self, tmp_path: Path) -> None:
        from eval_harness.cli import main

        out = tmp_path / "tasks.jsonl"
        ret = main([
            "generate", "--seed", "1", "--num-tasks", "10",
            "--families", "sequence,analogy",
            "--output", str(out),
        ])
        assert ret == 0
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        for line in lines:
            obj = json.loads(line)
            assert obj["family"] in ("sequence", "analogy")

    def test_evaluate_mock_creates_run(self, tmp_path: Path) -> None:
        from eval_harness.cli import main

        tasks_file = tmp_path / "tasks.jsonl"
        runs_dir = tmp_path / "runs"
        main(["generate", "--seed", "42", "--num-tasks", "20", "--output", str(tasks_file)])
        ret = main([
            "evaluate",
            "--tasks", str(tasks_file),
            "--outputs", str(tmp_path / "nonexistent.jsonl"),
            "--mock",
            "--runs-dir", str(runs_dir),
            "--tag", "smoke",
        ])
        assert ret == 0
        assert any(runs_dir.glob("*.json"))

    def test_evaluate_with_real_outputs(self, tmp_path: Path) -> None:
        from eval_harness.cli import main
        from eval_harness.generator import generate_tasks

        tasks_file = tmp_path / "tasks.jsonl"
        main(["generate", "--seed", "7", "--num-tasks", "16", "--output", str(tasks_file)])

        # Build a perfect-score outputs file
        tasks = generate_tasks(seed=7, num_tasks=16)
        outputs_file = tmp_path / "outputs.jsonl"
        with outputs_file.open("w") as fh:
            for t in tasks:
                fh.write(json.dumps({"task_id": t.task_id, "answer": t.expected}) + "\n")

        runs_dir = tmp_path / "runs"
        ret = main([
            "evaluate",
            "--tasks", str(tasks_file),
            "--outputs", str(outputs_file),
            "--runs-dir", str(runs_dir),
            "--tag", "perfect",
        ])
        assert ret == 0
        # Check run record has overall == 1.0
        from eval_harness.tracker import load_runs
        runs = load_runs(runs_dir)
        assert len(runs) == 1
        assert runs[0].overall == pytest.approx(1.0)

    def test_compare_improvement_exit_zero(self, tmp_path: Path) -> None:
        from eval_harness.cli import main
        from eval_harness.tracker import RunRecord, save_run

        runs_dir = tmp_path / "runs"
        r1 = _make_run(0.60, timestamp="20260101T000000Z", run_id="aaaa0001", tag="baseline")
        r2 = _make_run(0.80, timestamp="20260201T000000Z", run_id="bbbb0002", tag="current")
        save_run(r1, runs_dir)
        save_run(r2, runs_dir)

        ret = main(["compare", "--runs-dir", str(runs_dir), "--threshold", "0.05"])
        assert ret == 0  # improvement → exit 0

    def test_compare_regression_exit_one(self, tmp_path: Path) -> None:
        from eval_harness.cli import main
        from eval_harness.tracker import save_run

        runs_dir = tmp_path / "runs"
        r1 = _make_run(0.80, timestamp="20260101T000000Z", run_id="cccc0003", tag="baseline")
        r2 = _make_run(0.60, timestamp="20260201T000000Z", run_id="dddd0004", tag="regressed")
        save_run(r1, runs_dir)
        save_run(r2, runs_dir)

        ret = main(["compare", "--runs-dir", str(runs_dir), "--threshold", "0.05"])
        assert ret == 1  # regression → exit 1

    def test_trend_outputs_table(self, tmp_path: Path, capsys) -> None:
        from eval_harness.cli import main
        from eval_harness.tracker import save_run

        runs_dir = tmp_path / "runs"
        for i in range(3):
            save_run(_make_run(0.5 + i * 0.1, timestamp=f"2026010{i+1}T000000Z",
                               run_id=f"e{i:07d}"), runs_dir)

        ret = main(["trend", "--runs-dir", str(runs_dir), "--last", "5"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Timestamp" in captured.out

    def test_full_cycle_determinism(self, tmp_path: Path) -> None:
        """Run generate+evaluate twice with the same seed; scores must be identical."""
        from eval_harness.cli import main
        from eval_harness.tracker import load_runs

        def run_cycle(run_tag: str):
            tasks_file = tmp_path / f"tasks_{run_tag}.jsonl"
            runs_dir = tmp_path / f"runs_{run_tag}"
            main(["generate", "--seed", "42", "--num-tasks", "40", "--output", str(tasks_file)])
            main([
                "evaluate",
                "--tasks", str(tasks_file),
                "--outputs", "ignored.jsonl",
                "--mock",
                "--mock-score", "0.8",
                "--runs-dir", str(runs_dir),
                "--tag", run_tag,
            ])
            return load_runs(runs_dir)[0]

        r1 = run_cycle("first")
        r2 = run_cycle("second")
        assert r1.overall == pytest.approx(r2.overall)
        assert r1.task_hash == r2.task_hash
