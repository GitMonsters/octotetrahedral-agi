"""Tests for the stress-test scripts and collect_artifacts helper.

Validates:
- collect_artifacts.py writes all required artifact files
- Stress fixture data files exist and are valid JSONL
- Shell scripts are present and executable
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# collect_artifacts helper
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "stress"
REPO_ROOT = Path(__file__).resolve().parent.parent


class TestCollectArtifacts:
    """collect_artifacts.py writes all required metadata files."""

    def test_required_files_created(self, tmp_path: Path) -> None:
        from scripts.stress.collect_artifacts import collect

        collect(
            out_dir=tmp_path,
            suite="quick",
            seed=1337,
            command="scripts/stress/quick_suite.sh",
            exit_code=0,
        )

        assert (tmp_path / "run_config.json").exists()
        assert (tmp_path / "commit_sha.txt").exists()
        assert (tmp_path / "environment.txt").exists()
        assert (tmp_path / "summary.md").exists()

    def test_run_config_content(self, tmp_path: Path) -> None:
        from scripts.stress.collect_artifacts import collect

        collect(
            out_dir=tmp_path,
            suite="nightly",
            seed=42,
            command="scripts/stress/repro_seeded.sh",
            exit_code=1,
        )

        config = json.loads((tmp_path / "run_config.json").read_text())
        assert config["suite"] == "nightly"
        assert config["seed"] == 42
        assert config["exit_code"] == 1
        assert "timestamp" in config
        assert "command" in config

    def test_commit_sha_nonempty(self, tmp_path: Path) -> None:
        from scripts.stress.collect_artifacts import collect

        collect(out_dir=tmp_path, suite="quick", seed=1337, command="test", exit_code=0)

        sha = (tmp_path / "commit_sha.txt").read_text().strip()
        assert len(sha) > 0

    def test_environment_txt_contains_key_vars(self, tmp_path: Path) -> None:
        from scripts.stress.collect_artifacts import collect

        collect(out_dir=tmp_path, suite="quick", seed=1337, command="test", exit_code=0)

        env_text = (tmp_path / "environment.txt").read_text()
        assert "python=" in env_text
        assert "PYTHONHASHSEED=" in env_text

    def test_summary_pass_status(self, tmp_path: Path) -> None:
        from scripts.stress.collect_artifacts import collect

        collect(out_dir=tmp_path, suite="quick", seed=1337, command="test", exit_code=0)

        summary = (tmp_path / "summary.md").read_text()
        assert "PASS" in summary

    def test_summary_fail_status(self, tmp_path: Path) -> None:
        from scripts.stress.collect_artifacts import collect

        collect(out_dir=tmp_path, suite="nightly", seed=1337, command="test", exit_code=1)

        summary = (tmp_path / "summary.md").read_text()
        assert "FAIL" in summary

    def test_existing_summary_not_overwritten(self, tmp_path: Path) -> None:
        from scripts.stress.collect_artifacts import collect

        # Write a custom summary first
        custom = "# My custom summary\n\nCustom content.\n"
        (tmp_path / "summary.md").write_text(custom)

        collect(out_dir=tmp_path, suite="quick", seed=1337, command="test", exit_code=0)

        # Should not have been overwritten
        assert (tmp_path / "summary.md").read_text() == custom

    def test_creates_out_dir_if_missing(self, tmp_path: Path) -> None:
        from scripts.stress.collect_artifacts import collect

        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        collect(out_dir=nested, suite="quick", seed=1337, command="test", exit_code=0)
        assert nested.exists()
        assert (nested / "run_config.json").exists()

    def test_cli_entrypoint(self, tmp_path: Path) -> None:
        """The CLI main() function should return 0 and create artifacts."""
        from scripts.stress.collect_artifacts import main

        rc = main(["--out-dir", str(tmp_path), "--suite", "quick", "--seed", "42"])
        assert rc == 0
        assert (tmp_path / "run_config.json").exists()


# ---------------------------------------------------------------------------
# Fixture data files
# ---------------------------------------------------------------------------


class TestStressFixtures:
    """Verify that the stress data fixture files exist and contain valid data."""

    def test_distribution_shift_tasks_exists(self) -> None:
        tasks_file = REPO_ROOT / "data" / "stress" / "distribution_shift" / "tasks.jsonl"
        assert tasks_file.exists(), f"Missing fixture: {tasks_file}"

    def test_distribution_shift_tasks_valid_jsonl(self) -> None:
        tasks_file = REPO_ROOT / "data" / "stress" / "distribution_shift" / "tasks.jsonl"
        lines = tasks_file.read_text().strip().splitlines()
        assert len(lines) > 0
        for line in lines:
            task = json.loads(line)
            assert "task_id" in task
            assert "family" in task
            assert "prompt" in task
            assert "expected" in task

    def test_soak_mix_exists(self) -> None:
        soak_file = REPO_ROOT / "data" / "stress" / "soak" / "mix.jsonl"
        assert soak_file.exists(), f"Missing fixture: {soak_file}"

    def test_soak_mix_valid_jsonl(self) -> None:
        soak_file = REPO_ROOT / "data" / "stress" / "soak" / "mix.jsonl"
        lines = soak_file.read_text().strip().splitlines()
        assert len(lines) > 0
        for line in lines:
            task = json.loads(line)
            assert "task_id" in task

    def test_fault_fixtures_exist(self) -> None:
        faults_dir = REPO_ROOT / "data" / "stress" / "faults"
        assert faults_dir.is_dir()
        # At least one fault file must exist
        fault_files = list(faults_dir.glob("*.json"))
        assert len(fault_files) >= 1, "No fault fixture files found"

    def test_empty_fault_fixture_is_empty(self) -> None:
        empty_file = REPO_ROOT / "data" / "stress" / "faults" / "empty.json"
        assert empty_file.exists()
        assert empty_file.read_text().strip() == ""

    def test_malformed_fault_fixture_is_invalid_json(self) -> None:
        malformed = REPO_ROOT / "data" / "stress" / "faults" / "malformed_json.json"
        assert malformed.exists()
        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed.read_text())


# ---------------------------------------------------------------------------
# Shell script presence and executability
# ---------------------------------------------------------------------------


class TestStressScripts:
    """Shell scripts must exist and be executable."""

    @pytest.mark.parametrize(
        "script_name",
        [
            "quick_suite.sh",
            "distribution_shift.sh",
            "repro_seeded.sh",
            "soak.sh",
            "fault_injection.sh",
        ],
    )
    def test_script_exists(self, script_name: str) -> None:
        script = SCRIPTS_DIR / script_name
        assert script.exists(), f"Missing script: {script}"

    @pytest.mark.parametrize(
        "script_name",
        [
            "quick_suite.sh",
            "distribution_shift.sh",
            "repro_seeded.sh",
            "soak.sh",
            "fault_injection.sh",
        ],
    )
    def test_script_is_executable(self, script_name: str) -> None:
        script = SCRIPTS_DIR / script_name
        assert script.exists(), f"Missing script: {script}"
        mode = script.stat().st_mode
        assert mode & stat.S_IXUSR, f"Script not executable: {script}"

    def test_collect_artifacts_exists(self) -> None:
        py_helper = SCRIPTS_DIR / "collect_artifacts.py"
        assert py_helper.exists(), f"Missing helper: {py_helper}"

    def test_scripts_have_shebang(self) -> None:
        for sh in SCRIPTS_DIR.glob("*.sh"):
            first_line = sh.read_text().splitlines()[0]
            assert first_line.startswith("#!/"), (
                f"{sh.name}: missing shebang (got: {first_line!r})"
            )

    def test_scripts_use_set_euo_pipefail(self) -> None:
        for sh in SCRIPTS_DIR.glob("*.sh"):
            content = sh.read_text()
            assert "set -euo pipefail" in content, (
                f"{sh.name}: missing 'set -euo pipefail'"
            )


# ---------------------------------------------------------------------------
# GitHub Actions workflow files
# ---------------------------------------------------------------------------


class TestWorkflowFiles:
    """Verify that the CI workflow files exist and reference the stress scripts."""

    def test_stress_quick_workflow_exists(self) -> None:
        wf = REPO_ROOT / ".github" / "workflows" / "stress-quick.yml"
        assert wf.exists(), f"Missing workflow: {wf}"

    def test_stress_nightly_workflow_exists(self) -> None:
        wf = REPO_ROOT / ".github" / "workflows" / "stress-nightly.yml"
        assert wf.exists(), f"Missing workflow: {wf}"

    def test_stress_quick_references_script(self) -> None:
        wf = REPO_ROOT / ".github" / "workflows" / "stress-quick.yml"
        content = wf.read_text()
        assert "quick_suite.sh" in content

    def test_stress_nightly_references_scripts(self) -> None:
        wf = REPO_ROOT / ".github" / "workflows" / "stress-nightly.yml"
        content = wf.read_text()
        assert "repro_seeded.sh" in content
        assert "distribution_shift.sh" in content
        assert "fault_injection.sh" in content
        assert "soak.sh" in content

    def test_stress_nightly_has_schedule(self) -> None:
        wf = REPO_ROOT / ".github" / "workflows" / "stress-nightly.yml"
        content = wf.read_text()
        assert "schedule:" in content
        assert "cron:" in content

    def test_stress_quick_uploads_artifacts(self) -> None:
        wf = REPO_ROOT / ".github" / "workflows" / "stress-quick.yml"
        content = wf.read_text()
        assert "upload-artifact" in content
