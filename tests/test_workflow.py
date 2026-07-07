"""Tests for the compound workflow orchestrator (workflow.py).

These tests verify:
- The monitoring namespace fix (``from monitoring import InferenceMonitor``)
- Imports and initialization of the compound workflow
- Health-check integration
- Single-inference integration
- Evaluate integration
- CLI entrypoint basic execution
- Context-manager lifecycle
"""

from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# 1. Monitoring namespace fix
# ---------------------------------------------------------------------------


def test_monitoring_package_exports_inference_monitor():
    """InferenceMonitor must be importable from the monitoring package."""
    from monitoring import CoherenceAlert, InferenceMonitor, MonitoringStats

    monitor = InferenceMonitor()
    assert callable(monitor.record)
    assert callable(monitor.stats)
    assert callable(monitor.reset)
    # Verify type annotations are also importable
    assert CoherenceAlert is not None
    assert MonitoringStats is not None


def test_monitoring_inference_monitor_module_importable():
    """monitoring.inference_monitor sub-module must also be importable."""
    from monitoring import InferenceMonitor
    from monitoring.inference_monitor import InferenceMonitor as IM

    # Both import paths must resolve to the same class
    assert IM is InferenceMonitor


# ---------------------------------------------------------------------------
# 2. Workflow imports
# ---------------------------------------------------------------------------


def test_workflow_imports():
    """workflow module must import without errors."""
    import workflow

    from workflow import CompoundWorkflow, WorkflowConfig

    assert CompoundWorkflow is not None
    assert WorkflowConfig is not None
    assert hasattr(workflow, "main")


# ---------------------------------------------------------------------------
# 3. Workflow initialization
# ---------------------------------------------------------------------------


def test_workflow_initialize_and_shutdown():
    from workflow import CompoundWorkflow

    wf = CompoundWorkflow()
    assert not wf._initialized

    wf.initialize()
    assert wf._initialized
    assert wf.service is not None
    assert wf.monitor is not None

    wf.shutdown()
    assert not wf._initialized


def test_workflow_initialize_idempotent():
    from workflow import CompoundWorkflow

    wf = CompoundWorkflow()
    wf.initialize()
    service_id = id(wf._service)
    wf.initialize()  # second call — should not re-create
    assert id(wf._service) == service_id
    wf.shutdown()


def test_workflow_context_manager():
    from workflow import CompoundWorkflow

    with CompoundWorkflow() as wf:
        assert wf._initialized
    assert not wf._initialized


def test_workflow_service_raises_before_initialize():
    from workflow import CompoundWorkflow

    wf = CompoundWorkflow()
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = wf.service


# ---------------------------------------------------------------------------
# 4. Health check
# ---------------------------------------------------------------------------


def test_workflow_health_check():
    from workflow import CompoundWorkflow

    with CompoundWorkflow() as wf:
        status = wf.health_check(num_tests=2)

    assert status["healthy"] is True
    assert status["model_loaded"] is True
    assert status["self_test_passed"] is True
    assert len(status["self_test_details"]) == 2


def test_workflow_boot_convenience():
    from workflow import CompoundWorkflow

    wf = CompoundWorkflow()
    status = wf.boot()

    assert status["healthy"] is True
    wf.shutdown()


# ---------------------------------------------------------------------------
# 5. Single inference
# ---------------------------------------------------------------------------


def test_workflow_infer_basic():
    from workflow import CompoundWorkflow

    with CompoundWorkflow() as wf:
        result = wf.infer([0.5] * 8, task_signal="reasoning")

    assert result["error"] is None
    assert 0.0 <= result["coherence"] <= 1.0
    assert len(result["limb_states"]) == 8


def test_workflow_infer_with_request_id():
    from workflow import CompoundWorkflow

    with CompoundWorkflow() as wf:
        result = wf.infer([0.1] * 8, request_id="test-wf-001")

    assert result["request_id"] == "test-wf-001"


def test_workflow_infer_monitor_records():
    from workflow import CompoundWorkflow

    with CompoundWorkflow() as wf:
        wf.infer([0.5] * 8, task_signal="language")
        stats = wf.monitor.stats()

    assert stats["total_inferences"] >= 1


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------


def test_workflow_evaluate_returns_summary():
    from workflow import CompoundWorkflow

    with CompoundWorkflow() as wf:
        summary = wf.evaluate(num_tasks=4, seed=99)

    assert summary["total_tasks"] == 4
    assert summary["seed"] == 99
    assert 0.0 <= summary["mean_score"] <= 1.0
    assert 0.0 <= summary["pass_rate"] <= 1.0
    assert isinstance(summary["by_family"], dict)


def test_workflow_evaluate_all_pipeline_stages():
    """Evaluate should exercise model init, inference, and scoring without error."""
    from workflow import CompoundWorkflow

    with CompoundWorkflow() as wf:
        # Health check first (as the CLI does)
        status = wf.health_check(num_tests=1)
        assert status["healthy"] is True

        summary = wf.evaluate(num_tasks=8, seed=0)

    assert summary["total_tasks"] == 8


# ---------------------------------------------------------------------------
# 7. CLI entrypoint
# ---------------------------------------------------------------------------


def test_workflow_main_health_check_mode(capsys):
    from workflow import main

    exit_code = main(["--mode", "health-check", "--health-tests", "1"])
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["healthy"] is True


def test_workflow_main_inference_mode(capsys):
    from workflow import main

    exit_code = main([
        "--mode", "inference",
        "--limb-states", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
        "--task-signal", "reasoning",
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["error"] is None


def test_workflow_main_inference_defaults(capsys):
    """Inference mode with no --limb-states should use default 0.5 vector."""
    from workflow import main

    exit_code = main(["--mode", "inference"])
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["error"] is None


def test_workflow_main_evaluate_mode(capsys):
    from workflow import main

    exit_code = main([
        "--mode", "evaluate",
        "--num-tasks", "4",
        "--eval-seed", "7",
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["total_tasks"] == 4
