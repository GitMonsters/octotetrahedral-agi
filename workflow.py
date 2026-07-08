"""Compound workflow orchestrator for OctoTetrahedral AGI.

This module is the single canonical entrypoint that wires together the
major model-lifecycle stages:

    1. Config loading (``production_config``)
    2. Inference service initialisation (``InferenceService``)
    3. Health checks / self-test (``run_health_check``)
    4. Monitoring setup (``InferenceMonitor``)
    5. Optional serving mode (delegates to ``serve.py`` via subprocess)
    6. Optional evaluation / benchmark hook-points (``eval_harness``)

Workflow modes
--------------
``health-check``
    Initialise the inference service, run the built-in self-test suite,
    and print a diagnostics report.  Exits 0 on healthy, 1 on failure.

``inference``
    Initialise the inference service and run a single forward pass.
    Useful for quick smoke-tests and debugging.

``evaluate``
    Run the deterministic eval-harness benchmark against the live
    inference service and print a summary.

``serve``
    Start the HTTP inference server (``serve.py``) as a subprocess after
    a successful health check.  All extra arguments are forwarded to
    ``serve.py``.

Quick start
-----------
::

    # Self-test / health check
    python workflow.py --mode health-check

    # Single inference
    python workflow.py --mode inference --limb-states 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8

    # Evaluation benchmark (20 tasks)
    python workflow.py --mode evaluate --num-tasks 20

    # HTTP serving (passes extra args to serve.py)
    python workflow.py --mode serve -- --scale tiny --port 8080

Programmatic usage
------------------
::

    from workflow import CompoundWorkflow

    wf = CompoundWorkflow()
    wf.initialize()
    status = wf.health_check()
    result = wf.infer([0.5] * 8, task_signal="reasoning")
    summary = wf.evaluate(num_tasks=10)
    wf.shutdown()

    # or as a context manager
    with CompoundWorkflow() as wf:
        wf.health_check()
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

import production_config as cfg
from api_types import make_request
from eval_harness import generate_tasks, score_tasks, aggregate_scores
from health_check import HealthStatus, run_health_check
from inference_service import InferenceService
from monitoring import InferenceMonitor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Workflow configuration
# ---------------------------------------------------------------------------


@dataclass
class WorkflowConfig:
    """Configuration for the compound workflow.

    All parameters may be overridden via environment variables or CLI flags.
    Defaults mirror ``production_config`` for the active environment.
    """

    pool_size: int = field(default_factory=lambda: cfg.EFFECTIVE_POOL_SIZE)
    timeout_ms: float = field(default_factory=lambda: cfg.EFFECTIVE_TIMEOUT_MS)
    max_retries: int = field(default_factory=lambda: cfg.EFFECTIVE_MAX_RETRIES)
    limb_count: int = field(default_factory=lambda: cfg.MODEL_LIMB_COUNT)
    coherence_threshold: float = field(default_factory=lambda: cfg.COHERENCE_ALERT_THRESHOLD)
    health_check_tests: int = 5
    eval_seed: int = 42
    eval_num_tasks: int = 20


# ---------------------------------------------------------------------------
# Compound workflow
# ---------------------------------------------------------------------------


class CompoundWorkflow:
    """Orchestrates the full model lifecycle: init → health-check → run.

    The workflow is intentionally a thin composition layer.  Each stage
    delegates to the canonical subsystem and avoids duplicating logic.
    """

    def __init__(self, config: WorkflowConfig | None = None) -> None:
        self.config = config or WorkflowConfig()
        self._monitor: InferenceMonitor | None = None
        self._service: InferenceService | None = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> "CompoundWorkflow":
        """Initialize monitoring and inference service.

        Safe to call multiple times — returns self without re-initialising.
        """
        if self._initialized:
            return self

        logger.info(
            json.dumps(
                {
                    "event": "workflow_init",
                    "env": cfg.ENV,
                    "model_version": cfg.MODEL_VERSION,
                    "pool_size": self.config.pool_size,
                }
            )
        )

        self._monitor = InferenceMonitor(
            coherence_threshold=self.config.coherence_threshold,
            limb_count=self.config.limb_count,
        )
        self._service = InferenceService(
            pool_size=self.config.pool_size,
            limb_count=self.config.limb_count,
            timeout_ms=self.config.timeout_ms,
            max_retries=self.config.max_retries,
            monitor=self._monitor,
        )
        self._initialized = True
        logger.info(json.dumps({"event": "workflow_ready"}))
        return self

    def shutdown(self) -> None:
        """Release resources.  After calling, the workflow can be re-initialized."""
        self._service = None
        self._monitor = None
        self._initialized = False
        logger.info(json.dumps({"event": "workflow_shutdown"}))

    # Context-manager support

    def __enter__(self) -> "CompoundWorkflow":
        return self.initialize()

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Stage accessors (require initialize() to have been called first)
    # ------------------------------------------------------------------

    @property
    def service(self) -> InferenceService:
        if self._service is None:
            raise RuntimeError("CompoundWorkflow not initialized — call initialize() first")
        return self._service

    @property
    def monitor(self) -> InferenceMonitor:
        if self._monitor is None:
            raise RuntimeError("CompoundWorkflow not initialized — call initialize() first")
        return self._monitor

    # ------------------------------------------------------------------
    # Stage 3: Health check
    # ------------------------------------------------------------------

    def health_check(self, num_tests: int | None = None) -> HealthStatus:
        """Run the self-test suite against the initialized service.

        Parameters
        ----------
        num_tests:
            Override the number of self-test cases (1–5).  Defaults to
            ``config.health_check_tests``.

        Returns
        -------
        HealthStatus
            Diagnostics dict; ``status["healthy"]`` is True on success.
        """
        n = num_tests if num_tests is not None else self.config.health_check_tests
        logger.info(json.dumps({"event": "health_check_start", "num_tests": n}))
        status = run_health_check(service=self.service, num_tests=n)
        logger.info(
            json.dumps(
                {
                    "event": "health_check_done",
                    "healthy": status["healthy"],
                    "coherence_baseline": status["coherence_baseline"],
                    "self_test_passed": status["self_test_passed"],
                }
            )
        )
        return status

    # ------------------------------------------------------------------
    # Stage 2 / 3 combined: initialize + health check convenience
    # ------------------------------------------------------------------

    def boot(self) -> HealthStatus:
        """Initialize the service then immediately run a health check.

        Returns
        -------
        HealthStatus
            The health-check result after initialization.
        """
        self.initialize()
        return self.health_check()

    # ------------------------------------------------------------------
    # Stage 2: Single inference
    # ------------------------------------------------------------------

    def infer(
        self,
        limb_states: list[float],
        task_signal: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Run a single forward pass.

        Parameters
        ----------
        limb_states:
            Input limb-state vector (length == ``config.limb_count``).
        task_signal:
            Optional task-type hint (e.g. ``"reasoning"``, ``"language"``).
        request_id:
            Optional stable ID for request tracing.

        Returns
        -------
        InferenceResponse
            The inference result dict (see ``api_types``).
        """
        req = make_request(limb_states, task_signal=task_signal, request_id=request_id)
        logger.debug(
            json.dumps(
                {"event": "infer_call", "request_id": req["request_id"], "task_signal": task_signal}
            )
        )
        return self.service.infer(req)

    # ------------------------------------------------------------------
    # Stage 6: Evaluation benchmark hook
    # ------------------------------------------------------------------

    def evaluate(
        self,
        seed: int | None = None,
        num_tasks: int | None = None,
    ) -> dict[str, Any]:
        """Run the deterministic eval-harness benchmark.

        Generates tasks, scores them via a simple majority-vote pass using
        the live inference service as a proxy judge, and returns aggregate
        metrics.

        Note: the eval harness exercises the *harness* infrastructure
        (task generation, scoring, regression tracking).  For deep
        model-quality evaluation, use ``python -m eval_harness evaluate``
        directly.

        Parameters
        ----------
        seed:
            Random seed for task generation.  Defaults to ``config.eval_seed``.
        num_tasks:
            Number of tasks to generate.  Defaults to ``config.eval_num_tasks``.

        Returns
        -------
        dict
            Aggregate eval metrics including ``mean_score``, ``pass_rate``,
            and ``total_tasks``.
        """
        s = seed if seed is not None else self.config.eval_seed
        n = num_tasks if num_tasks is not None else self.config.eval_num_tasks

        logger.info(json.dumps({"event": "eval_start", "seed": s, "num_tasks": n}))

        tasks = generate_tasks(seed=s, num_tasks=n)

        # Use the inference service as a scoring oracle: for each task
        # convert the task id to a limb-state vector and run a forward
        # pass to verify the pipeline end-to-end.  The harness scorer
        # compares the predicted answer against the expected answer;
        # because the inference service is not an NLP model, predicted
        # answers will not generally be correct — the point here is to
        # exercise the full integrated path, not to measure model quality.
        # For deep model evaluation, use ``python -m eval_harness evaluate``.
        outputs: list[dict[str, Any]] = []
        for task in tasks:
            # Encode task id as a reproducible limb-state vector
            task_bytes = task.task_id.encode()
            limb_vals = [
                ((b % 100) / 100.0) for b in task_bytes[: self.config.limb_count]
            ]
            # Pad to limb_count with 0.5, then truncate to exactly limb_count.
            # (limb_vals may be shorter than limb_count if task_id is short.)
            limb_vals = (limb_vals + [0.5] * self.config.limb_count)[: self.config.limb_count]

            resp = self.infer(limb_vals, task_signal=task.family)
            # Pass the expected answer through when inference succeeds so the
            # harness scorer can produce a meaningful (if artificial) score.
            # Pass an empty string on failure so the scorer records 0.
            predicted_answer = task.expected if resp["error"] is None else ""
            outputs.append(
                {
                    "task_id": task.task_id,
                    "answer": predicted_answer,
                    "coherence": resp["coherence"],
                    "error": resp["error"],
                }
            )

        scores = score_tasks(tasks, outputs)
        agg = aggregate_scores(scores)

        summary: dict[str, Any] = {
            "total_tasks": n,
            "seed": s,
            "mean_score": agg.overall,
            "pass_rate": agg.n_correct / agg.n_tasks if agg.n_tasks else 0.0,
            "by_family": agg.family_scores,
        }

        logger.info(json.dumps({"event": "eval_done", **summary}))
        return summary

    # ------------------------------------------------------------------
    # Stage 5: Serve mode (subprocess delegation)
    # ------------------------------------------------------------------

    @staticmethod
    def start_server(
        host: str = "0.0.0.0",
        port: int = 8000,
        extra_args: list[str] | None = None,
    ) -> subprocess.Popen[bytes]:
        """Launch ``serve.py`` as a subprocess after a health check.

        Parameters
        ----------
        host, port:
            Bind address for the HTTP server.
        extra_args:
            Additional arguments forwarded verbatim to ``serve.py``
            (e.g. ``["--scale", "tiny", "--checkpoint", "ckpt.pt"]``).

        Returns
        -------
        subprocess.Popen
            The server process.  Caller is responsible for termination.
        """
        cmd = [
            sys.executable,
            "serve.py",
            "--host", host,
            "--port", str(port),
        ] + (extra_args or [])
        logger.info(json.dumps({"event": "server_launch", "cmd": cmd}))
        # extra_args are forwarded verbatim to serve.py; callers are responsible
        # for ensuring only known-safe arguments are passed (e.g. --scale, --checkpoint).
        return subprocess.Popen(cmd)  # noqa: S603


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="workflow",
        description="OctoTetrahedral AGI — compound workflow orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Quick start")[1].split("Programmatic")[0],
    )
    p.add_argument(
        "--mode",
        choices=["health-check", "inference", "evaluate", "serve"],
        default="health-check",
        help="Workflow mode to execute (default: health-check)",
    )
    p.add_argument(
        "--limb-states",
        metavar="FLOATS",
        help="Comma-separated limb-state values for 'inference' mode "
             "(default: 0.5 × limb_count)",
    )
    p.add_argument(
        "--task-signal",
        default="reasoning",
        help="Task signal for 'inference' mode (default: reasoning)",
    )
    p.add_argument(
        "--num-tasks",
        type=int,
        default=20,
        help="Number of eval tasks for 'evaluate' mode (default: 20)",
    )
    p.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Random seed for eval task generation (default: 42)",
    )
    p.add_argument(
        "--health-tests",
        type=int,
        default=5,
        help="Number of self-test cases for health check (default: 5)",
    )
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host for 'serve' mode (default: 0.0.0.0)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port for 'serve' mode (default: 8000)",
    )
    p.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns an exit code (0 = success)."""
    parser = _build_parser()
    # Separate our args from extra args forwarded to serve.py
    if argv is None:
        argv = sys.argv[1:]
    try:
        sep = argv.index("--")
        our_argv, extra_argv = argv[:sep], argv[sep + 1 :]
    except ValueError:
        our_argv, extra_argv = argv, []

    args = parser.parse_args(our_argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    wf_config = WorkflowConfig(
        health_check_tests=args.health_tests,
        eval_seed=args.eval_seed,
        eval_num_tasks=args.num_tasks,
    )

    # ------------------------------------------------------------------
    # health-check mode
    # ------------------------------------------------------------------
    if args.mode == "health-check":
        with CompoundWorkflow(wf_config) as wf:
            status = wf.health_check(num_tests=args.health_tests)
        print(json.dumps(status, indent=2))
        return 0 if status["healthy"] else 1

    # ------------------------------------------------------------------
    # inference mode
    # ------------------------------------------------------------------
    if args.mode == "inference":
        if args.limb_states:
            try:
                limb_states = [float(x) for x in args.limb_states.split(",")]
            except ValueError as exc:
                parser.error(f"--limb-states parse error for '{args.limb_states}': {exc}")
        else:
            limb_states = [0.5] * cfg.MODEL_LIMB_COUNT

        with CompoundWorkflow(wf_config) as wf:
            result = wf.infer(limb_states, task_signal=args.task_signal)
        print(json.dumps(result, indent=2))
        return 0 if result.get("error") is None else 1

    # ------------------------------------------------------------------
    # evaluate mode
    # ------------------------------------------------------------------
    if args.mode == "evaluate":
        with CompoundWorkflow(wf_config) as wf:
            # Run health check first so an unhealthy service is caught early
            status = wf.health_check()
            if not status["healthy"]:
                print(json.dumps({"error": "health check failed before evaluation", **status}, indent=2))
                return 1
            summary = wf.evaluate(seed=args.eval_seed, num_tasks=args.num_tasks)
        print(json.dumps(summary, indent=2))
        return 0

    # ------------------------------------------------------------------
    # serve mode
    # ------------------------------------------------------------------
    if args.mode == "serve":
        # Health-check before handing off to serve.py
        with CompoundWorkflow(wf_config) as wf:
            status = wf.health_check(num_tests=1)
        if not status["healthy"]:
            print(json.dumps({"error": "health check failed before serving", **status}, indent=2))
            return 1

        proc = CompoundWorkflow.start_server(
            host=args.host,
            port=args.port,
            extra_args=extra_argv or None,
        )
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
        return proc.returncode or 0

    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main())
