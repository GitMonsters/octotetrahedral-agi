"""
Cognition Tracer — Lightweight function-call tracing for ARC/RE-ARC solvers.

Captures per-run execution traces (function calls, durations, call counts)
that feed into the multilayer cognition graph pipeline.

Schema follows the 2.1/2.2 spec ("Per-run log entry") with full support for:
  - checkpoint_id / split (train/dev/heldout_family)
  - run_metadata (timestamp, seed, runner_version)
  - structured behavior block with robustness probes
  - intermediate_stats (segmented objects, rules, grid dimensions)
  - rich trace events with event_id, args_summary, return_summary
  - per-function aggregated trace stats
  - optional re_arc block (task_id, example_index, difficulty, generator_params)

Usage:
    tracer = CognitionTracer()
    with tracer.trace(solver_id="009d5c81", puzzle_id="009d5c81",
                      family="color_map", checkpoint_id="v1", split="dev"):
        result = solve(grid)
    tracer.set_success(True)
    record = tracer.last_record
"""

from __future__ import annotations

import ast
import datetime
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

RUNNER_VERSION = "0.3.0"


# ---------------------------------------------------------------------------
# Sub-record dataclasses (matches 2.1/2.2 schema)
# ---------------------------------------------------------------------------

@dataclass
class RunMetadata:
    """Runner-level provenance (schema §run_metadata)."""
    timestamp: str = ""            # ISO 8601 UTC
    seed: int = 0
    runner_version: str = RUNNER_VERSION

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.utcnow().isoformat() + "Z"


@dataclass
class RobustnessVariant:
    """Result of one robustness probe (schema §behavior.robustness.variants)."""
    variant_type: str              # "color_remap" | "grid_resize" | "noise" | …
    success: bool = False
    runtime_ms: float = 0.0


@dataclass
class RunBehavior:
    """Behavioral outcome block (schema §behavior)."""
    success: bool = False
    num_examples_used: int = 1
    total_runtime_ms: float = 0.0
    num_candidates_evaluated: int = 0
    num_search_steps: int = 0
    robustness_tested: bool = False
    robustness_variants: list[RobustnessVariant] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "num_examples_used": self.num_examples_used,
            "total_runtime_ms": self.total_runtime_ms,
            "num_candidates_evaluated": self.num_candidates_evaluated,
            "num_search_steps": self.num_search_steps,
            "robustness": {
                "tested": self.robustness_tested,
                "variants": [
                    {"variant_type": v.variant_type,
                     "success": v.success,
                     "runtime_ms": v.runtime_ms}
                    for v in self.robustness_variants
                ],
            },
        }

    @staticmethod
    def from_dict(d: dict) -> "RunBehavior":
        rob = d.get("robustness", {})
        variants = [
            RobustnessVariant(
                variant_type=v["variant_type"],
                success=v.get("success", False),
                runtime_ms=v.get("runtime_ms", 0.0),
            )
            for v in rob.get("variants", [])
        ]
        return RunBehavior(
            success=d.get("success", False),
            num_examples_used=d.get("num_examples_used", 1),
            total_runtime_ms=d.get("total_runtime_ms", 0.0),
            num_candidates_evaluated=d.get("num_candidates_evaluated", 0),
            num_search_steps=d.get("num_search_steps", 0),
            robustness_tested=rob.get("tested", False),
            robustness_variants=variants,
        )


@dataclass
class IntermediateStats:
    """Key intermediate statistics gathered during solving (schema §intermediate_stats)."""
    num_segmented_objects: int = 0
    num_rules_hypothesized: int = 0
    num_rules_pruned: int = 0
    avg_grid_width: float = 0.0
    avg_grid_height: float = 0.0

    def to_dict(self) -> dict:
        return {
            "num_segmented_objects": self.num_segmented_objects,
            "num_rules_hypothesized": self.num_rules_hypothesized,
            "num_rules_pruned": self.num_rules_pruned,
            "avg_grid_width": self.avg_grid_width,
            "avg_grid_height": self.avg_grid_height,
        }

    @staticmethod
    def from_dict(d: dict) -> "IntermediateStats":
        return IntermediateStats(
            num_segmented_objects=d.get("num_segmented_objects", 0),
            num_rules_hypothesized=d.get("num_rules_hypothesized", 0),
            num_rules_pruned=d.get("num_rules_pruned", 0),
            avg_grid_width=d.get("avg_grid_width", 0.0),
            avg_grid_height=d.get("avg_grid_height", 0.0),
        )


@dataclass
class ReArcInfo:
    """RE-ARC–specific provenance (schema §re_arc, optional)."""
    task_id: str = ""
    example_index: int = 0
    difficulty: dict[str, float] = field(default_factory=dict)
    task_metadata: dict[str, Any] = field(default_factory=dict)
    generator_params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trace event (schema §trace.events)
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    """A single function call captured during solver execution."""
    event_id: str           # e.g. "e1", "e2" — unique within a run
    func_name: str          # Python function name (function_id in schema)
    module_name: str        # semantic module label (perception, search, …)
    file_path: str          # absolute source path (empty for builtins)
    start_ms: float         # ms since run start
    end_ms: float = 0.0     # ms since run start at return
    depth: int = 0          # call stack depth at time of call
    call_count: int = 1     # accumulated when same function called again
    args_summary: dict[str, Any] = field(default_factory=dict)
    return_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return max(self.end_ms - self.start_ms, 0.0)

    # Back-compat alias used by cognition_graph.py
    @property
    def duration_us(self) -> float:
        return self.duration_ms * 1_000


# ---------------------------------------------------------------------------
# RunRecord — top-level log entry (matches 2.1/2.2 schema)
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """Complete trace record for a single (solver, checkpoint, puzzle) run."""
    # Identity
    solver_id: str
    solver_family: str          # e.g. "specialized", "pipeline", "baseline"
    checkpoint_id: str          # e.g. "v1", "step_200k", or ""
    puzzle_id: str
    puzzle_family: str          # e.g. "tiling", "symmetry", "color_map"
    split: str                  # "train" | "dev" | "heldout_family"

    # Sub-records
    run_metadata: RunMetadata = field(default_factory=RunMetadata)
    behavior: RunBehavior = field(default_factory=RunBehavior)
    intermediate_stats: IntermediateStats = field(default_factory=IntermediateStats)
    re_arc: Optional[ReArcInfo] = None
    events: list[TraceEvent] = field(default_factory=list)

    # Derived (populated by finalize())
    unique_functions: int = 0
    total_calls: int = 0
    hot_functions: list[str] = field(default_factory=list)

    # Legacy shims so callers that read .success / .total_runtime_ms keep working
    @property
    def success(self) -> bool:
        return self.behavior.success

    @success.setter
    def success(self, v: bool) -> None:
        self.behavior.success = v

    @property
    def total_runtime_ms(self) -> float:
        return self.behavior.total_runtime_ms

    def finalize(self) -> None:
        self.unique_functions = len({e.func_name for e in self.events})
        self.total_calls = sum(e.call_count for e in self.events)
        by_count = sorted(self.events, key=lambda e: e.call_count, reverse=True)
        self.hot_functions = [e.func_name for e in by_count[:5]]

    # ------------------------------------------------------------------
    # Serialization (schema §trace)
    # ------------------------------------------------------------------

    def to_jsonl_line(self) -> str:
        """Serialize to a single JSONL line following the 2.1/2.2 schema."""
        trace_events = [
            {
                "event_id": e.event_id,
                "type": "function_call",
                "function_id": e.func_name,
                "module": e.module_name,
                "start_time_ms": e.start_ms,
                "end_time_ms": e.end_ms,
                "args_summary": e.args_summary,
                "return_summary": e.return_summary,
            }
            for e in self.events
        ]
        per_function = []
        fn_agg: dict[str, dict] = {}
        for e in self.events:
            if e.func_name not in fn_agg:
                fn_agg[e.func_name] = {"function_id": e.func_name,
                                        "num_calls": 0, "total_time_ms": 0.0}
            fn_agg[e.func_name]["num_calls"] += e.call_count
            fn_agg[e.func_name]["total_time_ms"] += e.duration_ms * e.call_count
        per_function = list(fn_agg.values())

        doc: dict[str, Any] = {
            "solver_id": self.solver_id,
            "solver_family": self.solver_family,
            "checkpoint_id": self.checkpoint_id,
            "puzzle_id": self.puzzle_id,
            "puzzle_family": self.puzzle_family,
            "split": self.split,
            "run_metadata": asdict(self.run_metadata),
            "behavior": self.behavior.to_dict(),
            "intermediate_stats": self.intermediate_stats.to_dict(),
            "trace": {
                "events": trace_events,
                "aggregated": {"per_function": per_function},
            },
            # derived
            "unique_functions": self.unique_functions,
            "total_calls": self.total_calls,
            "hot_functions": self.hot_functions,
        }
        if self.re_arc is not None:
            doc["re_arc"] = asdict(self.re_arc)
        return json.dumps(doc)

    @staticmethod
    def from_jsonl_line(line: str) -> "RunRecord":
        """Deserialize from a JSONL line."""
        d = json.loads(line)
        trace_raw = d.pop("trace", {})
        events = [
            TraceEvent(
                event_id=ev.get("event_id", f"e{i}"),
                func_name=ev.get("function_id", ev.get("fn", "")),
                module_name=ev.get("module", ev.get("mod", "")),
                file_path="",
                start_ms=ev.get("start_time_ms", 0.0),
                end_ms=ev.get("end_time_ms", 0.0),
                depth=ev.get("depth", 0),
                call_count=ev.get("call_count", ev.get("calls", 1)),
                args_summary=ev.get("args_summary", {}),
                return_summary=ev.get("return_summary", {}),
            )
            for i, ev in enumerate(trace_raw.get("events", []))
        ]

        re_arc_raw = d.pop("re_arc", None)
        re_arc = ReArcInfo(**re_arc_raw) if re_arc_raw else None

        run_meta_raw = d.pop("run_metadata", {})
        behavior_raw = d.pop("behavior", {})
        istats_raw = d.pop("intermediate_stats", {})

        # Strip derived fields that aren't constructor args
        for key in ("unique_functions", "total_calls", "hot_functions"):
            d.pop(key, None)

        rec = RunRecord(
            solver_id=d.get("solver_id", ""),
            solver_family=d.get("solver_family", ""),
            checkpoint_id=d.get("checkpoint_id", ""),
            puzzle_id=d.get("puzzle_id", ""),
            puzzle_family=d.get("puzzle_family", ""),
            split=d.get("split", "dev"),
            run_metadata=RunMetadata(**run_meta_raw) if run_meta_raw else RunMetadata(),
            behavior=RunBehavior.from_dict(behavior_raw),
            intermediate_stats=IntermediateStats.from_dict(istats_raw),
            re_arc=re_arc,
            events=events,
        )
        rec.finalize()
        return rec


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class CognitionTracer:
    """
    Wraps solver execution with sys.settrace to capture all function calls
    within the target module (solver file), recording timing and call counts.

    Only functions defined in files matching `target_file_prefix` are traced
    to avoid enormous traces from Python internals.
    """

    def __init__(self, target_file_prefix: Optional[str] = None):
        """
        Args:
            target_file_prefix: Only trace functions in files whose path starts
                with this prefix.  Defaults to arc-puzzle-catalog root.
        """
        if target_file_prefix is None:
            # Default: only trace code inside the arc-puzzle-catalog tree
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            target_file_prefix = here
        self.target_file_prefix = target_file_prefix
        self.last_record: Optional[RunRecord] = None

    @contextmanager
    def trace(
        self,
        solver_id: str,
        puzzle_id: str,
        puzzle_family: str,
        solver_family: str = "specialized",
        checkpoint_id: str = "",
        split: str = "dev",
        seed: int = 0,
    ):
        """Context manager that traces all function calls during the `with` block.

        Args:
            solver_id:      Solver identifier (e.g. task ID for specialized solvers).
            puzzle_id:      Puzzle identifier.
            puzzle_family:  High-level family label (e.g. "tiling").
            solver_family:  Solver family ("specialized", "pipeline", "baseline").
            checkpoint_id:  Optional checkpoint label (e.g. "step_200k").
            split:          Data split ("train" | "dev" | "heldout_family").
            seed:           RNG seed used for the run.

        Example:
            with tracer.trace("009d5c81", "009d5c81", "color_map", split="dev"):
                result = solve(grid)
            tracer.set_success(True)
        """
        events: list[TraceEvent] = []
        call_stack: list[TraceEvent] = []
        func_registry: dict[str, TraceEvent] = {}   # func_key → latest event
        event_counter: list[int] = [0]

        run_start = time.perf_counter()

        def _elapsed_ms() -> float:
            return (time.perf_counter() - run_start) * 1_000

        def _local_trace(frame, event, arg):
            """Per-function local tracer (handles 'return')."""
            if event == "return":
                fn = frame.f_code.co_name
                fp = frame.f_code.co_filename or ""
                key = f"{fp}:{fn}"
                if key in func_registry:
                    func_registry[key].end_ms = _elapsed_ms()
                if call_stack and call_stack[-1].func_name == fn:
                    call_stack.pop()
            return _local_trace

        def _global_trace(frame, event, arg):
            """Global tracer — fires on every 'call' event."""
            if event != "call":
                return None
            fp = frame.f_code.co_filename or ""
            # Skip builtins and external libraries
            if fp and not fp.startswith(self.target_file_prefix):
                return None
            fn = frame.f_code.co_name
            # Skip uninteresting dunder names
            if fn.startswith("__") and fn not in ("__init__",):
                return None
            mod = _file_to_module(fp)
            key = f"{fp}:{fn}"
            if key in func_registry:
                func_registry[key].call_count += 1
                return _local_trace
            event_counter[0] += 1
            ev = TraceEvent(
                event_id=f"e{event_counter[0]}",
                func_name=fn,
                module_name=mod,
                file_path=fp,
                start_ms=_elapsed_ms(),
                depth=len(call_stack),
            )
            func_registry[key] = ev
            events.append(ev)
            call_stack.append(ev)
            return _local_trace

        old_trace = sys.gettrace()
        sys.settrace(_global_trace)
        try:
            yield
        finally:
            sys.settrace(old_trace)
            elapsed_ms = _elapsed_ms()

        record = RunRecord(
            solver_id=solver_id,
            solver_family=solver_family,
            checkpoint_id=checkpoint_id,
            puzzle_id=puzzle_id,
            puzzle_family=puzzle_family,
            split=split,
            run_metadata=RunMetadata(seed=seed),
            behavior=RunBehavior(total_runtime_ms=elapsed_ms),
            events=events,
        )
        record.finalize()
        self.last_record = record

    def set_success(self, success: bool) -> None:
        """Call immediately after the trace context to mark success."""
        if self.last_record is not None:
            self.last_record.behavior.success = success

    def set_intermediate_stats(self, **kwargs: Any) -> None:
        """Populate intermediate_stats on the last record.

        Valid kwargs: num_segmented_objects, num_rules_hypothesized,
                      num_rules_pruned, avg_grid_width, avg_grid_height
        """
        if self.last_record is None:
            return
        ist = self.last_record.intermediate_stats
        for k, v in kwargs.items():
            if hasattr(ist, k):
                setattr(ist, k, v)

    def add_robustness_probe(
        self, variant_type: str, success: bool, runtime_ms: float = 0.0
    ) -> None:
        """Record one robustness probe result on the last record."""
        if self.last_record is None:
            return
        self.last_record.behavior.robustness_tested = True
        self.last_record.behavior.robustness_variants.append(
            RobustnessVariant(variant_type=variant_type, success=success, runtime_ms=runtime_ms)
        )

    def set_re_arc(self, task_id: str, example_index: int, **kwargs: Any) -> None:
        """Attach RE-ARC provenance to the last record."""
        if self.last_record is None:
            return
        self.last_record.re_arc = ReArcInfo(
            task_id=task_id,
            example_index=example_index,
            difficulty=kwargs.get("difficulty", {}),
            task_metadata=kwargs.get("task_metadata", {}),
            generator_params=kwargs.get("generator_params", {}),
        )


# ---------------------------------------------------------------------------
# JSONL log utilities
# ---------------------------------------------------------------------------

class TraceLog:
    """Appends RunRecords to a JSONL file; supports iteration."""

    def __init__(self, path: str):
        self.path = path

    def append(self, record: RunRecord) -> None:
        with open(self.path, "a") as fh:
            fh.write(record.to_jsonl_line() + "\n")

    def __iter__(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield RunRecord.from_jsonl_line(line)

    def all_records(self) -> list[RunRecord]:
        return list(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_to_module(fp: str) -> str:
    """Convert an absolute file path to a short module label."""
    if not fp:
        return "<builtin>"
    # e.g. .../solves/009d5c81/solver.py  → "009d5c81.solver"
    parts = fp.replace("\\", "/").split("/")
    # Drop .py extension from last part
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    # Keep last 2 meaningful segments
    return ".".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def extract_static_functions(source: str) -> list[dict]:
    """
    Parse Python source and return a list of function info dicts:
        {"name": str, "lineno": int, "calls": [str], "docstring": str}
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    class FuncVisitor(ast.NodeVisitor):
        def __init__(self):
            self.functions: list[dict] = []
            self._current: list[str] = []

        def visit_FunctionDef(self, node):
            doc = ast.get_docstring(node) or ""
            calls: list[str] = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        calls.append(child.func.id)
                    elif isinstance(child.func, ast.Attribute):
                        calls.append(child.func.attr)
            self.functions.append({
                "name": node.name,
                "lineno": node.lineno,
                "calls": calls,
                "docstring": doc[:120],
            })
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    v = FuncVisitor()
    v.visit(tree)
    return v.functions
