"""Extract a reusable concept library from the 562 verified solvers.

Strategy (DreamCoder-style): instead of hand-writing a DSL, we mine the
verified solver corpus for small, self-contained helper functions
(component detection, orientations, flood fill, etc.). Each entry keeps
its source task id + the helper source so the synthesizer can import
them directly or imitate their patterns.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field

from .config import HarnessConfig


@dataclass
class LibraryEntry:
    task_id: str
    name: str
    source: str
    datasets: list[str] = field(default_factory=list)
    import_line: str = ""


# Functions/classes we consider reusable generic primitives vs task-specific glue.
_GENERIC_NAMES = {
    "_components", "components", "connected_components", "flood_fill",
    "_find_legend", "_orientations", "orientations", "rotations", "reflections",
    "_normalise", "normalize", "get_components", "find_objects", "objects",
    "bbox", "bounding_box", "transpose", "rotate", "reflect", "pad", "crop",
    "find_color", "count_colors", "color_counts", "neighbors", "neighbours",
    "canonical", "canonicalize", "to_tuple", "from_tuple",
}
_GENERIC_PREFIXES = ("_", "get_", "find_", "extract_", "compute_", "count_", "build_", "apply_")


def _helper_names(node: ast.FunctionDef) -> bool:
    """Heuristic: functions with generic-ish names are library candidates."""
    name = node.name
    if name == "solve":
        return False
    if name.startswith("__"):
        return False
    if name in _GENERIC_NAMES:
        return True
    if name.startswith(_GENERIC_PREFIXES) and len(node.body) < 80:
        return True
    return False


def _strip_docstring(src: str) -> str:
    return re.sub(r'^\s*""".*?"""\s*', "", src, count=1, flags=re.S).strip()


def extract_from_solver(solver_path: str, task_id: str) -> list[LibraryEntry]:
    """Pull generic helper functions out of one solver.py file."""
    with open(solver_path) as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    entries = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and _helper_names(node):
            seg = ast.get_source_segment(src, node) or ""
            seg = _strip_docstring(seg)
            if not seg:
                continue
            imports = "\n".join(l for l in src.splitlines() if re.match(r"^(import|from)\s", l))[:400]
            entries.append(
                LibraryEntry(
                    task_id=task_id,
                    name=node.name,
                    source=seg,
                    import_line=imports,
                )
            )
    return entries


def build_library(cfg: HarnessConfig) -> list[LibraryEntry]:
    """Build the library from all verified solvers."""
    solvers_root = os.path.join(cfg.solvers_root, "solves")
    catalog = json.load(open(os.path.join(cfg.solvers_root, "catalog.json")))
    datasets_by_id = {x["id"]: x.get("datasets", []) for x in catalog}
    entries: list[LibraryEntry] = []
    for task_id in sorted(os.listdir(solvers_root)):
        d = os.path.join(solvers_root, task_id)
        sp = os.path.join(d, "solver.py")
        if not os.path.isfile(sp):
            continue
        for e in extract_from_solver(sp, task_id):
            e.datasets = datasets_by_id.get(task_id, [])
            entries.append(e)
    return entries


def write_library(cfg: HarnessConfig, entries: list[LibraryEntry]) -> str:
    """Write library as a single importable module + index json."""
    os.makedirs(cfg.library_root, exist_ok=True)
    lines = ['"""Auto-extracted concept library from verified ARC solvers."""', ""]
    lines.append("from collections import deque, Counter, defaultdict")
    lines.append("import itertools")
    lines.append("import numpy as np")
    lines.append("import copy")
    lines.append("from typing import List, Tuple, Set, Dict, Optional")
    lines.append("")
    index = []
    for e in entries:
        idx = len(index)
        header = f"\n# === {e.name} (source task {e.task_id}) ==="
        lines.append(header)
        lines.append(e.source.rstrip())
        lines.append("")
        index.append(
            {
                "idx": idx,
                "name": e.name,
                "task_id": e.task_id,
                "datasets": e.datasets,
            }
        )
    lib_path = os.path.join(cfg.library_root, "concepts.py")
    idx_path = os.path.join(cfg.library_root, "index.json")
    with open(lib_path, "w") as f:
        f.write("\n".join(lines))
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=1)
    return lib_path
