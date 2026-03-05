"""
TTCCVTLR — Test-Time Critical Consequential Visualization Thinking Learning & Reasoning

A multi-phase cognitive loop that runs per-task at inference time. Unlike
plain TTT (which only learns weights), TTCCVTLR integrates six cognitive
capabilities into an iterative refinement loop:

  ┌──────────────────────────────────────────────────────┐
  │                   TTCCVTLR Loop                      │
  │                                                      │
  │  ┌───────────┐   ┌──────────┐   ┌─────────────┐    │
  │  │ VISUALIZE │──▶│  THINK   │──▶│   LEARN     │    │
  │  │ (V-phase) │   │ (T-phase)│   │  (L-phase)  │    │
  │  └───────────┘   └──────────┘   └──────┬──────┘    │
  │       ▲                                 │           │
  │       │                                 ▼           │
  │  ┌────┴──────┐   ┌──────────┐   ┌─────────────┐    │
  │  │ CRITIQUE  │◀──│ CONSEQUEN│◀──│   REASON    │    │
  │  │ (CC-phase)│   │ (C-phase)│   │  (R-phase)  │    │
  │  └───────────┘   └──────────┘   └─────────────┘    │
  │                                                      │
  │  Loop until: confidence > threshold OR budget spent  │
  └──────────────────────────────────────────────────────┘

Each phase:
  V — Visualization: Extract structured visual features from input→output
      diffs (change masks, color flows, object movements, symmetry axes)
  T — Thinking: Generate ranked hypotheses from visual features using
      multiple strategy generators (geometric, chromatic, topological)
  L — Learning: Per-task weight adaptation via gradient descent on
      augmented training examples (TTT-style)
  R — Reasoning: Deductive elimination — test hypotheses against ALL
      training pairs, prune any that fail even once
  C — Consequential: Evaluate downstream consequences — does the
      prediction make physical sense? Is it self-consistent?
  CC — Critical: Meta-evaluate — compare competing predictions, pick
       the one with highest combined score, or loop again if none pass

Usage:
    engine = TTCCVTLREngine()
    result = engine.solve(task)
    if result['solved']:
        print(result['prediction'])
        print(result['reasoning_trace'])  # Full cognitive trace
"""

import time
import copy
import math
import json
import hashlib
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import Counter, defaultdict
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from core.primitives import CompositionEngine, Grid, BASE_PRIMITIVES
from core.few_shot import (
    FewShotAbstractor, find_objects, background_color,
    grid_dims, grid_equal, extract_subgrid,
)
from core.hypothesis import Hypothesis, HypothesisEngine


# ============================================================================
# Visual Feature Extraction (V-phase)
# ============================================================================

class VisualFeatures:
    """Structured representation of visual transformation features."""

    def __init__(self):
        self.change_mask: Optional[np.ndarray] = None
        self.change_ratio: float = 0.0
        self.color_flow: Dict[int, Counter] = {}   # src_color → Counter(dst_colors)
        self.same_size: bool = False
        self.size_ratio: Tuple[float, float] = (1.0, 1.0)
        self.new_colors: Set[int] = set()
        self.removed_colors: Set[int] = set()
        self.preserved_cells_ratio: float = 0.0
        self.symmetry_axes: List[str] = []          # 'h', 'v', 'd1', 'd2'
        self.object_count_in: int = 0
        self.object_count_out: int = 0
        self.objects_moved: bool = False
        self.grid_tiled: bool = False
        self.border_changed: bool = False
        self.interior_changed: bool = False
        self.sparse_change: bool = False             # <30% cells changed

    def summary(self) -> str:
        parts = []
        if self.same_size:
            parts.append(f"same-size, {self.change_ratio:.0%} changed")
        else:
            parts.append(f"resize {self.size_ratio[0]:.1f}x{self.size_ratio[1]:.1f}")
        if self.new_colors:
            parts.append(f"new colors: {self.new_colors}")
        if self.sparse_change:
            parts.append("sparse")
        if self.symmetry_axes:
            parts.append(f"symmetry: {self.symmetry_axes}")
        return ", ".join(parts)


def extract_visual_features(inp: List[List[int]], out: List[List[int]]) -> VisualFeatures:
    """V-phase: Extract structured visual features from a single input→output pair."""
    vf = VisualFeatures()
    a = np.array(inp)
    b = np.array(out)
    h_in, w_in = a.shape
    h_out, w_out = b.shape

    vf.same_size = (h_in == h_out and w_in == w_out)
    vf.size_ratio = (h_out / max(h_in, 1), w_out / max(w_in, 1))

    # Color analysis
    in_colors = set(a.flatten())
    out_colors = set(b.flatten())
    vf.new_colors = out_colors - in_colors
    vf.removed_colors = in_colors - out_colors

    if vf.same_size:
        vf.change_mask = (a != b)
        vf.change_ratio = float(vf.change_mask.mean())
        vf.preserved_cells_ratio = 1.0 - vf.change_ratio
        vf.sparse_change = vf.change_ratio < 0.3

        # Color flow: for each changed cell, track src→dst
        changed_y, changed_x = np.where(vf.change_mask)
        for y, x in zip(changed_y, changed_x):
            src = int(a[y, x])
            dst = int(b[y, x])
            if src not in vf.color_flow:
                vf.color_flow[src] = Counter()
            vf.color_flow[src][dst] += 1

        # Border vs interior changes
        border = np.zeros_like(vf.change_mask)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        vf.border_changed = bool(np.any(vf.change_mask & border))
        interior = ~border
        vf.interior_changed = bool(np.any(vf.change_mask & interior))
    else:
        vf.change_ratio = 1.0  # Different size = everything changed

    # Symmetry detection on output
    if h_out > 1 and w_out > 1:
        if np.array_equal(b, b[::-1, :]):
            vf.symmetry_axes.append('v')
        if np.array_equal(b, b[:, ::-1]):
            vf.symmetry_axes.append('h')
        if h_out == w_out:
            if np.array_equal(b, b.T):
                vf.symmetry_axes.append('d1')

    # Object counting
    vf.object_count_in = _count_objects(a)
    vf.object_count_out = _count_objects(b)

    # Tiling detection: does output tile input?
    if not vf.same_size and h_out >= h_in and w_out >= w_in:
        if h_out % h_in == 0 and w_out % w_in == 0:
            tiled = True
            for ty in range(0, h_out, h_in):
                for tx in range(0, w_out, w_in):
                    patch = b[ty:ty+h_in, tx:tx+w_in]
                    if patch.shape == a.shape and not np.array_equal(patch, a):
                        tiled = False
            vf.grid_tiled = tiled

    return vf


def _count_objects(grid: np.ndarray) -> int:
    """Count non-background connected components."""
    bg = int(Counter(grid.flatten()).most_common(1)[0][0])
    visited = np.zeros_like(grid, dtype=bool)
    count = 0
    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            if grid[y, x] != bg and not visited[y, x]:
                # BFS flood fill
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if cy < 0 or cy >= h or cx < 0 or cx >= w:
                        continue
                    if visited[cy, cx] or grid[cy, cx] == bg:
                        continue
                    visited[cy, cx] = True
                    stack.extend([(cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)])
                count += 1
    return count


def aggregate_visual_features(task: Dict) -> Dict[str, Any]:
    """Aggregate visual features across all training examples."""
    features = []
    for ex in task['train']:
        vf = extract_visual_features(ex['input'], ex['output'])
        features.append(vf)

    agg = {
        'all_same_size': all(f.same_size for f in features),
        'avg_change_ratio': np.mean([f.change_ratio for f in features]),
        'all_sparse': all(f.sparse_change for f in features),
        'common_new_colors': set.intersection(*[f.new_colors for f in features]) if features else set(),
        'any_symmetry': any(f.symmetry_axes for f in features),
        'consistent_symmetry': _consistent_axes(features),
        'size_change_consistent': len(set(f.size_ratio for f in features)) == 1,
        'features': features,
    }

    # Detect consistent color flow across examples
    if agg['all_same_size']:
        flows = [f.color_flow for f in features]
        agg['consistent_color_flow'] = _merge_color_flows(flows)
    else:
        agg['consistent_color_flow'] = {}

    return agg


def _consistent_axes(features: List[VisualFeatures]) -> List[str]:
    """Find symmetry axes that appear in ALL examples."""
    if not features:
        return []
    common = set(features[0].symmetry_axes)
    for f in features[1:]:
        common &= set(f.symmetry_axes)
    return sorted(common)


def _merge_color_flows(flows: List[Dict]) -> Dict[int, int]:
    """Find consistent color mappings across examples."""
    if not flows:
        return {}
    combined: Dict[int, Counter] = {}
    for flow in flows:
        for src, dst_counter in flow.items():
            if src not in combined:
                combined[src] = Counter()
            combined[src] += dst_counter
    # Extract dominant mapping
    mapping = {}
    for src, dst_counter in combined.items():
        dominant = dst_counter.most_common(1)[0][0]
        mapping[src] = dominant
    return mapping


# ============================================================================
# Thinking Phase (T-phase) — Hypothesis Generation from Visual Features
# ============================================================================

class ThinkingEngine:
    """Generate ranked hypotheses from visual features."""

    def __init__(self):
        self.composer = CompositionEngine(max_depth=3)

    def think(
        self, task: Dict, visual_agg: Dict[str, Any], prior_failures: List[str] = None
    ) -> List[Hypothesis]:
        """Generate hypotheses informed by visual analysis.
        
        Strategies are prioritized by visual features:
        - Sparse same-size → color mapping / recoloring
        - Tiled → repetition / reflection
        - Object count changes → object operations
        - Symmetry detected → symmetry-based transforms
        """
        hypotheses = []
        prior_failures = set(prior_failures or [])

        # Strategy 1: Color flow mapping (if consistent across examples)
        if visual_agg.get('consistent_color_flow'):
            h = self._color_flow_hypothesis(task, visual_agg['consistent_color_flow'])
            if h and h.name not in prior_failures:
                hypotheses.append(h)

        # Strategy 2: Composition search (informed by feature constraints)
        depth = 2 if visual_agg.get('all_sparse') else 3
        comp_hyps = self._composition_hypotheses(task, depth, prior_failures)
        hypotheses.extend(comp_hyps)

        # Strategy 3: Symmetry-based (if output has consistent symmetry)
        if visual_agg.get('consistent_symmetry'):
            sym_hyps = self._symmetry_hypotheses(
                task, visual_agg['consistent_symmetry'], prior_failures
            )
            hypotheses.extend(sym_hyps)

        # Strategy 4: Object manipulation
        obj_hyps = self._object_hypotheses(task, visual_agg, prior_failures)
        hypotheses.extend(obj_hyps)

        # Strategy 5: Conditional recoloring (the most common unsolved pattern)
        if visual_agg.get('all_same_size') and visual_agg.get('all_sparse'):
            recolor_hyps = self._conditional_recolor_hypotheses(task, visual_agg, prior_failures)
            hypotheses.extend(recolor_hyps)

        return hypotheses

    def _color_flow_hypothesis(self, task: Dict, flow: Dict[int, int]) -> Optional[Hypothesis]:
        """Build a hypothesis from consistent color flow."""
        def apply_flow(grid: Grid, m=flow) -> Grid:
            return [[m.get(c, c) for c in row] for row in grid]
        return Hypothesis(
            name=f"color_flow({flow})",
            apply_fn=apply_flow,
            source='visual_color_flow',
            confidence=0.85,
        )

    def _composition_hypotheses(
        self, task: Dict, depth: int, prior_failures: Set[str]
    ) -> List[Hypothesis]:
        solutions = self.composer.search(task['train'], max_depth=depth)
        hyps = []
        for sol in solutions:
            prog = sol['program']
            name = ' → '.join(prog)
            if name in prior_failures:
                continue
            h = Hypothesis(
                name=name,
                apply_fn=lambda g, p=prog: self.composer.apply_program(p, g),
                source='composition',
                confidence=0.9,
            )
            hyps.append(h)
        return hyps

    def _symmetry_hypotheses(
        self, task: Dict, axes: List[str], prior_failures: Set[str]
    ) -> List[Hypothesis]:
        hyps = []
        for axis in axes:
            name = f"enforce_symmetry_{axis}"
            if name in prior_failures:
                continue

            if axis == 'v':
                def apply_fn(g: Grid) -> Grid:
                    a = np.array(g)
                    h = len(g) // 2
                    top = a[:h, :]
                    return np.vstack([top, top[::-1, :]]).tolist()
            elif axis == 'h':
                def apply_fn(g: Grid) -> Grid:
                    a = np.array(g)
                    w = len(g[0]) // 2
                    left = a[:, :w]
                    return np.hstack([left, left[:, ::-1]]).tolist()
            else:
                continue

            hyps.append(Hypothesis(name=name, apply_fn=apply_fn, source='symmetry', confidence=0.7))
        return hyps

    def _object_hypotheses(
        self, task: Dict, visual_agg: Dict, prior_failures: Set[str]
    ) -> List[Hypothesis]:
        """Generate object-based hypotheses (sort, filter, transform objects)."""
        hyps = []

        # Hypothesis: keep only largest object
        name = "keep_largest_object"
        if name not in prior_failures:
            def keep_largest(g: Grid) -> Grid:
                a = np.array(g)
                bg = int(Counter(a.flatten()).most_common(1)[0][0])
                objects = _find_objects_with_bounds(a, bg)
                if not objects:
                    return g
                largest = max(objects, key=lambda o: o['size'])
                out = np.full_like(a, bg)
                for y, x in largest['cells']:
                    out[y, x] = a[y, x]
                return out.tolist()
            hyps.append(Hypothesis(name=name, apply_fn=keep_largest, source='object', confidence=0.5))

        # Hypothesis: fill enclosed regions
        name = "fill_enclosed"
        if name not in prior_failures:
            def fill_enclosed(g: Grid) -> Grid:
                a = np.array(g)
                h, w = a.shape
                bg = int(Counter(a.flatten()).most_common(1)[0][0])
                # BFS from edges to find reachable bg cells
                reachable = np.zeros((h, w), dtype=bool)
                stack = []
                for y in range(h):
                    for x in range(w):
                        if (y == 0 or y == h-1 or x == 0 or x == w-1) and a[y, x] == bg:
                            stack.append((y, x))
                            reachable[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < h and 0 <= nx < w and not reachable[ny, nx] and a[ny, nx] == bg:
                            reachable[ny, nx] = True
                            stack.append((ny, nx))
                # Fill unreachable bg cells with most common non-bg color
                non_bg = [c for c in a.flatten() if c != bg]
                fill_color = Counter(non_bg).most_common(1)[0][0] if non_bg else bg
                out = a.copy()
                for y in range(h):
                    for x in range(w):
                        if a[y, x] == bg and not reachable[y, x]:
                            out[y, x] = fill_color
                return out.tolist()
            hyps.append(Hypothesis(name=name, apply_fn=fill_enclosed, source='object', confidence=0.5))

        return hyps

    def _conditional_recolor_hypotheses(
        self, task: Dict, visual_agg: Dict, prior_failures: Set[str]
    ) -> List[Hypothesis]:
        """Generate hypotheses based on neighbor-conditional recoloring."""
        hyps = []

        # Hypothesis: recolor cells based on neighbor count
        name = "neighbor_count_recolor"
        if name not in prior_failures:
            # Learn the rule from first example
            train = task['train']
            try:
                rule = _learn_neighbor_rule(train)
                if rule is not None:
                    def apply_rule(g: Grid, r=rule) -> Grid:
                        return _apply_neighbor_rule(g, r)
                    hyps.append(Hypothesis(
                        name=name, apply_fn=apply_rule,
                        source='conditional_recolor', confidence=0.6,
                    ))
            except Exception:
                pass

        return hyps


def _find_objects_with_bounds(grid: np.ndarray, bg: int) -> List[Dict]:
    """Find connected components with bounding boxes."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []
    for y in range(h):
        for x in range(w):
            if grid[y, x] != bg and not visited[y, x]:
                cells = []
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if cy < 0 or cy >= h or cx < 0 or cx >= w:
                        continue
                    if visited[cy, cx] or grid[cy, cx] == bg:
                        continue
                    visited[cy, cx] = True
                    cells.append((cy, cx))
                    stack.extend([(cy+1,cx),(cy-1,cx),(cy,cx+1),(cy,cx-1)])
                if cells:
                    ys = [c[0] for c in cells]
                    xs = [c[1] for c in cells]
                    objects.append({
                        'cells': cells,
                        'size': len(cells),
                        'bbox': (min(ys), min(xs), max(ys), max(xs)),
                        'color': int(grid[cells[0][0], cells[0][1]]),
                    })
    return objects


def _learn_neighbor_rule(train: List[Dict]) -> Optional[Dict]:
    """Learn a rule: cell color changes based on neighbor colors."""
    # For each changed cell, record (cell_color, neighbor_pattern) → new_color
    rules: Dict[Tuple, Counter] = {}
    for ex in train:
        a = np.array(ex['input'])
        b = np.array(ex['output'])
        h, w = a.shape
        if a.shape != b.shape:
            return None
        for y in range(h):
            for x in range(w):
                if a[y, x] != b[y, x]:
                    # Count neighbor colors
                    neighbors = []
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w:
                            neighbors.append(int(a[ny, nx]))
                    key = (int(a[y, x]), tuple(sorted(Counter(neighbors).items())))
                    if key not in rules:
                        rules[key] = Counter()
                    rules[key][int(b[y, x])] += 1

    if not rules:
        return None

    # Extract dominant mapping
    mapping = {}
    for key, counter in rules.items():
        mapping[key] = counter.most_common(1)[0][0]
    return mapping


def _apply_neighbor_rule(grid: Grid, rule: Dict) -> Grid:
    """Apply a learned neighbor-based recoloring rule."""
    a = np.array(grid)
    h, w = a.shape
    out = a.copy()
    for y in range(h):
        for x in range(w):
            neighbors = []
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w:
                    neighbors.append(int(a[ny, nx]))
            key = (int(a[y, x]), tuple(sorted(Counter(neighbors).items())))
            if key in rule:
                out[y, x] = rule[key]
    return out.tolist()


# ============================================================================
# Learning Phase (L-phase) — Per-Task Weight Adaptation
# ============================================================================

class PerTaskLearner:
    """TTT-style per-task neural learning with augmentation."""

    def __init__(self, device: str = None):
        if not HAS_TORCH:
            self.device = None
            return
        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def learn_and_predict(
        self,
        task: Dict,
        num_steps: int = 2000,
        num_aug: int = 200,
        verbose: bool = False,
    ) -> Optional[List[np.ndarray]]:
        """Train a fresh model on this task's examples and predict test outputs."""
        if not HAS_TORCH:
            return None

        try:
            from arc_ttt_v2 import solve_task
            return solve_task(task, device=self.device, verbose=verbose)
        except Exception:
            return None


# ============================================================================
# Reasoning Phase (R-phase) — Deductive Elimination
# ============================================================================

class ReasoningEngine:
    """Deductive reasoning: test hypotheses and eliminate failures."""

    @staticmethod
    def eliminate(
        hypotheses: List[Hypothesis], train: List[Dict]
    ) -> List[Hypothesis]:
        """Test all hypotheses against training data, keep only perfect ones."""
        survivors = []
        for h in hypotheses:
            score = h.test(train)
            if score == 1.0:
                survivors.append(h)
        return survivors

    @staticmethod
    def rank(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Rank by accuracy, then by Occam's razor (shorter name ≈ simpler)."""
        return sorted(
            hypotheses,
            key=lambda h: (h.accuracy, -len(h.name), h.confidence),
            reverse=True,
        )

    @staticmethod
    def near_misses(
        hypotheses: List[Hypothesis], train: List[Dict], threshold: float = 0.8
    ) -> List[Tuple[Hypothesis, float]]:
        """Find hypotheses that almost work (for refinement hints)."""
        near = []
        for h in hypotheses:
            score = h.test(train)
            if threshold <= score < 1.0:
                near.append((h, score))
        return sorted(near, key=lambda x: x[1], reverse=True)


# ============================================================================
# Consequential Analysis (C-phase) — Self-Consistency Check
# ============================================================================

class ConsequentialAnalyzer:
    """Evaluate predictions for self-consistency and physical plausibility."""

    @staticmethod
    def check_consistency(prediction: Grid, task: Dict) -> Dict[str, Any]:
        """Check if a prediction is self-consistent with the task's patterns."""
        train = task['train']
        pred = np.array(prediction)
        scores = {}

        # 1. Color palette consistency
        train_out_colors = set()
        for ex in train:
            for row in ex['output']:
                train_out_colors.update(row)
        pred_colors = set(pred.flatten())
        novel = pred_colors - train_out_colors
        scores['color_consistent'] = len(novel) == 0

        # 2. Size consistency
        out_sizes = set()
        for ex in train:
            out = np.array(ex['output'])
            out_sizes.add(out.shape)
        # Check if pred size matches a pattern
        train_in = np.array(task['test'][0]['input'])
        expected_sizes = set()
        for ex in train:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            h_ratio = out.shape[0] / max(inp.shape[0], 1)
            w_ratio = out.shape[1] / max(inp.shape[1], 1)
            eh = int(round(train_in.shape[0] * h_ratio))
            ew = int(round(train_in.shape[1] * w_ratio))
            expected_sizes.add((eh, ew))
        scores['size_consistent'] = pred.shape in expected_sizes or pred.shape in out_sizes

        # 3. Density consistency (similar ratio of non-bg cells)
        bg = int(Counter(train_in.flatten()).most_common(1)[0][0])
        train_densities = []
        for ex in train:
            out = np.array(ex['output'])
            train_densities.append(float(np.mean(out != bg)))
        pred_density = float(np.mean(pred != bg))
        avg_density = np.mean(train_densities) if train_densities else 0.5
        scores['density_consistent'] = abs(pred_density - avg_density) < 0.3

        # 4. Non-trivial (not just returning input)
        test_in = np.array(task['test'][0]['input'])
        if test_in.shape == pred.shape:
            scores['non_trivial'] = not np.array_equal(test_in, pred)
        else:
            scores['non_trivial'] = True

        # Combined score
        checks = [scores['color_consistent'], scores['size_consistent'],
                   scores['density_consistent'], scores['non_trivial']]
        scores['overall'] = sum(checks) / len(checks)

        return scores

    @staticmethod
    def compare_predictions(
        candidates: List[Tuple[Grid, str, float]], task: Dict
    ) -> List[Tuple[Grid, str, float, Dict]]:
        """Compare multiple candidate predictions, augmenting each with
        consistency scores."""
        analyzer = ConsequentialAnalyzer()
        results = []
        for pred, method, confidence in candidates:
            consistency = analyzer.check_consistency(pred, task)
            combined_score = confidence * 0.6 + consistency['overall'] * 0.4
            results.append((pred, method, combined_score, consistency))
        return sorted(results, key=lambda x: x[2], reverse=True)


# ============================================================================
# Critical Meta-Evaluation (CC-phase) — Decision & Loop Control
# ============================================================================

class CriticalEvaluator:
    """Meta-level decision: accept a prediction or loop again."""

    def __init__(self, confidence_threshold: float = 0.8):
        self.threshold = confidence_threshold

    def should_accept(
        self, candidates: List[Tuple[Grid, str, float, Dict]], round_num: int
    ) -> Tuple[bool, Optional[Grid], str]:
        """Decide whether to accept the best candidate or loop again.
        
        Returns (accept, prediction, reason).
        """
        if not candidates:
            return False, None, "no candidates"

        best = candidates[0]
        pred, method, score, consistency = best

        # Accept if high confidence
        if score >= self.threshold:
            return True, pred, f"accepted (score={score:.2f}, method={method})"

        # Accept if we've looped enough (diminishing returns)
        if round_num >= 3:
            return True, pred, f"budget exhausted after {round_num} rounds (score={score:.2f})"

        # Accept if consistency checks all pass even with lower score
        if consistency.get('overall', 0) >= 0.9 and score >= 0.5:
            return True, pred, f"consistent prediction (score={score:.2f})"

        return False, None, f"low confidence (score={score:.2f}), looping"


# ============================================================================
# TTCCVTLR Engine — The Full Loop
# ============================================================================

class TTCCVTLREngine:
    """
    Test-Time Critical Consequential Visualization Thinking Learning & Reasoning.
    
    Orchestrates the full cognitive loop:
    V → T → L → R → C → CC → (loop or accept)
    """

    def __init__(
        self,
        max_rounds: int = 3,
        confidence_threshold: float = 0.75,
        use_neural_learning: bool = True,
        device: str = None,
        timeout_seconds: float = 60.0,
    ):
        self.max_rounds = max_rounds
        self.timeout = timeout_seconds
        self.thinker = ThinkingEngine()
        self.learner = PerTaskLearner(device) if use_neural_learning else None
        self.reasoner = ReasoningEngine()
        self.analyzer = ConsequentialAnalyzer()
        self.critic = CriticalEvaluator(confidence_threshold)
        # Also use the existing HypothesisEngine for Phase 1
        self.hypothesis_engine = HypothesisEngine(
            max_composition_depth=3, timeout_seconds=15.0,
        )

    def solve(self, task: Dict, verbose: bool = False) -> Dict[str, Any]:
        """
        Run the full TTCCVTLR loop on a single task.
        
        Returns:
            dict with:
                solved: bool
                prediction: Grid or None
                predictions: list of (Grid, method, score) — top candidates
                reasoning_trace: list of dicts describing each round
                visual_summary: str
                time_ms: float
        """
        t0 = time.time()
        trace = []
        failed_hypotheses: List[str] = []
        best_candidates: List[Tuple[Grid, str, float, Dict]] = []
        test_input = task['test'][0]['input']

        # ──── V-phase: Visualization ────────────────────────────────────
        if verbose:
            print("═══ TTCCVTLR Loop ═══")
            print("▶ V-phase: Extracting visual features...")

        visual_agg = aggregate_visual_features(task)
        visual_summary = "; ".join(f.summary() for f in visual_agg['features'])

        trace.append({
            'phase': 'V',
            'visual_summary': visual_summary,
            'same_size': visual_agg['all_same_size'],
            'avg_change': visual_agg['avg_change_ratio'],
        })

        if verbose:
            print(f"  Visual: {visual_summary}")

        # ──── Main loop ─────────────────────────────────────────────────
        for round_num in range(self.max_rounds):
            elapsed = time.time() - t0
            if elapsed > self.timeout:
                trace.append({'phase': 'timeout', 'round': round_num})
                break

            round_trace = {'round': round_num, 'phases': {}}

            if verbose:
                print(f"\n── Round {round_num} ──")

            # ──── T-phase: Thinking ─────────────────────────────────────
            if verbose:
                print("▶ T-phase: Generating hypotheses...")

            hypotheses = self.thinker.think(task, visual_agg, failed_hypotheses)

            # Also include HypothesisEngine's validated hypotheses (round 0)
            if round_num == 0:
                try:
                    he_result = self.hypothesis_engine.solve(task)
                    # Use the already-tested hypotheses from HE directly —
                    # they carry the real apply_fn that works on any input
                    for he_hyp in he_result.get('hypotheses', []):
                        if he_hyp.accuracy == 1.0:
                            he_hyp.source = 'hypothesis_engine'
                            he_hyp.confidence = 0.95
                            hypotheses.insert(0, he_hyp)
                except Exception:
                    pass

            round_trace['phases']['T'] = {
                'num_hypotheses': len(hypotheses),
                'names': [h.name for h in hypotheses[:5]],
            }

            if verbose:
                print(f"  Generated {len(hypotheses)} hypotheses")

            # ──── R-phase: Reasoning (deductive elimination) ────────────
            if verbose:
                print("▶ R-phase: Deductive elimination...")

            survivors = self.reasoner.eliminate(hypotheses, task['train'])
            near = self.reasoner.near_misses(hypotheses, task['train'])

            round_trace['phases']['R'] = {
                'survivors': len(survivors),
                'near_misses': len(near),
            }

            if verbose:
                print(f"  {len(survivors)} perfect, {len(near)} near-misses")
                for h in survivors[:3]:
                    print(f"    ✓ {h.name} (src={h.source})")
                for h, score in near[:2]:
                    print(f"    ~ {h.name} ({score:.0%})")

            # Generate predictions from survivors
            candidates = []
            for h in survivors:
                try:
                    pred = h.apply_fn(test_input)
                    if pred is not None:
                        candidates.append((pred, h.name, h.confidence))
                except Exception:
                    pass

            # ──── L-phase: Learning (TTT on same-size tasks) ────────────
            if (round_num == 0
                    and visual_agg['all_same_size']
                    and not candidates
                    and self.learner is not None
                    and (time.time() - t0) < self.timeout * 0.7):
                if verbose:
                    print("▶ L-phase: Per-task neural learning...")

                ttt_preds = self.learner.learn_and_predict(task, verbose=verbose)
                if ttt_preds is not None:
                    for pi, pred_arr in enumerate(ttt_preds):
                        pred_list = pred_arr.tolist()
                        candidates.append((pred_list, 'ttt_neural', 0.65))

                round_trace['phases']['L'] = {'ttt_ran': True, 'ttt_produced': ttt_preds is not None}
            else:
                round_trace['phases']['L'] = {'ttt_ran': False}

            # ──── C-phase: Consequential analysis ───────────────────────
            if verbose:
                print("▶ C-phase: Consequence evaluation...")

            scored = self.analyzer.compare_predictions(candidates, task)

            round_trace['phases']['C'] = {
                'num_candidates': len(scored),
                'top_score': scored[0][2] if scored else 0,
            }

            if verbose and scored:
                for pred, method, score, cons in scored[:3]:
                    flags = [k for k, v in cons.items() if k != 'overall' and v is True]
                    print(f"  {method}: score={score:.2f} [{', '.join(flags)}]")

            # ──── CC-phase: Critical decision ───────────────────────────
            if verbose:
                print("▶ CC-phase: Critical evaluation...")

            accept, prediction, reason = self.critic.should_accept(scored, round_num)

            round_trace['phases']['CC'] = {'accept': accept, 'reason': reason}

            if verbose:
                print(f"  Decision: {reason}")

            # Track best across all rounds
            best_candidates = scored if scored else best_candidates

            if accept and prediction is not None:
                elapsed_ms = (time.time() - t0) * 1000
                trace.append(round_trace)
                return {
                    'solved': True,
                    'prediction': prediction,
                    'predictions': [(p, m, s) for p, m, s, _ in scored[:2]],
                    'reasoning_trace': trace,
                    'visual_summary': visual_summary,
                    'time_ms': elapsed_ms,
                    'rounds': round_num + 1,
                    'method': reason,
                }

            # Record failed hypotheses for next round
            for h in hypotheses:
                if h.accuracy < 1.0:
                    failed_hypotheses.append(h.name)

            trace.append(round_trace)

        # ──── Final: return best effort ─────────────────────────────────
        elapsed_ms = (time.time() - t0) * 1000
        prediction = None
        if best_candidates:
            prediction = best_candidates[0][0]

        return {
            'solved': prediction is not None,
            'prediction': prediction,
            'predictions': [(p, m, s) for p, m, s, _ in best_candidates[:2]] if best_candidates else [],
            'reasoning_trace': trace,
            'visual_summary': visual_summary,
            'time_ms': elapsed_ms,
            'rounds': self.max_rounds,
            'method': 'best_effort' if prediction else 'unsolved',
        }

    def refine_with_neural_hint(
        self, task: Dict, neural_prediction: Grid, verbose: bool = False
    ) -> Dict[str, Any]:
        """Neural→Symbolic feedback: use a neural prediction as a seed.
        
        When the neural model produces a prediction, evaluate it with the
        ConsequentialAnalyzer and optionally use it as a starting point
        for symbolic refinement (e.g., fix color inconsistencies).
        
        This closes the loop: Neural → TTCCVTLR → refined prediction.
        """
        consistency = self.analyzer.check_consistency(neural_prediction, task)
        
        if consistency.get('overall', 0) >= 0.75:
            # Neural prediction is self-consistent — accept it
            return {
                'accepted': True,
                'prediction': neural_prediction,
                'consistency': consistency,
                'method': 'neural_accepted',
            }
        
        # Try to fix the prediction symbolically
        # If color palette is wrong, remap to valid colors
        if not consistency.get('color_consistent', True):
            fixed = self._fix_color_palette(neural_prediction, task)
            if fixed is not None:
                fix_consistency = self.analyzer.check_consistency(fixed, task)
                if fix_consistency.get('overall', 0) > consistency.get('overall', 0):
                    return {
                        'accepted': True,
                        'prediction': fixed,
                        'consistency': fix_consistency,
                        'method': 'neural_color_fixed',
                    }
        
        return {
            'accepted': consistency.get('overall', 0) >= 0.5,
            'prediction': neural_prediction,
            'consistency': consistency,
            'method': 'neural_low_confidence',
        }

    def _fix_color_palette(self, prediction: Grid, task: Dict) -> Optional[Grid]:
        """Remap prediction colors to match the training output palette."""
        # Collect valid output colors from training
        valid_colors = set()
        for ex in task['train']:
            for row in ex['output']:
                valid_colors.update(row)
        
        pred = np.array(prediction)
        pred_colors = set(pred.flatten())
        invalid = pred_colors - valid_colors
        
        if not invalid:
            return None  # Already valid
        
        # For each invalid color, map to nearest valid color by frequency
        train_color_freq = Counter()
        for ex in task['train']:
            for row in ex['output']:
                train_color_freq.update(row)
        
        remap = {}
        for ic in invalid:
            # Map to most common valid color
            remap[ic] = train_color_freq.most_common(1)[0][0]
        
        fixed = pred.copy()
        for old_c, new_c in remap.items():
            fixed[pred == old_c] = new_c
        
        return fixed.tolist()


# ============================================================================
# Convenience: solve a task file
# ============================================================================

def solve_task(task: Dict, verbose: bool = False, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper for TTCCVTLREngine.solve()."""
    engine = TTCCVTLREngine(**kwargs)
    return engine.solve(task, verbose=verbose)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TTCCVTLR: Test-Time Critical Consequential '
                                                 'Visualization Thinking Learning & Reasoning')
    parser.add_argument('--data', default='ARC_AMD_TRANSFER/data/ARC-AGI/data/training')
    parser.add_argument('--task', type=str, help='Single task ID to solve')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-ttt', action='store_true', help='Disable TTT neural learning')
    args = parser.parse_args()

    import os

    if args.task:
        path = os.path.join(args.data, f'{args.task}.json')
        with open(path) as f:
            data = json.load(f)
        result = solve_task(data, verbose=True, use_neural_learning=not args.no_ttt)
        print(f"\n{'='*50}")
        print(f"Solved: {result['solved']}")
        print(f"Method: {result['method']}")
        print(f"Rounds: {result['rounds']}")
        print(f"Time: {result['time_ms']:.0f}ms")
        if result['solved'] and 'output' in data['test'][0]:
            expected = data['test'][0]['output']
            correct = result['prediction'] == expected
            print(f"Correct: {correct}")
    else:
        files = sorted(os.listdir(args.data))
        if args.limit:
            files = files[:args.limit]

        solved = 0
        total = 0
        times = []

        for tf in files:
            if not tf.endswith('.json'):
                continue
            tid = tf.replace('.json', '')
            with open(os.path.join(args.data, tf)) as f:
                data = json.load(f)

            total += 1
            result = solve_task(
                data, verbose=args.verbose,
                use_neural_learning=not args.no_ttt,
            )
            times.append(result['time_ms'])

            gt = data['test'][0].get('output')
            correct = result['solved'] and gt and result['prediction'] == gt

            if correct:
                solved += 1
            status = '✓' if correct else '✗'
            print(f"[{total:3d}] {status} {tid}: {result['method']} ({result['time_ms']:.0f}ms)")

        print(f"\n{'='*50}")
        print(f"TTCCVTLR Results: {solved}/{total} ({100*solved/max(total,1):.1f}%)")
        print(f"Avg time: {np.mean(times):.0f}ms, Median: {np.median(times):.0f}ms")
