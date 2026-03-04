"""
Hypothesis Engine — Form and Test Hypotheses from 2-3 Examples

The core AGI capability: given a few examples, generate candidate
transformation hypotheses, test them exhaustively, and select the
best one. Combines:

  1. FewShotAbstractor — structural rule extraction
  2. CompositionEngine — program synthesis from primitives
  3. HypothesisRanker — score and select among competing hypotheses
  4. Color mapping search — enumerate bijective color transforms
  5. Parameterized search — try operations with different parameters

The engine implements a generate-test-refine loop:
  Generate → Test against training examples → Rank → Apply to test input
"""

from typing import List, Dict, Tuple, Optional, Any, Set
from collections import Counter
import time
import copy

from core.few_shot import FewShotAbstractor, find_objects, background_color, grid_dims, grid_equal, extract_subgrid
from core.primitives import CompositionEngine, Grid, BASE_PRIMITIVES


class Hypothesis:
    """A candidate transformation hypothesis."""

    def __init__(
        self,
        name: str,
        apply_fn,
        source: str = 'unknown',
        confidence: float = 0.0,
    ):
        self.name = name
        self.apply_fn = apply_fn  # Grid → Grid
        self.source = source
        self.confidence = confidence
        self.train_correct = 0
        self.train_total = 0

    def test(self, examples: List[Dict]) -> float:
        """Test hypothesis against examples, return accuracy."""
        self.train_total = len(examples)
        self.train_correct = 0
        for ex in examples:
            try:
                pred = self.apply_fn(ex['input'])
                if pred == ex['output']:
                    self.train_correct += 1
            except Exception:
                pass
        return self.train_correct / max(self.train_total, 1)

    @property
    def accuracy(self) -> float:
        return self.train_correct / max(self.train_total, 1)

    def __repr__(self):
        return f'Hypothesis({self.name}, acc={self.accuracy:.0%}, src={self.source})'


class HypothesisEngine:
    """
    Generate, test, and rank hypotheses for ARC tasks.
    
    Usage:
        engine = HypothesisEngine()
        result = engine.solve(task)
        if result['solved']:
            prediction = result['prediction']
    """

    def __init__(self, max_composition_depth: int = 2, timeout_seconds: float = 30.0):
        self.abstractor = FewShotAbstractor()
        self.composer = CompositionEngine(max_depth=max_composition_depth)
        self.timeout = timeout_seconds

    def solve(self, task: Dict, test_idx: int = 0) -> Dict[str, Any]:
        """
        Attempt to solve an ARC task.
        
        Args:
            task: dict with 'train' (list of {input, output}) and 'test' (list of {input})
            test_idx: which test case to solve
            
        Returns:
            dict with:
              - solved: bool
              - prediction: Grid or None
              - hypotheses: list of tested hypotheses
              - method: str describing how it was solved
              - time_ms: float
        """
        t0 = time.time()
        train = task['train']
        test_input = task['test'][test_idx]['input']

        hypotheses: List[Hypothesis] = []

        # Phase 1: Few-shot abstraction (fast structural analysis)
        abstracted_rules = self.abstractor.abstract(train)
        for rule in abstracted_rules:
            h = self._rule_to_hypothesis(rule)
            if h:
                hypotheses.append(h)

        # Phase 2: Composition search (program synthesis)
        if not self._has_perfect(hypotheses, train):
            elapsed = time.time() - t0
            if elapsed < self.timeout:
                comp_solutions = self.composer.search(train, max_depth=2)
                for sol in comp_solutions:
                    program = sol['program']
                    h = Hypothesis(
                        name=' → '.join(program),
                        apply_fn=lambda g, p=program: self.composer.apply_program(p, g),
                        source='composition',
                        confidence=0.9,
                    )
                    hypotheses.append(h)

        # Phase 3: Color mapping enumeration
        if not self._has_perfect(hypotheses, train):
            cmap_hyps = self._generate_color_mapping_hypotheses(train)
            hypotheses.extend(cmap_hyps)

        # Phase 4: Object-based hypotheses
        if not self._has_perfect(hypotheses, train):
            obj_hyps = self._generate_object_hypotheses(train)
            hypotheses.extend(obj_hyps)

        # Phase 5: Pattern replication hypotheses
        if not self._has_perfect(hypotheses, train):
            pattern_hyps = self._generate_pattern_hypotheses(train)
            hypotheses.extend(pattern_hyps)

        # Test all hypotheses
        for h in hypotheses:
            h.test(train)

        # Rank: perfect matches first, then by confidence
        hypotheses.sort(key=lambda h: (h.accuracy, h.confidence), reverse=True)

        # Apply best hypothesis
        prediction = None
        method = 'none'
        solved = False

        for h in hypotheses:
            if h.accuracy == 1.0:
                try:
                    prediction = h.apply_fn(test_input)
                    method = f'{h.source}: {h.name}'
                    solved = True
                    break
                except Exception:
                    continue

        elapsed_ms = (time.time() - t0) * 1000
        return {
            'solved': solved,
            'prediction': prediction,
            'hypotheses': hypotheses[:10],  # Top 10
            'method': method,
            'time_ms': elapsed_ms,
            'num_hypotheses_tested': len(hypotheses),
        }

    def _has_perfect(self, hypotheses: List[Hypothesis], train: List[Dict]) -> bool:
        """Quick check if any hypothesis already perfectly solves training."""
        for h in hypotheses:
            if h.test(train) == 1.0:
                return True
        return False

    def _rule_to_hypothesis(self, rule: Dict) -> Optional[Hypothesis]:
        """Convert an abstracted rule to a testable hypothesis."""
        rtype = rule['type']
        params = rule['params']

        if rtype == 'geometric':
            transforms = {
                'rotate_90': lambda g: [list(r) for r in zip(*g[::-1])],
                'rotate_180': lambda g: [r[::-1] for r in g[::-1]],
                'rotate_270': lambda g: [list(r) for r in zip(*[row[::-1] for row in g])],
                'flip_h': lambda g: [r[::-1] for r in g],
                'flip_v': lambda g: g[::-1],
                'transpose': lambda g: [list(r) for r in zip(*g)],
            }
            t = params['transform']
            if t in transforms:
                return Hypothesis(t, transforms[t], 'abstraction', rule['confidence'])

        elif rtype == 'color_map':
            mapping = params['mapping']
            return Hypothesis(
                f'color_map({mapping})',
                lambda g, m=mapping: [[m.get(c, c) for c in row] for row in g],
                'abstraction',
                rule['confidence'],
            )

        elif rtype == 'scale':
            factor = params['factor']
            def scale_fn(g, f=factor):
                result = []
                for row in g:
                    new_row = []
                    for c in row:
                        new_row.extend([c] * f)
                    for _ in range(f):
                        result.append(new_row[:])
                return result
            return Hypothesis(f'scale_{factor}x', scale_fn, 'abstraction', rule['confidence'])

        elif rtype == 'tile':
            tr, tc = params['rows'], params['cols']
            def tile_fn(g, tr=tr, tc=tc):
                tiled = []
                for _ in range(tr):
                    for row in g:
                        tiled.append(row * tc)
                return tiled
            return Hypothesis(f'tile_{tr}x{tc}', tile_fn, 'abstraction', rule['confidence'])

        elif rtype == 'gravity':
            direction = params['direction']
            from core.few_shot import FewShotAbstractor
            ab = FewShotAbstractor()
            return Hypothesis(
                f'gravity_{direction}',
                lambda g, d=direction: ab._apply_gravity(g, d),
                'abstraction',
                rule['confidence'],
            )

        elif rtype == 'extract_largest_object':
            def extract_largest(g):
                bg = background_color(g)
                objs = find_objects(g, bg)
                if not objs:
                    return g
                largest = max(objs, key=lambda o: o['size'])
                return extract_subgrid(g, largest['bbox'])
            return Hypothesis('extract_largest', extract_largest, 'abstraction', rule['confidence'])

        elif rtype == 'extract_smallest_object':
            def extract_smallest(g):
                bg = background_color(g)
                objs = find_objects(g, bg)
                if not objs:
                    return g
                smallest = min(objs, key=lambda o: o['size'])
                return extract_subgrid(g, smallest['bbox'])
            return Hypothesis('extract_smallest', extract_smallest, 'abstraction', rule['confidence'])

        elif rtype == 'add_border':
            color = params['color']
            width = params['width']
            def add_border(g, c=color, w=width):
                rows, cols = len(g), len(g[0])
                new_rows = rows + 2*w
                new_cols = cols + 2*w
                result = [[c]*new_cols for _ in range(new_rows)]
                for r in range(rows):
                    for ci in range(cols):
                        result[r+w][ci+w] = g[r][ci]
                return result
            return Hypothesis(f'add_border(color={color})', add_border, 'abstraction', rule['confidence'])

        elif rtype == 'remove_border':
            width = params['width']
            def remove_border(g, w=width):
                return [row[w:-w] for row in g[w:-w]]
            return Hypothesis('remove_border', remove_border, 'abstraction', rule['confidence'])

        elif rtype == 'symmetry_completion':
            axis = params['axis']
            if axis == 'vertical':
                from core.primitives import p_mirror_h_complete
                return Hypothesis('mirror_h_complete', p_mirror_h_complete, 'abstraction', rule['confidence'])

        return None

    def _generate_color_mapping_hypotheses(self, train: List[Dict]) -> List[Hypothesis]:
        """Generate hypotheses based on color relationships."""
        hypotheses = []
        # Try swapping each pair of colors
        all_colors = set()
        for ex in train:
            all_colors |= {c for row in ex['input'] for c in row}
            all_colors |= {c for row in ex['output'] for c in row}
        
        for a in all_colors:
            for b in all_colors:
                if a >= b:
                    continue
                def swap_fn(g, a=a, b=b):
                    return [[b if c == a else (a if c == b else c) for c in row] for row in g]
                hypotheses.append(Hypothesis(
                    f'swap({a},{b})', swap_fn, 'color_search', 0.5,
                ))
        return hypotheses

    def _generate_object_hypotheses(self, train: List[Dict]) -> List[Hypothesis]:
        """Generate hypotheses based on object-level analysis."""
        hypotheses = []
        
        # Try: for each color, extract all cells of that color as a cropped grid
        all_colors = set()
        for ex in train:
            all_colors |= {c for row in ex['input'] for c in row}

        for color in all_colors:
            def extract_color(g, c=color):
                rows, cols = len(g), len(g[0])
                bg = background_color(g)
                # Find bounding box of this color
                r1, r2, c1_, c2 = rows, 0, cols, 0
                found = False
                for r in range(rows):
                    for ci in range(cols):
                        if g[r][ci] == c:
                            r1 = min(r1, r)
                            r2 = max(r2, r)
                            c1_ = min(c1_, ci)
                            c2 = max(c2, ci)
                            found = True
                if not found:
                    return g
                return [row[c1_:c2+1] for row in g[r1:r2+1]]
            hypotheses.append(Hypothesis(
                f'extract_color({color})', extract_color, 'object_search', 0.4,
            ))

        # Try: mask — keep only one color, rest becomes background
        bg_candidates = set()
        for ex in train:
            bg_candidates.add(background_color(ex['input']))

        for bg in bg_candidates:
            for keep_color in all_colors:
                if keep_color == bg:
                    continue
                def mask_fn(g, bg=bg, kc=keep_color):
                    return [[c if c == kc else bg for c in row] for row in g]
                hypotheses.append(Hypothesis(
                    f'mask(keep={keep_color},bg={bg})', mask_fn, 'object_search', 0.3,
                ))

        return hypotheses

    def _generate_pattern_hypotheses(self, train: List[Dict]) -> List[Hypothesis]:
        """Generate hypotheses for pattern-based transformations."""
        hypotheses = []

        # Check if output is a sub-region of input
        for ex in train:
            ir, ic = grid_dims(ex['input'])
            or_, oc = grid_dims(ex['output'])
            if or_ < ir or oc < ic:
                # Output is smaller — try all possible crops
                for r_off in range(ir - or_ + 1):
                    for c_off in range(ic - oc + 1):
                        cropped = [row[c_off:c_off+oc] for row in ex['input'][r_off:r_off+or_]]
                        if cropped == ex['output']:
                            def crop_fn(g, ro=r_off, co=c_off, oh=or_, ow=oc):
                                return [row[co:co+ow] for row in g[ro:ro+oh]]
                            hypotheses.append(Hypothesis(
                                f'crop({r_off},{c_off},{or_},{oc})', crop_fn, 'pattern', 0.6,
                            ))
                break  # Only analyze first example for crop positions

        return hypotheses


# ============================================================================
# Convenience function
# ============================================================================

def solve_arc_task(task: Dict, timeout: float = 30.0) -> Dict[str, Any]:
    """One-line interface to solve an ARC task."""
    engine = HypothesisEngine(max_composition_depth=2, timeout_seconds=timeout)
    return engine.solve(task)
