"""
RE-ARC v47: Smart Pattern Matching + Hybrid Voting

Improvements over v46:
1. Match solvers by COLOR PATTERN similarity (not random)
2. Match solvers by GRID SIZE similarity (prefer same-sized grids)
3. Transformation TYPE matching (rotation/symmetry/compression/expansion)
4. Hybrid voting: Different solver types vote together

Strategy:
- Compute color histogram of test input
- Find solvers whose training inputs have similar color histograms
- Try size-similar solvers first (higher priority)
- Ensemble vote across all successful predictions
- This should break past 3.33% by finding MORE relevant solvers
"""

import json
import numpy as np
from pathlib import Path
import importlib.util
import sys
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import hashlib

sys.path.insert(0, '/Users/evanpieser')


def _numpy_to_python(obj):
    """Recursively convert numpy types to Python types"""
    if isinstance(obj, np.ndarray):
        return _numpy_to_python(obj.tolist())
    elif isinstance(obj, (np.integer, np.floating)):
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    elif isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_python(item) for item in obj]
    else:
        return obj


class SmartPatternREARCSolver:
    """RE-ARC v47: Smart pattern matching + hybrid voting"""
    
    def __init__(self):
        self.solvers = {}
        self.loaded_solvers = {}
        self.solver_cache = {}
        self.solver_features = {}  # Cache solver characteristics
        self.voting_cache = {}
        self._load_catalog_solvers()
        self._analyze_solver_features()
    
    def _load_catalog_solvers(self):
        """Load all catalog solvers"""
        print("Loading solvers for smart pattern matching...")
        solver_dir = Path("/Users/evanpieser")
        
        count = 0
        for solver_file in sorted(solver_dir.rglob("*_solver.py")):
            stem = solver_file.stem
            if stem.endswith("_solver"):
                task_id = stem[:-7]
                if len(task_id) == 8 and task_id.isalnum():
                    self.solvers[task_id] = solver_file
                    count += 1
        
        print(f"Loaded {count} solvers")
    
    def _analyze_solver_features(self):
        """Analyze characteristics of each solver from its training data"""
        print("Analyzing solver features...")
        
        # Try to load ARC-AGI training data if available
        rearc_path = "/Users/evanpieser/Downloads/re-arc_test_challenges-2026-05-06T02-34-30.json"
        
        try:
            with open(rearc_path) as f:
                data = json.load(f)
            
            # Store basic statistics for quick matching
            for task_id, task in data.items():
                if task.get('train'):
                    pair = task['train'][0]
                    inp = np.array(pair['input'])
                    
                    # Compute features
                    colors = np.unique(inp)
                    color_hist = np.histogram(inp.flatten(), bins=10, range=(0, 10))[0]
                    
                    self.solver_features[task_id] = {
                        'size': inp.shape,
                        'num_colors': len(colors),
                        'color_hist': color_hist.tolist(),
                        'complexity': np.sum(color_hist > 0),
                    }
        except:
            print("Could not analyze features")
    
    def _color_similarity(self, hist1: List[float], hist2: List[float]) -> float:
        """Compute similarity between color histograms (0..1)"""
        h1 = np.array(hist1)
        h2 = np.array(hist2)
        
        # Normalize
        h1 = h1 / (np.sum(h1) + 1e-10)
        h2 = h2 / (np.sum(h2) + 1e-10)
        
        # Cosine similarity
        similarity = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-10)
        return float(similarity)
    
    def _size_similarity(self, size1: Tuple, size2: Tuple) -> float:
        """Compute size similarity (1.0 if same, decays with difference)"""
        if len(size1) != len(size2):
            return 0.0
        
        ratio = np.prod(np.array(size1)) / (np.prod(np.array(size2)) + 1e-10)
        # Prefer sizes within 0.5x to 2x of each other
        if 0.5 <= ratio <= 2.0:
            return 1.0 - abs(ratio - 1.0) * 0.5
        else:
            return max(0, 1.0 - abs(np.log2(ratio)))
    
    def _get_best_solver_matches(self, test_input: np.ndarray, max_solvers: int = 80) -> List[str]:
        """Get best solver IDs based on pattern similarity"""
        
        # Compute test input features
        test_colors = np.unique(test_input)
        test_hist = np.histogram(test_input.flatten(), bins=10, range=(0, 10))[0]
        test_size = test_input.shape
        
        # Score each solver
        scores = []
        for solver_id, features in self.solver_features.items():
            color_sim = self._color_similarity(test_hist.tolist(), features['color_hist'])
            size_sim = self._size_similarity(test_size, features['size'])
            
            # Combined score: prefer color match slightly over size match
            score = color_sim * 0.6 + size_sim * 0.4
            scores.append((score, solver_id))
        
        # Sort and return top solvers
        scores.sort(reverse=True)
        return [solver_id for _, solver_id in scores[:max_solvers]]
    
    def _load_solver_module(self, task_id: str):
        """Dynamically load solver"""
        if task_id in self.loaded_solvers:
            return self.loaded_solvers[task_id]
        
        solver_file = self.solvers.get(task_id)
        if not solver_file:
            return None
        
        try:
            spec = importlib.util.spec_from_file_location(f"solver_{task_id}", solver_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"solver_{task_id}"] = module
            spec.loader.exec_module(module)
            self.loaded_solvers[task_id] = module
            return module
        except:
            return None
    
    def _get_solver_function(self, module):
        """Extract solve function"""
        for func_name in ['solve', 'solve_task', 'transform', 'process']:
            if hasattr(module, func_name):
                return getattr(module, func_name)
        
        for name in dir(module):
            if not name.startswith('_'):
                attr = getattr(module, name)
                if callable(attr):
                    return attr
        return None
    
    def _test_solver(self, solver_func, train_pairs: List, max_test: int = 1) -> bool:
        """Test if solver works"""
        try:
            sample = min(max_test, len(train_pairs))
            for pair in train_pairs[:sample]:
                inp = np.array(pair['input'])
                expected = np.array(pair['output'])
                pred = solver_func(inp)
                
                if not np.array_equal(pred, expected):
                    return False
            
            return True
        except:
            return False
    
    def _compute_output_hash(self, outputs: List[Any]) -> str:
        """Compute hash for voting"""
        hash_str = ""
        for out in outputs:
            if isinstance(out, np.ndarray):
                hash_str += str(hash(out.tobytes()))
            elif isinstance(out, list):
                hash_str += str(hash(tuple(map(tuple, out))))
            else:
                hash_str += str(hash(str(out)))
        return hashlib.md5(hash_str.encode()).hexdigest()[:8]
    
    def _try_smart_solvers(self, task: Dict, task_id: str) -> Tuple[bool, Optional[List[np.ndarray]]]:
        """Try smart-ranked solvers"""
        
        test_items = task.get('test', [])
        train_pairs = task.get('train', [])
        
        if not test_items or not train_pairs:
            return False, None
        
        if task_id in self.voting_cache:
            return True, self.voting_cache[task_id]
        
        # Get test input for smart ranking
        test_input = np.array(test_items[0]['input'])
        
        # Get best matching solvers
        ranked_solvers = self._get_best_solver_matches(test_input, max_solvers=80)
        
        all_predictions = []
        
        # Try ranked solvers
        for solver_id in ranked_solvers:
            # Try cache first
            if solver_id in self.solver_cache:
                try:
                    test_predictions = []
                    for test_item in test_items:
                        inp = np.array(test_item['input'])
                        pred = self.solver_cache[solver_id](inp)
                        test_predictions.append(pred)
                    
                    all_predictions.append(test_predictions)
                    
                    if len(all_predictions) >= 3:  # Enough predictions for voting
                        break
                    continue
                except:
                    pass
            
            # Load and test new solver
            module = self._load_solver_module(solver_id)
            if not module:
                continue
            
            solver_func = self._get_solver_function(module)
            if not solver_func:
                continue
            
            if self._test_solver(solver_func, train_pairs, max_test=1):
                self.solver_cache[solver_id] = solver_func
                
                try:
                    test_predictions = []
                    for test_item in test_items:
                        inp = np.array(test_item['input'])
                        pred = solver_func(inp)
                        test_predictions.append(pred)
                    
                    all_predictions.append(test_predictions)
                    
                    if len(all_predictions) >= 3:  # Enough for voting
                        break
                except:
                    pass
        
        if all_predictions:
            # Voting: pick most common prediction
            hashes = defaultdict(list)
            for preds in all_predictions:
                h = self._compute_output_hash(preds)
                hashes[h].append(preds)
            
            most_common = max(hashes.keys(), key=lambda h: len(hashes[h]))
            voted_predictions = hashes[most_common][0]
            self.voting_cache[task_id] = voted_predictions
            return True, voted_predictions
        
        return False, None
    
    def _try_geometric(self, task: Dict) -> Optional[List[np.ndarray]]:
        """Try geometric transforms"""
        train_pairs = task.get('train', [])
        test_items = task.get('test', [])
        
        if not train_pairs or not test_items:
            return None
        
        pair = train_pairs[0]
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        
        for k in [1, 2, 3]:
            if np.array_equal(np.rot90(inp, k), out):
                return [np.rot90(np.array(item['input']), k) for item in test_items]
        
        if np.array_equal(np.fliplr(inp), out):
            return [np.fliplr(np.array(item['input'])) for item in test_items]
        if np.array_equal(np.flipud(inp), out):
            return [np.flipud(np.array(item['input'])) for item in test_items]
        if np.array_equal(inp.T, out):
            return [np.array(item['input']).T for item in test_items]
        
        return None
    
    def _fallback(self, task: Dict) -> List[np.ndarray]:
        """Fallback: identity"""
        test_items = task.get('test', [])
        return [np.array(item['input']) for item in test_items]
    
    def solve_task(self, task: Dict, task_id: str) -> List[np.ndarray]:
        """Solve with smart pattern matching"""
        
        # 1. Smart solver matching (main)
        success, predictions = self._try_smart_solvers(task, task_id)
        if success and predictions:
            return predictions
        
        # 2. Geometric transforms
        predictions = self._try_geometric(task)
        if predictions:
            return predictions
        
        # 3. Fallback
        return self._fallback(task)
    
    def solve_rearc_dataset(self, dataset_path: str, output_path: str):
        """Solve with smart pattern matching"""
        
        with open(dataset_path) as f:
            rearc_data = json.load(f)
        
        submission = {}
        
        print("\n" + "=" * 70)
        print("RE-ARC v47: Smart Pattern Matching + Hybrid Voting")
        print("=" * 70)
        print(f"Solving {len(rearc_data)} tasks with smart pattern matching...")
        
        for i, (task_id, task) in enumerate(rearc_data.items()):
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1:3d}/{len(rearc_data)}] Cache: {len(self.solver_cache)} | Vote: {len(self.voting_cache)}")
            
            predictions = self.solve_task(task, task_id)
            submission[task_id] = []
            
            for pred in predictions:
                pred_clean = _numpy_to_python(pred)
                submission[task_id].append(pred_clean)
        
        submission = _numpy_to_python(submission)
        
        with open(output_path, 'w') as f:
            json.dump(submission, f, indent=2)
        
        print("\n" + "=" * 70)
        print(f"SMART PATTERN MATCHING STATISTICS")
        print("=" * 70)
        print(f"Total tasks: {len(submission)}")
        print(f"Total predictions: {sum(len(preds) for preds in submission.values())}")
        print(f"Solver cache: {len(self.solver_cache)}")
        print(f"Voting hits: {len(self.voting_cache)}")
        print("=" * 70 + "\n")
        
        return submission


def main():
    solver = SmartPatternREARCSolver()
    
    dataset_path = "/Users/evanpieser/Downloads/re-arc_test_challenges-2026-05-06T02-34-30.json"
    output_path = "/Users/evanpieser/Desktop/72%/octotetrahedral_rearc_v47_smart_pattern.json"
    
    solver.solve_rearc_dataset(dataset_path, output_path)


if __name__ == "__main__":
    main()
