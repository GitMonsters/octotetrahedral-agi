"""
Generic ARC-AGI-3 Game Solvers
BFS-based solvers for click and keyboard games.
"""

import copy
import time
from collections import deque
from typing import Optional, List, Tuple
import heapq

from arcengine import GameAction, GameState


class GenericClickSolver:
    """BFS solver for click-based games."""
    
    def __init__(self, env, verbose: bool = False):
        self.env = env
        self.verbose = verbose
    
    def _get_click_targets(self) -> List[Tuple[int, int]]:
        """Find all clickable positions from current frame."""
        game = self.env._game
        targets = []
        
        # Method 1: Look for sys_click tagged sprites
        try:
            level = game.current_level
            for attr in dir(level):
                if attr.startswith('_'):
                    continue
                try:
                    val = getattr(level, attr)
                    if hasattr(val, 'tags') and 'sys_click' in val.tags:
                        targets.append((val.x, val.y))
                except:
                    pass
        except:
            pass
        
        # Method 2: Grid scan for non-background pixels
        if not targets:
            try:
                frame = self.env._cur_frame.frame[0] if hasattr(self.env, '_cur_frame') else None
                if frame is not None:
                    # Sample grid positions
                    h, w = frame.shape[:2] if len(frame.shape) >= 2 else (64, 64)
                    for y in range(0, h, 4):
                        for x in range(0, w, 4):
                            targets.append((x, y))
            except:
                # Fallback: grid of positions
                for y in range(0, 64, 8):
                    for x in range(0, 64, 8):
                        targets.append((x, y))
        
        return targets[:100]  # Limit to 100 targets
    
    def solve_level(self, max_depth: int = 20, max_nodes: int = 50000, 
                   timeout: float = 60.0) -> Optional[List[Tuple[int, int]]]:
        """BFS to find click sequence that completes level."""
        game = self.env._game
        target_level = game.level_index + 1 if hasattr(game, 'level_index') else 1
        
        # Get initial click targets
        click_targets = self._get_click_targets()
        if self.verbose:
            print(f"  {len(click_targets)} click targets")
        
        # BFS
        t0 = time.time()
        initial_game = copy.deepcopy(game)
        queue = deque([(initial_game, [])])
        visited = set()
        nodes = 0
        
        while queue and time.time() - t0 < timeout:
            game_copy, path = queue.popleft()
            nodes += 1
            
            if nodes > max_nodes or len(path) >= max_depth:
                if len(path) >= max_depth:
                    continue
                break
            
            for x, y in click_targets:
                self.env._game = copy.deepcopy(game_copy)
                try:
                    frame = self.env.step(GameAction.ACTION6, data={'x': x, 'y': y})
                except:
                    continue
                
                # Check win
                if hasattr(self.env._game, 'level_index'):
                    if self.env._game.level_index >= target_level:
                        if self.verbose:
                            print(f"  Solved in {len(path)+1} clicks, {nodes} nodes")
                        self.env._game = initial_game
                        return path + [(x, y)]
                elif frame.levels_completed >= target_level:
                    if self.verbose:
                        print(f"  Solved in {len(path)+1} clicks, {nodes} nodes")
                    self.env._game = initial_game
                    return path + [(x, y)]
                
                if frame.state == GameState.GAME_OVER:
                    continue
                
                # State key
                try:
                    key = hash(frame.frame[0].tobytes())
                except:
                    key = hash(str(id(self.env._game)))
                
                if key not in visited:
                    visited.add(key)
                    queue.append((copy.deepcopy(self.env._game), path + [(x, y)]))
        
        self.env._game = initial_game
        if self.verbose:
            print(f"  No solution: {nodes} nodes, {len(visited)} states")
        return None


class GenericKeyboardSolver:
    """BFS solver for keyboard-based games."""
    
    def __init__(self, env, verbose: bool = False):
        self.env = env
        self.verbose = verbose
        self.AMAP = {a.value: a for a in GameAction}
    
    def solve_level(self, max_depth: int = 50, max_nodes: int = 100000,
                   timeout: float = 60.0, actions: List[int] = None) -> Optional[List[int]]:
        """BFS to find action sequence that completes level."""
        if actions is None:
            actions = [1, 2, 3, 4]  # Default: arrow keys
        
        game = self.env._game
        target_level = game.level_index + 1 if hasattr(game, 'level_index') else 1
        
        t0 = time.time()
        initial_game = copy.deepcopy(game)
        queue = deque([(initial_game, [])])
        visited = set()
        nodes = 0
        
        while queue and time.time() - t0 < timeout:
            game_copy, path = queue.popleft()
            nodes += 1
            
            if nodes > max_nodes or len(path) >= max_depth:
                if len(path) >= max_depth:
                    continue
                break
            
            for action in actions:
                self.env._game = copy.deepcopy(game_copy)
                try:
                    frame = self.env.step(self.AMAP[action])
                except:
                    continue
                
                # Check win
                if hasattr(self.env._game, 'level_index'):
                    if self.env._game.level_index >= target_level:
                        if self.verbose:
                            print(f"  Solved in {len(path)+1} moves, {nodes} nodes")
                        self.env._game = initial_game
                        return path + [action]
                elif frame.levels_completed >= target_level:
                    if self.verbose:
                        print(f"  Solved in {len(path)+1} moves, {nodes} nodes")
                    self.env._game = initial_game
                    return path + [action]
                
                if frame.state == GameState.GAME_OVER:
                    continue
                
                # State key
                try:
                    key = hash(frame.frame[0].tobytes())
                except:
                    key = hash(str(id(self.env._game)))
                
                if key not in visited:
                    visited.add(key)
                    queue.append((copy.deepcopy(self.env._game), path + [action]))
        
        self.env._game = initial_game
        if self.verbose:
            print(f"  No solution: {nodes} nodes, {len(visited)} states")
        return None
