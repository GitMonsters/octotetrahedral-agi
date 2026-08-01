"""
Tetrahedral Grid Graph Implementation
For OctoTetrahedral AGI - ARC Prize 2026

Provides tetrahedral coordinate system for geometric reasoning on ARC grids.
"""

from collections import deque
from typing import Tuple, Dict, List, Optional
import numpy as np


class TetrahedralGridGraph:
    """Tetrahedral grid graph with barycentric coordinates."""
    
    def __init__(self, size: int = 10):
        self.size = size
        self.cells: Dict[Tuple[int, int, int, int], Dict] = {}
        self.adjacency: Dict[Tuple[int, int, int, int], List] = {}
        self._build_graph()
    
    def _build_graph(self):
        """Build tetrahedral grid structure."""
        for a in range(self.size):
            for b in range(self.size):
                for c in range(self.size):
                    d = self.size - (a + b + c)
                    if d >= 0:
                        coord = (a, b, c, d)
                        self.cells[coord] = {'value': 0, 'color': 0}
                        self.adjacency[coord] = self._get_neighbors(coord)
    
    def _get_neighbors(self, coord: Tuple) -> List[Tuple]:
        """Get 12 neighbors in tetrahedral space."""
        a, b, c, d = coord
        neighbors = []
        
        directions = [
            (1, -1, 0, 0), (-1, 1, 0, 0),
            (1, 0, -1, 0), (-1, 0, 1, 0),
            (1, 0, 0, -1), (-1, 0, 0, 1),
            (0, 1, -1, 0), (0, -1, 1, 0),
            (0, 1, 0, -1), (0, -1, 0, 1),
            (0, 0, 1, -1), (0, 0, -1, 1),
        ]
        
        for da, db, dc, dd in directions:
            na, nb, nc, nd = a+da, b+db, c+dc, d+dd
            if all(x >= 0 for x in [na, nb, nc, nd]):
                neighbor = (na, nb, nc, nd)
                if neighbor in self.cells:
                    neighbors.append(neighbor)
        
        return neighbors
    
    def distance(self, coord1: Tuple, coord2: Tuple) -> int:
        """Tetrahedral distance between points."""
        a1, b1, c1, d1 = coord1
        a2, b2, c2, d2 = coord2
        return (abs(a1-a2) + abs(b1-b2) + abs(c1-c2) + abs(d1-d2)) // 2
    
    def shortest_path(self, start: Tuple, end: Tuple) -> Optional[List[Tuple]]:
        """BFS shortest path in tetrahedral grid."""
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
            
            for neighbor in self.adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def rotate_120(self, coord: Tuple) -> Tuple:
        """120° rotation (cyclic permutation)."""
        a, b, c, d = coord
        return (b, c, a, d)
    
    def rotate_240(self, coord: Tuple) -> Tuple:
        """240° rotation."""
        a, b, c, d = coord
        return (c, a, b, d)
    
    def reflect_axis(self, coord: Tuple, axis: int) -> Tuple:
        """Reflect across axis."""
        coords = list(coord)
        coords[axis] = -coords[axis]
        return tuple(coords)


def arc_grid_to_tetrahedral(grid: List[List[int]]) -> Dict[Tuple, int]:
    """Convert ARC rectangular grid to tetrahedral representation."""
    tet_grid = {}
    
    if not grid:
        return tet_grid
    
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    for y in range(height):
        for x in range(width):
            color = grid[y][x]
            # Simple mapping: (x, y) -> (x, y, 0, 0)
            a = x
            b = y
            c = 0
            d = 0
            coord = (a, b, c, d)
            tet_grid[coord] = color
    
    return tet_grid


def tetrahedral_to_arc_grid(tet_grid: Dict[Tuple, int]) -> List[List[int]]:
    """Convert tetrahedral back to rectangular ARC grid."""
    if not tet_grid:
        return []
    
    max_x = max(coord[0] for coord in tet_grid.keys())
    max_y = max(coord[1] for coord in tet_grid.keys())
    
    grid = [[0] * (max_x + 1) for _ in range(max_y + 1)]
    
    for (a, b, c, d), color in tet_grid.items():
        if 0 <= b < len(grid) and 0 <= a < len(grid[0]):
            grid[b][a] = color
    
    return grid


def find_tet_patterns(tet_grid: Dict[Tuple, int], grid_graph: TetrahedralGridGraph) -> Dict:
    """Find geometric patterns in tetrahedral grid."""
    patterns = {
        'colors': set(),
        'connected_components': [],
        'symmetries': []
    }
    
    # Find unique colors
    patterns['colors'] = set(tet_grid.values())
    
    # Find connected components
    visited = set()
    for coord in tet_grid:
        if coord not in visited:
            component = _bfs_component(coord, tet_grid, grid_graph, visited)
            patterns['connected_components'].append(component)
    
    return patterns


def _bfs_component(start: Tuple, grid: Dict, graph: TetrahedralGridGraph, visited: set) -> List:
    """BFS to find connected component."""
    component = []
    queue = deque([start])
    visited.add(start)
    
    while queue:
        coord = queue.popleft()
        component.append(coord)
        
        for neighbor in graph.adjacency[coord]:
            if neighbor not in visited and neighbor in grid:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return component


def detect_rotation_tet(input_tet: Dict, output_tet: Dict, grid_graph: TetrahedralGridGraph) -> Optional[str]:
    """Detect if output is rotated version of input."""
    if len(input_tet) != len(output_tet):
        return None
    
    # Try 120° rotation
    rotated_120 = {grid_graph.rotate_120(k): v for k, v in input_tet.items()}
    if rotated_120 == output_tet:
        return 'rotate_120'
    
    # Try 240° rotation
    rotated_240 = {grid_graph.rotate_240(k): v for k, v in input_tet.items()}
    if rotated_240 == output_tet:
        return 'rotate_240'
    
    return None


def apply_rotation_tet(tet_grid: Dict[Tuple, int], rotation: str, 
                       grid_graph: TetrahedralGridGraph) -> Dict[Tuple, int]:
    """Apply rotation to tetrahedral grid."""
    if rotation == 'rotate_120':
        return {grid_graph.rotate_120(k): v for k, v in tet_grid.items()}
    elif rotation == 'rotate_240':
        return {grid_graph.rotate_240(k): v for k, v in tet_grid.items()}
    else:
        return tet_grid


# Example usage
if __name__ == '__main__':
    # Create tetrahedral grid
    grid = TetrahedralGridGraph(size=5)
    print(f"Grid cells: {len(grid.cells)}")
    
    # Create sample ARC grid
    arc_grid = [
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ]
    
    # Convert to tetrahedral
    tet_grid = arc_grid_to_tetrahedral(arc_grid)
    print(f"Tetrahedral points: {len(tet_grid)}")
    
    # Convert back
    recovered = tetrahedral_to_arc_grid(tet_grid)
    print(f"Recovered grid matches: {recovered == arc_grid}")
    
    # Find patterns
    patterns = find_tet_patterns(tet_grid, grid)
    print(f"Colors found: {patterns['colors']}")
    print(f"Connected components: {len(patterns['connected_components'])}")
