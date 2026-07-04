#!/usr/bin/env python3
"""
Solver Abstractions for ARC-AGI Trait-Based Architecture

Provides abstract base classes and utilities used by the refactored solver
pipeline:
  - Solver       — base class for all ARC solvers
  - CompoundTrait — multi-solver composition via voting
  - TransformTrait — grid transformation capability
  - BBoxTrait    — bounding-box / region-extraction capability
  - GridUtils    — shared static grid operations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Trait base classes
# ---------------------------------------------------------------------------

class CompoundTrait(ABC):
    """Mixin for solvers that compose multiple sub-solvers."""

    @abstractmethod
    def add_subsolver(self, solver: "Solver", priority: int = 0) -> None:
        """Register a sub-solver."""

    @abstractmethod
    def compose_solutions(
        self, solutions: List[Tuple[List[List[int]], float]]
    ) -> List[List[int]]:
        """Combine candidate solutions into a single best solution."""


class TransformTrait(ABC):
    """Mixin for solvers that can apply grid transformations."""

    @abstractmethod
    def can_rotate(self) -> bool:
        """Return True if the solver supports rotation."""

    @abstractmethod
    def can_flip(self) -> bool:
        """Return True if the solver supports flipping."""

    @abstractmethod
    def can_scale(self) -> bool:
        """Return True if the solver supports scaling."""

    @abstractmethod
    def apply_transform(
        self, grid: List[List[int]], transform_type: str, **kwargs: Any
    ) -> List[List[int]]:
        """Apply a named transformation to a grid."""


class BBoxTrait(ABC):
    """Mixin for solvers that work with bounding boxes."""

    @abstractmethod
    def extract_bounding_box(
        self, grid: List[List[int]]
    ) -> Tuple[int, int, int, int]:
        """Return (min_row, max_row, min_col, max_col) of the foreground."""

    @abstractmethod
    def extract_region(
        self, grid: List[List[int]], bbox: Tuple[int, int, int, int]
    ) -> List[List[int]]:
        """Extract the rectangular region described by bbox."""

    @abstractmethod
    def detect_background_color(self, grid: List[List[int]]) -> int:
        """Return the most-common (background) colour value."""


# ---------------------------------------------------------------------------
# Solver base class
# ---------------------------------------------------------------------------

class Solver(ABC):
    """
    Abstract base class for all ARC-AGI solvers.

    Attributes:
        name   — human-readable solver identifier
        traits — dict mapping trait class name → trait instance
    """

    def __init__(
        self,
        name: str = "Solver",
        traits: Optional[List[Any]] = None,
    ) -> None:
        self.name: str = name
        self.traits: Dict[str, Any] = {}
        for trait in (traits or []):
            self.traits[type(trait).__name__] = trait

    def get_trait(self, trait_class: Type[T]) -> Optional[T]:
        """Return the registered trait instance for *trait_class*, or None."""
        return self.traits.get(trait_class.__name__)  # type: ignore[return-value]

    @abstractmethod
    def solve(self, grid: Any) -> Any:
        """Solve a task or grid.  Signature may vary by concrete subclass."""

    def __repr__(self) -> str:
        trait_names = list(self.traits.keys())
        return f"{self.__class__.__name__}(name={self.name!r}, traits={trait_names})"

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# GridUtils — shared static grid operations
# ---------------------------------------------------------------------------

class GridUtils:
    """Static helper methods for 2-D grid operations."""

    @staticmethod
    def find_background_color(grid: List[List[int]]) -> int:
        """Return the most-frequent colour in *grid* (background heuristic)."""
        counts: Counter = Counter()
        for row in grid:
            counts.update(row)
        return counts.most_common(1)[0][0] if counts else 0

    @staticmethod
    def find_bounding_box(
        grid: List[List[int]], bg_color: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Return (min_row, max_row, min_col, max_col) of all cells whose value
        differs from *bg_color*, or None if every cell is background.
        """
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        min_r = rows
        max_r = -1
        min_c = cols
        max_c = -1
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg_color:
                    if r < min_r:
                        min_r = r
                    if r > max_r:
                        max_r = r
                    if c < min_c:
                        min_c = c
                    if c > max_c:
                        max_c = c
        if max_r == -1:
            return None
        return (min_r, max_r, min_c, max_c)

    @staticmethod
    def extract_region(
        grid: List[List[int]],
        r_start: int,
        r_end: int,
        c_start: int,
        c_end: int,
    ) -> List[List[int]]:
        """
        Extract rows *r_start..r_end* and columns *c_start..c_end* (inclusive).
        """
        return [row[c_start: c_end + 1] for row in grid[r_start: r_end + 1]]

    @staticmethod
    def rotate_90(grid: List[List[int]]) -> List[List[int]]:
        """Rotate *grid* 90 degrees clockwise."""
        if not grid or not grid[0]:
            return grid
        rows, cols = len(grid), len(grid[0])
        return [[grid[rows - 1 - r][c] for r in range(rows)] for c in range(cols)]

    @staticmethod
    def rotate_180(grid: List[List[int]]) -> List[List[int]]:
        """Rotate *grid* 180 degrees."""
        return [row[::-1] for row in reversed(grid)]

    @staticmethod
    def rotate_270(grid: List[List[int]]) -> List[List[int]]:
        """Rotate *grid* 270 degrees clockwise (= 90 counter-clockwise)."""
        if not grid or not grid[0]:
            return grid
        rows, cols = len(grid), len(grid[0])
        return [[grid[r][cols - 1 - c] for r in range(rows)] for c in range(cols)]

    @staticmethod
    def flip_horizontal(grid: List[List[int]]) -> List[List[int]]:
        """Flip *grid* left-to-right."""
        return [row[::-1] for row in grid]

    @staticmethod
    def flip_vertical(grid: List[List[int]]) -> List[List[int]]:
        """Flip *grid* top-to-bottom."""
        return list(reversed(grid))
