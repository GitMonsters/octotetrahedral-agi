"""
Perception Module — Parse frame observations into structured game state.

Extracts grid regions, objects, colors, spatial relationships, and tracks
changes between frames to build a world model.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GameObject:
    """A detected object in the game frame."""
    object_id: int
    color: int
    pixels: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]  # min_r, min_c, max_r, max_c
    size: int
    center: tuple[float, float]


@dataclass
class FrameAnalysis:
    """Structured analysis of a single frame."""
    frame: np.ndarray
    objects: list[GameObject]
    color_counts: dict[int, int]
    unique_colors: set[int]
    regions: dict[int, list[tuple[int, int]]]  # color -> pixel coords
    grid_hash: int
    changed_pixels: int = 0
    changed_regions: list[tuple[int, int, int, int]] = field(default_factory=list)


class PerceptionModule:
    """Extracts structured information from raw 64x64 game frames."""

    def __init__(self):
        self.prev_frame: Optional[np.ndarray] = None
        self.frame_count: int = 0
        self.background_color: Optional[int] = None

    def analyze(self, frame: np.ndarray) -> FrameAnalysis:
        """Analyze a frame and extract objects, regions, changes."""
        self.frame_count += 1

        color_counts = {}
        regions: dict[int, list[tuple[int, int]]] = {}
        for r in range(frame.shape[0]):
            for c in range(frame.shape[1]):
                val = int(frame[r, c])
                color_counts[val] = color_counts.get(val, 0) + 1
                if val not in regions:
                    regions[val] = []
                regions[val].append((r, c))

        # Detect background as most frequent color
        if self.background_color is None:
            self.background_color = max(color_counts, key=color_counts.get)

        # Extract objects via connected components (non-background)
        objects = self._extract_objects(frame)

        # Compute changes from previous frame
        changed_pixels = 0
        changed_regions = []
        if self.prev_frame is not None:
            diff_mask = frame != self.prev_frame
            changed_pixels = int(np.sum(diff_mask))
            if changed_pixels > 0:
                changed_regions = self._find_change_regions(diff_mask)

        grid_hash = hash(frame.tobytes())

        analysis = FrameAnalysis(
            frame=frame.copy(),
            objects=objects,
            color_counts=color_counts,
            unique_colors=set(color_counts.keys()),
            regions=regions,
            grid_hash=grid_hash,
            changed_pixels=changed_pixels,
            changed_regions=changed_regions,
        )

        self.prev_frame = frame.copy()
        return analysis

    def _extract_objects(self, frame: np.ndarray) -> list[GameObject]:
        """Extract connected components as objects (flood fill)."""
        visited = np.zeros_like(frame, dtype=bool)
        objects = []
        obj_id = 0

        for r in range(frame.shape[0]):
            for c in range(frame.shape[1]):
                color = int(frame[r, c])
                if visited[r, c] or color == self.background_color:
                    continue

                # BFS flood fill
                pixels = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    pixels.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < frame.shape[0] and 0 <= nc < frame.shape[1]
                                and not visited[nr, nc]
                                and int(frame[nr, nc]) == color):
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                if len(pixels) < 2:
                    continue

                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]
                bbox = (min(rows), min(cols), max(rows), max(cols))
                center = (sum(rows) / len(rows), sum(cols) / len(cols))

                objects.append(GameObject(
                    object_id=obj_id,
                    color=color,
                    pixels=pixels,
                    bbox=bbox,
                    size=len(pixels),
                    center=center,
                ))
                obj_id += 1

        return objects

    def _find_change_regions(self, diff_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Find bounding boxes of changed regions."""
        regions = []
        visited = np.zeros_like(diff_mask)

        rows, cols = np.where(diff_mask)
        if len(rows) == 0:
            return regions

        # Simple clustering: divide into grid sectors
        sector_size = 16
        sectors: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for r, c in zip(rows, cols):
            key = (r // sector_size, c // sector_size)
            if key not in sectors:
                sectors[key] = []
            sectors[key].append((int(r), int(c)))

        for pixels in sectors.values():
            rs = [p[0] for p in pixels]
            cs = [p[1] for p in pixels]
            regions.append((min(rs), min(cs), max(rs), max(cs)))

        return regions

    def get_movement_vector(self, analysis: FrameAnalysis) -> Optional[tuple[int, int]]:
        """Detect primary movement direction from frame changes."""
        if not analysis.changed_regions or self.prev_frame is None:
            return None

        # Compare object centers between frames
        if len(analysis.objects) > 0 and self.prev_frame is not None:
            prev_objects = self._extract_objects(self.prev_frame)
            if prev_objects and analysis.objects:
                # Match by color and size
                for obj in analysis.objects:
                    for prev_obj in prev_objects:
                        if obj.color == prev_obj.color and abs(obj.size - prev_obj.size) < 5:
                            dr = obj.center[0] - prev_obj.center[0]
                            dc = obj.center[1] - prev_obj.center[1]
                            if abs(dr) > 0.5 or abs(dc) > 0.5:
                                return (int(np.sign(dr)), int(np.sign(dc)))
        return None

    def reset(self):
        """Reset perception state for new game."""
        self.prev_frame = None
        self.frame_count = 0
        self.background_color = None
