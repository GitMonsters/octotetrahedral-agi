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

    def detect_shapes(self, frame: np.ndarray) -> dict:
        """Detect rectangular, diamond, and circular shapes in frame."""
        shapes = {'rectangles': [], 'diamonds': [], 'sparse': []}
        visited = np.zeros_like(frame, dtype=bool)
        bg = max(np.unique(frame, return_counts=True)[1]) if frame.size > 0 else 0
        bg_color = np.unique(frame)[np.argmax(np.unique(frame, return_counts=True)[1])]
        
        for r in range(frame.shape[0]):
            for c in range(frame.shape[1]):
                if visited[r, c] or frame[r, c] == bg_color:
                    continue
                
                color = int(frame[r, c])
                pixels = []
                queue = [(r, c)]
                visited[r, c] = True
                
                while queue:
                    cr, cc = queue.pop(0)
                    pixels.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < frame.shape[0] and 0 <= nc < frame.shape[1]
                                and not visited[nr, nc] and frame[nr, nc] == color):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                if len(pixels) < 2:
                    continue
                
                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]
                min_r, max_r = min(rows), max(rows)
                min_c, max_c = min(cols), max(cols)
                height = max_r - min_r + 1
                width = max_c - min_c + 1
                fill_ratio = len(pixels) / (height * width) if height * width > 0 else 0
                
                shape_info = {
                    'color': color,
                    'bbox': (min_r, min_c, max_r, max_c),
                    'size': len(pixels),
                    'height': height,
                    'width': width,
                    'fill_ratio': fill_ratio,
                    'aspect_ratio': width / height if height > 0 else 1.0,
                }
                
                if fill_ratio > 0.8:
                    shapes['rectangles'].append(shape_info)
                elif fill_ratio > 0.5:
                    shapes['diamonds'].append(shape_info)
                else:
                    shapes['sparse'].append(shape_info)
        
        return shapes

    def detect_symmetry(self, frame: np.ndarray) -> dict:
        """Detect rotational and reflectional symmetry."""
        symmetry = {
            'rotational_90': False,
            'rotational_180': False,
            'rotational_270': False,
            'horizontal_flip': False,
            'vertical_flip': False,
            'scores': {}
        }
        
        h, w = frame.shape
        
        # Check 180° rotational symmetry
        rot180 = np.rot90(frame, 2)
        sym180_score = np.sum(frame == rot180) / frame.size if frame.size > 0 else 0
        symmetry['scores']['rotational_180'] = float(sym180_score)
        if sym180_score > 0.95:
            symmetry['rotational_180'] = True
        
        # Check horizontal flip
        h_flip = np.fliplr(frame)
        h_sym_score = np.sum(frame == h_flip) / frame.size if frame.size > 0 else 0
        symmetry['scores']['horizontal_flip'] = float(h_sym_score)
        if h_sym_score > 0.95:
            symmetry['horizontal_flip'] = True
        
        # Check vertical flip
        v_flip = np.flipud(frame)
        v_sym_score = np.sum(frame == v_flip) / frame.size if frame.size > 0 else 0
        symmetry['scores']['vertical_flip'] = float(v_sym_score)
        if v_sym_score > 0.95:
            symmetry['vertical_flip'] = True
        
        return symmetry

    def cluster_colors(self, frame: np.ndarray) -> dict:
        """Analyze color distribution and clustering."""
        unique_colors, counts = np.unique(frame, return_counts=True)
        total_pixels = frame.size
        
        clusters = {}
        for color, count in zip(unique_colors, counts):
            fraction = count / total_pixels if total_pixels > 0 else 0
            clusters[int(color)] = {
                'count': int(count),
                'fraction': float(fraction),
                'percentage': float(fraction * 100),
            }
        
        sorted_by_freq = sorted(clusters.items(), key=lambda x: x[1]['count'], reverse=True)
        
        return {
            'color_clusters': clusters,
            'dominant_color': sorted_by_freq[0][0] if sorted_by_freq else None,
            'dominant_fraction': sorted_by_freq[0][1]['fraction'] if sorted_by_freq else 0,
            'unique_count': len(unique_colors),
            'sorted_by_frequency': sorted_by_freq,
        }

    def analyze_scaling(self, frame1: np.ndarray, frame2: np.ndarray) -> dict:
        """Detect scaling relationship between two grids."""
        h1, w1 = frame1.shape
        h2, w2 = frame2.shape
        
        scale_factors = []
        if h2 % h1 == 0 and w2 % w1 == 0:
            sy = h2 // h1
            sx = w2 // w1
            if sy == sx:
                scale_factors.append(sy)
                # Verify by upscaling frame1
                expanded = np.repeat(np.repeat(frame1, sy, axis=0), sx, axis=1)
                if expanded.shape == frame2.shape:
                    match = np.sum(expanded == frame2) / frame2.size if frame2.size > 0 else 0
                    if match > 0.9:
                        return {
                            'scaled': True,
                            'scale_factor': int(sy),
                            'match_ratio': float(match),
                        }
        
        return {
            'scaled': False,
            'scale_factor': None,
            'match_ratio': 0.0,
        }

    def measure_connectivity(self, frame: np.ndarray) -> dict:
        """Measure connectivity patterns in the grid."""
        bg = np.unique(frame, return_counts=True)[0][np.argmax(np.unique(frame, return_counts=True)[1])]
        
        # Count objects and their sizes
        visited = np.zeros_like(frame, dtype=bool)
        object_sizes = []
        
        for r in range(frame.shape[0]):
            for c in range(frame.shape[1]):
                if visited[r, c] or frame[r, c] == bg:
                    continue
                
                size = 0
                queue = [(r, c)]
                visited[r, c] = True
                
                while queue:
                    cr, cc = queue.pop(0)
                    size += 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < frame.shape[0] and 0 <= nc < frame.shape[1]
                                and not visited[nr, nc] and frame[nr, nc] != bg):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                if size > 1:
                    object_sizes.append(size)
        
        total_non_bg = np.sum(frame != bg)
        
        return {
            'object_count': len(object_sizes),
            'object_sizes': object_sizes,
            'avg_object_size': float(np.mean(object_sizes)) if object_sizes else 0.0,
            'max_object_size': int(max(object_sizes)) if object_sizes else 0,
            'min_object_size': int(min(object_sizes)) if object_sizes else 0,
            'fragmentation': float(len(object_sizes) / (total_non_bg / 10 + 1)) if total_non_bg > 0 else 0.0,
            'is_connected': len(object_sizes) == 1,
        }

    def reset(self):
        """Reset perception state for new game."""
        self.prev_frame = None
        self.frame_count = 0
        self.background_color = None
