"""
Geometric Patterns from Confucius SDK Implementation Data
Converts session metrics and architecture into visual geometric patterns
"""

import math
import json
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


# ============================================================================
# GEOMETRIC PATTERN GENERATION
# ============================================================================

@dataclass
class Point:
    """2D Point"""
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def rotate(self, angle: float, center: 'Point' = None) -> 'Point':
        """Rotate point around center"""
        if center is None:
            center = Point(0, 0)
        
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        x = self.x - center.x
        y = self.y - center.y
        
        new_x = x * cos_a - y * sin_a + center.x
        new_y = x * sin_a + y * cos_a + center.y
        
        return Point(new_x, new_y)
    
    def scale(self, factor: float, center: 'Point' = None) -> 'Point':
        """Scale point from center"""
        if center is None:
            center = Point(0, 0)
        
        return Point(
            center.x + (self.x - center.x) * factor,
            center.y + (self.y - center.y) * factor
        )
    
    def __str__(self) -> str:
        return f"({self.x:.1f}, {self.y:.1f})"


class GeometricPattern:
    """Base class for geometric patterns"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.points: List[Point] = []
        self.lines: List[Tuple[Point, Point]] = []
        self.polygons: List[List[Point]] = []
    
    def add_point(self, p: Point) -> None:
        """Add a point"""
        self.points.append(p)
    
    def add_line(self, p1: Point, p2: Point) -> None:
        """Add a line between two points"""
        self.lines.append((p1, p2))
    
    def add_polygon(self, points: List[Point]) -> None:
        """Add a polygon"""
        self.polygons.append(points)
    
    def to_svg(self, width: int = 800, height: int = 800) -> str:
        """Convert to SVG format"""
        svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
        svg += f'<title>{self.name}</title>\n'
        svg += f'<desc>{self.description}</desc>\n'
        
        # Background
        svg += f'<rect width="{width}" height="{height}" fill="#f8f9fa"/>\n'
        
        # Grid
        svg += self._add_grid(width, height)
        
        # Polygons (filled)
        for polygon in self.polygons:
            points_str = " ".join([f"{p.x},{p.y}" for p in polygon])
            svg += f'<polygon points="{points_str}" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>\n'
        
        # Lines
        for p1, p2 in self.lines:
            svg += f'<line x1="{p1.x}" y1="{p1.y}" x2="{p2.x}" y2="{p2.y}" stroke="#455a64" stroke-width="1.5"/>\n'
        
        # Points
        for p in self.points:
            svg += f'<circle cx="{p.x}" cy="{p.y}" r="4" fill="#d32f2f"/>\n'
        
        # Title
        svg += f'<text x="20" y="30" font-size="20" font-weight="bold" fill="#000">{self.name}</text>\n'
        
        svg += '</svg>\n'
        return svg
    
    def _add_grid(self, width: int, height: int) -> str:
        """Add grid background"""
        grid = ""
        step = 50
        
        # Vertical lines
        for x in range(0, width + 1, step):
            grid += f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" stroke="#eceff1" stroke-width="0.5"/>\n'
        
        # Horizontal lines
        for y in range(0, height + 1, step):
            grid += f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" stroke="#eceff1" stroke-width="0.5"/>\n'
        
        return grid
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pattern statistics"""
        if len(self.points) < 2:
            return {}
        
        distances = []
        for i in range(len(self.points)):
            for j in range(i + 1, len(self.points)):
                distances.append(self.points[i].distance_to(self.points[j]))
        
        return {
            "name": self.name,
            "point_count": len(self.points),
            "line_count": len(self.lines),
            "polygon_count": len(self.polygons),
            "distances": {
                "min": min(distances) if distances else 0,
                "max": max(distances) if distances else 0,
                "avg": sum(distances) / len(distances) if distances else 0,
            }
        }


# ============================================================================
# SESSION METRICS PATTERNS
# ============================================================================

class PhaseCompletionHexagon(GeometricPattern):
    """8 phases as hexagonal pattern"""
    
    def __init__(self):
        super().__init__(
            "Phase Completion Hexagon",
            "8 Phases of Confucius SDK arranged in hexagonal geometry"
        )
        
        center = Point(400, 400)
        radius = 200
        phases = [
            "Memory", "Notes", "Orchestrator", "Meta-Agent",
            "Extensions", "Integration", "Production", "Advanced"
        ]
        
        # Create hexagon vertices
        for i, phase in enumerate(phases):
            angle = (i / 8) * 2 * math.pi
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            
            point = Point(x, y)
            self.add_point(point)
        
        # Connect all points
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]
            self.add_line(p1, p2)
        
        # Add radial lines from center
        for point in self.points:
            self.add_line(center, point)
        
        # Create polygon from all points
        self.add_polygon(self.points)
        
        # Add center point
        self.add_point(center)


class TestCoveragePyramid(GeometricPattern):
    """Test coverage as pyramid structure"""
    
    def __init__(self):
        super().__init__(
            "Test Coverage Pyramid",
            "140+ tests distributed across 8 phases in pyramid structure"
        )
        
        # Test distribution by phase
        test_counts = [7, 8, 50, 40, 35, 11, 0, 0]  # Phase 7&8 embedded in system
        
        base_y = 700
        base_x = 400
        base_width = 700
        
        # Create pyramid layers
        for level in range(len(test_counts)):
            width = base_width * (1 - level * 0.11)
            height = 60
            
            y = base_y - (level * 70)
            left_x = base_x - width / 2
            right_x = base_x + width / 2
            
            # Create rectangle for this level
            tl = Point(left_x, y)
            tr = Point(right_x, y)
            br = Point(right_x, y + height)
            bl = Point(left_x, y + height)
            
            self.add_polygon([tl, tr, br, bl])
            
            # Add test count label points
            mid_point = Point(base_x, y + height / 2)
            self.add_point(mid_point)


class PerformanceMetricsSpiral(GeometricPattern):
    """Performance metrics as logarithmic spiral"""
    
    def __init__(self):
        super().__init__(
            "Performance Metrics Spiral",
            "System performance metrics visualized as logarithmic spiral"
        )
        
        center = Point(400, 400)
        
        # Metrics in order with times
        metrics = [
            ("Memory Composition", 0.04),
            ("Pattern Search", 0.01),
            ("Extension Execution", 1.0),
            ("Health Check", 24.7),
            ("Orchestration Iteration", 120.39),
        ]
        
        # Create spiral using logarithmic growth
        num_points = 100
        max_time = max(m[1] for m in metrics)
        
        for i in range(num_points):
            t = (i / num_points) * 4 * math.pi
            r = 50 + (i / num_points) * 200
            
            x = center.x + r * math.cos(t)
            y = center.y + r * math.sin(t)
            
            self.add_point(Point(x, y))
        
        # Connect spiral
        for i in range(len(self.points) - 1):
            self.add_line(self.points[i], self.points[i + 1])


class CodeStructureTreeOfLife(GeometricPattern):
    """Code structure as Tree of Life (Kabbalah)"""
    
    def __init__(self):
        super().__init__(
            "Code Structure - Tree of Life",
            "SDK architecture mapped to Tree of Life geometric pattern"
        )
        
        # Tree of Life Sephiroth positions (simplified)
        sephiroth = {
            "1": Point(400, 50),      # Crown - SDK
            "2": Point(300, 150),     # Wisdom - Memory
            "3": Point(500, 150),     # Understanding - Notes
            "4": Point(200, 250),     # Mercy - Orchestrator
            "5": Point(600, 250),     # Severity - Extensions
            "6": Point(400, 350),     # Beauty - Meta-Agent
            "7": Point(200, 450),     # Victory - Production
            "8": Point(600, 450),     # Splendor - Advanced
            "9": Point(400, 550),     # Foundation - Integration
            "10": Point(400, 650),    # Kingdom - Deployment
        }
        
        for name, point in sephiroth.items():
            self.add_point(point)
        
        # Traditional paths of Tree of Life
        paths = [
            ("1", "2"), ("1", "3"), ("2", "3"), ("2", "4"), ("3", "5"),
            ("4", "6"), ("5", "6"), ("4", "7"), ("6", "8"), ("6", "9"),
            ("7", "9"), ("8", "9"), ("9", "10"),
        ]
        
        for p1_id, p2_id in paths:
            self.add_line(sephiroth[p1_id], sephiroth[p2_id])
        
        # Create circles around key nodes
        for point in [sephiroth["1"], sephiroth["6"], sephiroth["10"]]:
            # Approximate circle with polygon
            circle_points = []
            for i in range(12):
                angle = (i / 12) * 2 * math.pi
                x = point.x + 30 * math.cos(angle)
                y = point.y + 30 * math.sin(angle)
                circle_points.append(Point(x, y))
            self.add_polygon(circle_points)


class ComponentInteractionMandala(GeometricPattern):
    """Component interactions as mandala pattern"""
    
    def __init__(self):
        super().__init__(
            "Component Interaction Mandala",
            "5 core components with 8-fold symmetry interaction pattern"
        )
        
        center = Point(400, 400)
        components = ["Memory", "Notes", "Orchestrator", "Meta-Agent", "Extensions"]
        
        # Create outer ring (8 symmetry)
        outer_radius = 250
        for i in range(8):
            angle = (i / 8) * 2 * math.pi
            x = center.x + outer_radius * math.cos(angle)
            y = center.y + outer_radius * math.sin(angle)
            self.add_point(Point(x, y))
        
        # Create inner ring (5 components)
        inner_radius = 120
        for i, comp in enumerate(components):
            angle = (i / 5) * 2 * math.pi
            x = center.x + inner_radius * math.cos(angle)
            y = center.y + inner_radius * math.sin(angle)
            self.add_point(Point(x, y))
        
        # Connect outer to inner
        for outer_point in self.points[:8]:
            for inner_point in self.points[8:]:
                dist = outer_point.distance_to(inner_point)
                if dist < 300 and dist > 150:
                    self.add_line(outer_point, inner_point)
        
        # Create inner mandala rings
        for radius in [30, 60, 90]:
            ring_points = []
            for i in range(16):
                angle = (i / 16) * 2 * math.pi
                x = center.x + radius * math.cos(angle)
                y = center.y + radius * math.sin(angle)
                ring_points.append(Point(x, y))
            self.add_polygon(ring_points)
        
        self.add_point(center)


class DataFlowSacredGeometry(GeometricPattern):
    """Data flow as sacred geometry (Flower of Life)"""
    
    def __init__(self):
        super().__init__(
            "Data Flow - Flower of Life",
            "Memory -> Notes -> Orchestrator -> Extensions flow in sacred geometry"
        )
        
        center = Point(400, 400)
        circle_radius = 80
        
        # Create overlapping circles in flower pattern
        circles = []
        for i in range(6):
            angle = (i / 6) * 2 * math.pi
            cx = center.x + circle_radius * math.cos(angle)
            cy = center.y + circle_radius * math.sin(angle)
            circles.append(Point(cx, cy))
        
        # Center circle
        circles.insert(0, center)
        
        # Draw circles as polygons
        for center_point in circles:
            circle_points = []
            for i in range(12):
                angle = (i / 12) * 2 * math.pi
                x = center_point.x + circle_radius * math.cos(angle)
                y = center_point.y + circle_radius * math.sin(angle)
                circle_points.append(Point(x, y))
            self.add_polygon(circle_points)
        
        # Add center points
        for point in circles:
            self.add_point(point)


class ArchitectureGoldenRectangle(GeometricPattern):
    """Architecture layout using golden rectangle divisions"""
    
    def __init__(self):
        super().__init__(
            "Architecture - Golden Rectangle",
            "SDK architecture divided by golden ratio rectangles"
        )
        
        # Golden ratio
        phi = (1 + math.sqrt(5)) / 2
        
        # Start with full rectangle
        w, h = 700, 700
        x, y = 50, 50
        
        # Create golden rectangle subdivisions
        rectangles = []
        current_x, current_y = x, y
        current_w, current_h = w, h
        
        for i in range(5):
            # Create rectangle
            rect = [
                Point(current_x, current_y),
                Point(current_x + current_w, current_y),
                Point(current_x + current_w, current_y + current_h),
                Point(current_x, current_y + current_h),
            ]
            self.add_polygon(rect)
            rectangles.append(rect)
            
            # Subdivide using golden ratio
            if current_w > current_h:
                # Divide width
                new_w = current_w / phi
                current_x += (current_w - new_w)
                current_w = new_w
            else:
                # Divide height
                new_h = current_h / phi
                current_y += (current_h - new_h)
                current_h = new_h
        
        # Add key points at rectangle intersections
        for rect in rectangles:
            for point in rect:
                self.add_point(point)


class FileStructureGeometricTree(GeometricPattern):
    """File structure as geometric tree"""
    
    def __init__(self):
        super().__init__(
            "File Structure - Geometric Tree",
            "7 core files arranged in tree topology"
        )
        
        # Root
        root = Point(400, 50)
        self.add_point(root)
        
        # Layer 1: Core components
        layer1 = [
            ("Memory", 250),
            ("Notes", 400),
            ("Orchestrator", 550),
        ]
        
        layer1_points = []
        for name, x in layer1:
            p = Point(x, 150)
            self.add_point(p)
            layer1_points.append(p)
            self.add_line(root, p)
        
        # Layer 2: Support components
        layer2 = [
            ("Meta-Agent", 150),
            ("Extensions", 400),
            ("Production", 650),
        ]
        
        layer2_points = []
        y = 250
        for i, (name, x) in enumerate(layer2):
            p = Point(x, y)
            self.add_point(p)
            layer2_points.append(p)
            # Connect to nearest layer1 point
            self.add_line(layer1_points[i % len(layer1_points)], p)
        
        # Layer 3: Advanced
        layer3 = [
            ("Advanced Extensions", 400),
        ]
        
        y = 350
        for name, x in layer3:
            p = Point(x, y)
            self.add_point(p)
            # Connect to layer2 points
            for p2 in layer2_points:
                if abs(p.x - p2.x) < 200:
                    self.add_line(p2, p)
        
        # Create connecting rings
        for layer in [layer1_points, layer2_points]:
            ring_points = []
            for i in range(8):
                angle = (i / 8) * 2 * math.pi
                for point in layer:
                    x = point.x + 40 * math.cos(angle)
                    y = point.y + 40 * math.sin(angle)
                    ring_points.append(Point(x, y))


# ============================================================================
# METRIC PATTERNS
# ============================================================================

class TestDistributionFractal(GeometricPattern):
    """Test distribution as fractal pattern"""
    
    def __init__(self):
        super().__init__(
            "Test Distribution Fractal",
            "140+ tests distributed in self-similar fractal pattern"
        )
        
        # Test counts per phase
        tests = [7, 8, 50, 40, 35, 11, 12, 12]
        colors_by_phase = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", 
                          "#9b59b6", "#1abc9c", "#34495e", "#e91e63"]
        
        # Create Sierpinski triangle-like pattern
        self._draw_fractal_triangles(Point(400, 100), 300, tests, 0)
    
    def _draw_fractal_triangles(self, apex: Point, size: float, 
                               tests: List[int], depth: int, max_depth: int = 3):
        """Recursively draw fractal triangles"""
        if depth > max_depth or size < 20:
            return
        
        # Create equilateral triangle
        angle1 = -math.pi / 2
        angle2 = angle1 + 2 * math.pi / 3
        angle3 = angle1 + 4 * math.pi / 3
        
        p1 = Point(apex.x + size * math.cos(angle1), 
                  apex.y + size * math.sin(angle1))
        p2 = Point(apex.x + size * math.cos(angle2), 
                  apex.y + size * math.sin(angle2))
        p3 = Point(apex.x + size * math.cos(angle3), 
                  apex.y + size * math.sin(angle3))
        
        self.add_polygon([p1, p2, p3])
        
        # Recursive calls
        if depth < max_depth:
            new_size = size / 2
            self._draw_fractal_triangles(p1, new_size, tests, depth + 1, max_depth)
            self._draw_fractal_triangles(p2, new_size, tests, depth + 1, max_depth)
            self._draw_fractal_triangles(p3, new_size, tests, depth + 1, max_depth)


# ============================================================================
# GENERATION & EXPORT
# ============================================================================

def generate_all_patterns() -> List[GeometricPattern]:
    """Generate all geometric patterns"""
    return [
        PhaseCompletionHexagon(),
        TestCoveragePyramid(),
        PerformanceMetricsSpiral(),
        CodeStructureTreeOfLife(),
        ComponentInteractionMandala(),
        DataFlowSacredGeometry(),
        ArchitectureGoldenRectangle(),
        FileStructureGeometricTree(),
        TestDistributionFractal(),
    ]


def export_patterns_to_html(patterns: List[GeometricPattern], 
                           output_file: str = "geometric_patterns.html") -> None:
    """Export all patterns to single HTML file"""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Confucius SDK - Geometric Patterns</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { 
            color: white; 
            text-align: center; 
            margin-bottom: 40px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .patterns-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(850px, 1fr));
            gap: 40px;
            margin-bottom: 40px;
        }
        .pattern-card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            transition: transform 0.3s ease;
        }
        .pattern-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0,0,0,0.4);
        }
        .pattern-card h2 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.5em;
        }
        .pattern-card p {
            color: #666;
            margin-bottom: 20px;
            font-size: 0.95em;
        }
        svg {
            border: 1px solid #eee;
            border-radius: 8px;
            display: block;
            margin: 0 auto;
        }
        .stats {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            color: #666;
            font-size: 0.9em;
        }
        .stat-item {
            display: inline-block;
            margin-right: 20px;
            margin-top: 10px;
        }
        .stat-label {
            font-weight: bold;
            color: #667eea;
        }
        .footer {
            text-align: center;
            color: white;
            margin-top: 40px;
            padding: 20px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Confucius SDK - Geometric Patterns</h1>
        <div class="patterns-grid">
"""
    
    for pattern in patterns:
        stats = pattern.get_statistics()
        
        html += f"""
        <div class="pattern-card">
            <h2>{pattern.name}</h2>
            <p>{pattern.description}</p>
            {pattern.to_svg()}
            <div class="stats">
"""
        
        if stats:
            html += f"""
                <div class="stat-item">
                    <span class="stat-label">Points:</span> {stats.get('point_count', 0)}
                </div>
                <div class="stat-item">
                    <span class="stat-label">Lines:</span> {stats.get('line_count', 0)}
                </div>
                <div class="stat-item">
                    <span class="stat-label">Polygons:</span> {stats.get('polygon_count', 0)}
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    html += """
        </div>
        <div class="footer">
            <p>Generated from Confucius SDK Implementation Session</p>
            <p>Mathematical patterns extracted from 8-phase architecture, 140+ tests, and ~5,500 lines of code</p>
            <p>January 29, 2026 - Production Ready System</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✓ Patterns exported to {output_file}")


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GEOMETRIC PATTERNS FROM CONFUCIUS SDK SESSION")
    print("="*80)
    
    patterns = generate_all_patterns()
    
    print(f"\nGenerated {len(patterns)} geometric patterns:\n")
    
    for i, pattern in enumerate(patterns, 1):
        stats = pattern.get_statistics()
        print(f"{i}. {pattern.name}")
        print(f"   {pattern.description}")
        if stats:
            print(f"   Points: {stats.get('point_count', 0)}, "
                  f"Lines: {stats.get('line_count', 0)}, "
                  f"Polygons: {stats.get('polygon_count', 0)}")
        print()
    
    print("Exporting patterns to HTML...")
    export_patterns_to_html(patterns)
    
    print("\n" + "="*80)
    print("✓ GEOMETRIC PATTERNS COMPLETE")
    print("="*80)
    print("\nView the patterns: open geometric_patterns.html in your browser")
    print("\nPatterns generated from:")
    print("  • 8 phases of SDK development")
    print("  • 140+ test cases")
    print("  • ~5,500 lines of production code")
    print("  • 5 core components with 8 lifecycle phases")
    print("  • Multiple mathematical structures and ratios")
    print("\n" + "="*80)
