def solve(grid):
    """
    ARC-AGI Task 32616820: Remove corruption blocks and reconstruct clean repeating pattern.
    
    The rule: Large solid color blocks are corruption overlaying a repeating tile pattern.
    1. Detect the pattern period (6 or 7 rows)  
    2. Identify corruption as large connected regions (10+ cells)
    3. Reconstruct the clean pattern by removing corruption
    """
    height = len(grid)
    width = len(grid[0])
    
    # Find large connected regions (corruption)
    def find_connected_region(start_r, start_c, target_color, visited):
        if (start_r < 0 or start_r >= height or 
            start_c < 0 or start_c >= width or
            (start_r, start_c) in visited or
            grid[start_r][start_c] != target_color):
            return []
        
        visited.add((start_r, start_c))
        region = [(start_r, start_c)]
        
        # Check 4 directions
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            region.extend(find_connected_region(start_r + dr, start_c + dc, target_color, visited))
        
        return region
    
    # Identify corruption colors - colors that form large connected regions
    visited = set()
    corruption_colors = set()
    
    for r in range(height):
        for c in range(width):
            if (r, c) not in visited:
                color = grid[r][c]
                region = find_connected_region(r, c, color, visited)
                if len(region) >= 10:  # Large region threshold
                    corruption_colors.add(color)
    
    # Try both possible pattern periods
    best_result = None
    best_match_rate = 0
    
    for pattern_height in [6, 7]:
        # Build clean pattern by removing corruption
        pattern = []
        
        for pos in range(pattern_height):
            # Collect all rows at this position in the repeating cycle
            candidates = []
            for r in range(pos, height, pattern_height):
                candidates.append(grid[r])
            
            if not candidates:
                pattern.append([0] * width)
                continue
            
            # Reconstruct clean row
            clean_row = []
            for c in range(width):
                # Get all values at this column position across candidates
                values = [candidate[c] for candidate in candidates]
                
                # Filter out corruption colors
                clean_values = [v for v in values if v not in corruption_colors]
                
                if clean_values:
                    # Use most common clean value
                    from collections import Counter
                    most_common = Counter(clean_values).most_common(1)[0][0]
                    clean_row.append(most_common)
                else:
                    # All values corrupted - infer from horizontal pattern
                    inferred = 0
                    # Try to find a repeating horizontal pattern
                    for period in [6, 7]:
                        if c >= period and c - period < len(clean_row):
                            inferred = clean_row[c - period]
                            break
                    clean_row.append(inferred)
            
            pattern.append(clean_row)
        
        # Test how well this pattern matches the uncorrupted parts of input
        matches = 0
        total_uncorrupted = 0
        for r in range(height):
            expected_row = pattern[r % pattern_height]
            actual_row = grid[r]
            for c in range(width):
                if actual_row[c] not in corruption_colors:
                    total_uncorrupted += 1
                    if expected_row[c] == actual_row[c]:
                        matches += 1
        
        match_rate = matches / total_uncorrupted if total_uncorrupted > 0 else 0
        
        # Keep the best pattern found
        if match_rate > best_match_rate:
            best_match_rate = match_rate
            best_result = []
            for r in range(height):
                best_result.append(pattern[r % pattern_height][:])
    
    return best_result if best_result else grid