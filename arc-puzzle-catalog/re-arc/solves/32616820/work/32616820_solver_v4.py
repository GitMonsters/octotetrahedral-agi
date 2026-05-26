def solve(grid):
    """
    ARC-AGI Task 32616820: Remove single corruption color and reconstruct clean repeating pattern.
    
    Key insights:
    1. The corruption is always a single color that forms large blocks
    2. Pattern periods are either 6 or 7 rows
    3. The correct value at each position is the non-corruption value that appears
    """
    height = len(grid)
    width = len(grid[0])
    
    best_result = None
    best_score = 0
    
    # Try both pattern periods
    for pattern_height in [6, 7]:
        # For each possible corruption color, test the reconstruction
        all_colors = set()
        for r in range(height):
            for c in range(width):
                all_colors.add(grid[r][c])
        
        for corruption_color in all_colors:
            pattern = []
            
            # Build pattern for this corruption color assumption
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
                    
                    # Filter out the corruption color
                    clean_values = [v for v in values if v != corruption_color]
                    
                    if clean_values:
                        # Use the most common clean value
                        from collections import Counter
                        most_common = Counter(clean_values).most_common(1)[0][0]
                        clean_row.append(most_common)
                    else:
                        # All values are corruption - try to infer from horizontal pattern
                        inferred = 0
                        for period in [6, 7]:
                            if c >= period and c - period < len(clean_row):
                                inferred = clean_row[c - period]
                                break
                        clean_row.append(inferred)
                
                pattern.append(clean_row)
            
            # Score this pattern by checking perfect repetition
            perfect_matches = 0
            total_checks = 0
            
            for r in range(height - pattern_height):
                pattern_row = pattern[r % pattern_height]
                next_cycle_row = pattern[(r + pattern_height) % pattern_height]
                for c in range(width):
                    total_checks += 1
                    if pattern_row[c] == next_cycle_row[c]:
                        perfect_matches += 1
            
            score = perfect_matches / total_checks if total_checks > 0 else 0
            
            # Keep the best pattern
            if score > best_score:
                best_score = score
                best_result = []
                for r in range(height):
                    best_result.append(pattern[r % pattern_height][:])
    
    return best_result if best_result else grid