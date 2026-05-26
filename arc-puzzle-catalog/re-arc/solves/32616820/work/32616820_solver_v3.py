def solve(grid):
    """
    ARC-AGI Task 32616820: Remove corruption blocks and reconstruct clean repeating pattern.
    
    Refined approach:
    1. Test both pattern periods (6 and 7)
    2. For each period, identify corruption as colors that form rectangular blocks
    3. Exclude background color (0) from corruption detection
    4. Reconstruct clean pattern by removing only actual corruption
    """
    height = len(grid)
    width = len(grid[0])
    
    best_result = None
    best_match_rate = 0
    
    for pattern_height in [6, 7]:
        # Build pattern by cleaning corruption for this period
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
                
                # Strategy: Remove values that appear in long consecutive runs
                # This targets rectangular corruption blocks
                clean_values = []
                for val in values:
                    is_corruption = False
                    # Check if this value forms long horizontal runs in any candidate
                    for candidate in candidates:
                        max_run = 0
                        current_run = 0
                        for cell_val in candidate:
                            if cell_val == val:
                                current_run += 1
                                max_run = max(max_run, current_run)
                            else:
                                current_run = 0
                        
                        # Values forming runs of 5+ cells are likely corruption
                        # But exclude color 0 (background) from this rule
                        if max_run >= 5 and val != 0:
                            is_corruption = True
                            break
                    
                    if not is_corruption:
                        clean_values.append(val)
                
                if clean_values:
                    # Use most common clean value
                    from collections import Counter
                    most_common = Counter(clean_values).most_common(1)[0][0]
                    clean_row.append(most_common)
                else:
                    # All values corrupted - infer from pattern
                    inferred = 0
                    # Try to find horizontal repetition
                    for period in [6, 7]:
                        if c >= period and c - period < len(clean_row):
                            inferred = clean_row[c - period]
                            break
                    clean_row.append(inferred)
            
            pattern.append(clean_row)
        
        # Test pattern by checking repeating structure 
        pattern_match_score = 0
        total_comparisons = 0
        
        # Check vertical repetition
        for r in range(height - pattern_height):
            expected_row = pattern[r % pattern_height]
            next_cycle_row = pattern[(r + pattern_height) % pattern_height]
            for c in range(width):
                total_comparisons += 1
                if expected_row[c] == next_cycle_row[c]:
                    pattern_match_score += 1
        
        match_rate = pattern_match_score / total_comparisons if total_comparisons > 0 else 0
        
        # Keep the best pattern found
        if match_rate > best_match_rate:
            best_match_rate = match_rate
            best_result = []
            for r in range(height):
                best_result.append(pattern[r % pattern_height][:])
    
    return best_result if best_result else grid