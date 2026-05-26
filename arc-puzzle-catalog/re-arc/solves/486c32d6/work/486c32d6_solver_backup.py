def transform(grid):
    """
    ARC-AGI Task 486c32d6: Pattern Replication
    
    Rule: For repeating patterns, propagate minority anomalous values horizontally
    across all pattern cycles when there's a clear majority.
    """
    from collections import Counter
    
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    
    for r in range(rows):
        row = grid[r]
        
        # Skip uniform rows (separator lines)
        if len(set(row)) <= 1:
            continue
            
        # Try pattern lengths in order of preference
        for pattern_len in [4, 3, 5]:
            if pattern_len * 2 > cols:
                continue
                
            complete_cycles = cols // pattern_len
            if complete_cycles < 2:
                continue
            
            # Check if this pattern length is valid
            valid_pattern = True
            any_anomalies = False
            
            for pattern_pos in range(pattern_len):
                values = []
                for cycle in range(complete_cycles):
                    pos = cycle * pattern_len + pattern_pos
                    if pos < cols:
                        values.append(row[pos])
                
                counter = Counter(values)
                
                if len(counter) > 1:  # Has anomalies
                    any_anomalies = True
                    # This is still a valid pattern if there's a clear majority
                    sorted_counts = counter.most_common()
                    if sorted_counts[0][1] <= sorted_counts[1][1]:  # No clear majority
                        valid_pattern = False
                        break
            
            if valid_pattern:
                # Apply transformations if there are any anomalies
                if any_anomalies:
                    for pattern_pos in range(pattern_len):
                        values = []
                        for cycle in range(complete_cycles):
                            pos = cycle * pattern_len + pattern_pos
                            if pos < cols:
                                values.append(row[pos])
                        
                        counter = Counter(values)
                        
                        if len(counter) > 1:  # Has anomalies
                            sorted_counts = counter.most_common()
                            most_common_val, most_common_count = sorted_counts[0]
                            
                            # Propagate minority values
                            for val, count in sorted_counts[1:]:
                                if count < most_common_count:
                                    # Propagate this minority value to ALL positions
                                    for cycle in range(complete_cycles):
                                        pos = cycle * pattern_len + pattern_pos
                                        if pos < cols:
                                            result[r][pos] = val
                                    break
                
                break  # Stop trying other pattern lengths - we found a valid one
    
    return result

solve = transform