def solve(grid):
    """
    ARC-AGI Task 32616820: Remove corruption blocks to reveal clean repeating pattern.
    
    Corrected algorithm:
    1. Try different vertical periods (6, 7) 
    2. For each period, reconstruct the clean pattern by voting
    3. Identify corruption as cells where the clean pattern differs from input
    4. Return the best reconstruction
    """
    from collections import Counter
    
    height = len(grid)
    width = len(grid[0])
    
    best_result = None
    best_score = -1
    
    # Test different vertical repeat periods
    for period in [6, 7]:
        # For each position in the pattern cycle, collect values and vote
        pattern = []
        
        for pattern_pos in range(period):
            pattern_row = []
            
            for col in range(width):
                # Collect all values at this pattern position and column
                values = []
                for row in range(pattern_pos, height, period):
                    values.append(grid[row][col])
                
                if values:
                    # Use majority vote
                    most_common = Counter(values).most_common(1)[0][0]
                    pattern_row.append(most_common)
                else:
                    pattern_row.append(0)
            
            pattern.append(pattern_row)
        
        # Reconstruct full grid using this pattern
        reconstructed = []
        for row in range(height):
            reconstructed.append(pattern[row % period][:])
        
        # Score this reconstruction by how well it matches input
        matches = 0
        total = 0
        for row in range(height):
            for col in range(width):
                total += 1
                if reconstructed[row][col] == grid[row][col]:
                    matches += 1
        
        score = matches / total if total > 0 else 0
        
        if score > best_score:
            best_score = score
            best_result = reconstructed
    
    return best_result if best_result else grid