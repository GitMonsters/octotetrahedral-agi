import json

def analyze_symmetry_creation():
    """Analyze if the pattern creates 4-fold symmetry"""
    task = json.load(open('apr12_tasks/124b77f2.json'))
    
    def create_4fold_symmetric(inp):
        """Create 4-fold symmetric version by reflecting all non-bg cells"""
        bg = inp[0][0]
        rows, cols = len(inp), len(inp[0])
        result = [row[:] for row in inp]  # Copy input
        
        # For each non-background cell, create all 3 reflections
        changes_made = []
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] != bg:
                    val = inp[r][c]
                    
                    # Horizontal reflection (across vertical center)
                    hc = cols - 1 - c
                    if 0 <= r < rows and 0 <= hc < cols and result[r][hc] == bg:
                        result[r][hc] = val
                        changes_made.append(f"H: ({r},{c}){val} -> ({r},{hc})")
                    
                    # Vertical reflection (across horizontal center)  
                    vr = rows - 1 - r
                    if 0 <= vr < rows and 0 <= c < cols and result[vr][c] == bg:
                        result[vr][c] = val
                        changes_made.append(f"V: ({r},{c}){val} -> ({vr},{c})")
                        
                    # Both reflections (180 degree rotation)
                    br = rows - 1 - r
                    bc = cols - 1 - c  
                    if 0 <= br < rows and 0 <= bc < cols and result[br][bc] == bg:
                        result[br][bc] = val
                        changes_made.append(f"B: ({r},{c}){val} -> ({br},{bc})")
        
        return result, changes_made
    
    # Test this hypothesis on first pair
    pair = task['train'][0]
    inp, out = pair['input'], pair['output']
    predicted, changes = create_4fold_symmetric(inp)
    
    # Compare prediction with actual output
    matches = 0
    total = 0
    differences = []
    
    for r in range(len(inp)):
        for c in range(len(inp[0])):
            total += 1
            if predicted[r][c] == out[r][c]:
                matches += 1
            else:
                differences.append((r, c, predicted[r][c], out[r][c]))
    
    print(f"4-fold symmetry hypothesis: {matches}/{total} matches")
    print(f"Changes made: {len(changes)}")
    print("Sample changes:", changes[:10])
    print(f"Differences: {len(differences)}")
    print("Sample differences:", differences[:10])
    
    return predicted, out

if __name__ == "__main__":
    analyze_symmetry_creation()