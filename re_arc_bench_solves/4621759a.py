from collections import Counter

def transform(grid):
    """For each row, fill with the most common color in that row."""
    result = []
    for row in grid:
        counter = Counter(row)
        dominant_color = counter.most_common(1)[0][0]
        result.append([dominant_color] * len(row))
    return result
