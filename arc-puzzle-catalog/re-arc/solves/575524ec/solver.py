from collections import Counter


def _extract_segments(grid):
    h, w = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    segments = []
    for x in range(w):
        y = 0
        while y < h:
            if grid[y][x] != bg:
                color = grid[y][x]
                y0 = y
                while y + 1 < h and grid[y + 1][x] == color:
                    y += 1
                segments.append({"x": x, "y0": y0, "y1": y, "h": y - y0 + 1})
            y += 1
    return bg, segments


def transform(grid):
    bg, segments = _extract_segments(grid)
    out = [row[:] for row in grid]
    heights = sorted({seg["h"] for seg in segments}, reverse=True)
    color_for_height = {}
    for i, height in enumerate(heights):
        color_for_height[height] = 4 if i < 2 else 6 if i == 2 else 1

    for seg in segments:
        new_color = color_for_height[seg["h"]]
        for y in range(seg["y0"], seg["y1"] + 1):
            out[y][seg["x"]] = new_color

    if len(heights) >= 4:
        third_height = heights[2]
        shortest_height = heights[3]
        third_segments = [seg for seg in segments if seg["h"] == third_height]
        shortest_segments = [seg for seg in segments if seg["h"] == shortest_height]

        if len(third_segments) == 1 and len(shortest_segments) >= 2:
            seg = third_segments[0]
            others = [other for other in segments if other is not seg]
            if others:
                nearest = min(others, key=lambda other: abs(other["x"] - seg["x"]))
                dx = -1 if nearest["x"] < seg["x"] else 1
                x = seg["x"] + dx
                if 0 <= x < len(grid[0]) and all(out[y][x] == bg for y in range(seg["y0"], seg["y1"] + 1)):
                    y0, y1 = Counter((s["y0"], s["y1"]) for s in shortest_segments).most_common(1)[0][0]
                    for y in range(y0, y1 + 1):
                        out[y][x] = color_for_height[shortest_height]

        if len(third_segments) == 1 and len(shortest_segments) == 1:
            third = third_segments[0]
            shortest = shortest_segments[0]
            supporting = [
                seg
                for seg in segments
                if color_for_height[seg["h"]] == 4 and seg["y0"] <= third["y0"] and seg["y1"] >= third["y1"]
            ]
            right = sorted((seg for seg in supporting if seg["x"] > third["x"]), key=lambda seg: seg["x"])
            if right:
                base = right[0]
                offset = third["x"] - base["x"]
                if abs(offset) <= 3:
                    for target in right[1:]:
                        x = target["x"] + offset
                        if 0 <= x < len(grid[0]) and all(out[y][x] == bg for y in range(third["y0"], third["y1"] + 1)):
                            for y in range(third["y0"], third["y1"] + 1):
                                out[y][x] = color_for_height[third_height]
            x = third["x"] - 1
            if 0 <= x < len(grid[0]) and all(out[y][x] == bg for y in range(shortest["y0"], shortest["y1"] + 1)):
                for y in range(shortest["y0"], shortest["y1"] + 1):
                    out[y][x] = color_for_height[shortest_height]

    return out
