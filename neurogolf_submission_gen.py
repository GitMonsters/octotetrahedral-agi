#!/usr/bin/env python3
"""
NeuroGolf 2026 Submission Builder
Combines 266 library ONNX files + generates identity fallback for 134 missing tasks.
Output: neurogolf_submission.zip with task001.onnx ... task400.onnx
"""
import json, zipfile, shutil
from pathlib import Path
import numpy as np

LIBRARY_DIR  = Path.home() / "kaggle_data/neurogolf/submission"
DATA_FILE    = Path.home() / "kaggle_data/arc-agi-2/arc-agi_training_challenges.json"
PRIMS_CSV    = Path.home() / "kaggle_data/neurogolf/arc_primitives.csv"
OUT_DIR      = Path.home() / "kaggle_data/neurogolf_submission"
ZIP_FILE     = Path.home() / "neurogolf_submission.zip"

# ── ONNX helpers ──────────────────────────────────────────────────────────────

def make_identity_onnx() -> bytes:
    """Minimal identity network: output = input. 0 params."""
    from onnx import helper, TensorProto
    X = helper.make_tensor_value_info('input',  TensorProto.FLOAT, [1, 10, 30, 30])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10, 30, 30])
    node = helper.make_node('Identity', inputs=['input'], outputs=['output'])
    graph = helper.make_graph([node], 'identity', [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 10
    import onnx
    onnx.checker.check_model(model)
    return model.SerializeToString()


def make_color_remap_onnx(color_map: dict) -> bytes:
    """
    Build a color-remapping ONNX network.
    color_map: {src_color: dst_color} for color indices 0-9
    Uses Gather + one-hot encode trick to remap channels.
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    # Build permutation matrix [10, 10] where perm[i,j] = 1 if src j -> dst i
    perm = np.zeros((10, 10), dtype=np.float32)
    for src in range(10):
        dst = color_map.get(src, src)  # default: identity
        perm[dst, src] = 1.0

    X = helper.make_tensor_value_info('input',  TensorProto.FLOAT, [1, 10, 30, 30])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10, 30, 30])

    # perm_w: [10, 10, 1, 1] conv weight to remap channels
    perm_w = perm.reshape(10, 10, 1, 1)
    perm_init = numpy_helper.from_array(perm_w, name="perm_w")

    conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'perm_w'],
        outputs=['output'],
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
    )

    graph = helper.make_graph([conv_node], 'color_remap', [X], [Y], [perm_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model.SerializeToString()


def infer_color_map(task: dict) -> dict | None:
    """Try to detect a simple color mapping from training pairs."""
    if not task.get('train'):
        return None
    mapping_per_pair = []
    for pair in task['train']:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        if inp.shape != out.shape:
            return None
        cmap = {}
        for src, dst in zip(inp.flatten(), out.flatten()):
            src, dst = int(src), int(dst)
            if src in cmap:
                if cmap[src] != dst:
                    return None  # inconsistent
            else:
                cmap[src] = dst
        mapping_per_pair.append(cmap)

    # Intersect: all pairs must agree
    final = {}
    for c in range(10):
        vals = set()
        for mp in mapping_per_pair:
            if c in mp:
                vals.add(mp[c])
        if len(vals) == 1:
            final[c] = vals.pop()
        elif len(vals) == 0:
            final[c] = c  # not seen, keep identity
        else:
            return None  # conflicting
    return final


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import onnx

    # Load ARC training tasks (sorted = task001 mapping)
    with open(DATA_FILE) as f:
        arc_train = json.load(f)
    arc_keys = sorted(arc_train.keys())  # task001=arc_keys[0], etc.

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"library": 0, "color_remap": 0, "identity_fallback": 0}
    print("NeuroGolf 2026 Submission Builder")
    print(f"Tasks: 400 | Library files: {len(list(LIBRARY_DIR.glob('*.onnx')))}")
    print("=" * 60)

    for task_num in range(1, 401):
        task_id = f"task{task_num:03d}"
        out_path = OUT_DIR / f"{task_id}.onnx"
        library_path = LIBRARY_DIR / f"{task_id}.onnx"

        # 1. Use existing library ONNX if available
        if library_path.exists():
            shutil.copy(library_path, out_path)
            stats["library"] += 1
            status = "lib"
        else:
            # 2. Try to synthesize a color-remap ONNX
            arc_idx = task_num - 1
            if arc_idx < len(arc_keys):
                arc_task = arc_train[arc_keys[arc_idx]]
                color_map = infer_color_map(arc_task)
                if color_map is not None:
                    data = make_color_remap_onnx(color_map)
                    out_path.write_bytes(data)
                    stats["color_remap"] += 1
                    status = "colmap"
                else:
                    # 3. Identity fallback
                    data = make_identity_onnx()
                    out_path.write_bytes(data)
                    stats["identity_fallback"] += 1
                    status = "id"
            else:
                data = make_identity_onnx()
                out_path.write_bytes(data)
                stats["identity_fallback"] += 1
                status = "id"

        print(f"  {task_id}  [{status}]")

    # Zip it up
    print("=" * 60)
    print(f"Library:        {stats['library']}/400")
    print(f"Color remap:    {stats['color_remap']}/400")
    print(f"Identity fbk:   {stats['identity_fallback']}/400")
    print(f"\nBuilding {ZIP_FILE} ...")

    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        for task_num in range(1, 401):
            task_id = f"task{task_num:03d}"
            src = OUT_DIR / f"{task_id}.onnx"
            if src.exists():
                zf.write(src, f"{task_id}.onnx")

    size_mb = ZIP_FILE.stat().st_size / 1024 / 1024
    print(f"Done: {ZIP_FILE}  ({size_mb:.2f} MB)")
    with zipfile.ZipFile(ZIP_FILE) as zf:
        entries = len(zf.namelist())
    print(f"Zip contains: {entries} files")


if __name__ == "__main__":
    main()
