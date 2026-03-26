"""Quick script to inspect KULeuven MAT file structure."""
import sys
import numpy as np

mat_path = sys.argv[1] if len(sys.argv) > 1 else "data/KULeuven/S1.mat"

# Try scipy first
try:
    import scipy.io as sio
    mat = sio.loadmat(mat_path, squeeze_me=False)
    print("=== Loaded with scipy.io.loadmat ===")
    for k in sorted(mat.keys()):
        if k.startswith("__"):
            continue
        v = mat[k]
        shape = getattr(v, "shape", "N/A")
        dtype = getattr(v, "dtype", "N/A")
        print(f"  {k}: type={type(v).__name__}, shape={shape}, dtype={dtype}")

    # Drill into 'trials' if present
    if "trials" in mat:
        trials = mat["trials"]
        print(f"\n=== trials: shape={trials.shape}, dtype={trials.dtype} ===")
        n_trials = trials.shape[1] if trials.ndim == 2 else trials.shape[0]
        print(f"  Number of trials: {n_trials}")

        # Inspect first trial
        t0 = trials[0, 0] if trials.ndim == 2 else trials[0]
        print(f"\n=== First trial: type={type(t0).__name__}, dtype={t0.dtype} ===")
        if hasattr(t0, "dtype") and t0.dtype.names:
            for name in t0.dtype.names:
                field = t0[name]
                if hasattr(field, "shape"):
                    print(f"  .{name}: shape={field.shape}, dtype={field.dtype}")
                else:
                    print(f"  .{name}: type={type(field).__name__}")

                # Recurse one level
                if hasattr(field, "dtype") and field.dtype.names:
                    inner = field[0, 0] if field.ndim >= 2 else field[0] if field.ndim == 1 else field
                    for n2 in inner.dtype.names:
                        f2 = inner[n2]
                        sh = getattr(f2, "shape", "N/A")
                        dt = getattr(f2, "dtype", "N/A")
                        print(f"    .{name}.{n2}: shape={sh}, dtype={dt}")
                        # show value if small
                        if hasattr(f2, "size") and f2.size <= 5:
                            print(f"      value={f2}")

        # Also try indexing as object array
        elif t0.dtype == object:
            inner = t0.flat[0]
            print(f"  inner type: {type(inner).__name__}")
            if hasattr(inner, "dtype") and inner.dtype.names:
                for name in inner.dtype.names:
                    field = inner[name]
                    sh = getattr(field, "shape", "N/A")
                    dt = getattr(field, "dtype", "N/A")
                    print(f"  .{name}: shape={sh}, dtype={dt}")

    # Check if there's a different key with trials
    for k in sorted(mat.keys()):
        if k.startswith("__"):
            continue
        v = mat[k]
        if hasattr(v, "shape") and v.dtype == object and v.size > 1:
            print(f"\n=== {k} looks like cell array, size={v.size} ===")
            first = v.flat[0]
            if hasattr(first, "dtype") and first.dtype.names:
                print(f"  First element fields: {first.dtype.names}")

except NotImplementedError as e:
    print(f"scipy failed (probably v7.3 HDF5): {e}")
    print("\n=== Trying h5py ===")
    import h5py
    with h5py.File(mat_path, "r") as f:
        def visitor(name, obj):
            extra = ""
            if isinstance(obj, h5py.Dataset):
                extra = f"  shape={obj.shape} dtype={obj.dtype}"
            print(f"  {name}: {type(obj).__name__}{extra}")
        f.visititems(visitor)

        # Check trials specifically
        if "trials" in f:
            trials = f["trials"]
            print(f"\n=== trials: shape={trials.shape}, dtype={trials.dtype} ===")
            # In HDF5 MAT v7.3, cell arrays are stored as arrays of object references
            if trials.dtype == h5py.ref_dtype:
                n = trials.shape[1] if trials.ndim == 2 else trials.shape[0]
                print(f"  Number of trials (refs): {n}")
                ref0 = trials[0, 0] if trials.ndim == 2 else trials[0]
                t0 = f[ref0]
                print(f"  First trial keys: {list(t0.keys())}")
                for key in t0.keys():
                    item = t0[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"    {key}: shape={item.shape}, dtype={item.dtype}")
                    elif isinstance(item, h5py.Group):
                        print(f"    {key}: Group with keys={list(item.keys())}")
                        for k2 in item.keys():
                            i2 = item[k2]
                            if isinstance(i2, h5py.Dataset):
                                print(f"      {key}.{k2}: shape={i2.shape}, dtype={i2.dtype}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
