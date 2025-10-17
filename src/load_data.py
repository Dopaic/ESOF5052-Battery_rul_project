#------------------------ Header Comment ----------------------#
#Load NASA-style Li-ion battery .mat files, extract per-cycle capacity,
#perform robust cleaning (warm-up trimming, local anomaly filtering, median smoothing)

# src/load_data.py — Enhanced cleaning + show-all
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from collections import deque

# ---- Parameters ----
BASELINE_WINDOW = 10        # use median of first 10 cycles as baseline
DROP_THRESHOLD = 0.30       # cliff drop threshold
RECOVER_THRESHOLD = 0.90    # recovery ratio for restart segment
MIN_LOW_SEGMENT = 2         # continuous valley length
MIN_CYCLES_KEEP = 10
WARMUP_JUMP = 0.20          # jump detection in early cycles (warm-up)
LOOKAHEAD = 8               # lookahead for warm-up detection

# Compatible with different datasets/scripts for naming "capacity"
CAP_KEYS = [
    'Capacity','capacity','Qd','Q','Discharge_Capacity','discharge_capacity',
    'Capacity_measured','Q_discharge','QDischarge'
]

#Load a MATLAB .mat file and return a Python dict-like structure
#Tries scipy.io.loadmat (v7.2 and below). If it fails (e.g., v7.3 HDF5),
#falls back to h5py and converts groups/datasets to Python-native types.
#Parameters：
#path : str (Path to .mat file)
#Returns :
#Dict[str, Any] (Pythonized tree/dict built from the .mat contents)
def _load_mat(path: str) -> Dict[str, Any]:
    try:
        from scipy.io import loadmat
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        for k in list(data.keys()):
            if k.startswith('__'): del data[k]
        return data
    except Exception:
        pass
    import h5py
    def _h5_to_py(obj):
        if isinstance(obj, h5py.Dataset):
            arr = obj[()]
            if isinstance(arr, (bytes, np.bytes_)):
                try: return arr.decode('utf-8', errors='ignore')
                except Exception: return str(arr)
            return np.array(arr)
        elif isinstance(obj, h5py.Group):
            return {k: _h5_to_py(obj[k]) for k in obj.keys()}
        else: return obj
    out = {}
    with h5py.File(path, 'r') as f:
        for k in f.keys(): out[k] = _h5_to_py(f[k])
    return out

#Safely extract a field, key, or attribute named `name` from mixed MATLAB-to-Python
#converted objects (dict / object / structured ndarray / HDF5 node).
#Parameters:
#obj : Any(dict, object, numpy.void, array, etc.)
#name : str
#default : Any, optional(Default value to return if retrieval fails (default is None))
def _get_attr(obj, name, default=None):
    if isinstance(obj, dict): return obj.get(name, default)
    if hasattr(obj, name): return getattr(obj, name)
    try:
        if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names: return obj[name]
    except Exception: pass
    try: return obj[name]
    except Exception: return default

#Normalizes arbitrary input into a list, converts lists/tuples to lists,
#converts arrays to flat lists, and flattens arrays into one dimension.
#- None → []
#- list/tuple → list
#- ndarray → flattened list
#- others → [x]
def _as_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    if isinstance(x, np.ndarray): return list(x.ravel())
    return [x]

def _safe_mean(x):
    try:
        arr = np.asarray(x).astype(float)
        if arr.size == 0 or not np.isfinite(arr).any(): return None
        return float(np.nanmean(arr[np.isfinite(arr)]))
    except Exception: return None

#Scans Dict to locate a field/key/attr named (case-insensitive) 'cycle'. Returns its value once found.
#Parameters
#tree : AnyRoot object/dict/sequence.
#Returns
#The 'cycle' container if found, else None.
def _find_cycle_container(tree: Any):
    q = deque([tree]); seen=set()
    while q:
        node = q.popleft()
        nid = id(node)
        if nid in seen: continue
        seen.add(nid)
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k).lower()=='cycle': return v
            for v in node.values(): q.append(v)
        elif isinstance(node, (list, tuple, np.ndarray)):
            for v in _as_list(node): q.append(v)
        else:
            names = [n for n in dir(node) if not n.startswith('_')]
            for n in names:
                try:
                    if n.lower()=='cycle': return getattr(node, n)
                except Exception: pass
            for n in names:
                try: q.append(getattr(node, n))
                except Exception: pass
    return None

#Apply a simple sliding-window median filter to smooth the capacity series.
#Parameters
#x : np.ndarray
#win : int   Size of the sliding window. Typically an odd number (3, 5, 7...).
#Returns
#np.ndarray   Smoothed array (same length as input).
def _median_filter(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) < win: return x
    pad = win//2
    xx = np.pad(x, (pad, pad), mode='edge')
    return np.array([np.median(xx[i:i+win]) for i in range(len(x))], dtype=float)

# The function targets two common in battery capacity curves:
#   1) Cliff drops: if the relative drop between consecutive points exceeds
#   DROP_THRESHOLD, the post-drop point is marked as an outlier.
#   2) Recoverable valleys: a contiguous low segment (length >= MIN_LOW_SEGMENT)
#   below a baseline that subsequently recovers to >= RECOVER_THRESHOLD × baseline
#   is treated as transient noise and removed.
#Parameters
#cap : np.ndarray 1D capacity array (positive, ordered by cycle).
#Returns
#np.ndarray : Filtered capacity series. If filtering would leave fewer than 2 points,
#returns a copy of the original series.
def _clean_local_anomalies(cap: np.ndarray) -> np.ndarray:
    if len(cap) < 3: return cap.copy()
    c = cap.copy()
    keep = np.ones_like(c, dtype=bool)
    prev = c[:-1]; curr = c[1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        drop = (prev - curr) / (np.maximum(prev, 1e-9))
    cliff_idx = np.where(drop > DROP_THRESHOLD)[0] + 1
    keep[cliff_idx] = False
    i = 0
    while i < len(c):
        j = i + 1
        baseline = c[i-1] if i > 0 else c[0]
        while j < len(c) and c[j] < baseline * (1 - DROP_THRESHOLD):
            j += 1
        if (j - i) >= MIN_LOW_SEGMENT:
            k = j; recovered = False
            while k < len(c):
                if c[k] >= baseline * RECOVER_THRESHOLD:
                    recovered = True; break
                k += 1
            if recovered:
                keep[i:j] = False; i = j; continue
        i += 1
    filtered = c[keep]
    return filtered if len(filtered) >= 2 else c.copy()

#Some battery datasets show an artificial increase at the beginning
#(e.g., due to early calibration or formation cycles). This function
#detects the first point within the initial `lookahead` cycles whose
#capacity exceeds (1 + jump) × baseline median and trims the curve
#from that index onward.
#Parameters：
#cap : np.ndarray (Capacity series)
#lookahead : int, optional (Window length to inspect warm-up)
# min_keep : int, optional (Relative jump threshold)
#jump : float, optional (Minimum number of data points to retain after trimming (default: 5))
#Returns :
def _trim_warmup(cap, lookahead=LOOKAHEAD, jump=WARMUP_JUMP, min_keep=5):
    if len(cap) <= lookahead:
        return cap
    base = np.median(cap[:min(lookahead, len(cap))])
    idxs = np.where(cap[:lookahead] >= base * (1 + jump))[0]
    if idxs.size > 0:
        cut = int(idxs[0])
        cut = max(0, min(cut, len(cap) - min_keep))
        return cap[cut:]
    return cap

#Main entry: load/clean all .mat files under `data_dir` and write outputs.
#Parameters：
#data_dir : str (Directory containing .mat files)
#Returns :
# pd.DataFrame (Long-form table with columns: file, cycle, capacity, soh)
#Returns None if nothing usable remains.
def load_battery_data(data_dir: str) -> pd.DataFrame:
    mats = [f for f in os.listdir(data_dir) if f.lower().endswith('.mat')]#列出目录下所有.mat
    if not mats:
        print('No .mat files found in', data_dir); return None

    os.makedirs('outputs', exist_ok=True)
    kept_frames: List[pd.DataFrame] = []
    removed_log: List[str] = []

    for fname in sorted(mats):
        path = os.path.join(data_dir, fname)
        root = _load_mat(path)
        cycles = _find_cycle_container(root)
        if cycles is None:
            removed_log.append(f"{fname}\tno_cycle_container")
            continue

        caps: List[float] = []
        for c in _as_list(cycles):
            data_obj = _get_attr(c, 'data')
            cap = None
            if data_obj is not None:
                for key in CAP_KEYS:
                    cap = _get_attr(data_obj, key)
                    if cap is not None: break
            if cap is None:
                for key in CAP_KEYS:
                    cap = _get_attr(c, key)
                    if cap is not None: break
            cap_val = _safe_mean(cap)
            if cap_val is not None and np.isfinite(cap_val) and cap_val > 0:
                caps.append(float(cap_val))

        if len(caps) < 2:
            removed_log.append(f"{fname}\ttoo_few_points")
            continue

        cap = np.asarray(caps, dtype=float)
        cap = _trim_warmup(cap)
        cap = _clean_local_anomalies(cap)
        cap = _median_filter(cap, win=3)

        if len(cap) < MIN_CYCLES_KEEP:
            removed_log.append(f"{fname}\ttoo_short_after_cleaning({len(cap)})")
            continue

        df = pd.DataFrame({
            "file": fname,
            "cycle": np.arange(1, len(cap)+1, dtype=int),
            "capacity": cap
        })

        q0 = float(np.median(df["capacity"].iloc[:min(BASELINE_WINDOW, len(df))]))
        if not np.isfinite(q0) or q0 <= 0:
            removed_log.append(f"{fname}\tinvalid_q0")
            continue

        df["soh"] = df["capacity"] / q0
        kept_frames.append(df)

    if not kept_frames:
        with open(os.path.join('outputs', 'cleaning_report_enhanced.txt'), 'w', encoding='utf-8') as f:
            f.write("No usable series after enhanced cleaning.\n")
        return None

    out = pd.concat(kept_frames, ignore_index=True)
    out.to_csv(os.path.join('outputs', 'clean_dataset_enhanced.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join('outputs', 'cleaning_report_enhanced.txt'), 'w', encoding='utf-8') as f:
        f.write("Enhanced cleaning report (方案B, no exclusion, all curves shown):\n")
        for line in removed_log:
            f.write(line + "\n")
    print("✅ Enhanced cleaning complete -> outputs/clean_dataset_enhanced.csv")
    print("   All curves retained for visualization.")
    return out