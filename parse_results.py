import os
import re
import sys
from statistics import mean, stdev

def extract(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s.startswith("all "):
                parts = s.split()
                if len(parts) >= 6:
                    return {"P": float(parts[-4]), "R": float(parts[-3]), "mAP50": float(parts[-2])}
    return None

def find_txt_in_subdirs(path, path_suffixes=None):
    """Walk subdirectories to find the first .txt result file,
    optionally requiring all path_suffixes appear somewhere in the dirpath."""
    for dirpath, dirnames, filenames in os.walk(path):
        if path_suffixes and not all(s in dirpath for s in path_suffixes):
            continue
        txts = [f for f in filenames if f.endswith(".txt")]
        if txts:
            return os.path.join(dirpath, txts[0])
    return None

def find_txt_in_folder_only(path):
    """Find the first .txt directly inside path, not in subdirectories."""
    txts = [f for f in os.listdir(path)
            if f.endswith(".txt") and os.path.isfile(os.path.join(path, f))]
    if txts:
        return os.path.join(path, txts[0])
    return None

def main(root_dir, suffixes=None):
    exact_pattern = re.compile(r"^N(1|3|5|10|30)-(1|2|3)$")
    suffix_pattern = re.compile(r"^N(1|3|5|10|30)-(1|2|3)(-.*)?$")

    data = {}
    for folder in os.listdir(root_dir):
        path = os.path.join(root_dir, folder)
        if not os.path.isdir(path):
            continue

        if suffixes:
            m = suffix_pattern.match(folder)
            if not m:
                continue

            if not all(s in folder for s in suffixes):
                folder_suffixes = [s for s in suffixes if s in folder]
                path_suffixes = [s for s in suffixes if s not in folder]
            else:
                folder_suffixes = suffixes
                path_suffixes = []

            if folder_suffixes and not all(s in folder for s in folder_suffixes):
                continue

            txt = find_txt_in_subdirs(path, path_suffixes if path_suffixes else None)

        else:
            # No suffixes: only match exact folder names like N1-1, N3-2, etc.
            m = exact_pattern.match(folder)
            if not m:
                continue

            # Only look for .txt directly in this folder
            txt = find_txt_in_folder_only(path)

        if not txt:
            continue

        vals = extract(txt)
        if vals:
            N, seed = int(m.group(1)), int(m.group(2))
            data.setdefault(N, []).append(vals)

    Ns = sorted(data.keys())
    if not Ns:
        print("No results found.")
        return

    print("N:     " + "  ".join(f"{N:>10}" for N in Ns))
    for metric in ["P", "R", "mAP50"]:
        row = []
        for N in Ns:
            vals = [r[metric] for r in data[N]]
            std = stdev(vals) if len(vals) > 1 else 0.0
            row.append(f"{mean(vals):.4f} ± {std:.4f}")
        print(f"{metric:>5}: " + "  ".join(f"{v:>10}" for v in row))

if __name__ == "__main__":
    root = sys.argv[1]
    suffixes = sys.argv[2:] if len(sys.argv) > 2 else None
    main(root, suffixes)