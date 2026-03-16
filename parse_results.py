import os
import sys
import re
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


def main(root_dir, suffix=None):
    if suffix:
        pattern = re.compile(rf"N(1|3|5|10|30)-(1|2|3)-{re.escape(suffix)}$")
    else:
        pattern = re.compile(r"N(1|3|5|10|30)-(1|2|3)$")

    data = {}

    for folder in os.listdir(root_dir):
        path = os.path.join(root_dir, folder)
        if not os.path.isdir(path):
            continue

        m = pattern.match(folder)
        if not m:
            continue

        N, seed = int(m.group(1)), int(m.group(2))

        txts = [f for f in os.listdir(path) if f.endswith(".txt")]
        if not txts:
            continue

        vals = extract(os.path.join(path, txts[0]))
        if vals:
            data.setdefault(N, []).append(vals)

    Ns = sorted(data.keys())
    print("N:     " + "  ".join(f"{N:>10}" for N in Ns))

    for metric in ["P", "R", "mAP50"]:
        row = []
        for N in Ns:
            vals = [r[metric] for r in data[N]]
            std = stdev(vals) if len(vals) > 1 else 0.0
            row.append(f"{mean(vals):.4f} ± {std:.4f}")
        print(f"{metric:>5}: " + "  ".join(f"{v:>10}" for v in row))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_results.py <root_directory> [suffix]")
        sys.exit(1)

    root = sys.argv[1]
    suffix = sys.argv[2] if len(sys.argv) > 2 else None

    main(root, suffix)