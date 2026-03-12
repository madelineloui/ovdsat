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


def main(root_dir):
    pattern = re.compile(r"zeroshot_(1|2|3)$")
    rows = []

    for folder in os.listdir(root_dir):
        path = os.path.join(root_dir, folder)
        if not os.path.isdir(path):
            continue
        if not pattern.match(folder):
            continue
        txts = [f for f in os.listdir(path) if f.endswith(".txt")]
        if not txts:
            continue
        vals = extract(os.path.join(path, txts[0]))
        if vals:
            rows.append(vals)

    if not rows:
        print("No zeroshot folders found.")
        return

    for metric in ["P", "R", "mAP50"]:
        vals = [r[metric] for r in rows]
        std = stdev(vals) if len(vals) > 1 else 0.0
        print(f"{metric:>5}: {mean(vals):.4f} ± {std:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_zeroshot.py <root_directory>")
        sys.exit(1)
    main(sys.argv[1])