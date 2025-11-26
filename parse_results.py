import os
import sys
import re
from statistics import mean

def extract_map50_from_file(filepath):
    """Extract the mAP50 value for the 'all' row from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith("all"):
                parts = line.split()
                if len(parts) >= 6:
                    return float(parts[5])  # mAP50 is 6th column
    return None

def main(root_dir):
    pattern = re.compile(r"N(\d+)-(\d+)")
    data = {}  # {N: {seed: value}}

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            match = pattern.match(folder_name)
            if match:
                N = int(match.group(1))
                seed = int(match.group(2))
                txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
                if not txt_files:
                    continue
                txt_path = os.path.join(folder_path, txt_files[0])
                val = extract_map50_from_file(txt_path)
                if val is not None:
                    data.setdefault(N, {})[seed] = val

    # Print in sorted order
    for N in sorted(data.keys()):
        print(f"\n=== N={N} ===")
        for seed in sorted(data[N].keys()):
            val = data[N][seed]
            #print(f"Seed {seed}: mAP50 = {val:.4f}")
            print(f'{val:.4f}')
        avg_val = mean(data[N].values())
        print(f"Average for N={N}: {avg_val:.5f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_results.py <root_directory>")
        sys.exit(1)
    main(sys.argv[1])
