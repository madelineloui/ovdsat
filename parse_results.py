import os
import sys
import re
from statistics import mean

def extract_map50_from_file(filepath):
    """
    Extract the mAP50 values for:
      - 'all'
      - 'total base'
      - 'total new'
    from a text file.
    """
    vals = {"all": None, "base": None, "new": None}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()

            # 'all' row
            if stripped.startswith("all "):
                parts = stripped.split()
                if len(parts) >= 6:
                    vals["all"] = float(parts[-2])  # mAP50 (2nd to last)

            # 'total base' row
            elif stripped.startswith("total base"):
                parts = stripped.split()
                if len(parts) >= 6:
                    vals["base"] = float(parts[-2])  # mAP50 (2nd to last)

            # 'total new' row
            elif stripped.startswith("total new"):
                parts = stripped.split()
                if len(parts) >= 6:
                    vals["new"] = float(parts[-2])  # mAP50 (2nd to last)

    return vals

def main(root_dir):
    pattern = re.compile(r"N(\d+)-(\d+)")
    # data[N][seed] = {"all": ..., "base": ..., "new": ...}
    data = {}

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
                vals = extract_map50_from_file(txt_path)

                # require at least 'all' to be present; adjust if you want stricter
                if vals["all"] is not None:
                    data.setdefault(N, {})[seed] = vals

    # Print in sorted order
    for N in sorted(data.keys()):
        print(f"\n=== N={N} ===")
        all_list, base_list, new_list = [], [], []

        for seed in sorted(data[N].keys()):
            v = data[N][seed]
            all_val = v["all"]
            base_val = v["base"]
            new_val = v["new"]

            # per-seed line: all, base, new
            print(f"{all_val:.4f} {base_val:.4f} {new_val:.4f}")

            all_list.append(all_val)
            if base_val is not None:
                base_list.append(base_val)
            if new_val is not None:
                new_list.append(new_val)

        # Averages
        print(f"Average all  for N={N}: {mean(all_list):.5f}")
        if base_list:
            print(f"Average base for N={N}: {mean(base_list):.5f}")
        if new_list:
            print(f"Average new  for N={N}: {mean(new_list):.5f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_results.py <root_directory>")
        sys.exit(1)
    main(sys.argv[1])
