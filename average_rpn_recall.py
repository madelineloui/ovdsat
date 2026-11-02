import os
import argparse
import pandas as pd

def main(args):
    results_dir = f'run/eval/rpn/{args.dataset}'
    results = []

    for split in args.splits:
        path = f'{results_dir}/M{split}/rpn_recall_{args.dataset}.txt'
        rows = []

        with open(path) as f:
            next(f)  # Skip header
            for line in f:
                if not line.strip():
                    continue
                name, instances, recall = line.rsplit(maxsplit=2)
                rows.append({
                    'Class': name.strip(),
                    'Instances': int(instances),
                    'Recall': float(recall),
                    'Split': split
                })

        results.append(pd.DataFrame(rows))

    df = pd.concat(results, ignore_index=True)
    summary = df.groupby('Class', sort=False).agg(
        Instances=('Instances', 'sum'),
        Mean=('Recall', 'mean'),
        Std=('Recall', 'std')
    ).reset_index()

    lines = [f'{"Class":25s} {"Instances":>10s} {"Mean":>10s} {"Std":>10s}']

    for _, row in summary.iterrows():
        lines.append(
            f'{row["Class"]:25s} '
            f'{int(row["Instances"]):10d} '
            f'{row["Mean"]:10.4f} '
            f'{row["Std"]:10.4f}'
        )

    result = '\n'.join(lines)
    print(result)

    output = os.path.join(results_dir, f'rpn_recall_{args.dataset}_avg.txt')
    with open(output, 'w') as f:
        f.write(result + '\n')

    print(f'\nSaved to {output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='simd')
    parser.add_argument('--splits', nargs='+', type=int, default=[1, 2, 3])
    main(parser.parse_args())