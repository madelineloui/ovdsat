
import argparse
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm
from utils_dir.backbones_utils import load_clip_to_cpu, load_backbone_and_tokenizer


def load_pt(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def normalize_name(name):
    aircraft = {'SU-35', 'C-130', 'C-17', 'C-5', 'F-16', 'TU-160', 'E-3', 'B-52',
                'P-3C', 'B-1B', 'E-8', 'TU-22', 'F-15', 'KC-135', 'F-22', 'FA-18',
                'TU-95', 'KC-1', 'SU-34', 'SU-24', 'propeller', 'chartered',
                'fighter', 'trainer'}
    if name in aircraft:
        name += ' aircraft'
    if name == 'other':
        name = 'others'
    return ''.join(c for c in name.lower() if c.isalnum())


def result(values, signed=False):
    avg = mean(values)
    spread = f'{stdev(values):.4f}' if len(values) > 1 else 'n/a'
    return f'{avg:+.4f} +/- {spread}' if signed else f'{avg:.4f} +/- {spread}'


def main(args):
    rng = random.Random(args.random_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = Path(args.test_dir or f'data/cropped_data/{args.dataset}/val')
    if not root.is_dir():
        raise FileNotFoundError(f'Validation directory missing: {root}')

    seeds = args.seeds[:args.max_splits] if args.max_splits else args.seeds
    for seed in seeds:
        split = root / f'{args.dataset}_val-{seed}'
        if not split.is_dir():
            raise FileNotFoundError(f'Matched validation split missing: {split}')

    labels = load_pt(f'run/text_prototypes/boxes/{args.dataset}/prototypes_clip-14.pt')['label_names']
    transform = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711])
    ])
    reports = []

    for backbone in args.backbones:
        print(f'\nBackbone: {backbone}; matched seeds/splits: {seeds}', flush=True)

        text_model, tokenizer = load_backbone_and_tokenizer(backbone)
        text_model = text_model.to(device).eval()
        prompts = [f'a satellite image of a {label}' for label in labels]

        with torch.no_grad():
            if backbone.startswith(('remoteclip', 'georsclip', 'openclip')):
                text = text_model.encode_text(tokenizer(prompts).to(device))
            else:
                tokens = tokenizer(prompts, return_tensors='pt', padding=True,
                                   truncation=True).to(device)
                text = text_model.get_text_features(**tokens)
            zero_shot = F.normalize(text.float(), dim=-1).cpu()
        del text_model

        visual_model = load_clip_to_cpu(backbone).to(device).eval()
        per_class, per_pair = defaultdict(list), defaultdict(list)

        # Evaluate each CoOp prototype on its matching validation split
        for seed in tqdm(seeds, desc=f'{backbone} pairs', unit='pair'):
            split = root / f'{args.dataset}_val-{seed}'
            variant = 1 if args.dataset == 'nwpu' else 2
            path = (
                f'run/coop_prototypes/boxes/{args.dataset}_N{args.n}-{seed}/'
                f'{args.dataset}_{backbone.split("-")[0]}_{variant}/'
                f'nctx{args.nctx}_cscFalse_ctp{args.ctp}/prototypes_{backbone}.pt'
            )
            data = load_pt(path)
            coop_names = [normalize_name(x) for x in data['label_names']]
            missing = [x for x in labels if normalize_name(x) not in coop_names]
            if missing:
                raise ValueError(f'Unmatched classes in {path}: {missing}')

            order = [coop_names.index(normalize_name(x)) for x in labels]
            coop = F.normalize(data['prototypes'][order].float(), dim=-1).cpu()

            classes = []
            for i, label in enumerate(labels):
                folder = split / label
                paths = sorted(p for p in folder.iterdir()
                               if p.suffix.lower() in ('.jpg', '.jpeg', '.png')) if folder.is_dir() else []
                if not paths:
                    continue
                if args.max_images:
                    paths = rng.sample(paths, min(args.max_images, len(paths)))
                classes.append((i, label, paths))
                if args.max_classes and len(classes) >= args.max_classes:
                    break

            if not classes:
                raise RuntimeError(f'No class image crops found in {split}')

            for i, label, paths in tqdm(
                classes, desc=f'{split.name} classes', unit='class', leave=False
            ):
                features = []
                with torch.no_grad():
                    for j in range(0, len(paths), args.batch_size):
                        batch = torch.stack([
                            transform(Image.open(p).convert('RGB'))
                            for p in paths[j:j + args.batch_size]
                        ]).to(device)
                        feats = visual_model.visual(batch.to(dtype=visual_model.dtype))
                        features.append(F.normalize(feats.float(), dim=-1).cpu())

                avg_image = F.normalize(torch.cat(features).mean(dim=0), dim=0)
                zs_sim = (avg_image @ zero_shot[i]).item()
                coop_sim = (avg_image @ coop[i]).item()
                delta = coop_sim - zs_sim

                per_class[label].append((zs_sim, coop_sim, delta, len(paths)))
                per_pair[seed].append(delta)

        del visual_model

        lines = [
            f'Backbone: {backbone} | Dataset: {args.dataset} | N={args.n}',
            f'Matched evaluation pairs: {", ".join(f"{args.dataset}_val-{s} / CoOp seed {s}" for s in seeds)}',
            'Mean +/- sample std across matched pairs (n/a if only one pair)',
            f'{"Class":<24} {"Zero-shot":>21} {"CoOp":>21} {"Delta":>22} {"Pairs":>6} {"Images":>7}'
        ]

        for label in labels:
            if label not in per_class:
                continue
            r = per_class[label]
            lines.append(
                f'{label:<24} {result([x[0] for x in r]):>21} '
                f'{result([x[1] for x in r]):>21} '
                f'{result([x[2] for x in r], True):>22} '
                f'{len(r):>6} {sum(x[3] for x in r):>7}'
            )

        overall = [mean(per_pair[s]) for s in seeds]
        lines.append(f'\nOverall mean delta: {result(overall, True)}')
        reports.append('\n'.join(lines))

    output = '\n\n'.join(reports) + '\n'
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(output)
    print('\n' + output + f'Saved report to {args.output}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='mar')
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--backbones', nargs='+', default=['georsclip-14'])
    p.add_argument('--test-dir', default=None)
    p.add_argument('--nctx', type=int, default=4)
    p.add_argument('--ctp', default='end')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-splits', type=int, default=0)
    p.add_argument('--max-classes', type=int, default=0)
    p.add_argument('--max-images', type=int, default=0)
    p.add_argument('--random-seed', type=int, default=None)
    p.add_argument('--output', default='run/embeddings/test_cosine_delta.txt')
    main(p.parse_args())
