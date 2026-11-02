import os
import json
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import init_dataloaders, get_base_new_classes
from models.rpn.box_rpn import BoxRPN
from utils_dir.metrics import box_iou
from utils_dir.nms import custom_xywh2xyxy

@torch.no_grad()
def eval_rpn_recall(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, loader = init_dataloaders(args)
    rpn = BoxRPN().to(device).eval()
    names = loader.dataset.get_categories()
    nc = len(names)

    # Get readable class names from COCO annotations
    with open(args.val_annotations_file) as f:
        categories = json.load(f)['categories']
    id_to_name = {cat['id']: cat['name'] for cat in categories}
    class_names = [
        id_to_name.get(int(c), str(c)) if str(c).isdigit() else str(c)
        for c in names
    ]

    matched = torch.zeros(nc, device=device)
    total = torch.zeros(nc, device=device)

    for images, boxes, labels, _ in tqdm(loader):
        images, boxes, labels = images.float().to(device), boxes.to(device), labels.to(device)
        proposals, scores = rpn(images)

        for i in range(len(images)):
            keep = (labels[i] >= 0) & (labels[i] < nc)
            gt_labels = labels[i, keep].long()
            if not len(gt_labels):
                continue

            gt_boxes = custom_xywh2xyxy(boxes[i, keep])
            total += torch.bincount(gt_labels, minlength=nc)

            valid = torch.isfinite(scores[i])
            idx = scores[i][valid].argsort(descending=True)[:300]
            prop = proposals[i][valid][idx]
            if len(prop):
                covered = box_iou(gt_boxes, prop).max(dim=1).values >= 0.5
                matched += torch.bincount(gt_labels[covered], minlength=nc)

    recall = matched / total.clamp(min=1)
    base, novel = get_base_new_classes(args.dataset)

    lines = [f'{"Class":25s} {"Instances":>10s} {"Recall@300_IoU0.5":>20s}']
    for c, name in enumerate(class_names):
        if total[c]:
            lines.append(f'{name:25s} {int(total[c]):10d} {recall[c].item():20.4f}')

    for group_name, group in [('all', names), ('base', base), ('novel', novel)]:
        ids = [i for i, name in enumerate(names) if name in group and total[i]]
        if ids:
            lines.append(f'{group_name:25s} {int(total[ids].sum()):10d} {recall[ids].mean().item():20.4f}')

    result = '\n'.join(lines)
    print(result)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, f'rpn_recall_{args.dataset}.txt'), 'w') as f:
            f.write(result + '\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--val_root_dir')
    parser.add_argument('--val_annotations_file')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--target_size', nargs=2, type=int, default=(560, 560))
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--annotations', type=str, default='box')
    args = parser.parse_args()
    eval_rpn_recall(args)