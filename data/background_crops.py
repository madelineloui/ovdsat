import random
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import json
from collections import defaultdict


def generate_background_crops_from_coco(
    annotations_path,
    images_dir,
    output_root,
    num_b=10,
    min_size=30,
    max_size=200,
    max_iter=100,
    max_images=None,
):
    images_dir = Path(images_dir)
    output_root = Path(output_root)
    background_dir = output_root / "background"
    background_dir.mkdir(parents=True, exist_ok=True)

    with open(annotations_path, "r") as f:
        coco = json.load(f)

    image_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        image_to_anns[ann["image_id"]].append(ann)

    images = coco["images"]
    if max_images is not None and len(images) > max_images:
        images = random.sample(images, max_images)

    for image_info in tqdm(images, desc="Images"):
        file_name = image_info["file_name"]
        image_id = image_info["id"]

        img_path = images_dir / file_name
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        anns = image_to_anns.get(image_id, [])

        saved = 0
        for _ in range(num_b):
            count = 0
            while count < max_iter:
                count += 1

                x = random.randint(0, w - max_size)
                y = random.randint(0, h - max_size)
                width = random.randint(min_size, max_size)
                height = random.randint(min_size, max_size)

                x2, y2 = x + width, y + height

                intersects = any(
                    x < ann["bbox"][0] + ann["bbox"][2] and ann["bbox"][0] < x2 and
                    y < ann["bbox"][1] + ann["bbox"][3] and ann["bbox"][1] < y2
                    for ann in anns
                )

                if not intersects:
                    crop = image.crop((x, y, x2, y2))
                    crop.save(background_dir / f"{Path(file_name).stem}_bg{saved}.jpg")
                    saved += 1
                    break

            if saved >= num_b:
                break


#==================== Example usage for train ====================

for N in [1, 3, 5, 10, 30]:
    for M in [1, 2, 3]:
        print(f'================ Processing train N={N}, M={M} ================')

        images_dir = "/home/gridsan/manderson/ovdsat/data/mar/JPEGImages"
        annotations_path = f"/home/gridsan/manderson/ovdsat/mar/dior/train_coco_subset_N{N}-{M}.json"
        output_root = f"/home/gridsan/manderson/ovdsat/data/cropped_data/mar/train/mar_N{N}-{M}"

        generate_background_crops_from_coco(
            annotations_path,
            images_dir,
            output_root,
        )
        print()


# # ==================== Example usage for val ====================

# for M in [1, 2, 3, 4, 5]:
#     print(f'================ Processing val M={M} ================')

#     images_dir = "/home/gridsan/manderson/ovdsat/data/dior/JPEGImages"
#     annotations_path = f"/home/gridsan/manderson/ovdsat/data/dior/val_coco-{M}.json"
#     output_root = f"/home/gridsan/manderson/ovdsat/data/cropped_data/dior/val/dior_val-{M}"
#     generate_background_crops_from_coco(annotations_path, images_dir, output_root, max_images=1000)
