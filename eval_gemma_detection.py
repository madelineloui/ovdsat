import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor

from datasets import get_base_new_classes
from utils_dir.metrics import box_iou


def normalize_label(label):
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


def get_prompt_categories(dataset, names):
    if dataset.lower() == "mar":
        return [name if "aircraft" in name.lower() else f"{name} aircraft" for name in names]

    if dataset.lower() == "dior":
        dior_names = [
            "airplane", "airport", "baseball field", "basketball court", "bridge", "chimney", "dam",
            "expressway service area", "expressway toll station", "golf field", "ground track field",
            "harbor", "overpass", "ship", "stadium", "storage tank", "tennis court", "train station",
            "vehicle", "windmill",
        ]
        readable_names = {normalize_label(name): name for name in dior_names}
        return [readable_names.get(normalize_label(name), name) for name in names]

    return names


def load_coco_annotations(annotation_file, dataset):
    with open(annotation_file, "r", encoding="utf-8") as file:
        coco = json.load(file)

    categories = sorted(coco["categories"], key=lambda category: category["id"])
    names = [category["name"] for category in categories]
    category_id_to_class = {category["id"]: i for i, category in enumerate(categories)}
    prompt_categories = get_prompt_categories(dataset, names)

    label_to_class = {normalize_label(name): i for i, name in enumerate(names)}
    for class_id, prompt_name in enumerate(prompt_categories):
        label_to_class[normalize_label(prompt_name)] = class_id

    annotations_by_image = defaultdict(list)

    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", 0):
            continue

        if annotation["category_id"] in category_id_to_class:
            annotations_by_image[annotation["image_id"]].append(annotation)

    return coco["images"], annotations_by_image, category_id_to_class, label_to_class, names, prompt_categories


def load_ground_truth(image_id, annotations_by_image, category_id_to_class):
    boxes = []
    labels = []

    for annotation in annotations_by_image.get(image_id, []):
        x, y, width, height = annotation["bbox"]

        if width <= 0 or height <= 0:
            continue

        boxes.append([x, y, x + width, y + height])
        labels.append(category_id_to_class[annotation["category_id"]])

    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    labels = torch.tensor(labels, dtype=torch.float32)

    return boxes, labels


def extract_json(text):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)

    if match:
        return json.loads(match.group(1))

    start = text.find("[")
    end = text.rfind("]")

    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"No JSON array found in response:\n{text}")


def parse_predictions(text, width, height, label_to_class):
    try:
        raw_predictions = extract_json(text)
    except Exception as error:
        print(f"\nCould not parse detections: {error}")
        return torch.zeros((0, 6), dtype=torch.float32)

    predictions = []
    unknown_labels = set()

    for item in raw_predictions:
        if not isinstance(item, dict) or "box_2d" not in item or "label" not in item:
            continue

        label = normalize_label(item["label"])

        if label not in label_to_class:
            unknown_labels.add(str(item["label"]))
            continue

        try:
            y1, x1, y2, x2 = map(float, item["box_2d"])
        except (TypeError, ValueError):
            continue

        x1 = np.clip(x1 / 1000.0 * width, 0, width)
        y1 = np.clip(y1 / 1000.0 * height, 0, height)
        x2 = np.clip(x2 / 1000.0 * width, 0, width)
        y2 = np.clip(y2 / 1000.0 * height, 0, height)

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        predictions.append([x1, y1, x2, y2, 1.0, label_to_class[label]])

    if unknown_labels:
        print(f"\nUnrecognized Gemma labels: {sorted(unknown_labels)}")

    return torch.tensor(predictions, dtype=torch.float32).reshape(-1, 6)


def process_batch(detections, labels, iouv):
    correct = np.zeros((detections.shape[0], iouv.shape[0]), dtype=bool)

    if len(detections) == 0 or len(labels) == 0:
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]

    for i, threshold in enumerate(iouv):
        matches = torch.where((iou >= threshold) & correct_class)

        if matches[0].numel() == 0:
            continue

        matches = torch.cat(
            (torch.stack(matches, 1), iou[matches[0], matches[1]][:, None]),
            1,
        ).cpu().numpy()

        if matches.shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        correct[matches[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def generate_predictions(model, processor, image, categories, max_new_tokens):
    category_text = ", ".join(categories)

    prompt = (
        "Detect every visible object belonging to these categories: "
        f"{category_text}. Return only a JSON array. Each item must have exactly this format: "
        '{"box_2d": [y1, x1, y2, x2], "label": "category"}. '
        "The label must exactly match one of the provided categories. Use integer coordinates normalized "
        "from 0 to 1000. Return [] if no listed objects are visible."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )

    input_device = model.get_input_embeddings().weight.device
    inputs = inputs.to(input_device)
    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    raw_response = processor.decode(
        generated_ids[0, input_length:],
        skip_special_tokens=False,
    )

    parsed_response = processor.parse_response(raw_response)

    if isinstance(parsed_response, dict):
        generated_text = (
            parsed_response.get("content")
            or parsed_response.get("response")
            or parsed_response.get("answer")
            or ""
        )
    else:
        generated_text = str(parsed_response)

    return generated_text.strip(), raw_response


def calculate_image_metrics(correct, num_predictions, num_ground_truth):
    tp = int(correct[:, 0].sum().item()) if num_predictions else 0
    fp = num_predictions - tp
    fn = num_ground_truth - tp

    precision = tp / num_predictions if num_predictions else 0.0
    recall = tp / num_ground_truth if num_ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return precision, recall, f1, tp, fp, fn


def get_best_same_class_ious(predictions, gt_boxes, gt_classes):
    predictions = predictions.detach().cpu()
    gt_boxes = gt_boxes.detach().cpu()
    gt_classes = gt_classes.detach().cpu()
    best_ious = []

    for prediction in predictions:
        matching = torch.where(gt_classes == prediction[5])[0]

        if matching.numel() == 0:
            best_ious.append(0.0)
            continue

        ious = box_iou(gt_boxes[matching], prediction[:4].reshape(1, 4))
        best_ious.append(float(ious.max().item()))

    return best_ious


def save_detection_figure(
    image,
    image_filename,
    predictions,
    gt_boxes,
    gt_classes,
    names,
    precision,
    recall,
    f1,
    figures_dir,
    figure_dpi,
):
    _, image_height = image.size

    predictions = predictions.detach().cpu()
    gt_boxes = gt_boxes.detach().cpu()
    gt_classes = gt_classes.detach().cpu()

    best_ious = get_best_same_class_ious(predictions, gt_boxes, gt_classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    for gt_box, gt_class in zip(gt_boxes, gt_classes):
        x1, y1, x2, y2 = gt_box.tolist()
        class_id = int(gt_class.item())

        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                edgecolor="lime",
                facecolor="none",
                linewidth=2,
            )
        )

        ax.text(
            x1,
            min(y2 + 5, image_height - 1),
            names[class_id],
            color="lime",
            fontsize=8,
            weight="bold",
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    for prediction, best_iou in zip(predictions, best_ious):
        x1, y1, x2, y2 = prediction[:4].tolist()
        class_id = int(prediction[5].item())

        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                edgecolor="red",
                facecolor="none",
                linewidth=2,
            )
        )

        ax.text(
            x1,
            max(y1 - 5, 0),
            f"{names[class_id]}\nIoU: {best_iou:.2f}",
            color="yellow",
            fontsize=8,
            weight="bold",
            verticalalignment="bottom",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    ax.set_title(
        f"{image_filename}\n"
        f"Precision@0.5: {precision:.4f}  |  "
        f"Recall@0.5: {recall:.4f}  |  "
        f"F1@0.5: {f1:.4f}"
    )

    ax.axis("off")
    fig.tight_layout()

    figure_path = figures_dir / f"{Path(image_filename).stem}.png"
    fig.savefig(figure_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    x = np.linspace(0.0, 1.0, 101)
    y = np.interp(x, mrec, mpre)

    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5)


def calculate_class_metrics(correct, confidence, predicted_classes, target_classes, num_classes):
    num_iou_thresholds = correct.shape[1]

    class_p = np.zeros(num_classes)
    class_r = np.zeros(num_classes)
    class_f1 = np.zeros(num_classes)
    class_ap = np.zeros((num_classes, num_iou_thresholds))

    predicted_classes = predicted_classes.astype(int)
    target_classes = target_classes.astype(int)

    for class_id in range(num_classes):
        pred_indices = np.where(predicted_classes == class_id)[0]
        num_targets = int(np.sum(target_classes == class_id))

        if len(pred_indices) == 0 or num_targets == 0:
            continue

        order = np.argsort(-confidence[pred_indices], kind="stable")
        class_correct = correct[pred_indices][order].astype(float)

        true_positives = np.cumsum(class_correct, axis=0)
        false_positives = np.cumsum(1.0 - class_correct, axis=0)

        recall_curve = true_positives / (num_targets + 1e-16)
        precision_curve = true_positives / (true_positives + false_positives + 1e-16)

        class_p[class_id] = precision_curve[-1, 0]
        class_r[class_id] = recall_curve[-1, 0]

        if class_p[class_id] + class_r[class_id] > 0:
            class_f1[class_id] = (
                2
                * class_p[class_id]
                * class_r[class_id]
                / (class_p[class_id] + class_r[class_id])
            )

        for iou_index in range(num_iou_thresholds):
            class_ap[class_id, iou_index] = compute_ap(
                recall_curve[:, iou_index],
                precision_curve[:, iou_index],
            )

    class_ap50 = class_ap[:, 0]
    class_ap5095 = class_ap.mean(axis=1)

    return class_p, class_r, class_f1, class_ap50, class_ap5095


def calculate_subset_metrics(indices, p, r, f1, ap50, ap5095):
    if not indices:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    indices = np.asarray(indices, dtype=int)

    return (
        float(p[indices].mean()),
        float(r[indices].mean()),
        float(f1[indices].mean()),
        float(ap50[indices].mean()),
        float(ap5095[indices].mean()),
    )


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        image_info,
        annotations_by_image,
        category_id_to_class,
        label_to_class,
        names,
        prompt_categories,
    ) = load_coco_annotations(args.val_annotations_file, args.dataset)

    if args.max_images is not None:
        image_info = image_info[:args.max_images]

    print("Dataset:", args.dataset)
    print("Images:", len(image_info))
    print("COCO categories:", names)
    print("Prompt categories:", prompt_categories)

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        device_map="auto",
        dtype=model_dtype,
        attn_implementation="sdpa",
    )

    model.eval()

    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats = []
    seen = 0

    save_dir = Path(args.save_dir)
    figures_dir = save_dir / "figures"
    predictions_file = save_dir / "predictions.jsonl"

    save_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(predictions_file, "w", encoding="utf-8") as output_file:
        for image_item in tqdm(image_info, total=len(image_info)):
            image_filename = image_item["file_name"]
            image_path = Path(args.val_root_dir) / image_filename

            if not image_path.exists():
                print(f"Missing image: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            gt_boxes, gt_classes = load_ground_truth(
                image_item["id"],
                annotations_by_image,
                category_id_to_class,
            )

            generated_text, raw_response = generate_predictions(
                model,
                processor,
                image,
                prompt_categories,
                args.max_new_tokens,
            )

            predictions = parse_predictions(
                generated_text,
                width,
                height,
                label_to_class,
            )

            predictions = predictions.to(device)
            gt_boxes = gt_boxes.to(device)
            gt_classes = gt_classes.to(device)

            correct = torch.zeros(
                (len(predictions), len(iouv)),
                dtype=torch.bool,
                device=device,
            )

            if len(predictions) and len(gt_classes):
                labels = torch.cat((gt_classes[:, None], gt_boxes), 1)
                correct = process_batch(predictions, labels, iouv)

            image_precision, image_recall, image_f1, image_tp, image_fp, image_fn = (
                calculate_image_metrics(correct, len(predictions), len(gt_classes))
            )

            # save_detection_figure(
            #     image,
            #     Path(image_filename).name,
            #     predictions,
            #     gt_boxes,
            #     gt_classes,
            #     names,
            #     image_precision,
            #     image_recall,
            #     image_f1,
            #     figures_dir,
            #     args.figure_dpi,
            # )

            stats.append(
                (
                    correct,
                    predictions[:, 4],
                    predictions[:, 5],
                    gt_classes,
                )
            )

            output_file.write(
                json.dumps(
                    {
                        "image_id": image_item["id"],
                        "image": str(image_path),
                        "raw_response": raw_response,
                        "response": generated_text,
                        "predictions": predictions.detach().cpu().tolist(),
                        "precision_0.5": image_precision,
                        "recall_0.5": image_recall,
                        "f1_0.5": image_f1,
                        "tp_0.5": image_tp,
                        "fp_0.5": image_fp,
                        "fn_0.5": image_fn,
                    }
                )
                + "\n"
            )

            output_file.flush()
            seen += 1

    if not stats:
        print("No images were evaluated.")
        return

    stats = [torch.cat(values, 0).cpu().numpy() for values in zip(*stats)]

    nc = len(names)
    nt = np.bincount(stats[3].astype(int), minlength=nc)

    class_p, class_r, class_f1, class_ap50, class_ap5095 = calculate_class_metrics(
        stats[0],
        stats[1],
        stats[2],
        stats[3],
        nc,
    )

    target_classes = np.where(nt > 0)[0]

    if len(target_classes):
        mp = class_p[target_classes].mean()
        mr = class_r[target_classes].mean()
        mf1 = class_f1[target_classes].mean()
        map50 = class_ap50[target_classes].mean()
        map5095 = class_ap5095[target_classes].mean()
    else:
        mp = mr = mf1 = map50 = map5095 = 0.0

    base_classes, new_classes = get_base_new_classes(args.dataset)

    base_names = {normalize_label(name) for name in base_classes}
    new_names = {normalize_label(name) for name in new_classes}

    base_indices = [
        i
        for i, name in enumerate(names)
        if normalize_label(name) in base_names
    ]

    new_indices = [
        i
        for i, name in enumerate(names)
        if normalize_label(name) in new_names
    ]

    mp_base, mr_base, mf1_base, map50_base, map5095_base = calculate_subset_metrics(
        base_indices,
        class_p,
        class_r,
        class_f1,
        class_ap50,
        class_ap5095,
    )

    mp_new, mr_new, mf1_new, map50_new, map5095_new = calculate_subset_metrics(
        new_indices,
        class_p,
        class_r,
        class_f1,
        class_ap50,
        class_ap5095,
    )

    print_format = "%22s" + "%11i" * 2 + "%11.4g" * 5

    header = (
        "%22s" + "%11s" * 7
    ) % (
        "Class",
        "Images",
        "Instances",
        "P",
        "R",
        "F1",
        "mAP50",
        "mAP50-95",
    )

    lines = [
        header,
        print_format
        % (
            "all",
            seen,
            nt.sum(),
            mp,
            mr,
            mf1,
            map50,
            map5095,
        ),
    ]

    for class_id, class_name in enumerate(names):
        lines.append(
            print_format
            % (
                class_name,
                seen,
                nt[class_id],
                class_p[class_id],
                class_r[class_id],
                class_f1[class_id],
                class_ap50[class_id],
                class_ap5095[class_id],
            )
        )

    lines.append(
        print_format
        % (
            "total base",
            seen,
            nt.sum(),
            mp_base,
            mr_base,
            mf1_base,
            map50_base,
            map5095_base,
        )
    )

    lines.append(
        print_format
        % (
            "total new",
            seen,
            nt.sum(),
            mp_new,
            mr_new,
            mf1_new,
            map50_new,
            map5095_new,
        )
    )

    results_file = save_dir / "results.txt"

    with open(results_file, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nSaved metrics to: {results_file}")
    print(f"Saved predictions to: {predictions_file}")
    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--val_root_dir", type=str, required=True)
    parser.add_argument("--val_annotations_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--figure_dpi", type=int, default=200)

    evaluate(parser.parse_args())