import os
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import init_dataloaders
from models.detector import OVDDetector
from datasets import get_base_new_classes
from utils_dir.nms import custom_xywh2xyxy
from utils_dir.metrics import ap_per_class, box_iou
from utils_dir.processing_utils import map_labels_to_prototypes


def prepare_model(args):
    '''
    Loads the model to evaluate given the input arguments and returns it.
    
    Args:
        args (argparse.Namespace): Input arguments
    '''

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Load prototypes and background prototypes
    prototypes = torch.load(args.prototypes_path)
    print(f'Using object prototypes from {args.prototypes_path}')
    bg_prototypes = torch.load(args.bg_prototypes_path) if args.bg_prototypes_path is not None else None
    if args.bg_prototypes_path is not None:
        print(f'Using background prototypes from {args.bg_prototypes_path}')
    model = OVDDetector(prototypes, bg_prototypes, scale_factor=args.scale_factor, backbone_type=args.backbone_type, target_size=args.target_size, classification=args.classification, text=args.t).to(device)
    #model.eval() 
    return model, device

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def reclassify(labels, sc_cat):
    """
    labels: array of numerical labels
    sc_cat: nested array of indices belonging to each super class
    """
    sc_labels = labels.clone()
    for i, sc in enumerate(sc_cat):
        for j, label_num in enumerate(labels):
            if label_num in sc:
                sc_labels[j] = i
    return sc_labels

def eval_detection(args, model, val_dataloader, device):
    seen = 0
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    
    sc = args.sc
    
    # Define superclass categories
    if sc:
        if args.dataset == 'simd':
            names = {
                0: 'car', 
                1: 'aircraft',
                2: 'boat',
                3: 'others'} 
            sc_cat = [[2,3,5,7,10,11,13],
                      [0,1,8,9,12,14],
                      [4],
                      [6]]   
        nc = len(names)
    else:
        names = model.classifier.get_categories()
        nc = val_dataloader.dataset.get_category_number()
        
    print('names', names)
    print('nc', nc)
    
    stats = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False):
            # if i > 50: # TODO debug
            #     break
            if args.classification != 'mask':
                images, boxes, labels, metadata = batch
                boxes = boxes.to(device)
            else:
                images, _, labels, masks, _ = batch
                loc = masks.float().to(device)
            
            # print(labels)
            # print(val_dataloader.dataset.get_categories())
            # print(model.classifier.get_categories())
            labels = map_labels_to_prototypes(val_dataloader.dataset.get_categories(), model.classifier.get_categories(), labels)
            #print(labels)
            images = images.float().to(device)
            labels = labels.to(device)

            preds = model(images, iou_thr=args.iou_thr, conf_thres=args.conf_thres, aggregation=args.aggregation)
            #print('preds\n', preds)

            for si, pred in enumerate(preds):
                keep = labels[si] > -1
                targets = labels[si, keep]
                targets_orig = targets.clone()
                if sc:
                    targets = reclassify(targets, sc_cat) # Reclassify using superclasses
                nl, npr = targets.shape[0], pred.shape[0]  # number of labels, predictions
                correct = torch.zeros(npr, len(iouv), dtype=torch.bool, device=device)  # init
                seen += 1

                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=device), targets[:]))
                    continue
                    
                predn = pred.clone()
                
                # Reclassify using superclasses
                if sc:
                    predn[:,-1] = reclassify(predn[:,-1], sc_cat)
                    
                if nl:
                    tbox = custom_xywh2xyxy(boxes[si, keep, :])  # target boxes
                    labelsn = torch.cat((targets[..., None], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)

                stats.append((correct, pred[:, 4], pred[:, 5], targets_orig[:]))
                
    # Use original categories
    names = model.classifier.get_categories()
    nc = val_dataloader.dataset.get_category_number()

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    
    print('stats: ', len(stats))
    
    #mp, mr, map50, map, ap_class = 0, 0, 0, 0, 0
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    print(s)
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        cat_type = '_sc' if sc else ''
        filename = f'results_{args.backbone_type}{cat_type}.txt'
        #filename = 'results_{}.txt'.format(args.backbone_type)
        save_file_path = os.path.join(args.save_dir, filename)
        base_classes, new_classes = get_base_new_classes(args.dataset)

        with open(save_file_path, 'w') as file:
            file.write('Class Images Instances P R mAP50 mAP50-95\n')
            file.write('%22s%11i%11i%11.4g%11.4g%11.4g%11.4g\n' % ('all', seen, nt.sum(), mp, mr, map50, map))

            if nc > 1 and len(stats):
                map50_base = map_base = mr_base = mp_base = 0
                map50_new = map_new = mr_new = mp_new = 0
                for i, c in enumerate(ap_class):
                    file.write('%22s%11i%11i%11.4g%11.4g%11.4g%11.4g\n' % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

                    if names[c] in base_classes:
                        map50_base += ap50[i]
                        map_base += ap[i]
                        mr_base += r[i]
                        mp_base += p[i]
                    elif names[c] in new_classes:
                        map50_new += ap50[i]
                        map_new += ap[i]
                        mr_new += r[i]
                        mp_new += p[i]
                map50_base /= len(base_classes)
                map_base /= len(base_classes)
                mr_base /= len(base_classes)
                mp_base /= len(base_classes)
                map50_new /= len(new_classes)
                map_new /= len(new_classes)
                mr_new /= len(new_classes)
                mp_new /= len(new_classes)
                file.write('%22s%11i%11i%11.4g%11.4g%11.4g%11.4g\n' % ('total base', seen, nt.sum(), mp_base, mr_base, map50_base, map_base))
                file.write('%22s%11i%11i%11.4g%11.4g%11.4g%11.4g\n' % ('total new', seen, nt.sum(), mp_new, mr_new, map50_new, map_new))
                print(f'wrote file to {save_file_path}')

def main(args):
    print('Setting up evaluation...')
    
    # ### TODO - for debugging
    # prototype = torch.load('/home/gridsan/manderson/ovdsat/run/init_prototypes/boxes/dior_N10-1/prototypes_clip-14.pt')
    # print('\nusual prototype shape:', prototype['prototypes'].shape)
    # print()

    # Initialize dataloader
    _, val_dataloader = init_dataloaders(args)


    # Load model
    model, device = prepare_model(args)

    # Perform training
    eval_detection(
        args, 
        model, 
        val_dataloader, 
        device
    )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--val_root_dir', type=str)
    parser.add_argument('--val_annotations_file', type=str)
    parser.add_argument('--annotations', type=str, default='box')
    parser.add_argument('--prototypes_path', type=str)
    parser.add_argument('--bg_prototypes_path', type=str, default=None)
    parser.add_argument('--aggregation', type=str, default='mean')
    parser.add_argument('--classification', type=str, default='box')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--backbone_type', type=str, default='dinov2')
    parser.add_argument('--target_size', nargs=2, type=int, metavar=('width', 'height'), default=(560, 560))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--scale_factor', nargs='+', type=int, default=2)
    parser.add_argument('--iou_thr', type=float, default=0.2)
    parser.add_argument('--conf_thres', type=float, default=0.001)
    parser.add_argument('--t', action='store_true', default=False) # if using text
    parser.add_argument('--sc', action='store_true', default=False)
    args = parser.parse_args()
    main(args)