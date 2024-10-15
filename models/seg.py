import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from utils_dir.backbones_utils import load_backbone, extract_backbone_features, prepare_image_for_backbone
import segmentation_models_pytorch as smp

class CustomSeg(torch.nn.Module):
     def __init__(self,
                num_classes,
                backbone_type='dinov2',
                segmodel_type='unet',
                target_size=(512,512),
                ):
        super().__init__()
        
        self.num_classes=num_classes
        self.backbone_type=backbone_type
        self.segmodel_type=segmodel_type
        
        bb = load_backbone(self.backbone_type)
        # TODO freeze bb
        
        if self.segmodel_type=='unet':
            self.model = smp.Unet(classes=self.num_classes, activation=None)
        elif self.segmodel_type=='unetpp':
            self.model = smp.UnetPlusPlus(classes=self.num_classes, activation=None)
        elif self.segmodel_type=='fpn':
            self.model = smp.FPN(classes=self.num_classes, activation=None)
        elif self.segmodel_type=='dlv3':
            self.model = smp.DeepLabV3(classes=self.num_classes, activation=None)
        elif self.segmodel_type=='dlv3p':
            self.model = smp.DeepLabV3Plus(classes=self.num_classes, activation=None)
        elif self.segmodel_type=='pan':
            self.model = smp.PAN(classes=self.num_classes, activation=None)
        elif self.segmodel_type=='pspnet':
            self.model = smp.PSPNet(classes=self.num_classes, activation=None)
        
        del self.model.encoder
        
        def forward(self, images):
            feats = extract_backbone_features(prepare_image_for_backbone(images, self.backbone_type))
            logits = self.model(feats)
            return logits
                                              
        