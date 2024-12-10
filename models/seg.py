import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from utils_dir.backbones_utils import load_backbone, extract_backbone_features, prepare_image_for_backbone
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import encoders
from segmentation_models_pytorch.encoders import get_encoder_names
from segmentation_models_pytorch.encoders._base import EncoderMixin
from pytorch_lightning import LightningModule

# Define LightningModule for model
class SegModel(LightningModule):
    def __init__(self, num_classes, backbone_type, segmodel_type, learning_rate=0.001):
        super(SegModel, self).__init__()
        self.model = CustomSeg(num_classes=num_classes, backbone_type=backbone_type, segmodel_type=segmodel_type)
        self.loss_fn = smp.losses.DiceLoss(mode='multiclass')
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        images, masks = images.cuda(), masks.cuda()  # Moving to GPU if available
        masks = torch.argmax(masks, dim=1)
        masks = masks.long()
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        images, masks = images.cuda(), masks.cuda()  # Moving to GPU if available
        masks = torch.argmax(masks, dim=1)
        masks = masks.long()
        outputs = self(images)
        val_loss = self.loss_fn(outputs, masks)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        
        # Threshold predictions for binary segmentation
        preds = (outputs > 0.5).float()
        return {"val_loss": val_loss, "preds": preds, "masks": masks}
    
    def test_step(self, batch, batch_idx):
        images, masks = batch
        images, masks = images.cuda(), masks.cuda()  # Moving to GPU if available
        masks = torch.argmax(masks, dim=1)
        masks = masks.long()
        outputs = self(images)
        test_loss = self.loss_fn(outputs, masks)
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)
        
        # Optionally return predictions and true masks for further evaluation
        return {"test_loss": test_loss, "preds": torch.argmax(outputs, dim=1), "masks": masks}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    
class CustomSeg(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 backbone_type='resnet50', #resnet50 / efficientnet-b4 / densenet121 / vgg16_bn / dinov2
                 segmodel_type='unet',
                 in_channels=3,
                 target_size=(512, 512)):
        super().__init__()

        self.num_classes = num_classes
        self.backbone_type = backbone_type
        self.segmodel_type = segmodel_type

        '''
        if 'resnet' in backbone_type:
            self.bb = smp.encoders.get_encoder(
                name=self.backbone_type, 
                weights="imagenet", 
                in_channels=in_channels
            )
        else:
            self.bb = load_backbone(self.backbone_type)
            # Freeze backbone
            for param in self.bb.parameters():
                param.requires_grad = False
        '''
        
        if self.backbone_type == 'dinov2':
            encoders["dinov2"] = {
                "encoder": DINOv2Encoder,
                "params": {
                    "model_name": "dinov2_vitl14",
                    "pretrained": True,
                    "depth": 5,
                },
            }
            encoder_wts = None
        else:
            encoder_wts = 'imagenet'

        if self.segmodel_type == 'unet':
            self.model = smp.Unet(
                encoder_name=backbone_type,
                encoder_weights=encoder_wts,
                classes=self.num_classes,
                activation=None)
        '''
        elif self.segmodel_type == 'unetpp':
            self.model = smp.UnetPlusPlus(classes=self.num_classes, activation=None)
        elif self.segmodel_type == 'fpn':
            self.model = smp.FPN(classes=self.num_classes, activation=None)
        elif self.segmodel_type == 'dlv3':
            self.model = smp.DeepLabV3(classes=self.num_classes, activation=None)
        elif self.segmodel_type == 'dlv3p':
            self.model = smp.DeepLabV3Plus(classes=self.num_classes, activation=None)
        elif self.segmodel_type == 'pan':
            self.model = smp.PAN(classes=self.num_classes, activation=None)
        elif self.segmodel_type == 'pspnet':
            self.model = smp.PSPNet(classes=self.num_classes, activation=None)
        '''
        
        #TODO: freeze encoder
        #del self.model.encoder  # Remove the encoder if not needed

    def forward(self, images):
        prep_images = prepare_image_for_backbone(images, self.backbone_type)
        #feats = extract_backbone_features(images=prep_images, model=self.bb, backbone_type=self.backbone_type)
        logits = self.model(prep_images)
        return logits

class DINOv2Encoder(nn.Module, EncoderMixin):
    def __init__(self, model_name='dinov2_vitl14', pretrained=True, depth=5, **kwargs):
        super().__init__()
        EncoderMixin.__init__(self)
        
        # Load the DINOv2 model
        self.dino_model = torch.hub.load('facebookresearch/dinov2', model_name)

        # Specify the expected output channels for each stage
        self._depth = depth
        self._in_channels = 3
        self._out_channels = [64, 128, 256, 512, 1024][:depth]

        # Set up hooks to extract intermediate features from DINOv2
        self.feature_maps = []
        self.hooks = []

        # Register hooks for specific layers
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.feature_maps.append(output)

        # Adjust based on the DINOv2 architecture
        for name, module in self.dino_model.named_modules():
            if "block" in name:  # Example: Customize to match the model layers
                self.hooks.append(module.register_forward_hook(hook_fn))

    def forward(self, x):
        self.feature_maps = []  # Reset feature maps
        _ = self.dino_model(x)  # Forward pass through DINOv2
        return self.feature_maps[:self._depth]

    def get_stages(self):
        # Optionally define stages if needed (depends on your feature extraction strategy)
        return [self.dino_model]

    def forward_stage(self, stage, x):
        # Forward through specific stages, useful for multi-stage processing
        return self.dino_model(x)