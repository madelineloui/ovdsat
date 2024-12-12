import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from utils_dir.backbones_utils import load_backbone, extract_backbone_features
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import encoders
from segmentation_models_pytorch.encoders import get_encoder_names
from segmentation_models_pytorch.encoders._base import EncoderMixin
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, jaccard_score, classification_report, confusion_matrix

def prepare_image_for_backbone(input_tensor, backbone_type):
    '''
    Preprocess an image for the backbone model given an input tensor and the backbone type.

    Args:
        input_tensor (torch.Tensor): Input tensor with shape (B, C, H, W)
        backbone_type (str): Backbone type
    '''
    
    if input_tensor.shape[1] == 4:
        input_tensor = input_tensor[:, :3, :, :]  # Discard the alpha channel (4th channel)

    # Define mean and std for normalization depending on the backbone type
    mean = torch.tensor([0.485, 0.456, 0.406]).to(input_tensor.device) if 'dinov2' in backbone_type else torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(input_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(input_tensor.device) if 'dinov2' in backbone_type else torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(input_tensor.device)
    
    # Scale the values to range from 0 to 1
    input_tensor /= 255.0
    
    # Normalize the tensor
    normalized_tensor = (input_tensor - mean[:, None, None]) / std[:, None, None]
    
    if backbone_type == 'dinov2':
        # Pad height and width to the nearest multiple of 224
        pad_h = (224 - normalized_tensor.shape[2] % 224) % 224
        pad_w = (224 - normalized_tensor.shape[3] % 224) % 224
        normalized_tensor = F.pad(normalized_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

    return normalized_tensor


class ChangeDetectionSegModel(LightningModule):
    def __init__(self, num_classes, backbone_type, segmodel_type, learning_rate=0.001, time_series_length=4):
        super(ChangeDetectionSegModel, self).__init__()
        self.model = CustomSeg(num_classes=num_classes, backbone_type=backbone_type, segmodel_type=segmodel_type)
        self.loss_fn = smp.losses.DiceLoss(mode='multiclass')
        self.learning_rate = learning_rate
        self.time_series_length = time_series_length

        # Buffers for storing previous frame predictions
        self.true_label_prev = None
        self.pred_label_prev = None
        self.frame_counter = 0

        # Aggregated results across batches for testing
        self.test_results = {
            "true_labels_all": [],
            "pred_labels_all": [],
            "true_bc_all": [],
            "pred_bc_all": [],
            "true_sc_all": [],
            "pred_sc_all": [],
        }

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        images, masks = batch
        masks = torch.argmax(masks, dim=1).long()
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        return loss, preds, masks

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, masks = self.step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        _, preds, masks = self.step(batch)

        # Flatten predictions and true labels for metrics
        true_labels = masks.flatten().cpu().numpy()
        pred_labels = preds.flatten().cpu().numpy()

        # Save frame-wise predictions for change detection
        if self.frame_counter > 0:
            pred_bc = (self.pred_label_prev != pred_labels)
            true_bc = (self.true_label_prev != true_labels)
            self.test_results["pred_bc_all"].extend(pred_bc)
            self.test_results["true_bc_all"].extend(true_bc)

            pred_sc = pred_labels[true_bc]
            true_sc = true_labels[true_bc]
            self.test_results["pred_sc_all"].extend(pred_sc)
            self.test_results["true_sc_all"].extend(true_sc)

        self.frame_counter += 1
        if self.frame_counter == self.time_series_length:
            self.frame_counter = 0
            self.true_label_prev = None
            self.pred_label_prev = None
        else:
            self.true_label_prev = true_labels
            self.pred_label_prev = pred_labels

        # Accumulate all predictions and labels for overall metrics
        self.test_results["true_labels_all"].extend(true_labels)
        self.test_results["pred_labels_all"].extend(pred_labels)

    def test_epoch_end(self, outputs):
        # Convert lists to numpy arrays for metric calculations
        results = {key: np.array(value) for key, value in self.test_results.items()}

        # Helper to calculate IoU for each class
        def calculate_class_iou(pred, true, num_classes):
            ious = []
            for c in range(num_classes):
                pred_c = pred == c
                true_c = true == c
                intersection = np.logical_and(pred_c, true_c).sum()
                union = np.logical_or(pred_c, true_c).sum()
                iou = intersection / union if union > 0 else 0.0
                ious.append(iou)
            return ious  # Return IoU for all classes

        # Helper to calculate accuracy
        def calculate_accuracy(pred, true):
            correct = (pred == true).sum()
            total = true.size
            return correct / total

        # Number of classes
        num_classes = self.model.num_classes

        # Calculate per-class IoUs for BC and SC
        bc_ious = calculate_class_iou(results["pred_bc_all"], results["true_bc_all"], num_classes=2)  # BC is binary
        sc_ious = calculate_class_iou(results["pred_sc_all"], results["true_sc_all"], num_classes=num_classes)

        # Calculate mean IoU for BC and SC
        bc_iou_mean = np.mean(bc_ious)
        sc_iou_mean = np.mean(sc_ious)

        # Calculate overall IoU (multiclass)
        overall_ious = calculate_class_iou(results["pred_labels_all"], results["true_labels_all"], num_classes=num_classes)
        overall_iou_mean = np.mean(overall_ious)

        # Calculate accuracy
        accuracy = calculate_accuracy(results["pred_labels_all"], results["true_labels_all"])

        # Log metrics
        self.log("test_bc_iou_mean", bc_iou_mean, prog_bar=True)
        self.log("test_sc_iou_mean", sc_iou_mean, prog_bar=True)
        self.log("test_overall_iou_mean", overall_iou_mean, prog_bar=True)
        self.log("test_accuracy", accuracy, prog_bar=True)
        self.log("test_scs", (bc_iou_mean + sc_iou_mean) / 2, prog_bar=True)

        # Reset results for next test run
        self.test_results = {key: [] for key in self.test_results}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    

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

'''
class SegModel(pl.LightningModule):
    def __init__(self, num_classes, backbone_type, segmodel_type, model_dir, exp_name, learning_rate=0.001):
        super(SegModel, self).__init__()
        self.model = CustomSeg(num_classes=num_classes, backbone_type=backbone_type, segmodel_type=segmodel_type)
        self.loss_fn = smp.losses.DiceLoss(mode='multiclass')
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Initialize lists to store test results
        self.true_labels_all = []
        self.pred_labels_all = []
        self.true_bc_all = []
        self.pred_bc_all = []
        self.true_sc_all = []
        self.pred_sc_all = []
        self.true_label_prev = None
        self.pred_label_prev = None
        self.scene_counter = 0

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)

        # Convert to predictions and targets
        true_labels = labels.argmax(dim=1).cpu().numpy().flatten()
        probs = F.softmax(outputs, dim=1)
        pred_labels = probs.argmax(dim=1).cpu().numpy().flatten()

        self.true_labels_all.extend(true_labels)
        self.pred_labels_all.extend(pred_labels)

        # Change detection logic
        if self.scene_counter > 0:
            pred_bc = (self.pred_label_prev != pred_labels)
            true_bc = (self.true_label_prev != true_labels)
            self.pred_bc_all.extend(pred_bc)
            self.true_bc_all.extend(true_bc)

            pred_sc = pred_labels[true_bc]
            true_sc = true_labels[true_bc]
            self.pred_sc_all.extend(pred_sc)
            self.true_sc_all.extend(true_sc)

        self.scene_counter += 1
        if self.scene_counter == 24:  # Assuming 24 time series images per scene
            self.scene_counter = 0
            self.true_label_prev = None
            self.pred_label_prev = None
        else:
            self.true_label_prev = true_labels
            self.pred_label_prev = pred_labels

        return {"test_loss": self.loss_fn(outputs, labels)}

    def test_epoch_end(self, outputs):
        results_file = "test_results.txt"  # Output file for results

        # Calculate test loss
        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("avg_test_loss", avg_test_loss)

        # Calculate metrics
        accuracy = accuracy_score(self.true_labels_all, self.pred_labels_all)
        iou_score = jaccard_score(
            y_true=self.true_labels_all,
            y_pred=self.pred_labels_all,
            labels=range(self.num_classes),
            zero_division=0,
            average='weighted'
        )
        bc_score = jaccard_score(
            y_true=self.true_bc_all,
            y_pred=self.pred_bc_all,
            average='binary'
        )
        sc_score = jaccard_score(
            y_true=self.true_sc_all,
            y_pred=self.pred_sc_all,
            labels=range(self.num_classes),
            zero_division=0,
            average='weighted'
        )
        scs_score = (bc_score + sc_score) / 2

        report = classification_report(
            self.true_labels_all,
            self.pred_labels_all,
            labels=range(self.num_classes),
            zero_division=0
        )
        cm = confusion_matrix(self.true_labels_all, self.pred_labels_all)

        iou_scores_per_class = jaccard_score(
            self.true_labels_all,
            self.pred_labels_all,
            average=None,
            labels=range(self.num_classes)
        )

        # Prepare results string
        results = []
        results.append(f"Test Loss: {avg_test_loss:.4f}")
        results.append(f"Accuracy: {accuracy:.4f}")
        results.append(f"IoU (Jaccard Index): {iou_score:.4f}")
        results.append(f"Binary Change (BC): {bc_score:.4f}")
        results.append(f"Semantic Change (SC): {sc_score:.4f}")
        results.append(f"Semantic Change Segmentation (SCS): {scs_score:.4f}")
        results.append("\nClassification Report:\n" + report)
        results.append("\nConfusion Matrix:\n" + str(cm))
        for i, iou_score in enumerate(iou_scores_per_class):
            results.append(f"Class {i} IoU: {iou_score:.4f}")

        # Write results to file
        with open(results_file, "w") as f:
            f.write("\n".join(results))

        # Optionally print to console as well
        print("\n".join(results))

        # Clear stored lists for next evaluation
        self.true_labels_all.clear()
        self.pred_labels_all.clear()
        self.true_bc_all.clear()
        self.pred_bc_all.clear()
        self.true_sc_all.clear()
        self.pred_sc_all.clear()
'''

    
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
            # Freeze encoder backbone
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
            self.model = smp.FPN(
                encoder_name="dinov2",
                encoder_weights=None,   # Use pretrained weights in DINOv2
                classes=self.num_classes,
                activation=None
            )
        elif self.backbone_type == 'densenet121':
            self.model = smp.Unet(
                encoder_name="densenet121",
                encoder_weights=None,
                classes=num_classes,
                activation=None
            )
            state_dict = torch.load("/home/gridsan/manderson/ovdsat/assets/densenet121.pth")
            self.model.encoder.load_state_dict(state_dict)
        else:
            self.model = smp.Unet(
                encoder_name=backbone_type,
                encoder_weights='imagenet',
                classes=self.num_classes,
                activation=None
            )
        '''
        if self.segmodel_type == 'unet':
            self.model = smp.Unet(
                encoder_name=backbone_type,
                encoder_weights=encoder_wts,
                classes=self.num_classes,
                activation=None)
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
        
        # Freeze encoder backbone
        for param in self.model.encoder.parameters():
            param.requires_grad = False

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
    '''
    def forward(self, x):
        self.feature_maps = []  # Reset feature maps
        _ = self.dino_model(x)  # Forward pass through DINOv2
        return self.feature_maps[:self._depth]
    '''
    
    def forward(self, x):
        self.feature_maps = []  # Reset feature maps
        _ = self.dino_model(x)  # Forward pass through DINOv2

        spatial_features = []

        for i, feature in enumerate(self.feature_maps[:self._depth]):
            if feature.ndim == 3:  # Sequence embeddings, e.g., [B, SeqLen, Channels]
                # Remove class token if present (assumes it's the first token)
                feature = feature[:, 1:, :]  # [B, SeqLen-1, Channels]

                # Calculate spatial dimensions from sequence length
                grid_size = int((feature.size(1)) ** 0.5)
                feature = feature.transpose(1, 2).view(
                    feature.size(0), feature.size(2), grid_size, grid_size
                )  # [B, Channels, H, W]

            elif feature.ndim == 4:  # Already in spatial format, e.g., [B, C, H, W]
                pass  # Keep as-is

            # Resize to match expected spatial dimensions (optional)
            target_size = (x.shape[2] // (2 ** (i + 1)), x.shape[3] // (2 ** (i + 1)))
            feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)

            spatial_features.append(feature)

        return spatial_features

    def get_stages(self):
        # Optionally define stages if needed (depends on your feature extraction strategy)
        return [self.dino_model]

    def forward_stage(self, stage, x):
        # Forward through specific stages, useful for multi-stage processing
        return self.dino_model(x)