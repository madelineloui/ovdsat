import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import CLIPModel
from utils_dir.backbones_utils import extract_backbone_features, load_backbone, load_backbone_and_tokenizer

class OVDBaseClassifier(torch.nn.Module):

    def __init__(self, prototypes, class_names, backbone_type='dinov2', target_size=(602,602), scale_factor=2, min_box_size=5, ignore_index=-1, text=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.min_box_size = min_box_size
        self.ignore_index = ignore_index
        self.backbone_type = backbone_type
        self.class_names = class_names
        self.text = text
        
        if isinstance(self.scale_factor, int):
            self.scale_factor = [self.scale_factor]

        # Initialize backbone
        if text:
            self.backbone, _ = load_backbone_and_tokenizer(backbone_type)  
        else:
            self.backbone = load_backbone(backbone_type)  
        
        # Initialize embedding as a learnable parameter
        self.embedding = torch.nn.Parameter(prototypes)
        self.num_classes = self.embedding.shape[0]
        
        if 'customclip' in self.backbone_type:
            ckpt_path = '/home/gridsan/manderson/ovdsat/weights/vlm4rs/customclip.pth'
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
            self.logit_scale = checkpoint['model.logit_scale']
        else:
            self.logit_scale = None

    def get_categories(self):
        return {idx: label for idx, label in enumerate(self.class_names)}

    def get_cosim_mini_batch(self, feats, embeddings, batch_size=100, normalize=False):
        '''
        Compute cosine similarity between features and protorype embeddings in mini-batches to avoid memory issues.

        Args:
            feats (torch.Tensor): Features with shape (B, K, D)
            embeddings (torch.Tensor): Embeddings with shape (N, D)
            batch_size (int): mini-batch size for computing the cosine similarity
            normalize (bool): Whether to normalize the cosine similarity maps
        '''

        num_feats = feats.shape[0]
        num_classes = embeddings.shape[0]
        patch_2d_size = int(np.sqrt(feats.shape[1]))

        cosim_list = []
        for start_idx in range(0, num_classes, batch_size):
            end_idx = min(start_idx + batch_size, num_classes)

            embedding_batch = embeddings[start_idx:end_idx]  # Get a batch of embeddings

            # Reshape and broadcast for cosine similarity
            B, K, D = feats.shape
            features_reshaped = feats.view(B, 1, K, D)
            embedding_reshaped = embedding_batch.view(1, end_idx - start_idx, 1, D)
            
            # if 'customclip' in self.backbone_type:
            #     print('here')
            #     dot_product = (features_reshaped @ embedding_reshaped).squeeze(-1)
            #     dot_product *= self.logit_scale.exp()
            # else:
            
            # print('in classifier')
            # print(feats.shape)
            # print(embedding_batch.shape)
            # print(features_reshaped.shape)
            # print(embedding_reshaped.shape)
            # print(torch.mean(features_reshaped)) #image
            # print(torch.mean(embedding_reshaped)) #text
            
            # Compute dot product (cosine similarity without normalization)
            dot_product = (features_reshaped * embedding_reshaped).sum(dim=3)
            
            #print(dot_product.shape)

            if normalize and not 'customclip' in self.backbone_type:
                # Compute norms
                feats_norm = torch.norm(features_reshaped, dim=3, keepdim=True).squeeze(-1)
                embedding_norm = torch.norm(embedding_reshaped, dim=3, keepdim=True).squeeze(-1)
                # Normalize
                dot_product /= (feats_norm * embedding_norm + 1e-8)  # Add epsilon for numerical stability
                
            if self.logit_scale:
                dot_product *= self.logit_scale.exp()
                
            #print(torch.mean(dot_product))
            
            # Append the similarity scores for this batch to the list
            cosim_list.append(dot_product)

        # Concatenate the similarity scores from different batches
        cosim = torch.cat(cosim_list, dim=1)

        # Reshape to 2D and return class similarity maps
        cosim = cosim.reshape(-1, num_classes, patch_2d_size, patch_2d_size)

        # Interpolate cosine similarity maps to original resolution
        cosim = F.interpolate(cosim, size=self.target_size, mode='bicubic')

        return cosim
    
    
class OVDBoxClassifier(OVDBaseClassifier):
    def __init__(self, prototypes, class_names, backbone_type='dinov2', target_size=(602,602), scale_factor=2, min_box_size=5, ignore_index=-1, text=False):
        super().__init__(prototypes, class_names, backbone_type, target_size, scale_factor, min_box_size, ignore_index, text)
        
    def forward(self, images, boxes, cls=None, normalize=False, return_cosim=False, aggregation='mean', k=10):
        '''
        Args:
            images (torch.Tensor): Input tensor with shape (B, C, H, W)
            boxes (torch.Tensor): Box coordinates with shape (B, max_boxes, 4)
            cls (torch.Tensor): Class labels with shape (B, max_boxes)
            normalize (bool): Whether to normalize the cosine similarity maps
            return_cosim (bool): Whether to return the cosine similarity maps
        '''
        
        scales = []

        for scale in self.scale_factor:
            feats = extract_backbone_features(images, self.backbone, self.backbone_type, scale_factor=scale, text=self.text)
            cosine_sim = self.get_cosim_mini_batch(feats, self.embedding, normalize=normalize)
            scales.append(cosine_sim)

        cosine_sim = torch.stack(scales).mean(dim=0)

         # Gather similarity values inside each box and compute average box similarity
        box_similarities = []
        B = images.shape[0]
        for b in range(B):
            image_boxes = boxes[b][:, :4]
    
            image_similarities = []
            count = 0
            for i, box in enumerate(image_boxes):
                x_min, y_min, x_max, y_max = box.int()
                
                if cls is not None:
                    if (x_min < 0 or
                        y_min < 0 or
                        y_max > self.target_size[0] or
                        x_max > self.target_size[0] or
                        y_max - y_min < self.min_box_size or
                        x_max - x_min < self.min_box_size):
                        count += 1
                        cls[b][i] = self.ignore_index   # If invalid box, assign the label to ignore it while computing the loss
                        image_similarities.append(torch.zeros(self.num_classes, device=images.device))
                        continue
                
                if aggregation == 'mean':
                    box_sim = cosine_sim[b, :, y_min:y_max, x_min:x_max].mean(dim=[1, 2])
                elif aggregation == 'max':
                    _,n,h,w = cosine_sim.shape
                    box_sim, _ = cosine_sim[b, :, y_min:y_max - 1, x_min:x_max - 1].reshape(n, -1).max(dim=1)
                elif aggregation == 'topk':
                    _,n,h,w = cosine_sim.shape
                    box_sim = cosine_sim[b, :, y_min:y_max - 1, x_min:x_max - 1].reshape(n, -1)
                    topk = k if k <= box_sim.shape[1] else box_sim.shape[1]
                    box_sim, _ = box_sim.topk(topk, dim=1)
                    box_sim = box_sim.mean(dim=1)
                else:
                    raise ValueError('Invalid aggregation method')
                
                has_nan = torch.isnan(box_sim).any().item()
                image_similarities.append(box_sim)
    
            box_similarities.append(torch.stack(image_similarities))
    
        box_similarities = torch.stack(box_similarities)  # Shape: [B, max_boxes, N]
        
        if return_cosim:
            return box_similarities.view(b, -1, self.num_classes), cosine_sim

        # Flatten box_logits and target_labels
        return box_similarities.view(B, -1, self.num_classes)
    
    
class OVDImageClassifier(OVDBaseClassifier):
    def __init__(self, prototypes, class_names, backbone_type='dinov2', target_size=(602,602), scale_factor=2, min_box_size=5, ignore_index=-1, text=False):
        super().__init__(prototypes, class_names, backbone_type, target_size, scale_factor, min_box_size, ignore_index, text)

    def forward(self, images, normalize=False, return_cosim=False, aggregation='mean', k=10):
        '''
        Args:
            images (torch.Tensor): Input tensor with shape (B, C, H, W)
            normalize (bool): Whether to normalize the cosine similarity maps
            return_cosim (bool): Whether to return the full cosine similarity maps
            aggregation (str): Aggregation method ('mean', 'max', 'topk')
            k (int): Number of top values to consider if aggregation is 'topk'
        
        Returns:
            image_similarities (torch.Tensor): Shape (B, N) – Aggregated similarity per image
            cosine_sim (torch.Tensor, optional): Shape (B, N, H, W) – Cosine similarity map (if return_cosim=True)
        '''
        
        scales = []

        for scale in self.scale_factor:
            feats = extract_backbone_features(images, self.backbone, self.backbone_type, scale_factor=scale, text=self.text)
            cosine_sim = self.get_cosim_mini_batch(feats, self.embedding, normalize=normalize)
            scales.append(cosine_sim)

        # Average across scales
        cosine_sim = torch.stack(scales).mean(dim=0)  # Shape: (B, N, H, W)

        # Apply aggregation
        if aggregation == 'mean':
            image_similarities = cosine_sim.mean(dim=[2, 3])  # Mean across spatial dimensions
        elif aggregation == 'max':
            image_similarities, _ = cosine_sim.flatten(2).max(dim=2)  # Max across spatial dimensions
        else:
            raise ValueError('Invalid aggregation method')

        if return_cosim:
            return image_similarities, cosine_sim  # Return both aggregated scores and full cosine similarity map

        return image_similarities  # Only return aggregated similarity scores



class OVDMaskClassifier(OVDBaseClassifier):
    def __init__(self, prototypes, class_names, backbone_type='dinov2', target_size=(602,602), scale_factor=2, min_box_size=5, ignore_index=-1):
        super().__init__(prototypes, class_names, backbone_type, target_size, scale_factor, min_box_size, ignore_index)
        
    def forward(self, images, masks, cls=None, normalize=False, return_cosim=False, aggregation='mean', k=10, batch_size=50):
        '''
        Args:
            images (torch.Tensor): Input tensor with shape (B, C, H, W)
            masks (torch.Tensor): Masks to classify with shape (B, max_masks, H, W)
            cls (torch.Tensor): Class labels with shape (B, max_masks)
            normalize (bool): Whether to normalize the cosine similarity maps
            return_cosim (bool): Whether to return the cosine similarity maps
            aggregation (str): Aggregation method for combining cosine similarity maps
            k (int): Number of top-k elements to consider if aggregation is 'topk'
            batch_size (int): Mini-batch size for processing masks
        '''

        # Initialize lists to store results from each mini-batch
        batch_mask_sims = []

        # Split masks into mini-batches
        num_batches = (masks.shape[1] + batch_size - 1) // batch_size
        mask_batches = torch.split(masks, batch_size, dim=1)

        for mask_batch in mask_batches:
            # Compute cosine similarity maps for the mini-batch
            cosine_sim = self.compute_cosine_similarity(images, mask_batch, normalize)

            # Multiply cosine similarity maps with masks element-wise
            masked_cosine_similarity = cosine_sim.unsqueeze(1) * mask_batch.unsqueeze(2)  # Shape: [B, M, N, H, W]

            # Aggregate masked cosine similarity maps
            mask_sim = self.aggregate(masked_cosine_similarity, aggregation, k)

            # Append the result of the mini-batch to the list
            batch_mask_sims.append(mask_sim)

        # Concatenate results from all mini-batches
        mask_sim = torch.cat(batch_mask_sims, dim=1)

        if return_cosim:
            return mask_sim, cosine_sim

        return mask_sim

    def compute_cosine_similarity(self, images, masks, normalize):
        scales = []

        for scale in self.scale_factor:
            feats = extract_backbone_features(images, self.backbone, self.backbone_type, scale_factor=scale)
            cosine_sim = self.get_cosim_mini_batch(feats, self.embedding, normalize=normalize)
            scales.append(cosine_sim)

        # Compute cosine similarity maps
        cosine_sim = torch.stack(scales).mean(dim=0)
        return cosine_sim

    def aggregate(self, masked_cosine_similarity, aggregation, k):
        if aggregation == 'mean':
            # Compute average cosine similarity values within masked areas
            mask_sim = masked_cosine_similarity.sum(dim=(-2, -1)) / (masked_cosine_similarity.shape[3] * masked_cosine_similarity.shape[4] + 1e-6)  # Shape: [B, M, N]
        elif aggregation == 'max':
            mask_sim, _ = torch.max(masked_cosine_similarity.view(masked_cosine_similarity.shape[0], masked_cosine_similarity.shape[1], masked_cosine_similarity.shape[2], -1), dim=-1)
        elif aggregation == 'topk':
            topk = min(k, masked_cosine_similarity.shape[3] * masked_cosine_similarity.shape[4])
            mask_sim, _ = torch.topk(masked_cosine_similarity.view(masked_cosine_similarity.shape[0], masked_cosine_similarity.shape[1], masked_cosine_similarity.shape[2], -1), topk, dim=-1)
            mask_sim = mask_sim.mean(dim=-1)
        else:
            raise ValueError('Invalid aggregation method')

        return mask_sim
    
"""
class ZeroShotClassifier(torch.nn.Module):
     def __init__(self, prototypes, class_names, backbone_type='clip-14', target_size=(602,602), scale_factor=2, min_box_size=5, ignore_index=-1):
        super().__init__()
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.min_box_size = min_box_size
        self.ignore_index = ignore_index
        self.backbone_type = backbone_type
        self.class_names = class_names
        
        if isinstance(self.scale_factor, int):
            self.scale_factor = [self.scale_factor]

        # Initialize backbone
        self.backbone = load_backbone(backbone_type)  
        
        # Initialize embedding as a learnable parameter
        self.embedding = torch.nn.Parameter(prototypes)
        self.num_classes = self.embedding.shape[0]
        
    def get_categories(self):
        return {idx: label for idx, label in enumerate(self.class_names)}
    
    def forward(self, images, boxes, text_embed, cls=None, normalize=False, return_cosim=False, aggregation='mean', k=10):
        '''
        Args:
            images (torch.Tensor): Input tensor with shape (B, C, H, W)
            boxes (torch.Tensor): Box coordinates with shape (B, max_boxes, 4)
            cls (torch.Tensor): Class labels with shape (B, max_boxes)
            normalize (bool): Whether to normalize the cosine similarity maps
            return_cosim (bool): Whether to return the cosine similarity maps
        '''
        
        # Text embed should be all the clip embedded class names
        # Extract bounding box cropping from images
        # Resize each crop to 224x224
        # Get CLIP image embedding of each crop
        # Get cos sim between image and all embedded class texts and get max to be the predicted class
        
        scales = []

        for scale in self.scale_factor:
            feats = extract_backbone_features(images, self.backbone, self.backbone_type, scale_factor=scale)
            cosine_sim = self.get_cosim_mini_batch(feats, self.embedding, normalize=normalize)
            scales.append(cosine_sim)

        cosine_sim = torch.stack(scales).mean(dim=0)

         # Gather similarity values inside each box and compute average box similarity
        box_similarities = []
        B = images.shape[0]
        for b in range(B):
            image_boxes = boxes[b][:, :4]
    
            image_similarities = []
            count = 0
            for i, box in enumerate(image_boxes):
                x_min, y_min, x_max, y_max = box.int()
                
                if cls is not None:
                    if (x_min < 0 or
                        y_min < 0 or
                        y_max > self.target_size[0] or
                        x_max > self.target_size[0] or
                        y_max - y_min < self.min_box_size or
                        x_max - x_min < self.min_box_size):
                        count += 1
                        cls[b][i] = self.ignore_index   # If invalid box, assign the label to ignore it while computing the loss
                        image_similarities.append(torch.zeros(self.num_classes, device=images.device))
                        continue
                
                if aggregation == 'mean':
                    box_sim = cosine_sim[b, :, y_min:y_max, x_min:x_max].mean(dim=[1, 2])
                elif aggregation == 'max':
                    _,n,h,w = cosine_sim.shape
                    box_sim, _ = cosine_sim[b, :, y_min:y_max - 1, x_min:x_max - 1].reshape(n, -1).max(dim=1)
                elif aggregation == 'topk':
                    _,n,h,w = cosine_sim.shape
                    box_sim = cosine_sim[b, :, y_min:y_max - 1, x_min:x_max - 1].reshape(n, -1)
                    topk = k if k <= box_sim.shape[1] else box_sim.shape[1]
                    box_sim, _ = box_sim.topk(topk, dim=1)
                    box_sim = box_sim.mean(dim=1)
                else:
                    raise ValueError('Invalid aggregation method')
                
                has_nan = torch.isnan(box_sim).any().item()
                image_similarities.append(box_sim)
    
            box_similarities.append(torch.stack(image_similarities))
    
        box_similarities = torch.stack(box_similarities)  # Shape: [B, max_boxes, N]
        
        if return_cosim:
            return box_similarities.view(b, -1, self.num_classes), cosine_sim

        # Flatten box_logits and target_labels
        return box_similarities.view(B, -1, self.num_classes)
"""