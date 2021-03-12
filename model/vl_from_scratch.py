"""
UNITER for end2end transformer pretraining with ViT visual backbone
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import F, functional

from .model import PretrainedModel
from .pretrain import UniterForPretraining
from .visual_transformer import ViT


class VLFromScratch(PretrainedModel):
    """ ViT Image Feature Generation with UNITER pretraining """
    def __init__(self, config, img_dim, img_label_dim, input_resolution):
        super().__init__(config)
        self.visual_transformer = ViT(config, img_dim, input_resolution)
        self.uniter = UniterForPretraining(config, img_dim, img_label_dim)
        self.apply(self.init_weights)

    def forward(self, batch, task, visual_feature_type="patch", compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        images = batch['image_tensors']

        # Compute normalized ViT image features
        vit_image_features = self.visual_transformer.forward(images)
        vit_image_features = vit_image_features / vit_image_features.norm(dim=-1, keepdim=True)
        if visual_feature_type == "patch":
            batch['img_feat'] = vit_image_features[:, 1:, :]  # Shape = [batch, grid ** 2, img_dim]
        elif visual_feature_type == "cls":
            batch['img_feat'] = vit_image_features[:, 0, :]  # Shape = [batch, 1, img_dim]

        uniter_output = self.uniter.forward(batch, task, compute_loss)
        return uniter_output
