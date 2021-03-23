"""
UNITER for end2end transformer pretraining with ViT visual backbone
"""
from collections import defaultdict

from data.data import get_gather_index, pad_tensors

from .model import PretrainedModel
from .pretrain import UniterForPretraining
from .visual_transformer import ViT

# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.nn.utils.rnn import pad_sequence



def get_feat_target(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)  # (n, m, d)
    feat_dim = img_feat.size(-1)
    feat_targets = img_feat[img_masks_ext].contiguous().view(
        -1, feat_dim)  # (s, d)
    return feat_targets


def mask_img_feat(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
    return img_feat_masked


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
            img_feat = vit_image_features[:, 1:, :]  # Shape = [batch, grid ** 2, img_dim]
        elif visual_feature_type == "cls":
            img_feat = vit_image_features[:, 0, :]  # Shape = [batch, 1, img_dim]
        else:
            raise ValueError('Invalid visual feature type')

        # Process ViT features according to task to create UNITER pretraining data
        if task == "mlm":
            batch['img_feat'] = img_feat
        elif task == "mrfr":
            img_masks = batch['img_masks']
            feat_targets = mrm.get_feat_target(img_feat, img_masks)
            img_feat = mrm.mask_img_feat(img_feat, img_masks)
            batch['img_feat'] = img_feat
            batch['feat_targets'] = feat_targets
        elif task == "itm":
            # Not implementing ITM hard negatives, only used for Retrieval task-specific finetuning
            # TODO(JaredFern): Implement ItmVal and ItmEval Datasets for downstream experiments
            attn_masks = batch['attn_masks']
            input_ids = batch['input_ids']
            img_pos_feats = batch['img_pos_feat']

            txt_lens = [i.size(0) for i in input_ids]
            num_bbs = [feat.size(0) for feat in img_feat]
            bs, max_tl = input_ids.size()
            out_size = attn_masks.size(1)

            img_feat = pad_tensors(img_feat, num_bbs)
            img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
            gather_index = get_gather_index(txt_lens, num_bbs, num_bbs, max_tl, out_size)

            batch['img_feat'] = img_feat
            batch['img_pos_feat'] = img_pos_feat
            batch['gather_index'] = gather_index
        else:
            # MRC Task is invalid for unsupervised patch-based visual features
            raise ValueError('Invalid task')

        uniter_output = self.uniter.forward(batch, task, compute_loss)
        return uniter_output
