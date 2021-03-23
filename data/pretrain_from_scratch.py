import torch
from toolz.sandbox import unzip
from torch.nn.utils.rnn import pad_sequence

from .data import DetectFeatTxtTokDataset, get_gather_index, pad_tensors
from .mlm import random_word
from .mrm import (_get_targets, get_feat_target, get_img_mask,
                  get_img_tgt_mask, mask_img_feat)


class VLFromScratchPretrainDataset(DetectFeatTxtTokDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_input_ids(self, txt_dump, mask=False):
        # text input
        input_ids = txt_dump['input_ids']
        type_ids = [0]*len(input_ids)
        if mask:
            input_ids, txt_labels = random_word(input_ids, self.txt_db.v_range, self.txt_db.mask)
        else:
            txt_labels = input_ids
        if mask:
            return input_ids, txt_labels
        else:
            return input_ids


def vl_from_scratch_pretrain_collate(input_ids, img_feats, img_pos_feats, attn_masks):
    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)  # TODO: Is this correct for raw image tensors?
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_tensors': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index}
    return batch


class MlmDatasetForVLFromScratch(VLFromScratchPretrainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_mlm_io(self, example):
        (input_ids, txt_type_ids, txt_labels) = self._get_input_ids(example, mask=True)
        return self.txt_db.combine_inputs(input_ids, txt_labels)

    def __getitem__(self, i):
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(example['img_fname'][0])

        # txt inputs, create mlm io
        input_ids, txt_labels = self.create_mlm_io(example)

        # Create attn mask over length of text tokens + special tokens + nbb's
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, txt_labels


def mlm_collate_for_vl_from_scratch(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks,
     txt_labels) = map(list, unzip(inputs))
    batch = vl_from_scratch_pretrain_collate(
        input_ids, txt_type_ids, img_feats,
        img_pos_feats, attn_masks)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)

    batch['txt_labels'] = txt_labels
    return batch


class MrfrDatasetForVLFromScratch(VLFromScratchPretrainDataset):
    def __init__(self, mask_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_prob = mask_prob

    def __getitem__(self, i):
        example = super().__getitem__(i)
        # text input
        input_ids, txt_type_ids = self._get_input_ids(example, mask=False)
        input_ids, txt_type_ids = self.combine_txt_inputs(
            input_ids, txt_type_ids)

        # image input features
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        img_mask = get_img_mask(self.mask_prob, num_bb)
        img_mask_tgt = get_img_tgt_mask(img_mask, len(input_ids))

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return (input_ids, txt_type_ids, img_feat, img_pos_feat,
                attn_masks, img_mask, img_mask_tgt)


def mrfr_collate_for_vl_from_scratch(inputs):
    (input_ids, txt_type_ids, img_feats, img_pos_feats,
     attn_masks, img_masks, img_mask_tgts) = map(list, unzip(inputs))

    batch = vcr_pretrain_collate(
        input_ids, txt_type_ids, img_feats,
        img_pos_feats, attn_masks)

    # mask features
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    feat_targets = get_feat_target(batch['img_feat'], img_masks)
    img_mask_tgt = pad_sequence(
        img_mask_tgts, batch_first=True, padding_value=0)
    batch['img_feat'] = mask_img_feat(batch['img_feat'], img_masks)
    batch['img_masks'] = img_masks
    batch['feat_targets'] = feat_targets
    batch['img_mask_tgt'] = img_mask_tgt

    return batch
