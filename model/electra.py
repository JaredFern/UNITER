from collections import defaultdict

import torch
from pretrain_vcr import UniterForPretrainingForVCR
from torch import nn

from model import PretrainedModel


class ElectraLoss(nn.Module):
    def __init__(self, loss_weights=(1.0, 50.0)):
        super().__init__()
        self.loss_weights = loss_weights
        self.gen_loss_fn = nn.CrossEntropyLoss()
        self.disc_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, gen_logits, disc_logits, attention_mask, gt_labels,
                is_masked, is_replaced):
        gen_loss = self.gen_loss_fn(gen_logits.float(), gt_labels[is_masked].float())
        disc_logits = disc_logits.masked_select(attention_mask)
        is_replaced = is_replaced.masked_select(attention_mask)
        disc_loss = self.disc_loss_fn(disc_logits.float(), is_replaced.float())
        return gen_loss * self.loss_weights[0] + disc_loss * self.loss_weights[1]


class UniterForElectraPretraining(PretrainedModel):
    def __init__(self, config, img_dim, img_label_dim):
        super().__init__(config)
        self.generator = UniterForPretrainingForVCR(config, img_dim, img_label_dim)
        self.discriminator = UniterForPretrainingForVCR(config, img_dim, img_label_dim)
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0., 1.)
        self.electra_loss = ElectraLoss()

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        if task == 'mlm':
            with torch.no_grad():
                txt_labels = batch['txt_labels']  # (B, L)
                is_mlm_applied = (txt_labels != -1)  # -1 if masking is not applied
                gen_logits = self.generator.forward_mlm(
                    input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, txt_labels, False)

                gen_tokens = self.generator.sample(gen_logits)  # (B, num_mlm_pos)
                generated = input_ids.clone()
                generated[txt_labels] = gen_tokens  # (B, L)

                is_replaced = is_mlm_applied.clone()
                is_replaced[is_mlm_applied] = (gen_tokens != txt_labels[is_mlm_applied])

            disc_logits = self.discriminator.forward_mlm(
                generated, position_ids, img_feat, img_pos_feat, attention_mask,
                gather_index, is_replaced, False)
            return self.electra_loss.forward_mlm(
                gen_logits, disc_logits, attention_mask, txt_labels,
                is_mlm_applied, is_replaced)
        elif task == 'mrfr':
            with torch.no_grad():
                img_mask_tgt = batch['img_mask_tgt']
                img_masks = batch['img_masks']
                mrfr_feat_target = batch['feat_targets']
                mrfr_gen_feats = self.forward_mrfr(
                    input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, img_masks, img_mask_tgt,
                    mrfr_feat_target, False)

                gen_feats = img_feat.clone()
                gen_feats[img_mask_tgt] = mrfr_gen_feats
                is_replaced = img_mask_tgt.clone()
            mrfr_disc_feats = self.discriminator.forward_mrfr(
                input_ids, position_ids, img_feat, img_pos_feat, attention_mask,
                gather_index, img_masks, img_mask_tgt, mrfr_feat_target, False)
            return self.electra_loss.mrfr_loss(
                mrfr_gen_feats, mrfr_disc_feats, attention_mask, gather_index,
                img_masks, img_feat, is_replaced, False)
        elif task == 'itm':
            # No Change: ITM Objective is already discriminative
            targets = batch['targets']
            ot_inputs = batch['ot_inputs']
            return self.discriminator.forward_itm(
                input_ids, position_ids, img_feat, img_pos_feat, attention_mask,
                gather_index, targets, ot_inputs, compute_loss)
        elif task.startswith('mrc'):
            # No change: MRC objective is already discriminative
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            return self.discriminator.forward_mrc(
                input_ids, position_ids, img_feat, img_pos_feat, attention_mask,
                gather_index, img_masks, img_mask_tgt, mrc_label_target, task,
                compute_loss)
        else:
            raise ValueError('invalid task')
