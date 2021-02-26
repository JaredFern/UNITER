from util import *
from tqdm import tqdm
from statistics import median


class Evaluator:
    def __init__(self, params, dataloader):
        self.params = params
        self.dataloader = dataloader
        self.factor = params.factor

    def i2t(self, model, is_test=False):
        model.eval()
        if is_test:
            ids = self.dataloader.test_ids
            dataloader = self.dataloader.test_dataloader
        else:
            ids = self.dataloader.val_ids
            dataloader = self.dataloader.eval_dataloader

        e_ts, e_vs, idd, caption_texts = self.get_embeddings(ids, dataloader, model)
        # construct plain e_vs => e_vsp
        e_vsp = []
        pids = [] # 1000
        idss = [] # 5000, it converts the text id to video id, and multiple text ids could project to the same video id.
        for n, id in enumerate(idd):
            pid = id.split('#')[0]
            idss.append(pid)
            if pid not in pids:
                pids.append(pid)
                e_vsp.append(e_vs[n])
        e_vsp = np.asarray(e_vsp)

        video_retrieved_captions = []
        len_pids = len(pids)
        pids = np.asarray(pids)
        idss = np.asarray(idss)
        sims = np.matmul(e_vsp, e_ts.T)
        correct_rank_list = []
        r_1, r_5, r_10 = 0., 0., 0.
        for n, id in enumerate(pids.tolist()):
            pid = id
            sorted_img_idx = (-np.asarray(sims[n])).argsort()  # The name is confusing, actually it is text idx.
            top_10_img_idx = sorted_img_idx[:10]
            top_pids = idss[top_10_img_idx]  # After converted by idss we get the videoid.
            if pid == top_pids[0]:
                r_1 += 1
                r_5 += 1
                r_10 += 1
            elif pid in top_pids[1:5]:
                r_5 += 1
                r_10 += 1
            elif pid in top_pids[5:10]:
                r_10 += 1
            correct_rank_list.append(Evaluator.get_correct_rank(pid, idss[sorted_img_idx]))

            if self.params.visualize_res:
                video_retrieved_captions.append((pid,
                                                 [caption_texts[idx] for idx in sorted_img_idx[:5] if idss[idx] == pid],
                                                 [caption_texts[i] for i in top_10_img_idx[:5]]))

        med_r = Evaluator.get_median(correct_rank_list)
        mean_r = Evaluator.get_mean(correct_rank_list)

        if self.params.visualize_res:
            out_file = 'i2t_{}'.format(self.params.retrive_out_file)
            with open(out_file, 'w') as f:
                for video, label_text, top_k_texts in video_retrieved_captions:
                    f.write('{0} {1}     {2}\n'.format(video, label_text, top_k_texts))
            print('The retrieved results has been written to {}'.format(out_file))

        return r_1 / len_pids, r_5 / len_pids, r_10 / len_pids, med_r, mean_r

    def caption_id_to_text(self, caption_ids):
        if len(caption_ids) == 0:
            return caption_ids
        idx2word = {idx: word for word, idx in self.dataloader.vocab.items()}
        return [' '.join([idx2word[word_id - 2] if word_id >= 2 else '<unk>' for word_id in caption]) for caption in caption_ids]

    def get_embeddings(self, ids, dataloader, model):
        e_vs = np.zeros((len(ids), self.factor*self.params.hidden_dim))
        e_ts = np.zeros((len(ids), self.factor*self.params.hidden_dim))

        count = 0
        idd = []
        caption_ids = []  # It is only used when we want to visualize the retrieval results.
        # inference and get embeddings
        for cap, cmask, vfeat, vmask, id in tqdm(dataloader, ascii=True, dynamic_ncols=True,
                                                 disable=self.params.disable_tqdm):
            cap, cmask, vfeat, vmask = to_cuda(cap, cmask, vfeat, vmask)
            idd += id
            e_t, e_v, _, _ = model(cap, cmask, vfeat, vmask)
            e_ts[count:count+len(id)] = e_t.data.cpu().numpy()
            e_vs[count:count+len(id)] = e_v.data.cpu().numpy()
            count += len(id)

            if self.params.visualize_res:
                cap = cap.data.cpu().numpy()
                lengths = np.sum(cmask.data.cpu().numpy(), axis=1, keepdims=False)
                caption_ids.extend([it_cap[:lengths[idx]] for idx, it_cap in enumerate(cap)])

        # Note that the len(idd) (real number of ids) might be different from the len(ids)
        # because some ids are removed if the corresponding videos doesn't exist.
        print('total len of idd:', len(idd))
        e_ts = e_ts[:len(idd), :]
        e_vs = e_vs[:len(idd), :]

        return e_ts, e_vs, idd, self.caption_id_to_text(caption_ids)

    @staticmethod
    def get_correct_rank(target, candidate_list):
        for idx, candidate in enumerate(candidate_list):
            if candidate == target:
                return idx
        raise ValueError("The target should be in the candidate_list!")

    @staticmethod
    def get_median(value_list):
        return median(value_list)

    @staticmethod
    def get_mean(value_list):
        return sum(value_list) / len(value_list)

    @staticmethod
    def t2i_remove_duplicate(idd, e_vs):
        # construct plain e_vs => e_vsp
        e_vsp = []
        pids = []
        for n, id in enumerate(idd):
            pid = id.split('#')[0]
            if pid not in pids:
                pids.append(pid)
                e_vsp.append(e_vs[n])
        e_vsp = np.asarray(e_vsp)
        pids = np.array(pids)
        return e_vsp, pids

    def t2i(self, model, is_test=False):
        model.eval()
        if is_test:
            ids = self.dataloader.test_ids
            dataloader = self.dataloader.test_dataloader
        else:
            ids = self.dataloader.val_ids
            dataloader = self.dataloader.eval_dataloader

        e_ts, e_vs, idd, caption_texts = self.get_embeddings(ids, dataloader, model)
        e_vsp, pids = Evaluator.t2i_remove_duplicate(idd, e_vs)

        caption_retrieved_videos = []
        sims = np.matmul(e_ts, e_vsp.T)
        correct_rank_list = []
        r_1, r_5, r_10 = 0., 0., 0.
        for n, id in enumerate(idd):
            pid = id.split('#')[0]
            sorted_img_idx = (-np.asarray(sims[n])).argsort()
            top_10_img_idx = sorted_img_idx[:10]
            top_pids = pids[top_10_img_idx]
            if pid == top_pids[0]:
                r_1 += 1
                r_5 += 1
                r_10 += 1
            elif pid in top_pids[1:5]:
                r_5 += 1
                r_10 += 1
            elif pid in top_pids[5:10]:
                r_10 += 1
            correct_rank_list.append(Evaluator.get_correct_rank(pid, pids[sorted_img_idx]))

            if self.params.visualize_res:
                caption_retrieved_videos.append((caption_texts[n], pid, top_pids[:5]))

        med_r = Evaluator.get_median(correct_rank_list)
        mean_r = Evaluator.get_mean(correct_rank_list)

        if self.params.visualize_res:
            out_file = 't2i_{}'.format(self.params.retrive_out_file)
            with open(out_file, 'w') as f:
                for caption_text, label_video, top_k_videos in caption_retrieved_videos:
                    f.write('{0} {1}     {2}\n'.format(caption_text, label_video, top_k_videos))
            print('The retrieved results has been written to {}'.format(out_file))

        return r_1 / len(ids), r_5 / len(ids), r_10 / len(ids), med_r, mean_r

    def t2i_inf(self, model, out_file, inf_type):
        """ Get the embedding by inference and store the embedding to disk.

        Used for ranking videos according to a query (for example, IACC3 dataset for AVS contest).
        Note that in the video inference, all texts are the same (it is the query we want to retrieve videos).
        And for test inference, all videos are the same.

        Args:
            model: The model used to do inference.
            out_file: The outfile to store the embedding.
            inf_type: 'text' or 'video'. If 'text', save text embedding; If 'video', save video embedding.
        """
        import pickle

        model.eval()
        ids = self.dataloader.inf_ids
        dataloader = self.dataloader.inf_dataloader

        e_ts, e_vs, idd, _ = self.get_embeddings(ids, dataloader, model)

        # We store the embedding to disk to avoid further redundant re-calculations.
        assert inf_type == 'text' or inf_type == 'video'
        out_embed = e_ts if inf_type == 'text' else e_vs
        with open(out_file, 'wb') as f:
            pickle.dump(out_embed, f, protocol=4)  # Use protocal 4 for the data larger than 4GB.
        print('The {0} embeddings with shape {1} is pickled to {2}.'.format(inf_type, out_embed.shape, out_file))
        # Also store the video ids for the video embedding.
        if inf_type == 'video':
            video_id_out_file = '{}_ids.txt'.format(out_file[:-4])
            with open(video_id_out_file, 'w') as fout:
                for video_id in idd:
                    fout.write('{}\n'.format(video_id.split("#")[0]))
            print('The corresponding video id is written to {}'.format(video_id_out_file))

        # The video name (including #) is stored in idd. e_ts is [video_num, embed_dim], e_vs is [video_num, embed_dim].
        # sim_score = np.sum(e_ts * e_vs, axis=1, keepdims=False)
        # top_k_idx = (-sim_score).argsort()[:top_k]
        # top_k_video_name = [idd[idx].split('#')[0] for idx in top_k_idx]
        # with open(out_file, 'w') as f:
        #     for video_name in top_k_video_name:
        #         f.write("{}\n".format(video_name))
        # print('The top {0}/{1} video names are written to {2}.'.format(top_k, len(idd), out_file))
