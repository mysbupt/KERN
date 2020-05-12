import os
import json
import yaml
import numpy as np
import random
random.seed(123)

import torch
torch.manual_seed(123)
from torch.utils.data import Dataset, DataLoader


class TrendDataset(Dataset):
    def __init__(self, conf, data, ids, norm, grps, eles, dist_mat):
        self.conf = conf
        self.data = data
        self.ids = ids
        self.norm = norm
        self.grps = grps
        self.eles = eles
        self.dist_mat = dist_mat
        self.neibors_mat = np.argsort(dist_mat, axis=1)
        self.ttl_num = len(data)
        self.seq_num = dist_mat.shape[0]

    def __len__(self):
        return len(self.data)

    def return_one(self, idx):
        each = self.data[idx]
        each_input = torch.Tensor(each[:self.conf["input_len"]])
        each_output = torch.Tensor(each[self.conf["input_len"]:])
        each_grp = torch.LongTensor([self.grps[idx]])
        each_ele = torch.LongTensor([self.eles[idx]])
        each_norm = torch.FloatTensor(self.norm[idx])

        _id = self.ids[idx]

        return [each_input, each_output, each_grp, each_ele, each_norm], _id

    def __getitem__(self, idx):
        self_cont, _id = self.return_one(idx)

        neibors = self.neibors_mat[_id].tolist()

        start_point = random.sample([x for x in range(self.seq_num-self.conf["sample_range"])], 1)[0]
        end_point = start_point + self.conf["sample_range"]
        sample_neibors = random.sample(neibors[start_point:end_point], 3)
        filtered_neibors = []
        for x in sample_neibors:
            if x != _id:
                filtered_neibors.append(x)
        filtered_neibors = sorted(filtered_neibors)

        close_item = filtered_neibors[0]
        ori_close_score = self.dist_mat[_id][close_item]
        close_score = torch.FloatTensor([ori_close_score])
        close_item_new = idx - (idx % self.seq_num) + close_item
        close_cont, _ = self.return_one(close_item_new)

        far_item = filtered_neibors[1]
        ori_far_score = self.dist_mat[_id][far_item]
        far_score = torch.FloatTensor([ori_far_score])
        far_item_new = idx - (idx % self.seq_num) + far_item
        far_cont, _ = self.return_one(far_item_new)

        if far_score >= close_score:
            return self_cont, close_cont, far_cont, close_score, far_score
        else:
            return self_cont, far_cont, close_cont, far_score, close_score


class TrendTestDataset(Dataset):
    def __init__(self, conf, data, norm, grps, eles):
        self.conf = conf
        self.data = data
        self.norm = norm
        self.grps = grps
        self.eles = eles

    def __len__(self):
        return len(self.data)

    def return_one(self, idx):
        each = self.data[idx]
        each_input = torch.Tensor(each[:self.conf["input_len"]])
        each_output = torch.Tensor(each[self.conf["input_len"]:])
        each_grp = torch.LongTensor([self.grps[idx]])
        each_ele = torch.LongTensor([self.eles[idx]])
        each_norm = torch.FloatTensor(self.norm[idx])

        return [each_input, each_output, each_grp, each_ele, each_norm]

    def __getitem__(self, idx):
        self_cont = self.return_one(idx)
        return self_cont


class TrendData(Dataset):
    def __init__(self, conf):
        self.conf = conf
        trends, grp_ids, ele_ids, self.time_num, trend_norm, self.grp_id_map, self.ele_id_map = self.get_ori_data()
        train_data, train_ids, train_norm, train_grps, train_eles, test_data, test_ids, test_norm, test_grps, test_eles = self.preprocess_data(trends, trend_norm, grp_ids, ele_ids)
        print(train_grps.shape)
        seq_num = self.dist_mat.shape[0]
        self.val_idx = random.sample([x for x in range(seq_num)], int(seq_num/2))
        self.test_idx = []
        for x in range(seq_num):
            if x not in self.val_idx:
                self.test_idx.append(x)
        self.train_set = TrendDataset(conf, train_data, train_ids, train_norm, train_grps, train_eles, self.dist_mat)
        self.train_loader = DataLoader(self.train_set, batch_size=conf["batch_size"], shuffle=True, num_workers=10)
        self.test_set = TrendTestDataset(conf, test_data, test_norm, test_grps, test_eles)
        self.test_loader = DataLoader(self.test_set, batch_size=conf["batch_size"], shuffle=True, num_workers=10)


    def get_ori_data(self):
        all_data = json.load(open(self.conf["data_path"]))
        all_data_norm = json.load(open(self.conf["data_norm_path"]))

        trends, grp_ids, ele_ids, trend_norm = [], [], [], []
        grp_ele_id, idx = {}, 0
        time_num = 0
        grp_id_map, ele_id_map = {}, {}
        for group_name, res in all_data.items():
            if group_name not in grp_id_map:
                grp_id_map[group_name] = len(grp_id_map)
            curr_grp_id = grp_id_map[group_name] 

            grp_ele_id[group_name] = {}
            for fashion_ele, seq in res.items():
                time_seq = [x[0] for x in seq]
                each_time_num = max(time_seq) + 1
                if each_time_num > time_num:
                    time_num = each_time_num

                if fashion_ele not in ele_id_map:
                    ele_id_map[fashion_ele] = len(ele_id_map)
                curr_ele_id = ele_id_map[fashion_ele]
                trends.append(seq)

                norm = all_data_norm[group_name][fashion_ele]
                norm = [float(x) for x in norm]
                trend_norm.append(norm)

                grp_ids.append(curr_grp_id)
                ele_ids.append(curr_ele_id)

                grp_ele_id[group_name][fashion_ele] = idx
                idx += 1

        trends = np.array(trends)

        if os.path.exists(self.conf["dist_mat_path"]):
            self.dist_mat = np.load(self.conf["dist_mat_path"])
        else:
            trends_for_train = trends[:, :-self.conf["output_len"], 1]
            self.dist_mat = self.generate_dist_mat(trends_for_train)
            np.save(self.conf["dist_mat_path"], self.dist_mat)

        json.dump(grp_ele_id, open("test_grp_ele_id_map_geostyle.json", "w"), indent=4)
        self.all_train_seq = trends[:, :-self.conf["output_len"], 1]
        return trends, np.array(grp_ids), np.array(ele_ids), time_num, np.array(trend_norm), grp_id_map, ele_id_map

    def generate_dist_mat(self, all_train):
        # all_train: [n_len, seq_len]
        n_len = all_train.shape[0]
        dist_mat = []
        for a_id, a in enumerate(all_train):
            a_broad = np.repeat(a[np.newaxis, :], n_len, axis=0) # [n_len, seq_len]
            mape = np.mean(np.abs(a_broad - all_train) / all_train, axis=-1)*100 # [n_len]
            dist_mat.append(mape)
        dist_mat = np.stack(dist_mat, axis=0) # [n_len, n_len]
        return dist_mat

    def preprocess_data(self, trends, trend_norm, grp_ids, ele_ids):
        ori_seq_len = trends.shape[1]
        ttl_len = self.conf["input_len"] + self.conf["output_len"]
        output_len = self.conf["output_len"]
        assert ori_seq_len > ttl_len + output_len
        train_data, train_ids, train_grps, train_eles, train_norm = [], [], [], [], []
        for i in range(ori_seq_len-ttl_len-output_len):
            train_data.append(trends[:, i:i+ttl_len])
            train_ids.append(np.array([j for j in range(trends.shape[0])]))
            train_norm.append(trend_norm)
            train_grps.append(grp_ids)
            train_eles.append(ele_ids)
        train_data = np.concatenate(train_data, axis=0)
        train_ids = np.concatenate(train_ids, axis=0)
        train_norm = np.concatenate(train_norm, axis=0)
        train_grps = np.concatenate(train_grps, axis=0)
        train_eles = np.concatenate(train_eles, axis=0)

        test_ids = np.array([j for j in range(trends.shape[0])])
        test_data = trends[:, -ttl_len:]
        test_norm = trend_norm[:, -ttl_len:]
        test_grps = grp_ids
        test_eles = ele_ids
        print("train data: ", train_data.shape)
        print("test data: ", test_data.shape)
        return train_data, train_ids, train_norm, train_grps, train_eles, test_data, test_ids, test_norm, test_grps, test_eles
