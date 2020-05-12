import json
import yaml
import numpy as np
import random
random.seed(1234)

import torch
torch.manual_seed(1234)
from torch.utils.data import Dataset, DataLoader


class TrendDataset(Dataset):
    def __init__(self, conf, data, ids, norm, grps, eles, city, gender, age, dist_mat):
        self.conf = conf
        self.data = data
        self.ids = ids
        self.norm = norm
        self.grps = grps
        self.city = city
        self.gender = gender
        self.age = age
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
        each_city = torch.LongTensor([self.city[idx]])
        each_gender = torch.LongTensor([self.gender[idx]])
        each_age = torch.LongTensor([self.age[idx]])
        each_ele = torch.LongTensor([self.eles[idx]])
        each_norm = torch.FloatTensor(self.norm[idx])

        _id = self.ids[idx]

        return [each_input, each_output, each_grp, each_ele, each_norm, each_city, each_gender, each_age], _id

    def __getitem__(self, idx):
        self_cont, _id = self.return_one(idx)

        neibors = self.neibors_mat[_id].tolist()
        # original sample methods, which is not effective
        """
        close_neibors = random.sample(neibors[0:self.thresh].tolist(), 2)
        far_neibors = random.sample(neibors[-self.thresh:].tolist(), 2)
        """
 
        # new sample methods, which can do hard negtive sample
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
    def __init__(self, conf, data, norm, grps, eles, city, gender, age):
        self.conf = conf
        self.data = data
        self.norm = norm
        self.grps = grps
        self.city = city
        self.gender = gender
        self.age = age
        self.eles = eles

    def __len__(self):
        return len(self.data)

    def return_one(self, idx):
        each = self.data[idx]
        each_input = torch.Tensor(each[:self.conf["input_len"]])
        each_output = torch.Tensor(each[self.conf["input_len"]:])
        each_grp = torch.LongTensor([self.grps[idx]])
        each_city = torch.LongTensor([self.city[idx]])
        each_gender = torch.LongTensor([self.gender[idx]])
        each_age = torch.LongTensor([self.age[idx]])
        each_ele = torch.LongTensor([self.eles[idx]])
        each_norm = torch.FloatTensor(self.norm[idx])

        return [each_input, each_output, each_grp, each_ele, each_norm, each_city, each_gender, each_age]

    def __getitem__(self, idx):
        self_cont = self.return_one(idx)
        return self_cont


class TrendData(Dataset):
    def __init__(self, conf):
        self.conf = conf
        self.adj, self.grp_id_map, self.ele_id_map = self.load_affiliation_adj()
        self.dist_mat = self.load_dist_mat()
        trends, grp_ids, ele_ids, self.time_num, trend_norm, city_ids, gender_ids, age_ids = self.get_ori_data()
        train_data, train_ids, train_norm, train_grps, train_eles, test_data, test_ids, test_norm, test_grps, test_eles, train_city, train_gender, train_age, test_city, test_gender, test_age = self.preprocess_data(trends, trend_norm, grp_ids, ele_ids, city_ids, gender_ids, age_ids)
        print(train_grps.shape, train_city.shape, train_gender.shape, train_age.shape)
        self.train_set = TrendDataset(conf, train_data, train_ids, train_norm, train_grps, train_eles, train_city, train_gender, train_age, self.dist_mat)
        self.train_loader = DataLoader(self.train_set, batch_size=conf["batch_size"], shuffle=True, num_workers=10)
        self.test_set = TrendTestDataset(conf, test_data, test_norm, test_grps, test_eles, test_city, test_gender, test_age)
        self.test_loader = DataLoader(self.test_set, batch_size=conf["batch_size"], shuffle=False, num_workers=10)

    def load_affiliation_adj(self):
        ele_id, grp_id = 0, 0
        ele_id_map, grp_id_map = {}, {}
        ori_adj = json.load(open(self.conf["group_affiliation_adj_path"]))
        for group, adj_data in ori_adj.items():
            grp_id_map[group] = grp_id
            grp_id += 1
            for k, res in adj_data.items():
                if k not in ele_id_map:
                    ele_id_map[k] = ele_id
                    ele_id += 1
                for v, score in res.items():
                    if v not in ele_id_map:
                        ele_id_map[v] = ele_id
                        ele_id += 1

        single_adj = json.load(open(self.conf["affiliation_adj_path"]))
        adj = np.zeros((len(ele_id_map), len(ele_id_map)), dtype=float)
        for k, res in single_adj.items():
            k_id = ele_id_map[k]
            for v, score in res.items():
                v_id = ele_id_map[v]
                adj[v_id][k_id] = score
        return adj, grp_id_map, ele_id_map 

    def load_affiliation_adj_old(self):
        ele_id, grp_id = 0, 0
        ele_id_map, grp_id_map = {}, {}
        ori_adj = json.load(open(self.conf["group_affiliation_adj_path"]))
        for group, adj_data in ori_adj.items():
            grp_id_map[group] = grp_id
            grp_id += 1
            for k, res in adj_data.items():
                if k not in ele_id_map:
                    ele_id_map[k] = ele_id
                    ele_id += 1
                for v, score in res.items():
                    if v not in ele_id_map:
                        ele_id_map[v] = ele_id
                        ele_id += 1
        adj = np.zeros((len(grp_id_map), len(ele_id_map), len(ele_id_map)), dtype=float)
        for group, adj_data in ori_adj.items():
            g_id = grp_id_map[group]
            for k, res in adj_data.items():
                k_id = ele_id_map[k]
                for v, score in res.items():
                    v_id = ele_id_map[v]
                    #adj[g_id][k_id][v_id] = score
                    adj[g_id][v_id][k_id] = score
        return adj, grp_id_map, ele_id_map 


    def load_dist_mat(self):
        dist_mat = np.load(self.conf["dist_mat_path"])
        return dist_mat


    def get_ori_data(self):
        all_data = json.load(open(self.conf["data_path"]))
        all_data_norm = json.load(open(self.conf["data_norm_path"]))

        city_id_map, gender_id_map, age_id_map = {}, {}, {"null": 0}
        trends, grp_ids, ele_ids, trend_norm, city_ids, gender_ids, age_ids = [], [], [], [], [], [], []
        grp_ele_id, idx = {}, 0
        time_num = 0
        city_id_g, gender_id_g, age_id_g = 0, 0, 1
        for group_name, res in all_data.items():
            comps = group_name.split("__")
            city_id, gender_id, age_id = 0, 0, 0
            for each in comps:
                if "city:" in each:
                    if each not in city_id_map:
                        city_id_map[each] = city_id_g
                        city_id_g += 1
                    city_id = city_id_map[each]
                if "gender:" in each:
                    if each not in gender_id_map:
                        gender_id_map[each] = gender_id_g
                        gender_id_g += 1
                    gender_id = gender_id_map[each]
                if "age:" in each:
                    if each not in age_id_map:
                        age_id_map[each] = age_id_g
                        age_id_g += 1
                    age_id = age_id_map[each]

            curr_grp_id = self.grp_id_map[group_name]
            grp_ele_id[group_name] = {}
            for fashion_ele, seq in res.items():
                time_seq = [x[0] for x in seq]
                each_time_num = max(time_seq) + 1
                if each_time_num > time_num:
                    time_num = each_time_num
                curr_ele_id = self.ele_id_map[fashion_ele]
                trends.append(seq)

                norm = all_data_norm[group_name][fashion_ele]
                norm = [float(x) for x in norm]
                trend_norm.append(norm)

                grp_ids.append(curr_grp_id)
                city_ids.append(city_id)
                gender_ids.append(gender_id)
                age_ids.append(age_id)
                ele_ids.append(curr_ele_id)

                grp_ele_id[group_name][fashion_ele] = idx
                idx += 1

        trends = np.array(trends)
        json.dump(grp_ele_id, open("./log/%s_%d/test_grp_ele_id_map.json" %(self.conf["dataset"], self.conf["output_len"]), "w"), indent=4)
        json.dump(city_id_map, open("city_id_map.json", "w"), indent=4)
        json.dump(gender_id_map, open("gender_id_map.json", "w"), indent=4)
        json.dump(age_id_map, open("age_id_map.json", "w"), indent=4)

        self.city_id_map = city_id_map
        self.gender_id_map = gender_id_map
        self.age_id_map = age_id_map
        self.all_train_seq = trends[:, :-self.conf["output_len"], 1]

        return trends, np.array(grp_ids), np.array(ele_ids), time_num, np.array(trend_norm), np.array(city_ids), np.array(gender_ids), np.array(age_ids)

    def preprocess_data(self, trends, trend_norm, grp_ids, ele_ids, city_ids, gender_ids, age_ids):
        ori_seq_len = trends.shape[1]
        ttl_len = self.conf["input_len"] + self.conf["output_len"]
        output_len = self.conf["output_len"]
        assert ori_seq_len > ttl_len + output_len
        train_data, train_ids, train_grps, train_eles, train_norm, train_city, train_gender, train_age = [], [], [], [], [], [], [], []
        for i in range(ori_seq_len-ttl_len-output_len):
            train_data.append(trends[:, i:i+ttl_len])
            train_ids.append(np.array([j for j in range(trends.shape[0])]))
            train_norm.append(trend_norm)
            train_grps.append(grp_ids)
            train_eles.append(ele_ids)
            train_city.append(city_ids)
            train_gender.append(gender_ids)
            train_age.append(age_ids)
        train_data = np.concatenate(train_data, axis=0)
        train_ids = np.concatenate(train_ids, axis=0)
        train_norm = np.concatenate(train_norm, axis=0)
        train_grps = np.concatenate(train_grps, axis=0)
        train_city = np.concatenate(train_city, axis=0)
        train_gender = np.concatenate(train_gender, axis=0)
        train_age = np.concatenate(train_age, axis=0)
        train_eles = np.concatenate(train_eles, axis=0)

        test_ids = np.array([j for j in range(trends.shape[0])])
        test_data = trends[:, -ttl_len:]
        test_norm = trend_norm[:, -ttl_len:]
        test_grps = grp_ids
        test_city = city_ids
        test_gender = gender_ids
        test_age = age_ids
        test_eles = ele_ids
        print("train data: ", train_data.shape)
        print("test data: ", test_data.shape)
        return train_data, train_ids, train_norm, train_grps, train_eles, test_data, test_ids, test_norm, test_grps, test_eles, train_city, train_gender, train_age, test_city, test_gender, test_age
