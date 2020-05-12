import os
import json
import yaml
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class KERN(nn.Module):
    def __init__(self, conf, adj=None):
        super(KERN, self).__init__()
        self.conf = conf
        if self.conf["use_grp_embed"] is True:
            self.enc_input_size = conf["feat_size"]*3 + 1 # grp_emb + ele_emb + time_emb + last_v
            self.dec_input_size = conf["feat_size"]*3 # grp_emb + ele_emb + time_emb
        else:
            self.enc_input_size = conf["feat_size"]*2 + 1 # ele_emb + time_emb + last_v
            self.dec_input_size = conf["feat_size"]*2 # ele_emb + time_emb
        device = torch.device(conf["device"])
        self.device = device

        self.encoder = nn.LSTM(self.enc_input_size, conf["rnn_hidden_size"])
        self.decoder = nn.LSTM(self.dec_input_size, conf["rnn_hidden_size"], bidirectional=True)

        self.grp_embeds = nn.Embedding(self.conf["grp_num"], self.conf["feat_size"])
        self.ele_embeds = nn.Embedding(self.conf["ele_num"], self.conf["feat_size"])
        if self.conf["dataset"] == "fit":
            self.city_embeds = nn.Embedding(self.conf["city_num"], self.conf["feat_size"])
            self.gender_embeds = nn.Embedding(self.conf["gender_num"], self.conf["feat_size"])
            self.age_embeds = nn.Embedding(self.conf["age_num"], self.conf["feat_size"])
            self.time_embeds = nn.Embedding(self.conf["time_num"], self.conf["feat_size"])

        self.enc_linear = nn.Linear(conf["rnn_hidden_size"], 1)
        self.dec_linear = nn.Linear(conf["rnn_hidden_size"]*2, 1)
        self.loss_function = nn.L1Loss()

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().to(device)
            self.adj_2 = torch.matmul(self.adj, self.adj) # [grp_num, ele_num, ele_num]
            self.adj_2 = F.normalize(self.adj_2, p=1, dim=1)

        self.city_gender_age_agg = nn.Linear(conf["feat_size"]*3, conf["feat_size"])
        self.agg_prop = nn.Linear(conf["feat_size"]*3, conf["feat_size"])
        self.agg_prog_exc = nn.Linear(conf["feat_size"]*2, conf["feat_size"])


    def predict(self, each_cont): 
        input_seq, output_seq, grp_id, ele_id, _, city_id, gender_id, age_id = each_cont

        grp_id = grp_id.squeeze(1) # [batch_size]
        city_id = city_id.squeeze(1) # [batch_size]
        gender_id = gender_id.squeeze(1) # [batch_size]
        age_id = age_id.squeeze(1) # [batch_size]
        ele_id = ele_id.squeeze(1) # [batch_size]

        city_embed = self.city_embeds(city_id) #[batch_size, feat_size]
        gender_embed = self.gender_embeds(gender_id) #[batch_size, feat_size]
        age_embed = self.age_embeds(age_id) #[batch_size, feat_size]
        ori_ele_embed = self.ele_embeds(ele_id) #[batch_size, feat_size]

        grp_embed = self.city_gender_age_agg(torch.cat([city_embed, gender_embed, age_embed], dim=1))

        # subsidiary relation
        # first order
        if self.conf["ext_kg"] is True:
            ele_embed_prop_1 = torch.matmul(self.adj, self.ele_embeds.weight) # [ele_num, feat_size]
            ele_embed_prop_2 = torch.matmul(self.adj_2, self.ele_embeds.weight) # [ele_num, feat_size]
            ele_embed_prop_1 = ele_embed_prop_1[ele_id, :] #[batch_size, feat_size] 
            ele_embed_prop_2 = ele_embed_prop_2[ele_id, :] #[batch_size, feat_size] 
            ele_embed = self.agg_prop(torch.cat([ori_ele_embed, ele_embed_prop_1, ele_embed_prop_2], dim=1))
        else:
            ele_embed = ori_ele_embed

        enc_time_embed = self.time_embeds(input_seq[:, :, 0].long()) #[batch_size, enc_seq_len, feat_size]
        dec_time_embed = self.time_embeds(output_seq[:, :, 0].long()) #[batch_size, dec_seq_len, feat_size]
        # encode part:
        # input_seq: [batch_size, enc_seq_len, 2]
        enc_seq_len = input_seq.shape[1]
        enc_grp_embed = grp_embed.unsqueeze(1).expand(-1, enc_seq_len, -1) #[batch_size, enc_seq_len, feat_size]
        enc_ele_embed = ele_embed.unsqueeze(1).expand(-1, enc_seq_len, -1) #[batch_size, enc_seq_len, feat_size]

        if self.conf["use_grp_embed"] is True:
            enc_input_seq = torch.cat([enc_grp_embed, enc_ele_embed, enc_time_embed, input_seq[:, :, 1].unsqueeze(-1)], dim=-1) #[batch_size, enc_seq_len, enc_input_size]
        else:
            enc_input_seq = torch.cat([enc_ele_embed, enc_time_embed, input_seq[:, :, 1].unsqueeze(-1)], dim=-1) #[batch_size, enc_seq_len, enc_input_size]
        enc_input_seq = enc_input_seq.permute(1, 0, 2) #[enc_seq_len, batch_size, enc_input_size]

        enc_outputs, (enc_hidden, enc_c) = self.encoder(enc_input_seq) #outputs: [seq_len, batch_size, hidden_size], hidden: [1, batch_size, hidden_size]

        enc_grd = input_seq[:, 1:, 1] #[batch_size, enc_seq_len-1]
        enc_output_feat = enc_outputs.permute(1, 0, 2)[:, 1:, :] #[batch_size, enc_seq_len-1, hidden_size]
        enc_pred = self.enc_linear(enc_output_feat).squeeze(-1) #[batch_size, enc_seq_len-1]

        # decode part:
        # output_seq: [batch_size, dec_seq_len, 2]
        dec_seq_len = output_seq.shape[1]
        dec_grp_embed = grp_embed.unsqueeze(1).expand(-1, dec_seq_len, -1) #[batch_size, dec_seq_len, feat_size]
        dec_ele_embed = ele_embed.unsqueeze(1).expand(-1, dec_seq_len, -1) #[batch_size, dec_seq_len, feat_size]
 
        if self.conf["use_grp_embed"] is True:
            dec_input_seq = torch.cat([dec_grp_embed, dec_ele_embed, dec_time_embed], dim=-1) #[batch_size, dec_seq_len, dec_input_size]
        else:
            dec_input_seq = torch.cat([dec_ele_embed, dec_time_embed], dim=-1) #[batch_size, dec_seq_len, dec_input_size]
        dec_input_seq = dec_input_seq.permute(1, 0, 2) #[dec_seq_len, batch_size, dec_input_size]

        dec_init_hidden = enc_hidden.expand(2, -1, -1) #[2, batch_size, hidden_size]
        dec_init_c = enc_c.expand(2, -1, -1) #[2, batch_size, hidden_size]
        dec_output_feat, _ = self.decoder(dec_input_seq, (dec_init_hidden.contiguous(), dec_init_c.contiguous())) #outputs: [seq_len, batch_size, hidden_size*2]

        dec_output_feat = dec_output_feat.permute(1, 0, 2) # [batch_size, seq_len, hidden_size*2]
        dec_grd = output_seq[:, :, 1] #[batch_size, dec_seq_len]
        dec_pred = self.dec_linear(dec_output_feat).squeeze(-1) #[batch_size, dec_seq_len]

        enc_loss = self.loss_function(enc_pred, enc_grd)
        dec_loss = self.loss_function(dec_pred, dec_grd)

        return enc_loss, dec_loss, dec_pred, enc_hidden.squeeze(0), enc_output_feat, dec_output_feat # [batch_size, hidden_size], [batch_size, enc_seq_len-1, hidden_size], [batch_size, seq_len, hidden_size*2] 


    def forward(self, self_cont, close_cont, far_cont, close_score, far_score):
        self_enc_loss, self_dec_loss, self_pred, self_enc_hidden, self_enc_output, self_dec_output = self.predict(self_cont)
        close_enc_loss, close_dec_loss, close_pred, close_enc_hidden, close_enc_output, close_dec_output = self.predict(close_cont)
        far_enc_loss, far_dec_loss, far_pred, far_enc_hidden, far_enc_output, far_dec_output = self.predict(far_cont)

        close_score = close_score.squeeze(1) # [batch_size]
        far_score = far_score.squeeze(1) # [batch_size]

        def cal_triplet_loss(self_dec_output, close_dec_output, far_dec_output):
            close_dist = (self_dec_output - close_dec_output).pow(2) # [batch_size, seq_len, hidden_size*2]
            close_dist = torch.sqrt(close_dist + 1e-8*torch.ones(close_dist.shape).to(self.device))
            close_dist = torch.sum(close_dist, dim=2) # [batch_size, seq_len]
            far_dist = (self_dec_output - far_dec_output).pow(2) # [batch_size, seq_len, hidden_size*2]
            far_dist = torch.sqrt(far_dist + 1e-8*torch.ones(far_dist.shape).to(self.device))
            far_dist = torch.sum(far_dist, dim=2) # [batch_size, seq_len]
            residual = close_dist - far_dist
            triplet_loss = torch.max(torch.zeros(residual.shape).to(self.device), residual) # [batch_size, seq_len]
            triplet_loss = torch.mean(torch.mean(triplet_loss, dim=1))
            return triplet_loss

        enc_triplet_loss = cal_triplet_loss(self_enc_output, close_enc_output, far_enc_output)
        dec_triplet_loss = cal_triplet_loss(self_dec_output, close_dec_output, far_dec_output)
        triplet_loss = enc_triplet_loss + dec_triplet_loss

        enc_loss = (self_enc_loss + close_enc_loss + far_enc_loss) / 3
        dec_loss = (self_dec_loss + close_dec_loss + far_dec_loss) / 3

        return enc_loss, dec_loss, triplet_loss
