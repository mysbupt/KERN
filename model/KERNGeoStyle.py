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


class KERNGeoStyle(nn.Module):
    def __init__(self, conf):
        super(KERNGeoStyle, self).__init__()
        self.conf = conf
        self.enc_input_size = conf["feat_size"]*3 + 1 # grp_emb + ele_emb + time_emb + last_v
        self.dec_input_size = conf["feat_size"]*3 # grp_emb + ele_emb + time_emb
        device = torch.device(conf["device"])
        self.device = device

        self.encoder = nn.LSTM(self.enc_input_size, conf["rnn_hidden_size"])
        self.decoder = nn.LSTM(self.dec_input_size, conf["rnn_hidden_size"], bidirectional=True)

        self.grp_embeds = nn.Embedding(self.conf["grp_num"], self.conf["feat_size"])
        self.ele_embeds = nn.Embedding(self.conf["ele_num"], self.conf["feat_size"])
        self.time_embeds = nn.Embedding(self.conf["time_num"], self.conf["feat_size"])

        self.enc_linear = nn.Linear(conf["rnn_hidden_size"], 1)
        self.dec_linear = nn.Linear(conf["rnn_hidden_size"]*2, 1)
        self.loss_function = nn.L1Loss()


    def predict(self, each_cont): 
        input_seq, output_seq, grp_id, ele_id, _ = each_cont

        grp_id = grp_id.squeeze(1) # [batch_size]
        ele_id = ele_id.squeeze(1) # [batch_size]

        grp_embed = self.grp_embeds(grp_id) #[batch_size, feat_size]
        ele_embed = self.ele_embeds(ele_id) #[batch_size, feat_size]

        enc_time_embed = self.time_embeds(input_seq[:, :, 0].long()) #[batch_size, enc_seq_len, feat_size]
        dec_time_embed = self.time_embeds(output_seq[:, :, 0].long()) #[batch_size, dec_seq_len, feat_size]
        # encode part:
        # input_seq: [batch_size, enc_seq_len, 2]
        enc_seq_len = input_seq.shape[1]
        enc_grp_embed = grp_embed.unsqueeze(1).expand(-1, enc_seq_len, -1) #[batch_size, enc_seq_len, feat_size]
        enc_ele_embed = ele_embed.unsqueeze(1).expand(-1, enc_seq_len, -1) #[batch_size, enc_seq_len, feat_size]

        enc_input_seq = torch.cat([enc_grp_embed, enc_ele_embed, enc_time_embed, input_seq[:, :, 1].unsqueeze(-1)], dim=-1) #[batch_size, enc_seq_len, enc_input_size]
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

        dec_input_seq = torch.cat([dec_grp_embed, dec_ele_embed, dec_time_embed], dim=-1) #[batch_size, dec_seq_len, dec_input_size]
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

        def cal_metric_loss(self_dec_output, close_dec_output, far_dec_output):
            close_dist = (self_dec_output - close_dec_output).pow(2) # [batch_size, seq_len, hidden_size*2]
            close_dist = torch.sqrt(close_dist + 1e-8*torch.ones(close_dist.shape).to(self.device))
            close_dist = torch.sum(close_dist, dim=2) # [batch_size, seq_len]
            far_dist = (self_dec_output - far_dec_output).pow(2) # [batch_size, seq_len, hidden_size*2]
            far_dist = torch.sqrt(far_dist + 1e-8*torch.ones(far_dist.shape).to(self.device))
            far_dist = torch.sum(far_dist, dim=2) # [batch_size, seq_len]
            residual = close_dist - far_dist
            metric_loss = torch.max(torch.zeros(residual.shape).to(self.device), residual) # [batch_size, seq_len]
            metric_loss = torch.mean(torch.mean(metric_loss, dim=1))
            return metric_loss

        enc_metric_loss = cal_metric_loss(self_enc_output, close_enc_output, far_enc_output)
        dec_metric_loss = cal_metric_loss(self_dec_output, close_dec_output, far_dec_output)
        metric_loss = enc_metric_loss + dec_metric_loss

        enc_loss = (self_enc_loss + close_enc_loss + far_enc_loss) / 3
        dec_loss = (self_dec_loss + close_dec_loss + far_dec_loss) / 3

        return enc_loss, dec_loss, metric_loss
