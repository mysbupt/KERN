import os
import json
import yaml
import numpy as np
from model.KERN import KERN
from model.KERNGeoStyle import KERNGeoStyle
import torch
torch.manual_seed(1234)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


def train(conf):
    ori_path = "%s_%d/" %(conf["dataset"], conf["output_len"])
    settings = []
    if conf["use_grp_embed"] is False:
        settings.append("NoGrpEmb")
    if conf["ext_kg"] is True:
        settings.append("ExtKG")
    if conf["int_kg"] is True:
        settings.append("IntKG_lambda:%.6f__SampleRange:%d" %(conf["triplet_lambda"], conf["sample_range"]))
    setting = ori_path + "__".join(settings)
        
    if not os.path.isdir("./runs"):
        os.makedirs("./runs")
    run_dir = "./runs/%s" %(setting)
    log = SummaryWriter(run_dir)
    log_dir = "./log/%s" %(setting)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    if conf["dataset"] == "fit":
        from utility import TrendDataset, TrendData
    else:
        from utility_geostyle import TrendDataset, TrendData

    dataset = TrendData(conf)
    conf["grp_num"] = len(dataset.grp_id_map)
    conf["ele_num"] = len(dataset.ele_id_map)
    conf["time_num"] = dataset.time_num
    if conf["dataset"] == "fit":
        conf["city_num"] = len(dataset.city_id_map)
        conf["gender_num"] = len(dataset.gender_id_map)
        conf["age_num"] = len(dataset.age_id_map)

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    conf["device"] = device

    for k, v in conf.items():
        print(k, v)

    if conf["dataset"] == "fit":
        model = KERN(conf, dataset.adj)
    else:
        model = KERNGeoStyle(conf)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=1e-4)
    #optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=int(conf["lr_decay_interval"]*len(dataset.train_set)/conf["batch_size"]), gamma=conf["lr_decay_gamma"])

    best_epoch, best_batch, best_mae, best_mape = 0, 0, 1, 100
    best_val_mae, best_val_mape, best_test_mae, best_test_mape = 1, 100, 1, 100
    for epoch in range(int(conf["epoch"])):
        print("\nEpoch: %d" %(epoch))
        loss_print, enc_loss_print, dec_loss_print, triplet_loss_print = 0, 0, 0, 0
        pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))
        
        ttl_batch = len(dataset.train_set) / conf["batch_size"]
        for batch_i, batch in pbar:
        #for batch_i, batch in enumerate(dataset.train_loader):
            model.train(True)
            exp_lr_scheduler.step() #adjust learning rate
            optimizer.zero_grad()

            self_cont, close_cont, far_cont, close_score, far_score = batch

            self_cont = [each.to(device) for each in self_cont]
            close_cont = [each.to(device) for each in close_cont]
            far_cont = [each.to(device) for each in far_cont]
            close_score = close_score.to(device)
            far_score = far_score.to(device)

            enc_loss, dec_loss, triplet_loss = model(self_cont, close_cont, far_cont, close_score, far_score)

            if conf["int_kg"] is True:
                loss = enc_loss + dec_loss + conf["triplet_lambda"] * triplet_loss
            else:
                loss = enc_loss + dec_loss
            loss.backward()                                                                                                
            optimizer.step()
            
            loss_print += loss.item()
            enc_loss_print += enc_loss.item()
            dec_loss_print += dec_loss.item()
            triplet_loss_print += triplet_loss.item()

            log.add_scalar('Loss/train', loss.item(), batch_i + epoch * len(dataset.train_set)/conf["batch_size"])
            log.add_scalar('EncLoss/train', enc_loss.item(), batch_i + epoch * len(dataset.train_set)/conf["batch_size"])
            log.add_scalar('DecLoss/train', dec_loss.item(), batch_i + epoch * len(dataset.train_set)/conf["batch_size"])
            log.add_scalar('MetricLoss/train', triplet_loss.item(), batch_i + epoch * len(dataset.train_set)/conf["batch_size"])

            pbar.set_description('L:{:.6f}, EL:{:.6f}, DL:{:.6f}, ML:{:.6f}'.format(loss_print/(batch_i+1), enc_loss_print/(batch_i+1), dec_loss_print/(batch_i+1), triplet_loss_print/(batch_i+1)))
                                                                                                                           
            if (batch_i+1) % int(conf["test_interval"]*ttl_batch) == 0:
                mae, mape, all_grd, all_pred, all_train, val_mae, val_mape, test_mae, test_mape, test_loss_print = evaluate(model, dataset, device, conf)
                log.add_scalar('MAE/all', mae, batch_i + epoch * ttl_batch)
                log.add_scalar('MAPE/all', mape, batch_i + epoch * ttl_batch)
                log.add_scalar('MAE/val', val_mae, batch_i + epoch * ttl_batch)
                log.add_scalar('MAPE/val', val_mape, batch_i + epoch * ttl_batch)
                log.add_scalar('MAE/test', test_mae, batch_i + epoch * ttl_batch)
                log.add_scalar('MAPE/test', test_mape, batch_i + epoch * ttl_batch)
                [loss_final, enc_loss_final, dec_loss_final] = test_loss_print
                log.add_scalar('Loss/test', loss_final, batch_i + epoch * ttl_batch)
                log.add_scalar('EncLoss/test', enc_loss_final, batch_i + epoch * ttl_batch)
                log.add_scalar('DecLoss/test', dec_loss_final, batch_i + epoch * ttl_batch)
                if mae <= best_mae and mape <= best_mape:
                    best_mae = mae
                    best_mape = mape
                    best_val_mae = val_mae
                    best_val_mape = val_mape
                    best_test_mae = test_mae
                    best_test_mape = test_mape
                    best_epoch = epoch
                    best_batch = batch_i
                    np.save("%s/all_grd_%s_%d" %(log_dir, conf["dataset"], conf["output_len"]), all_grd)
                    np.save("%s/all_pred_%s_%d" %(log_dir, conf["dataset"], conf["output_len"]), all_pred)
                    np.save("%s/all_train_%s_%d" %(log_dir, conf["dataset"], conf["output_len"]), all_train)
                    np.save("%s/ele_embed_%s_%d" %(log_dir, conf["dataset"], conf["output_len"]), model.ele_embeds.weight.detach().cpu().numpy())
                    if conf["dataset"] == "fit":
                       np.save("%s/city_embed_%s_%d" %(log_dir, conf["dataset"], conf["output_len"]), model.city_embeds.weight.detach().cpu().numpy())
                       np.save("%s/gender_embed_%s_%d" %(log_dir, conf["dataset"], conf["output_len"]), model.gender_embeds.weight.detach().cpu().numpy())
                       np.save("%s/age_embed_%s_%d" %(log_dir, conf["dataset"], conf["output_len"]), model.age_embeds.weight.detach().cpu().numpy())
                print("MAE: %.6f, MAPE: %.6f, VAL_MAE: %.6f, VAL_MAPE: %.6f, TEST_MAE: %.6f, TEST_MAPE: %.6f" %(mae, mape, val_mae, val_mape, test_mae, test_mape))
                print("BEST in epoch %d batch: %d, MAE: %.6f, MAPE: %.6f, VAL_MAE: %.6f, VAL_MAPE: %.6f, TEST_MAE: %.6f, TEST_MAPE: %.6f" %(best_epoch, best_batch, best_mae, best_mape, best_val_mae, best_val_mape, best_test_mae, best_test_mape))

    for k, v in conf.items():
        print(k, v)


def evaluate(model, dataset, device, conf):
    model.eval()                                                                                                           
    pbar = tqdm(enumerate(dataset.test_loader), total=len(dataset.test_loader))                                                              
    loss_print, enc_loss_print, dec_loss_print = 0, 0, 0
    all_grd, all_pred, all_norm = [], [], []
    for batch_i, batch in pbar:
        each_cont = [each.to(device) for each in batch]
        enc_loss, dec_loss, pred, _, _, _ = model.predict(each_cont)

        loss = enc_loss + dec_loss
        loss_print += loss.item()
        enc_loss_print += enc_loss.item()                                                                                      
        dec_loss_print += dec_loss.item()                                                                                      

        loss_final = loss_print/(batch_i+1)
        enc_loss_final = enc_loss_print/(batch_i+1)
        dec_loss_final = dec_loss_print/(batch_i+1)

        pbar.set_description('L:{:.6f}, EL:{:.6f}, DL:{:.6f}'.format(loss_print/(batch_i+1), enc_loss_print/(batch_i+1), dec_loss_print/(batch_i+1)))

        if conf["dataset"] == "fit":
            [input_seq, output_seq, grp_id, ele_id, norm, city_id, gender_id, age_id] = each_cont 
        else:
            [input_seq, output_seq, grp_id, ele_id, norm] = each_cont 

        all_grd.append(output_seq[:, :, 1].cpu())
        all_pred.append(pred.cpu())
        all_norm.append(norm.cpu())

    all_grd = torch.cat(all_grd, dim=0).numpy()
    all_pred = torch.cat(all_pred, dim=0).detach().numpy()
    all_norm = torch.cat(all_norm, dim=0).detach().numpy()
    all_train = dataset.all_train_seq

    if conf["denorm"] is True:
        all_grd = denorm(all_grd, all_norm)
        all_pred = denorm(all_pred, all_norm)
        all_train = denorm(all_train, all_norm)

    val_pred = all_pred[::2]
    val_grd = all_grd[::2]
    test_pred = all_pred[1::2]
    test_grd = all_grd[1::2]
    mae = np.mean(np.abs(all_pred-all_grd))
    mape = np.mean(np.abs(all_pred-all_grd)/all_grd)*100
    val_mae = np.mean(np.abs(val_pred-val_grd))
    val_mape = np.mean(np.abs(val_pred-val_grd)/val_grd)*100
    test_mae = np.mean(np.abs(test_pred-test_grd))
    test_mape = np.mean(np.abs(test_pred-test_grd)/test_grd)*100
    return mae, mape, all_grd, all_pred, all_train, val_mae, val_mape, test_mae, test_mape, [loss_final, enc_loss_final, dec_loss_final]


def denorm(seq, norms):
    # seq: [num_samples]
    # norms: [num_samples, 3] 2nd-dim: min, max, eps
    #seq = np.min(seq, 1)
    #seq = np.max(seq, 0)
    seq_len = seq.shape[-1]
    min_v = np.expand_dims(norms[:, 0], axis=1).repeat(seq_len, axis=1)
    max_v = np.expand_dims(norms[:, 1], axis=1).repeat(seq_len, axis=1)
    eps = np.expand_dims(norms[:, 2], axis=1).repeat(seq_len, axis=1)
    denorm_res = seq * (max_v - min_v) + min_v
    return denorm_res


def main():
    conf = yaml.load(open("./config.yaml"))
    dataset_name = "geostyle" # options: fit, geostyle
    conf = conf[dataset_name]
    conf["dataset"] = dataset_name

    train(conf)


if __name__ == "__main__":
    main()
