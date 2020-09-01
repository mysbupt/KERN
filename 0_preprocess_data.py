# This script do the preprocessing of the original GeoStyle dataset (not for FIT dataset). It basically consists of two parts:
# 1. Remove the sparse data, which leverages the same procedure with GeoStyle (refer: https://github.com/kavitabala/geostyle/blob/master/trends_events/benchmarking_test.py)
# 2. min-max normalization, which is crucial for training a whole model over all the time series.
# How to run this script:
# If you directly download the preprocessed dataset following the link in README.md, you don't need run this script.
# Otherwise, you need to download the original GeoStyle dataset from https://geostyle.cs.cornell.edu/static/data/metadata.pkl, save it to the current directory. Then run this script: python 0_preprocess_data.py. The processed data will be generated in the path: dataset/GeoStyle

import os
import numpy as np
from os.path import isfile
import pickle
import json


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def dump_geostyle_taxonomy(attributes, categories):
    for i in range(len(attributes)):
        attr = attributes[i]
        print(attr)
        for j in range(len(categories[i])):
            value = categories[i][j]
            print("\t%s" %(value))
        print("\n")


def preprocess_geostyle_data(eps):
    input_file = 'metadata.pkl'
    if not isfile(input_file):
        print("FileNotFound: Download 'metadata.pkl' and place it in the working directory")
        exit()
    data = unpickle(input_file)
    cities = sorted(data['cities'].keys())
    attributes = data['attributes']
    categories = data['categories']

    dump_geostyle_taxonomy(attributes, categories)
    
    new_data = {}
    new_data_norm = {}
    
    first_iter = True
    for i in range(len(attributes)):
        attr = attributes[i]
        for j in range(len(categories[i])):
            value = categories[i][j]
            attr_value = "__".join([attr, value])
            for cind, city in enumerate(cities):
                pos_tot = []
                datum = data['classifications'][city]
                # remove weeks with small amount of data from start and end
                if first_iter:
                    weeks = sorted(datum.keys())
                    weeks = weeks[5:-5]
                    first_iter = False
                for week in weeks:
                    week_num = int(week.split("_")[1]) - 1
                    pos_tot.append([np.sum(datum[week][:, i] == j), datum[week].shape[0], week_num])
    
                timestep, trend = [], []
                for k in range(len(pos_tot)):
                    if pos_tot[k][0] == 0:
                        pos_tot[k][0] = 1
                    elif pos_tot[k][0] == pos_tot[k][1]:
                        pos_tot[k][0] = pos_tot[k][0]-1
                    #timestep.append(int(pos_tot[k][2]/2))
                    timestep.append(int(pos_tot[k][2]))
                    trend.append(pos_tot[k][0]/float(pos_tot[k][1]))

                max_v = max(trend)
                min_v = min(trend)
                normed_trend = [max((x-min_v)/(max_v-min_v), eps) for x in trend]
                res = []
                for time_s, trend_v in zip(timestep, normed_trend):
                    res.append([time_s, trend_v])
    
                if city not in new_data:
                    new_data[city] = {attr_value: res}
                    new_data_norm[city] = {attr_value: [min_v, max_v, eps]}
                else:
                    new_data[city][attr_value] = res
                    new_data_norm[city][attr_value] = [min_v, max_v, eps]

    out_path = "./dataset/GeoStyle"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    json.dump(new_data, open(os.path.join(out_path, "geostyle_data.json"), "w"))
    json.dump(new_data_norm, open(os.path.join(out_path, "geostyle_data_norm.json"), "w"))


def main():
    eps=0.01
    preprocess_geostyle_data(eps)


if __name__ == "__main__":
    main()
