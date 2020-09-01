# This is the repo for paper: Knowledge Enhanced Neural Fashion Trend Forecasting

## Code Structure

1. The entry script is: train.py
2. The config file is: config.yaml
3. utility.py: the script for dataloader
4. model: the folder for model files

## How to Run
1. Download the [dataset](https://drive.google.com/open?id=1OtwOoHYMuLKy_Yjk-_rgjJL5uMWWhPn8), decompress it and put it on the top directory: tar -zxvf dataset.tgz

Note that, the downloaded files include both proprecessed datasets of GeoStyle and FIT. If you want to download the original GeoStyle dataset and reproduce the preprocessing, kindly run the script 0\_preprocess\_data.py. Detailed introduction is within the script.
2. Change the hyper-parameters in the configure file config.yaml.
3. Run: train.py


### Acknowledgement
This project is supported by the National Research Foundation, Prime Minister's Office, Singapore under its IRC@Singapore Funding Initiative.

<img src="https://github.com/mysbupt/KERN/blob/master/next.png" width = "297" height = "100" alt="next" align=center />
