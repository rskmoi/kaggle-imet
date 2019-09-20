#! /usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import click
from tqdm import tqdm
import numpy as np
import pandas as pd
from model.model import get_model
from util.util import make_output_dir
from config.config import load_config
from data.dataset import ImetDataset
import os
import glob
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference_for_submit(test_loader, model, img_size, pickle_name):
    inference_results = []
    with torch.no_grad():
        sigmoid = torch.nn.Sigmoid()
        valid_pbar = tqdm(test_loader, total=len(test_loader))
        for sample in valid_pbar:
            model.eval()
            images, ids = sample['image'].type(torch.FloatTensor).to(DEVICE), sample['id']
            logits = sigmoid(model(images))
            flipped = images[:,:,:,torch.arange(img_size-1, -1, -1)]
            logits_tta = sigmoid(model(flipped))
            logits = (logits + logits_tta) / 2.
            logits_arr = logits.data.cpu().numpy()
            for i in range(len(ids)):
                inference_results.append({"id": ids[i],
                                          "logit": logits_arr[i]})

    if not ".pickle" in pickle_name:
        pickle_name = "{}.pickle".format(pickle_name)
    pd.DataFrame(inference_results)[["id", "logit"]].to_pickle(pickle_name)


def get_average_dataframe(dir_pickles):
    path_pickles = glob.glob(os.path.join(dir_pickles, "*.pickle"))
    df_mean = pd.DataFrame()
    df_logit = None
    for cur_path in path_pickles:
        df = pd.read_pickle(cur_path)
        if df_logit is None:
            df_logit = df
            df_mean["id"] = df["id"]
        else:
            df_logit += df

    df_mean["logit"] = df_logit / len(path_pickles)
    return df_mean


@click.group()
def cmd():
    pass


@cmd.command()
@click.option("--config_path", default="", help="path to config yml file")
@click.option("--use_zoom_tta", is_flag=True, help="tta with different zoom")
@click.option("-f", is_flag=True, help="Delete output directory if already exists.")
def inference(config_path, use_zoom_tta, f):
    config = load_config(config_path)
    model = get_model(config.model.name, config.model.pretrained_model_path)
    make_output_dir(config, f)

    if use_zoom_tta:
        tta_zoom_list = [1.0, 0.9, 0.8]
    else:
        tta_zoom_list = [config.data.tta_zoom]

    for tta_zoom in tta_zoom_list:
        test_dataset = ImetDataset(batch_size=config.eval.batch_size, mode="test",
                                   img_size=config.data.img_size, tta_zoom=tta_zoom,
                                   valid_csv=config.data.valid_csv).get_loader()

        pickle_path = os.path.join(config.path.out,
                                   "{}_{}_{}.pickle".format(config.model.name,
                                                            os.path.basename(config.model.pretrained_model_path),
                                                            tta_zoom))

        inference_for_submit(test_dataset, model, config.data.img_size, pickle_name=pickle_path)


@cmd.command()
@click.option("--dir_pickles", help="path to inference result")
@click.option("--threshold", help="confidence threshold for making submission file")
@click.option("--csv_name", default="submission.csv", help="csv file")
def submit(dir_pickles, threshold, csv_name):
    results = []
    df = get_average_dataframe(dir_pickles)
    submit_pbar = tqdm(df.iterrows(), total=len(df))
    for _, v in submit_pbar:
        pred = ' '.join(list(map(str, np.where(v["logit"] > threshold)[0].tolist())))
        results.append({"id": v["id"],
                        "attribute_ids": pred})

    if ".csv" not in csv_name:
        csv_name = "{}.csv".format(csv_name)
    pd.DataFrame(results)[["id", "attribute_ids"]].to_csv(csv_name, index=False)


if __name__ == '__main__':
    cmd()