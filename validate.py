#! /usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import click
from tqdm import tqdm
import numpy as np
from sklearn.metrics import fbeta_score
import pandas as pd
from model.model import get_model
from util.util import make_output_dir
from config.config import load_config
from data.dataset import ImetDataset
import os
import glob
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def valid_one_epoch(valid_loader, model, epoch="?", threshold=0.5, flip_tta=False, pickle_name=None):
    model.eval()
    img_size = valid_loader.dataset.img_size
    valid_df_base = []
    with torch.no_grad():
        sigmoid = torch.nn.Sigmoid()
        fbetas = []
        valid_pbar = tqdm(valid_loader, total=len(valid_loader))
        for sample in valid_pbar:
            images, labels = sample['image'].type(torch.FloatTensor).to(DEVICE), \
                             sample['y'].type(torch.FloatTensor).to(DEVICE)
            logits = sigmoid(model(images))
            if flip_tta:
                # horizontal flipをしている
                flipped = images[:, :, :, torch.arange(img_size - 1, -1, -1)]
                logits_tta = sigmoid(model(flipped))
                logits = (logits + logits_tta) / 2.
            fbeta = fbeta_score(y_pred=np.uint8((logits.data.cpu().numpy() > threshold)),
                                y_true=labels.data.cpu().numpy(),
                                beta=2,
                                average="samples")
            fbetas.append(fbeta)
            valid_pbar.set_postfix(fbeta=np.mean(fbetas), epoch=epoch)

            if pickle_name:
                logits_arr = logits.data.cpu().numpy()
                labels_arr = labels.data.cpu().numpy()
                for i in range(len(labels_arr)):
                    valid_df_base.append({"logit": logits_arr[i], "label": labels_arr[i]})

    if pickle_name:
        if not ".pickle" in pickle_name:
            pickle_name = "{}.pickle".format(pickle_name)
        pd.DataFrame(valid_df_base).to_pickle(pickle_name)


def validate_from_results(dir_pickles, threshold):
    df = get_average_dataframe(dir_pickles)
    fbetas = []
    valid_pbar = tqdm(df.iterrows(), total=len(df))
    for _, v in valid_pbar:
        pred = np.uint8((v["logit"] > threshold))
        cur_fbeta = fbeta_score(y_true=v["label"], y_pred=pred, beta=2.)
        fbetas.append(cur_fbeta)
        valid_pbar.set_postfix(fbeta="{:4}".format(np.mean(fbetas)), threshold=threshold)


def get_average_dataframe(dir_pickles):
    path_pickles = glob.glob(os.path.join(dir_pickles, "*.pickle"))
    df_base = None
    for cur_path in path_pickles:
        df = pd.read_pickle(cur_path)
        if df_base is None:
            df_base = df
        else:
            df_base += df

    return df_base / len(path_pickles)


@click.group()
def cmd():
    pass


@cmd.command()
@click.option("--dir_pickles", default="", help="path to inference result")
def search(dir_pickles):
    for i in range(200, 300):
        validate_from_results(dir_pickles, 0.001 * i)


@cmd.command()
@click.option("--config_path", default="", help="path to config yml file")
@click.option("--use_zoom_tta", is_flag=True, help="tta with different zoom")
@click.option("-f", is_flag=True, help="Delete output directory if already exists.")
def validate(config_path, use_zoom_tta, f):
    config = load_config(config_path)
    model = get_model(config.model.name, config.model.pretrained_model_path)
    make_output_dir(config, f)

    if use_zoom_tta:
        tta_zoom_list = [1.0, 0.9, 0.8]
    else:
        tta_zoom_list = [config.data.tta_zoom]

    for tta_zoom in tta_zoom_list:
        valid_loader = ImetDataset(batch_size=config.eval.batch_size, mode="valid",
                                   img_size=config.data.img_size, tta_zoom=tta_zoom,
                                   valid_csv=config.data.valid_csv).get_loader()

        pickle_path = os.path.join(config.path.out,
                                   "{}_{}_{}.pickle".format(config.model.name,
                                                            os.path.basename(config.model.pretrained_model_path),
                                                            tta_zoom))
        valid_one_epoch(valid_loader, model, flip_tta=True, pickle_name=pickle_path)

    search(config.path.out)



if __name__ == '__main__':
    cmd()