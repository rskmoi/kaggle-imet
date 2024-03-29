#! /usr/bin/python3
# -*- coding: utf-8 -*-
from builtins import range
import torch
from data.dataset import ImetDataset
from tqdm import tqdm
import numpy as np
import click
from train_module.cyclic_scheduler import CyclicLRWithRestarts
from train_module.loss import FocalLoss
from model.model import get_model
from config.config import load_config
from validate import valid_one_epoch
import os
from util.util import make_output_dir
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(train_loader, model, optimizer, scheduler, criterion, epoch, out):
    model.train()
    train_pbar = tqdm(train_loader, total=len(train_loader))
    for sample in train_pbar:
        images, labels = sample['image'].type(torch.FloatTensor).to(DEVICE), \
                         sample['y'].type(torch.FloatTensor).to(DEVICE)

        images, labels_a, labels_b, lam = mixup(images, labels)
        optimizer.zero_grad()
        logits = model(images)
        loss = mixup_loss(criterion, logits, labels_a, labels_b, lam)
        loss.backward()
        optimizer.step()
        scheduler.batch_step()
        train_pbar.set_postfix(loss=loss.data.cpu().numpy(), epoch=epoch)

    torch.save(model.state_dict(), os.path.join(out, "{}epoch.pth".format(epoch)))


@click.command()
@click.option("--config_path", default="", help="path to config yml file")
@click.option("-f", is_flag=True, help="Delete output directory if already exists.")
def train(config_path, f):
    config = load_config(config_path)
    make_output_dir(config, f)

    train_loader = ImetDataset(batch_size=config.train.batch_size,
                               mode="train",
                               img_size=config.data.img_size,
                               train_csv=config.data.train_csv).get_loader()
    valid_loader = ImetDataset(batch_size=config.eval.batch_size,
                               mode="valid",
                               img_size=config.data.img_size,
                               valid_csv=config.data.valid_csv).get_loader()
    model = get_model(config.model.name, config.model.pretrained_model_path, config.model.multi)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    train_total = len(train_loader.dataset)
    scheduler = CyclicLRWithRestarts(optimizer, config.train.batch_size, train_total, restart_period=2, t_mult=1.)
    criterion = FocalLoss()

    for epoch in range(config.train.num_epochs):
        scheduler.step()
        train_one_epoch(train_loader, model, optimizer, scheduler, criterion, epoch, config.path.out)
        valid_one_epoch(valid_loader, model, epoch)

if __name__ == '__main__':
    train()
