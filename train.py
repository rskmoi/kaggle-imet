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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader

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


def train_one_epoch(rank, train_loader, model, optimizer, scheduler, criterion, epoch, out):
    model.train()
    train_pbar = tqdm(train_loader, total=len(train_loader))
    for sample in train_pbar:
        images, labels = sample['image'].type(torch.FloatTensor).to(rank), \
                         sample['y'].type(torch.FloatTensor).to(rank)

        images, labels_a, labels_b, lam = mixup(images, labels)
        optimizer.zero_grad()
        logits = model(images)
        loss = mixup_loss(criterion, logits, labels_a, labels_b, lam)
        loss.backward()
        optimizer.step()
        scheduler.batch_step()
        train_pbar.set_postfix(loss=loss.data.cpu().numpy(), epoch=epoch)

    torch.save(model.state_dict(), os.path.join(out, "{}epoch.pth".format(epoch)))


def _train(rank, batch_size, img_size, train_csv, model_name, pretrained_model_path, model_multi, lr, num_epochs, out):
    n_gpu = 1
    dist.init_process_group("gloo", rank=rank, world_size=n_gpu)
    train_dataset = ImetDataset(batch_size=batch_size,
                                mode="train",
                                img_size=img_size,
                                train_csv=train_csv).dataset
    # valid_dataset = ImetDataset(batch_size=config.eval.batch_size,
    #                             mode="valid",
    #                             img_size=config.data.img_size,
    #                             valid_csv=config.data.valid_csv).dataset
    model = get_model(model_name, pretrained_model_path, model_multi)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sampler = DistributedSampler(train_dataset, num_replicas=n_gpu, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    train_total = len(train_dataset)
    scheduler = CyclicLRWithRestarts(optimizer, batch_size, train_total, restart_period=2, t_mult=1.)
    criterion = FocalLoss()

    for epoch in range(num_epochs):
        scheduler.step()
        train_one_epoch(rank, train_loader, model, optimizer, scheduler, criterion, epoch, out)
        # valid_one_epoch(valid_loader, model, epoch)


@click.command()
@click.option("--config_path", default="", help="path to config yml file")
@click.option("-f", is_flag=True, help="Delete output directory if already exists.")
def train(config_path, f):
    config = load_config(config_path)
    make_output_dir(config, f)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print(config)
    batch_size = config.train.batch_size
    img_size = config.data.img_size
    train_csv = config.data.train_csv
    model_name = config.model.name
    pretrained_model_path = config.model.pretrained_model_path
    model_multi = config.model.multi
    lr = config.train.lr
    num_epochs = config.train.num_epochs
    out = config.path.out

    mp.spawn(_train, args=(batch_size, img_size, train_csv, model_name, pretrained_model_path, model_multi, lr, num_epochs, out), nprocs=1, join=True)


if __name__ == '__main__':
    train()