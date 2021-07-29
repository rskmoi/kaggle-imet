import os
import glob
import pickle
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class _Dataset(Dataset):
    def __init__(self, mode='train', tta_zoom=1.0, img_size=256, **kwargs):
        if mode == "train":
            train_csv = kwargs.get("train_csv", "./data/train_split_90pc.csv")
            self.df = pd.read_csv(train_csv)[-5000:]
        elif mode == "valid":
            valid_csv = kwargs.get("valid_csv", "./data/valid_split_10pc.csv")
            self.df = pd.read_csv(valid_csv)
        elif mode == "test":
            test_dir = kwargs.get("test_dir", "../input/test")
            self.df = self._get_test_df(test_dir)
        else:
            raise ValueError()

        self.mode = mode
        self.tta_zoom = tta_zoom
        self.img_size = img_size
        self.transform = self.make_transform(mode, tta_zoom)
        self.images = self._load_img_on_memory(self.df)
        print("Created Dataset. mode: {} files: {}".format(self.mode, len(self.df)))
    
    def _load_img_on_memory(self, df):
        images = []
        dir_image = "train"
        for k, v in tqdm(df.iterrows()):
            img_path = "../input/{}/{}.png".format(dir_image, v["id"])
            img = Image.open(img_path)
            images.append(img)
        return images

    def _get_test_df(self, test_dir):
        path_tests = sorted(glob.glob(os.path.join(test_dir, "*.png")))
        df_base = []
        for cur_path in path_tests:
            df_base.append({"id": os.path.basename(cur_path).replace(".png", "")})

        return pd.DataFrame(df_base)

    def make_transform(self, mode, tta_zoom):
        if mode == "train":
            return transforms.Compose([SquarePad(),
                                       transforms.Resize((self.img_size, self.img_size)),
                                       transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.66, 1.0), ratio=(1.0, 1.0)),
                                       transforms.RandomHorizontalFlip(),
                                       torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.8, saturation=0.5, hue=0.01),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           [0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])
                                       ])
        else:
            if tta_zoom == 1.0:
                return transforms.Compose([SquarePad(),
                                           transforms.Resize((self.img_size, self.img_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               [0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])
                                           ])
            else:
                return transforms.Compose([SquarePad(),
                                           transforms.Resize((self.img_size, self.img_size)),
                                           transforms.CenterCrop((self.img_size * tta_zoom, self.img_size * tta_zoom)),
                                           transforms.Resize((self.img_size, self.img_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               [0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])
                                           ])

    def __len__(self):
        return len(self.df)

    def make_label(self, dense_label):
        label = np.zeros((1103,), dtype=np.uint8)
        dense_label_split = dense_label.split(" ")
        for cur_label in dense_label_split:
            label[int(cur_label)] = 1.

        assert label.shape == (1103,)
        return label

    def __getitem__(self, idx):
        """
        train modeでは画像とカテゴリID、test modeでは画像とkey_idを返す
        :param idx:
        :return:
        """
        data = self.df.iloc[idx]
        sample = {}

        #dir_image = "train" if self.mode in ["train", "valid"] else "test"
        #img_path = "../input/{}/{}.png".format(dir_image, data["id"])
        #image = Image.open(img_path)
        image = self.images[idx]
        image = image.convert("RGB")

        if self.mode == 'train':
            sample['y'] = self.make_label(data['attribute_ids'])
        elif self.mode == "valid":
            sample['y'] = self.make_label(data['attribute_ids'])
        elif self.mode == 'test':
            sample['id'] = data["id"]
        
        try:
            sample['image'] = self.transform(image)
        except:
            pass
        return sample


class SquarePad:
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size[0], img.size[1]
        p = int(abs(h - w) / 2)
        if h < w:
            padding = (p, 0, p, 0)
        else:
            padding = (0, p, 0, p)
        m = torchvision.transforms.Pad(padding, padding_mode='edge')
        return m(img)


class ImetDataset:
    def __init__(self, batch_size, mode, tta_zoom=1.0, img_size=256, **kwargs):
        self.batch_size = batch_size
        self.num_workers = 4
        self.mode = mode
        self.dataset = _Dataset(mode=mode, tta_zoom=tta_zoom, img_size=img_size, **kwargs)

    def get_loader(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size,
                            shuffle=(self.mode == "train"), num_workers=self.num_workers)
        return loader


if __name__ == '__main__':
    data_loader = ImetDataset(batch_size=128, mode="train").get_loader()

    for sample in data_loader:
        images_arr = sample["image"].data.numpy()
        labels_arr = sample["y"].data.numpy()
        print(images_arr.shape)
        print(labels_arr.shape)

