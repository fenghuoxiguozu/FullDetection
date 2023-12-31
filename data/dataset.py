import torch
import os
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, dataloader
from .sampler import TrainSampler, InferSampler
from structures.instance import Instances, Bbox, Mask, ImageX, Label
from data.transforms import *


class ImagenetDataset(Dataset):
    def __init__(self, cfg, mode, transform):
        image_path = cfg[mode]
        label_dict = {value: key for key, value in cfg["names"].items()}
        self.image_list = []
        self.label_list = []
        for label_folder in os.listdir(image_path):
            label_path = os.path.join(image_path, label_folder)
            if os.path.isdir(label_path):
                images = glob.glob(label_path+'/*.[jb][pm][gp]')
                for image in images:
                    self.image_list.append(image)
                    self.label_list.append(label_dict[label_folder])
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        im = Image.open(img_path)
        if self.transform:
            im = self.transform(im)
        return {"im": im, "label": self.label_list[idx]}

    @staticmethod
    def collate_fn(batch):  # [{"im",CHW},{"label":INT}...]
        images = torch.stack([item["im"] for item in batch], dim=0)
        labels = torch.as_tensor([item["label"] for item in batch])
        return images, labels


class DetectDataset(Dataset):
    def __init__(self, cfg, mode, transform):
        image_path = cfg[mode]
        self.label_dict = {value: key for key, value in cfg["names"].items()}
        self.image_list = glob.glob(image_path+'/*.[jb][pm][gp]')
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label_path = img_path.replace("images", "labels").replace("jpg", "txt")
        with open(label_path, 'r') as f:
            txts = f.readlines()

        bboxes, labels = [], []
        for item in txts:
            result = item.replace('\n', '').split(' ')
            label, masks = result[0], result[1:]
            bboxes.append([float(mask) for mask in masks])
            labels.append(int(label))

        instance = Instances(image=ImageX(path=img_path), bboxes=Bbox(
            data=np.array(bboxes, dtype=np.float32), format='xywh'), labels=Label(data=np.array(labels, dtype=np.float32)))
        if self.transform:
            im = self.transform(instance)
        return instance

    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: instance List
        Return:
            images: tensor BCHW
            labels: tensor List
        """
        images = torch.stack([item._image.data for item in batch], dim=0)
        labels = torch.concatenate([item._labels.data for item in batch])
        return images, labels


data_transform = {
    "train": Compose([letterBox(imgsz=224), ToTensor()]),
    "val": Compose([letterBox(imgsz=224), ToTensor()])
}


def create_dataloader(cfg, mode):
    assert cfg["batch_size"] % cfg["world_size"] == 0, "batch_size 必须为world_size的倍数"
    # batch = cfg["batch_size"] // cfg["world_size"]
    dataset = DetectDataset(cfg, mode, transform=data_transform[mode])
    # if mode=="train":
    #     sampler = TrainSampler(size=len(dataset))
    #     batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch, drop_last=True)

    # elif mode=="val":
    #     sampler = InferSampler(size=len(dataset))
    #     batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    # return torch.utils.data.DataLoader(dataset,num_workers=8,batch_sampler=batch_sampler,collate_fn=dataset.collate_fn)
    return torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=8, collate_fn=dataset.collate_fn)
