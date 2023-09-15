import os
import torch
import torch.nn as nn
import utils.comm as comm
from torchvision.models import resnet50
from torch.nn.parallel import DistributedDataParallel


def build_backbone(cfg):
    model = resnet50()
    if cfg["resume"]:
        weight_path = cfg["resume"]
        assert os.path.exists(weight_path), "file {} does not exist.".format(weight_path)
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    # change fc layer structure
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, len(cfg["names"]))
    return model.cuda()