import os
import torch
import torch.nn as nn
# from backbone.resnet.resnet import resnet50
# from backbone.alexnet.alexnet import AlexNet
from structures.register import Registry

BACKBONE_REGISTRY = Registry("BACKBONE")


def build_backbone(cfg):
    backbone_name = cfg["BACKBONE"]
    include_top = True if cfg["task"] == "detect" else False
    model = BACKBONE_REGISTRY.get(backbone_name)(
        num_classes=len(cfg["names"]), include_top=include_top)
    if cfg["resume"]:
        weight_path = cfg["resume"]
        assert os.path.exists(
            weight_path), "file {} does not exist.".format(weight_path)
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    # assert isinstance(backbone, Backbone)
    return model.cuda()

    # model = resnet50(num_classes=len(cfg["names"]))
    # if cfg["resume"]:
    #     weight_path = cfg["resume"]
    #     assert os.path.exists(
    #         weight_path), "file {} does not exist.".format(weight_path)
    #     model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    # change fc layer structures
    # in_channel = model.fc.in_features
    # model.fc = nn.Linear(in_channel, len(cfg["names"]))
    # return
